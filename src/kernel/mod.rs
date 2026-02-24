pub mod agent_manager;
pub mod builder;
pub mod config;
pub mod event;
mod event_persistence;
mod harness_hooks;
mod mcp_runtime;
pub mod identity;
mod init;
pub mod policy;
pub mod session;
mod session_lifecycle;
mod run_loop;
mod task_lifecycle;
mod task_planning;
mod turn;

use agent_manager::AgentManager;
use anyhow::Result;
use builder::RuntimeBuilder;
use config::TurinConfig;
use event::TaskTerminalStatus;
use session::{QueuedTask, SessionState};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{error, info, instrument, warn};

use crate::harness::engine::HarnessEngine;
use crate::inference::embeddings::EmbeddingProvider;
use crate::inference::provider::{
    InferenceContent, InferenceMessage, InferenceRole, ProviderClient,
};
use crate::kernel::policy::RuntimePolicyManager;
use crate::persistence::manager::StoreManager;

use crate::tools::ToolContext;
use crate::tools::registry::ToolRegistry;
use mcp_runtime::McpClientEntry;
use notify::RecommendedWatcher;

/// The Turin Kernel — manages the agent loop, event system, and tool execution.
///
/// The Kernel has no opinions about agent behavior. It provides the physics:
/// transport, streaming, tool execution, persistence, and event hooks.
/// Harness scripts define the behavior.
pub struct Kernel {
    pub(crate) config: Arc<TurinConfig>,
    pub(crate) json: bool,
    pub(crate) tool_registry: ToolRegistry,
    pub(crate) store_manager: Arc<StoreManager>,
    pub(crate) agent_manager: Arc<AgentManager>,
    pub(crate) policy_manager: Arc<RuntimePolicyManager>,
    /// Thread-safe harness engine for hot-reloading
    pub(crate) harness: Arc<std::sync::Mutex<Option<HarnessEngine>>>,
    /// Watcher handle to keep it alive
    pub(crate) check_watcher: Option<RecommendedWatcher>,
    pub(crate) clients: HashMap<String, ProviderClient>,
    pub(crate) embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    /// Active session queue for harness interaction
    pub(crate) active_queue: crate::harness::globals::ActiveSessionQueue,

    pub(crate) mcp_clients: Vec<McpClientEntry>,
}

/// A pending tool call collected during streaming.
#[derive(Debug, Clone)]
pub(crate) struct PendingToolCall {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TaskExecutionResult {
    pub status: TaskTerminalStatus,
    pub task_turn_count: u32,
}

impl Kernel {
    /// Create a new builder for Kernel.
    pub fn builder(config: TurinConfig) -> RuntimeBuilder {
        RuntimeBuilder::new(config)
    }

    /// Access the store manager.
    pub fn store_manager(&self) -> &Arc<StoreManager> {
        &self.store_manager
    }

    /// Access the runtime policy manager.
    pub fn policy_manager(&self) -> &Arc<RuntimePolicyManager> {
        &self.policy_manager
    }

    /// Access the configuration.
    pub fn config(&self) -> &TurinConfig {
        &self.config
    }

    /// Lock the harness mutex.
    ///
    /// Panics if the mutex is poisoned (previous holder panicked).
    /// A poisoned harness is an unrecoverable state — continuing would
    /// risk executing tool calls with a partially-updated engine.
    pub fn lock_harness(&self) -> std::sync::MutexGuard<'_, Option<HarnessEngine>> {
        self.harness.lock().expect("harness mutex poisoned")
    }

    /// Get names of all loaded harness scripts.
    pub fn loaded_scripts(&self) -> Vec<String> {
        let lock = self.lock_harness();
        if let Some(ref engine) = *lock {
            engine.loaded_scripts().to_vec()
        } else {
            Vec::new()
        }
    }

    /// Add a provider client manually (e.g. for testing).
    pub fn add_client(&mut self, name: String, client: ProviderClient) {
        self.clients.insert(name, client);
    }

    /// Run a Lua script directly in the harness (for testing/verification).
    pub fn run_script(&self, script: &str) -> Result<()> {
        let mut harness_lock = self.lock_harness();
        if let Some(ref mut engine) = *harness_lock {
            engine.load_script_str(script)?;
        } else {
            anyhow::bail!("Harness not initialized");
        }
        Ok(())
    }

    /// Run the agent loop with the given prompt.
    #[instrument(skip(self, session), fields(session_id = %session.identity.session_id()))]
    pub async fn run(&mut self, session: &mut SessionState, prompt: Option<String>) -> Result<()> {
        // Ensure session is started
        self.start_session(session).await?;

        // Set active queue for harness
        {
            let mut aq = self.active_queue.lock().await;
            *aq = Some(session.queue.clone());
        }

        if let Some(p) = prompt {
            self.enqueue_initial_run_prompt(session, p).await;
        }

        while let Some((mut task, queue_depth_after_pop)) = self.dequeue_next_task(session).await {
            if !self
                .prepare_task_start(session, &mut task, queue_depth_after_pop)
                .await?
            {
                continue;
            }

            info!(task_id = %task.task_id, prompt = %task.prompt, "Running task");

            let task_result = match self.run_task(session, &task).await {
                Ok(result) => result,
                Err(e) => {
                    error!(task_id = %task.task_id, error = %e, "Task failed with runtime error");
                    let error_message = e.to_string();
                    let recovered = self
                        .handle_inference_error(session, &task, &error_message)
                        .await?;
                    self.complete_task(
                        session,
                        &task,
                        TaskTerminalStatus::Error,
                        0,
                        Some(error_message),
                    )
                    .await?;
                    if recovered {
                        continue;
                    }
                    return Err(e);
                }
            };

            self.complete_task(
                session,
                &task,
                task_result.status,
                task_result.task_turn_count,
                None,
            )
            .await?;
        }

        Ok(())
    }

    /// Execute a single task (one specific prompt) within the persistent session.
    #[instrument(skip(self, session, task), fields(task_id = %task.task_id))]
    async fn run_task(
        &mut self,
        session: &mut SessionState,
        task: &QueuedTask,
    ) -> Result<TaskExecutionResult> {
        let session_id = session.identity.session_id().to_string();
        let prompt = task.prompt.as_str();

        // Append user message to history
        session.history.push(InferenceMessage {
            role: InferenceRole::User,
            content: vec![InferenceContent::Text {
                text: prompt.to_string(),
            }],
            tool_call_id: None,
        });

        let tool_ctx = ToolContext {
            workspace_root: std::path::PathBuf::from(&self.config.kernel.workspace_root),
            session_id: session_id.clone(),
        };

        // Persist user message
        if let Ok(store) = self.store_manager.get_default().await {
            if let Some(iid) = session.internal_id {
                let _ = store
                    .insert_message(
                        iid,
                        session.turn_index,
                        "user",
                        &serde_json::json!([{"type": "text", "text": prompt}]),
                        None,
                    )
                    .await;
            } else {
                warn!("Session missing internal_id, skipping message persistence");
            }
        }

        // Set active session for harness globals (memory etc)
        {
            let harness = self.lock_harness();
            if let Some(ref engine) = *harness {
                engine.set_active_session(Some(&session_id), Some(session.mode.clone()));
            }
        }

        let mut task_turn_count = 0;
        let max_task_turns = self.config.kernel.max_turns;

        let task_status_result: Result<TaskTerminalStatus> = loop {
            if task_turn_count >= max_task_turns {
                error!(
                    max_turns = max_task_turns,
                    "Max turns reached for this task"
                );
                break Ok(TaskTerminalStatus::MaxTurns);
            }

            let turn_ctx = turn::TurnContext {
                task_id: task.task_id.clone(),
                plan_id: task.plan_id.clone(),
                task_turn_index: task_turn_count,
            };
            let completed_turn = match self.execute_turn(session, &tool_ctx, &turn_ctx).await {
                Ok(outcome) => outcome,
                Err(err) => break Err(err),
            };

            self.evaluate_token_usage(session.total_input_tokens, session.total_output_tokens);
            session.turn_index += 1;
            task_turn_count += 1;

            {
                let harness = self.lock_harness();
                if let Some(ref engine) = *harness
                    && let Some(m) = engine.get_active_session_mode()
                {
                    session.mode = m;
                }
            }

            if session.mode == crate::kernel::config::AgentMode::Stateless {
                match completed_turn {
                    turn::TurnOutcome::Continue | turn::TurnOutcome::Complete => {
                        break Ok(TaskTerminalStatus::Success);
                    }
                    turn::TurnOutcome::Rejected => {
                        break Ok(TaskTerminalStatus::Rejected);
                    }
                }
            }

            match completed_turn {
                turn::TurnOutcome::Continue => {}
                turn::TurnOutcome::Complete => {
                    break Ok(TaskTerminalStatus::Success);
                }
                turn::TurnOutcome::Rejected => {
                    break Ok(TaskTerminalStatus::Rejected);
                }
            }
        };

        // Clear active session
        {
            let harness = self.lock_harness();
            if let Some(ref engine) = *harness {
                engine.set_active_session(None, None);
            }
        }
        let task_status = task_status_result?;
        Ok(TaskExecutionResult {
            status: task_status,
            task_turn_count,
        })
    }

}
