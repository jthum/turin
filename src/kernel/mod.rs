pub mod config;
pub mod event;
pub mod builder;
pub mod session;
mod init;
mod turn;

use anyhow::{Context, Result};
use builder::RuntimeBuilder;
use session::SessionState;
use config::TurinConfig;
use event::{KernelEvent, LifecycleEvent};
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex as AsyncMutex};
use tracing::{info, warn, error, debug, instrument};
use std::collections::HashMap;

use crate::harness::engine::HarnessEngine;
use crate::harness::verdict::Verdict;
use crate::inference::provider::{
    InferenceContent, InferenceMessage, InferenceRole, ProviderClient,
};
use crate::persistence::state::StateStore;
use crate::tools::ToolContext;
use crate::tools::registry::ToolRegistry;
use crate::tools::mcp::McpToolProxy;
use mcp_sdk::client::McpClient;
use mcp_sdk::transport::StdioTransport;
use crate::inference::embeddings::EmbeddingProvider;
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
    pub(crate) state: Option<StateStore>,
    /// Thread-safe harness engine for hot-reloading
    pub(crate) harness: Arc<std::sync::Mutex<Option<HarnessEngine>>>,
    /// Watcher handle to keep it alive
    pub(crate) check_watcher: Option<RecommendedWatcher>,
    pub(crate) clients: HashMap<String, ProviderClient>,
    pub(crate) embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    /// Active session queue for harness interaction
    pub(crate) active_queue: crate::harness::globals::ActiveSessionQueue,
    pub(crate) mcp_clients: Vec<Arc<McpClient<mcp_sdk::transport::StdioTransport>>>,
}

/// A pending tool call collected during streaming.
#[derive(Debug, Clone)]
pub(crate) struct PendingToolCall {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}

impl Kernel {
    /// Create a new builder for Kernel.
    pub fn builder(config: TurinConfig) -> RuntimeBuilder {
        RuntimeBuilder::new(config)
    }

    /// Access the state store (if initialized).
    pub fn state(&self) -> Option<&StateStore> {
        self.state.as_ref()
    }

    /// Lock the harness mutex.
    ///
    /// Panics if the mutex is poisoned (previous holder panicked).
    /// A poisoned harness is an unrecoverable state — continuing would
    /// risk executing tool calls with a partially-updated engine.
    pub(crate) fn lock_harness(&self) -> std::sync::MutexGuard<'_, Option<HarnessEngine>> {
        self.harness.lock().expect("harness mutex poisoned")
    }

    /// Create a new session.
    pub fn create_session(&self) -> SessionState {
        let mut session = SessionState::new();
        // Spawn background persistence if state is available
        if let Some(ref store) = self.state {
            let mut rx = session.event_tx.subscribe();
            let store_clone = store.clone();
            let cancel = session.cancel_token.clone();
            let handle = tokio::spawn(async move {
                loop {
                    tokio::select! {
                        _ = cancel.cancelled() => break,
                        result = rx.recv() => {
                            match result {
                                Ok((session_id, event)) => {
                                    let event_type = event.event_type().to_string();
                                    let payload = serde_json::to_value(&event).unwrap_or_default();
                                    if let Err(e) = store_clone.insert_event(&session_id, &event_type, &payload).await {
                                        warn!(error = %e, "Background persistence error");
                                    }
                                }
                                Err(_) => break,
                            }
                        }
                    }
                }
            });
            session.event_task = Some(Arc::new(AsyncMutex::new(Some(handle))));
        }
        session
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

    /// Start a new session.
    pub async fn start_session(&self, session: &mut SessionState) -> Result<()> {
        if session.status == crate::kernel::session::SessionStatus::Active {
            return Ok(());
        }

        let session_id = session.id.clone();
        info!(session_id = %session_id, "Starting new agent session");
        
        // Emit AgentStart event
        self.persist_event(session, &KernelEvent::Lifecycle(LifecycleEvent::AgentStart {
            session_id: session_id.clone(),
        }));

        // Trigger on_agent_start harness hook
        {
            let harness = self.lock_harness();
            if let Some(ref engine) = *harness
                && let Err(e) = engine.evaluate("on_agent_start", serde_json::json!({ "session_id": session_id })) {
                     warn!(error = %e, "Harness on_agent_start failed");
                }
        }

        session.status = crate::kernel::session::SessionStatus::Active;
        Ok(())
    }

    /// End the session and emit AgentEnd event.
    pub async fn end_session(&self, session: &mut SessionState) -> Result<()> {
         if session.status == crate::kernel::session::SessionStatus::Inactive {
             return Ok(());
         }

         self.persist_event(session, &KernelEvent::Lifecycle(LifecycleEvent::AgentEnd {
            message_count: session.turn_index,
            total_input_tokens: session.total_input_tokens,
            total_output_tokens: session.total_output_tokens,
         }));
         
         // Cancel background event persistence task
         session.cancel_token.cancel();
         
         // Clear active queue
         {
             let mut aq = self.active_queue.lock().await;
             *aq = None;
         }
         
         session.status = crate::kernel::session::SessionStatus::Inactive;
         Ok(())
    }

    /// Run the agent loop with the given prompt.
    #[instrument(skip(self, session), fields(session_id = %session.id))]
    pub async fn run(&mut self, session: &mut SessionState, prompt: Option<String>) -> Result<()> {
        // Ensure session is started
        self.start_session(session).await?;

        // Set active queue for harness
        {
            let mut aq = self.active_queue.lock().await;
            *aq = Some(session.queue.clone());
        }

        if let Some(p) = prompt {
            session.queue.lock().await.push_back(p);
        }

        loop {
            // Pop next task
            {
                let mut q = session.queue.lock().await;
                if q.is_empty() {
                    debug!("Queue empty, ending run");
                    break;
                }
                let task = q.pop_front().unwrap();
                drop(q);
                
                info!(task = %task, "Running task");
                self.run_task(session, &task).await?;
            }
            
            // ─── Harness Hook: on_task_complete ─────────────────────
            // Triggered when the queue is explicitly empty.
            let mut recheck = false;
            
            let verdict_result = {
                let harness = self.lock_harness();
                if let Some(ref engine) = *harness {
                    let payload = serde_json::json!({
                        "session_id": session.id,
                        "turn_count": session.turn_index,
                    });
                     Some(engine.evaluate("on_task_complete", payload))
                } else {
                    None
                }
            };

            if let Some(result) = verdict_result {
                match result {
                    Ok(Verdict::Modify(new_tasks_val)) => {
                        if let Some(new_tasks) = new_tasks_val.as_array()
                            && !new_tasks.is_empty() {
                                let mut q = session.queue.lock().await;
                                for task in new_tasks {
                                    if let Some(t) = task.as_str() {
                                        q.push_back(t.to_string());
                                    }
                                }
                                info!(count = new_tasks.len(), "Validation failed or extended by harness; new tasks queued");
                                recheck = true;
                            }
                    },
                    Ok(Verdict::Reject(reason)) => {
                        warn!(reason = %reason, "Session ended with REJECTION from harness");
                        break;
                    },
                    Ok(_) => {},
                    Err(e) => {
                        warn!(error = %e, "Harness on_task_complete error");
                    }
                }
            }
            
            if recheck {
                continue;
            }
            break;
        }
        
        Ok(())
    }

    /// Add a prompt to the end of the queue.
    pub async fn queue_prompt(&self, session: &SessionState, prompt: String) {
        let mut q = session.queue.lock().await;
        q.push_back(prompt);
    }
    
    /// Execute a single task (one specific prompt) within the persistent session.
    #[instrument(skip(self, session, prompt), fields(task = %prompt))]
    async fn run_task(&mut self, session: &mut SessionState, prompt: &str) -> Result<()> {
        let session_id = session.id.clone();

        // Append user message to history
        session.history.push(InferenceMessage {
            role: InferenceRole::User,
            content: vec![InferenceContent::Text { text: prompt.to_string() }],
            tool_call_id: None,
        });

        let tool_ctx = ToolContext {
            workspace_root: std::path::PathBuf::from(&self.config.kernel.workspace_root),
            session_id: session_id.clone(),
        };

        // Persist user message
        if let Some(ref store) = self.state {
            let _ = store.insert_message(
                &session_id,
                session.turn_index,
                "user",
                &serde_json::json!([{"type": "text", "text": prompt}]),
                None,
            ).await;
        }

        let mut task_turn_count = 0;
        let max_task_turns = self.config.kernel.max_turns;

        loop {
            if task_turn_count >= max_task_turns {
                error!(max_turns = max_task_turns, "Max turns reached for this task");
                break;
            }

            let completed_turn = self.execute_turn(session, &tool_ctx).await?;

            self.evaluate_token_usage(session.total_input_tokens, session.total_output_tokens);
            session.turn_index += 1;
            task_turn_count += 1;

            if !completed_turn {
                break;
            }
        }
        Ok(())
    }

    /// Evaluate harness `on_tool_call` hook.
    ///
    /// Returns the composed verdict. If no harness is loaded, returns `Allow`.
    pub(crate) fn evaluate_tool_call(&self, name: &str, id: &str, args: &serde_json::Value) -> Verdict {
        let harness = self.lock_harness();
        if let Some(ref engine) = *harness {
            let payload = serde_json::json!({
                "name": name,
                "id": id,
                "args": args,
            });
            match engine.evaluate("on_tool_call", payload) {
                Ok(verdict) => {
                    if !verdict.is_allowed() {
                        info!(tool = %name, verdict = %verdict, "Harness verdict");
                    }
                    verdict
                }
                Err(e) => {
                    // Harness evaluation errors are non-fatal — default to ALLOW
                    warn!(error = %e, "Harness on_tool_call error");
                    Verdict::Allow
                }
            }
        } else {
            Verdict::Allow
        }
    }

    /// Evaluate harness `on_token_usage` hook.
    ///
    /// This fires after each turn. If a harness rejects, it logs a warning.
    /// Budget enforcement via token hooks is informational in v0.1 — a REJECT here
    /// logs but doesn't halt the loop (the harness can use `db.kv_set` to track state
    /// and reject tool calls instead).
    pub fn evaluate_token_usage(&self, input_tokens: u64, output_tokens: u64) {
        let harness = self.lock_harness();
        if let Some(ref engine) = *harness {
            let payload = serde_json::json!({
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            });
            match engine.evaluate("on_token_usage", payload) {
                Ok(verdict) => {
                    if verdict.is_rejected() {
                        warn!(reason = %verdict.reason().unwrap_or("budget exceeded"), "Token usage harness rejection");
                    }
                }
                Err(e) => {
                    warn!(error = %e, "Harness on_token_usage error");
                }
            }
        }
    }

    /// Persist an event to the state store in the background.
    #[instrument(skip(self, session, event), fields(event_type = %event.event_type()))]
    pub fn persist_event(&self, session: &SessionState, event: &KernelEvent) {
        // Allow harness to observe/intercept any event
        if let Ok(harness_guard) = self.harness.lock()
            && let Some(engine) = &*harness_guard {
                let payload = serde_json::to_value(event).unwrap_or_default();
                if let Ok(verdict) = engine.evaluate("on_kernel_event", payload)
                    && verdict.is_rejected() {
                        warn!(event_type = %event.event_type(), "Event REJECTED by harness on_kernel_event");
                        return;
                    }
                    // Note: MODIFY is ignored for general events for now to avoid complexity
            }
        self.persist_event_internal(&session.event_tx, &session.id, event);
    }

    /// Internal helper for persistence (used by parallel runners)
    fn persist_event_internal(&self, tx: &broadcast::Sender<(String, KernelEvent)>, session_id: &str, event: &KernelEvent) {
        if self.json {
            // In JSON mode, all events go to stdout as NDJSON
            println!("{}", serde_json::to_string(event).unwrap_or_default());
        }
        if tx.send((session_id.to_string(), event.clone())).is_err() {
            warn!("Event broadcast failed — no active receivers");
        }
    }

    /// Connect to an MCP server, initialize it, and register its tools.
    #[instrument(skip(self, args), fields(command = %command, args = ?args))]
    async fn spawn_mcp_server(&mut self, command: &str, args: &[String]) -> Result<usize> {
        let args_str: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        info!("Connecting to MCP server");

        let transport = StdioTransport::new(command, &args_str)
            .with_context(|| format!("Failed to spawn MCP process: {}", command))?;
        
        let client = McpClient::new(transport);
        client.initialize().await.with_context(|| "Failed to initialize MCP client")?;
        
        let list_result = client.list_tools().await.with_context(|| "Failed to list MCP tools")?;
        let count = list_result.tools.len();
        
        let client_arc = Arc::new(client);
        self.mcp_clients.push(client_arc.clone());

        for tool_def in list_result.tools {
            let proxy = McpToolProxy::new(client_arc.clone(), tool_def);
            self.tool_registry.register(Box::new(proxy))
                .with_context(|| "Failed to register MCP tool")?;
        }

        info!(count = count, "MCP tools registered");

        Ok(count)
    }
}
