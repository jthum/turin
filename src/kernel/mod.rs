pub mod builder;
pub mod config;
pub mod event;
mod init;
pub mod session;
mod turn;

use anyhow::{Context, Result};
use builder::RuntimeBuilder;
use config::TurinConfig;
use event::{KernelEvent, LifecycleEvent, TaskTerminalStatus};
use session::{PlanProgress, QueuedTask, SessionState};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex as AsyncMutex, broadcast};
use tracing::{debug, error, info, instrument, warn};

use crate::harness::engine::HarnessEngine;
use crate::harness::verdict::Verdict;
use crate::inference::embeddings::EmbeddingProvider;
use crate::inference::provider::{
    InferenceContent, InferenceMessage, InferenceRole, ProviderClient,
};
use crate::persistence::state::StateStore;
use crate::tools::ToolContext;
use crate::tools::mcp::McpToolProxy;
use crate::tools::registry::ToolRegistry;
use mcp_sdk::client::McpClient;
use mcp_sdk::transport::StdioTransport;
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

    pub(crate) mcp_clients: Vec<McpClientEntry>,
}

pub(crate) struct McpClientEntry {
    pub command: String,
    pub args: Vec<String>,
    pub client: Arc<McpClient<mcp_sdk::transport::StdioTransport>>,
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

    /// Access the state store (if initialized).
    pub fn state(&self) -> Option<&StateStore> {
        self.state.as_ref()
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
        info!(session_id = %session_id, "Starting new session");

        // Emit SessionStart event
        self.persist_event(
            session,
            &KernelEvent::Lifecycle(LifecycleEvent::SessionStart {
                session_id: session_id.clone(),
            }),
        );

        // Trigger on_session_start harness hook
        {
            let harness = self.lock_harness();
            if let Some(ref engine) = *harness
                && let Err(e) = engine.evaluate(
                    "on_session_start",
                    serde_json::json!({ "session_id": session_id }),
                )
            {
                warn!(error = %e, "Harness on_session_start failed");
            }
        }

        session.status = crate::kernel::session::SessionStatus::Active;
        Ok(())
    }

    /// End the session and emit SessionEnd event.
    pub async fn end_session(&self, session: &mut SessionState) -> Result<()> {
        if session.status == crate::kernel::session::SessionStatus::Inactive {
            return Ok(());
        }

        self.persist_event(
            session,
            &KernelEvent::Lifecycle(LifecycleEvent::SessionEnd {
                turn_count: session.turn_index,
                total_input_tokens: session.total_input_tokens,
                total_output_tokens: session.total_output_tokens,
            }),
        );

        {
            let harness = self.lock_harness();
            if let Some(ref engine) = *harness
                && let Err(e) = engine.evaluate(
                    "on_session_end",
                    serde_json::json!({
                        "session_id": session.id.clone(),
                        "turn_count": session.turn_index,
                        "total_input_tokens": session.total_input_tokens,
                        "total_output_tokens": session.total_output_tokens,
                    }),
                )
            {
                warn!(error = %e, "Harness on_session_end failed");
            }
        }

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
            let plan_id = uuid::Uuid::new_v4().to_string();
            session.plans.insert(
                plan_id.clone(),
                PlanProgress {
                    plan_id: plan_id.clone(),
                    title: "user_request".to_string(),
                    total_tasks: 1,
                    completed_tasks: 0,
                },
            );
            let mut q = session.queue.lock().await;
            q.push_back(QueuedTask::with_plan(
                p,
                plan_id,
                Some("user_request".to_string()),
            ));
        }

        loop {
            let (mut task, queue_depth_after_pop) = {
                let mut q = session.queue.lock().await;
                if q.is_empty() {
                    debug!("Queue empty, firing on_all_tasks_complete");
                    drop(q);
                    self.persist_event(
                        session,
                        &KernelEvent::Lifecycle(LifecycleEvent::AllTasksComplete {
                            session_id: session.id.clone(),
                        }),
                    );
                    let verdict = {
                        let harness = self.lock_harness();
                        if let Some(ref engine) = *harness {
                            match engine.evaluate(
                                "on_all_tasks_complete",
                                serde_json::json!({
                                    "session_id": session.id.clone(),
                                    "turn_count": session.turn_index,
                                }),
                            ) {
                                Ok(v) => Some(v),
                                Err(e) => {
                                    warn!(error = %e, "Harness on_all_tasks_complete failed");
                                    None
                                }
                            }
                        } else {
                            None
                        }
                    };

                    if let Some(Verdict::Modify(new_tasks_val)) = verdict {
                        let new_tasks = Self::parse_task_list(&new_tasks_val, None, None);
                        if !new_tasks.is_empty() {
                            let mut q = session.queue.lock().await;
                            for task in new_tasks {
                                q.push_back(task);
                            }
                            continue;
                        }
                    }

                    break;
                }
                let task = q.pop_front().expect("queue checked non-empty");
                let depth = q.len();
                (task, depth)
            };

            self.persist_event(
                session,
                &KernelEvent::Lifecycle(LifecycleEvent::TaskStart {
                    task_id: task.task_id.clone(),
                    plan_id: task.plan_id.clone(),
                    title: task.title.clone(),
                    prompt: task.prompt.clone(),
                    queue_depth: queue_depth_after_pop,
                }),
            );

            let task_start_verdict = {
                let harness = self.lock_harness();
                if let Some(ref engine) = *harness {
                    match engine.evaluate(
                        "on_task_start",
                        serde_json::json!({
                            "session_id": session.id.clone(),
                            "task_id": task.task_id.clone(),
                            "plan_id": task.plan_id.clone(),
                            "title": task.title.clone(),
                            "prompt": task.prompt.clone(),
                            "queue_depth": queue_depth_after_pop,
                        }),
                    ) {
                        Ok(v) => v,
                        Err(e) => {
                            warn!(error = %e, "Harness on_task_start error");
                            Verdict::Allow
                        }
                    }
                } else {
                    Verdict::Allow
                }
            };

            match task_start_verdict {
                Verdict::Reject(reason) => {
                    warn!(task_id = %task.task_id, reason = %reason, "Task rejected by on_task_start");
                    self.complete_task(session, &task, TaskTerminalStatus::Rejected, 0, None)
                        .await?;
                    continue;
                }
                Verdict::Modify(val) => {
                    if let Some(obj) = val.as_object() {
                        if let Some(prompt) = obj.get("prompt").and_then(|v| v.as_str()) {
                            task.prompt = prompt.to_string();
                        }
                        if let Some(title) = obj.get("title").and_then(|v| v.as_str()) {
                            task.title = Some(title.to_string());
                        }
                    }
                }
                Verdict::Escalate(reason) => {
                    warn!(task_id = %task.task_id, reason = %reason, "Task escalated at on_task_start; treating as rejected");
                    self.complete_task(session, &task, TaskTerminalStatus::Rejected, 0, None)
                        .await?;
                    continue;
                }
                Verdict::Allow => {}
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

    /// Add a prompt to the end of the queue as an implicit single-task plan.
    pub async fn queue_prompt(&self, session: &mut SessionState, prompt: String) {
        let plan_id = uuid::Uuid::new_v4().to_string();
        session.plans.insert(
            plan_id.clone(),
            PlanProgress {
                plan_id: plan_id.clone(),
                title: "queued_prompt".to_string(),
                total_tasks: 1,
                completed_tasks: 0,
            },
        );
        let mut q = session.queue.lock().await;
        q.push_back(QueuedTask::with_plan(
            prompt,
            plan_id,
            Some("queued_prompt".to_string()),
        ));
    }

    /// Execute a single task (one specific prompt) within the persistent session.
    #[instrument(skip(self, session, task), fields(task_id = %task.task_id))]
    async fn run_task(
        &mut self,
        session: &mut SessionState,
        task: &QueuedTask,
    ) -> Result<TaskExecutionResult> {
        let session_id = session.id.clone();
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
        if let Some(ref store) = self.state {
            let _ = store
                .insert_message(
                    &session_id,
                    session.turn_index,
                    "user",
                    &serde_json::json!([{"type": "text", "text": prompt}]),
                    None,
                )
                .await;
        }

        // Set active session for harness globals (memory etc)
        {
            let harness = self.lock_harness();
            if let Some(ref engine) = *harness {
                engine.set_active_session(Some(&session_id));
            }
        }

        let mut task_turn_count = 0;
        let max_task_turns = self.config.kernel.max_turns;

        let task_status = loop {
            if task_turn_count >= max_task_turns {
                error!(
                    max_turns = max_task_turns,
                    "Max turns reached for this task"
                );
                break TaskTerminalStatus::MaxTurns;
            }

            let turn_ctx = turn::TurnContext {
                task_id: task.task_id.clone(),
                plan_id: task.plan_id.clone(),
                task_turn_index: task_turn_count,
            };
            let completed_turn = match self.execute_turn(session, &tool_ctx, &turn_ctx).await {
                Ok(outcome) => outcome,
                Err(err) => {
                    let harness = self.lock_harness();
                    if let Some(ref engine) = *harness {
                        engine.set_active_session(None);
                    }
                    return Err(err);
                }
            };

            self.evaluate_token_usage(session.total_input_tokens, session.total_output_tokens);
            session.turn_index += 1;
            task_turn_count += 1;

            match completed_turn {
                turn::TurnOutcome::Continue => {}
                turn::TurnOutcome::Complete => {
                    break TaskTerminalStatus::Success;
                }
                turn::TurnOutcome::Rejected => {
                    break TaskTerminalStatus::Rejected;
                }
            }
        };

        // Clear active session
        {
            let harness = self.lock_harness();
            if let Some(ref engine) = *harness {
                engine.set_active_session(None);
            }
        }
        Ok(TaskExecutionResult {
            status: task_status,
            task_turn_count,
        })
    }

    pub(crate) fn parse_task_list(
        tasks_val: &serde_json::Value,
        default_plan_id: Option<&str>,
        default_title: Option<&str>,
    ) -> Vec<QueuedTask> {
        let Some(items) = tasks_val.as_array() else {
            return Vec::new();
        };

        items
            .iter()
            .filter_map(|item| {
                if let Some(prompt) = item.as_str() {
                    if let Some(plan_id) = default_plan_id {
                        return Some(QueuedTask::with_plan(
                            prompt.to_string(),
                            plan_id.to_string(),
                            default_title.map(ToString::to_string),
                        ));
                    }
                    return Some(QueuedTask::ad_hoc(prompt.to_string()));
                }

                let obj = item.as_object()?;
                let prompt = obj.get("prompt").and_then(|v| v.as_str())?;
                let plan_id = obj
                    .get("plan_id")
                    .and_then(|v| v.as_str())
                    .map(ToString::to_string)
                    .or_else(|| default_plan_id.map(ToString::to_string));
                let title = obj
                    .get("title")
                    .and_then(|v| v.as_str())
                    .map(ToString::to_string)
                    .or_else(|| default_title.map(ToString::to_string));
                match plan_id {
                    Some(plan_id) => {
                        Some(QueuedTask::with_plan(prompt.to_string(), plan_id, title))
                    }
                    None => Some(QueuedTask::ad_hoc(prompt.to_string())),
                }
            })
            .collect()
    }

    pub(crate) async fn complete_task(
        &mut self,
        session: &mut SessionState,
        task: &QueuedTask,
        status: TaskTerminalStatus,
        task_turn_count: u32,
        error_message: Option<String>,
    ) -> Result<()> {
        self.persist_event(
            session,
            &KernelEvent::Lifecycle(LifecycleEvent::TaskComplete {
                task_id: task.task_id.clone(),
                plan_id: task.plan_id.clone(),
                status,
                task_turn_count,
                error: error_message.clone(),
            }),
        );

        let verdict_result = {
            let harness = self.lock_harness();
            if let Some(ref engine) = *harness {
                Some(engine.evaluate(
                    "on_task_complete",
                    serde_json::json!({
                        "session_id": session.id.clone(),
                        "task_id": task.task_id.clone(),
                        "plan_id": task.plan_id.clone(),
                        "status": status,
                        "task_turn_count": task_turn_count,
                        "turn_count": session.turn_index,
                        "error": error_message,
                    }),
                ))
            } else {
                None
            }
        };

        if let Some(result) = verdict_result {
            match result {
                Ok(Verdict::Modify(new_tasks_val)) => {
                    let new_tasks = Self::parse_task_list(&new_tasks_val, None, None);
                    if !new_tasks.is_empty() {
                        let mut q = session.queue.lock().await;
                        for queued in new_tasks {
                            q.push_back(queued);
                        }
                        info!("on_task_complete queued additional tasks via MODIFY");
                    }
                }
                Ok(Verdict::Reject(reason)) => {
                    warn!(task_id = %task.task_id, reason = %reason, "on_task_complete rejected");
                }
                Ok(Verdict::Escalate(reason)) => {
                    warn!(task_id = %task.task_id, reason = %reason, "on_task_complete escalated");
                }
                Ok(Verdict::Allow) => {}
                Err(e) => {
                    warn!(error = %e, "Harness on_task_complete error");
                }
            }
        }

        if let Some(plan_id) = &task.plan_id {
            let completed_plan = if let Some(progress) = session.plans.get_mut(plan_id) {
                progress.completed_tasks += 1;
                if progress.is_complete() {
                    Some(progress.clone())
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(plan) = completed_plan {
                self.persist_event(
                    session,
                    &KernelEvent::Lifecycle(LifecycleEvent::PlanComplete {
                        plan_id: plan.plan_id.clone(),
                        title: plan.title.clone(),
                        total_tasks: plan.total_tasks,
                        completed_tasks: plan.completed_tasks,
                    }),
                );

                {
                    let harness = self.lock_harness();
                    if let Some(ref engine) = *harness
                        && let Err(e) = engine.evaluate(
                            "on_plan_complete",
                            serde_json::json!({
                                "session_id": session.id.clone(),
                                "plan_id": plan.plan_id.clone(),
                                "title": plan.title.clone(),
                                "total_tasks": plan.total_tasks,
                                "completed_tasks": plan.completed_tasks,
                            }),
                        )
                    {
                        warn!(error = %e, "Harness on_plan_complete failed");
                    }
                }

                session.plans.remove(plan_id);
            }
        }

        Ok(())
    }

    async fn handle_inference_error(
        &mut self,
        session: &mut SessionState,
        task: &QueuedTask,
        error: &str,
    ) -> Result<bool> {
        let verdict_result = {
            let harness = self.lock_harness();
            if let Some(ref engine) = *harness {
                engine.set_active_session(Some(&session.id));
                let result = engine.evaluate(
                    "on_inference_error",
                    serde_json::json!({
                        "session_id": session.id.clone(),
                        "task_id": task.task_id.clone(),
                        "plan_id": task.plan_id.clone(),
                        "turn_count": session.turn_index,
                        "error": error,
                    }),
                );
                engine.set_active_session(None);
                Some(result)
            } else {
                None
            }
        };

        if let Some(result) = verdict_result {
            match result {
                Ok(Verdict::Modify(new_tasks_val)) => {
                    let new_tasks = Self::parse_task_list(
                        &new_tasks_val,
                        task.plan_id.as_deref(),
                        task.title.as_deref(),
                    );
                    if !new_tasks.is_empty() {
                        let mut q = session.queue.lock().await;
                        for queued in new_tasks {
                            q.push_back(queued);
                        }
                        info!(
                            task_id = %task.task_id,
                            "on_inference_error queued additional tasks via MODIFY"
                        );
                        return Ok(true);
                    }
                }
                Ok(Verdict::Reject(reason)) => {
                    warn!(task_id = %task.task_id, reason = %reason, "on_inference_error rejected");
                }
                Ok(Verdict::Escalate(reason)) => {
                    warn!(task_id = %task.task_id, reason = %reason, "on_inference_error escalated");
                }
                Ok(Verdict::Allow) => {}
                Err(e) => {
                    warn!(error = %e, "Harness on_inference_error error");
                }
            }
        }

        Ok(false)
    }

    pub(crate) async fn cancel_queued_tasks(
        &mut self,
        session: &mut SessionState,
    ) -> Result<usize> {
        let drained_tasks: Vec<QueuedTask> = {
            let mut q = session.queue.lock().await;
            q.drain(..).collect()
        };

        let cancelled = drained_tasks.len();
        for queued in drained_tasks {
            self.complete_task(session, &queued, TaskTerminalStatus::Cancelled, 0, None)
                .await?;
        }
        Ok(cancelled)
    }

    /// Evaluate harness `on_tool_call` hook.
    ///
    /// Returns the composed verdict. If no harness is loaded, returns `Allow`.
    pub(crate) fn evaluate_tool_call(
        &self,
        name: &str,
        id: &str,
        args: &serde_json::Value,
    ) -> Verdict {
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
            && let Some(engine) = &*harness_guard
        {
            let payload = serde_json::to_value(event).unwrap_or_default();
            if let Ok(verdict) = engine.evaluate("on_kernel_event", payload)
                && verdict.is_rejected()
            {
                warn!(event_type = %event.event_type(), "Event REJECTED by harness on_kernel_event");
                return;
            }
            // Note: MODIFY is ignored for general events for now to avoid complexity
        }
        self.persist_event_internal(&session.event_tx, &session.id, event);
    }

    /// Internal helper for persistence (used by parallel runners)
    fn persist_event_internal(
        &self,
        tx: &broadcast::Sender<(String, KernelEvent)>,
        session_id: &str,
        event: &KernelEvent,
    ) {
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

        // Check for existing client
        if let Some(entry) = self
            .mcp_clients
            .iter()
            .find(|e| e.command == command && e.args == args)
        {
            info!(command = %command, "Reusing existing MCP client");
            // We can return the tool count from the existing client,
            // but we don't store tool count. We could just re-list or trust existing registry.
            // If the registry already has the tools, we might be fine.
            // But if specific tools were removed?
            // Simplest is to assume consistency.
            // But evaluate_tool_call needs tool execution logic, which relies on ToolRegistry.
            // If registry is wiped but client reused?
            // ToolRegistry persists in Kernel.

            // Let's re-list to be safe and ensure tools are registered?
            // Listing is cheap.
            let list_result = entry
                .client
                .list_tools()
                .await
                .with_context(|| "Failed to list MCP tools on reused client")?;
            let count = list_result.tools.len();

            // Update tool registry just in case
            for tool_def in list_result.tools {
                // McpToolProxy needs an Arc<McpClient>. The entry has it.
                let proxy = McpToolProxy::new(entry.client.clone(), tool_def);
                let _ = self.tool_registry.register(Box::new(proxy));
                // Ignore error if duplicate (register returns Result<()>)
            }
            return Ok(count);
        }

        info!("Connecting to MCP server");

        let transport = StdioTransport::new(command, &args_str)
            .with_context(|| format!("Failed to spawn MCP process: {}", command))?;

        let client = McpClient::new(transport);
        client
            .initialize()
            .await
            .with_context(|| "Failed to initialize MCP client")?;

        let list_result = client
            .list_tools()
            .await
            .with_context(|| "Failed to list MCP tools")?;
        let count = list_result.tools.len();

        let client_arc = Arc::new(client);
        self.mcp_clients.push(McpClientEntry {
            command: command.to_string(),
            args: args.to_vec(),
            client: client_arc.clone(),
        });

        for tool_def in list_result.tools {
            let proxy = McpToolProxy::new(client_arc.clone(), tool_def);
            self.tool_registry
                .register(Box::new(proxy))
                .with_context(|| "Failed to register MCP tool")?;
        }

        info!(count = count, "MCP tools registered");

        Ok(count)
    }
}
