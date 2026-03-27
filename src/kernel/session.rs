use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{Mutex, broadcast, mpsc};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::inference::provider::InferenceMessage;
use crate::kernel::event::KernelEvent;
use crate::kernel::identity::RuntimeIdentity;
use turin_types::ToolSelectionConfig;

/// One queued unit of work to be executed by the kernel.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QueuedTask {
    pub task_id: String,
    pub plan_id: Option<String>,
    pub title: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub tool_selection: Option<ToolSelectionConfig>,
    #[serde(default = "new_trace_id")]
    pub trace_id: String,
}

impl QueuedTask {
    pub fn ad_hoc(prompt: impl Into<String>) -> Self {
        Self {
            task_id: String::new(), // Assigned by SessionState
            plan_id: None,
            title: None,
            prompt: prompt.into(),
            tool_selection: None,
            trace_id: new_trace_id(),
        }
    }

    pub fn with_plan(
        prompt: impl Into<String>,
        plan_id: impl Into<String>,
        title: Option<String>,
    ) -> Self {
        Self {
            task_id: String::new(), // Assigned by SessionState
            plan_id: Some(plan_id.into()),
            title,
            prompt: prompt.into(),
            tool_selection: None,
            trace_id: new_trace_id(),
        }
    }

    pub fn with_inherited_trace(mut self, trace_id: Option<&str>) -> Self {
        if let Some(trace_id) = trace_id
            && !trace_id.is_empty()
        {
            self.trace_id = trace_id.to_string();
        }
        self
    }
}

/// Lightweight in-memory progress tracker for a plan.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PlanProgress {
    pub plan_id: String,
    pub title: String,
    pub total_tasks: usize,
    pub completed_tasks: usize,
}

impl PlanProgress {
    pub fn pending_tasks(&self) -> usize {
        self.total_tasks.saturating_sub(self.completed_tasks)
    }

    pub fn is_complete(&self) -> bool {
        self.completed_tasks >= self.total_tasks
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    Inactive,
    Active,
}

/// Holds the state of an active agent session.
pub struct SessionState {
    pub identity: RuntimeIdentity,
    pub internal_id: Option<i64>,
    pub history: Vec<InferenceMessage>,
    pub queue: Arc<Mutex<VecDeque<QueuedTask>>>,
    pub plans: HashMap<String, PlanProgress>,
    pub turn_index: u32,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    // Event channel for this session
    pub event_tx: broadcast::Sender<(Option<i64>, KernelEvent)>,
    /// Reliable durability lane (separate from observer fanout).
    pub durability_tx: Option<mpsc::UnboundedSender<(Option<i64>, KernelEvent)>>,
    pub event_task: Option<Arc<Mutex<Option<JoinHandle<()>>>>>,
    /// Token to cooperatively cancel the currently running task/turn.
    pub cancel_token: CancellationToken,
    // Internal counters for task scheduling
    pub next_task_id: u32,
    pub next_plan_id: u32,
    pub status: SessionStatus,
    pub mode: crate::kernel::config::AgentMode,
    pub stop_requested: bool,
    pub restored_from_persistence: bool,
}

impl Default for SessionState {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionState {
    pub fn new() -> Self {
        let session_id = uuid::Uuid::now_v7().simple().to_string();
        let (tx, _rx) = broadcast::channel(1024);
        Self {
            identity: RuntimeIdentity::new(session_id, "default"),
            internal_id: None,
            history: Vec::new(),
            queue: Arc::new(Mutex::new(VecDeque::new())),
            plans: HashMap::new(),
            turn_index: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            event_tx: tx,
            durability_tx: None,
            event_task: Some(Arc::new(Mutex::new(None))),
            cancel_token: CancellationToken::new(),
            next_task_id: 1,
            next_plan_id: 1,
            status: SessionStatus::Inactive,
            mode: crate::kernel::config::AgentMode::Auto,
            stop_requested: false,
            restored_from_persistence: false,
        }
    }
}

fn new_trace_id() -> String {
    format!("tr_{}", uuid::Uuid::now_v7().simple())
}
