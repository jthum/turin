use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{Mutex, broadcast};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::inference::provider::InferenceMessage;
use crate::kernel::event::KernelEvent;

/// One queued unit of work to be executed by the kernel.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QueuedTask {
    pub task_id: String,
    pub plan_id: Option<String>,
    pub title: Option<String>,
    pub prompt: String,
}

impl QueuedTask {
    pub fn ad_hoc(prompt: impl Into<String>) -> Self {
        Self {
            task_id: uuid::Uuid::new_v4().to_string(),
            plan_id: None,
            title: None,
            prompt: prompt.into(),
        }
    }

    pub fn with_plan(
        prompt: impl Into<String>,
        plan_id: impl Into<String>,
        title: Option<String>,
    ) -> Self {
        Self {
            task_id: uuid::Uuid::new_v4().to_string(),
            plan_id: Some(plan_id.into()),
            title,
            prompt: prompt.into(),
        }
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
    pub id: String,
    pub history: Vec<InferenceMessage>,
    pub queue: Arc<Mutex<VecDeque<QueuedTask>>>,
    pub plans: HashMap<String, PlanProgress>,
    pub turn_index: u32,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    // Event channel for this session
    pub event_tx: broadcast::Sender<(String, KernelEvent)>,
    pub event_task: Option<Arc<Mutex<Option<JoinHandle<()>>>>>,
    /// Token to cancel the background event persistence task.
    pub cancel_token: CancellationToken,
    pub status: SessionStatus,
}

impl Default for SessionState {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionState {
    pub fn new() -> Self {
        let (tx, _rx) = broadcast::channel(1024);
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            history: Vec::new(),
            queue: Arc::new(Mutex::new(VecDeque::new())),
            plans: HashMap::new(),
            turn_index: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            event_tx: tx,
            event_task: Some(Arc::new(Mutex::new(None))),
            cancel_token: CancellationToken::new(),
            status: SessionStatus::Inactive,
        }
    }
}
