use std::collections::{BTreeMap, HashSet, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

use tokio::sync::{Notify, oneshot};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::kernel::session::QueuedTask;
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::LinkedSessionCreate;

use super::{
    ExecutionStatusSnapshot, PeerAgentTaskResult, RuntimeControl, SessionContextOverrides,
    TaskPromotionCandidate,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct RuntimeSlotKey {
    pub(super) agent_id: String,
    pub(super) slot_id: String,
}

impl RuntimeSlotKey {
    pub(super) const DEFAULT_SLOT_ID: &str = "default";

    pub(super) fn default_for(agent_id: &str) -> Self {
        Self {
            agent_id: agent_id.to_string(),
            slot_id: Self::DEFAULT_SLOT_ID.to_string(),
        }
    }

    pub(super) fn linked_for_excluding(
        agent_id: &str,
        parent_session_reference: &str,
        thread_key: &str,
        excluded_slots: &HashSet<String>,
        lane_capacity: usize,
    ) -> Option<Self> {
        let mut hasher = DefaultHasher::new();
        parent_session_reference.hash(&mut hasher);
        thread_key.hash(&mut hasher);
        let lane_capacity = u64::try_from(lane_capacity).ok()?;
        if lane_capacity == 0 {
            return None;
        }
        let initial_lane = hasher.finish() % lane_capacity;
        (0..lane_capacity).find_map(|offset| {
            let lane = (initial_lane + offset) % lane_capacity;
            let slot_id = format!("linked_{lane}");
            (!excluded_slots.contains(&slot_id)).then(|| Self {
                agent_id: agent_id.to_string(),
                slot_id,
            })
        })
    }

    pub(super) fn is_linked(&self) -> bool {
        self.slot_id.starts_with("linked_")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PendingTaskState {
    Queued,
    Running,
    Cancelling,
}

#[derive(Debug, Clone)]
pub(super) struct PendingTaskRecord {
    pub(super) runtime_key: RuntimeSlotKey,
    pub(super) session_target: TaskSessionTarget,
    pub(super) trace_id: String,
    pub(super) title: Option<String>,
    pub(super) prompt_preview: String,
    pub(super) state: PendingTaskState,
    pub(super) runtime_task_id: Option<String>,
    pub(super) execution: ExecutionStatusSnapshot,
}

#[derive(Debug, Clone, Default)]
pub(super) struct TaskSessionTarget {
    pub(super) session_id: Option<String>,
    pub(super) store_selector: Option<StoreSelector>,
    pub(super) linked_parent_session_id: Option<i64>,
    pub(super) thread_key: Option<String>,
    pub(super) reserves_new_child: bool,
}

impl TaskSessionTarget {
    pub(super) fn matches_session(&self, session_id: &str) -> bool {
        self.session_id.as_deref().is_some_and(|target| {
            crate::kernel::session_refs::session_references_match(target, session_id)
        })
    }

    pub(super) fn belongs_to_family(
        &self,
        session_ids: &[String],
        persisted_ids: &HashSet<(StoreSelector, i64)>,
    ) -> bool {
        self.session_id.as_deref().is_some_and(|target| {
            session_ids.iter().any(|session_id| {
                crate::kernel::session_refs::session_references_match(target, session_id)
            })
        }) || self
            .store_selector
            .as_ref()
            .zip(self.linked_parent_session_id)
            .is_some_and(|(store, parent_id)| persisted_ids.contains(&(store.clone(), parent_id)))
    }
}

pub(super) fn task_prompt_preview(prompt: &str) -> String {
    const MAX_CHARS: usize = 240;

    let normalized = prompt.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= MAX_CHARS {
        return normalized;
    }

    let mut preview = normalized.chars().take(MAX_CHARS - 3).collect::<String>();
    preview.push_str("...");
    preview
}

pub(super) struct PeerAgentTaskEnvelope {
    pub(super) task: QueuedTask,
    pub(super) request_id: Option<String>,
    pub(super) result_tx: Option<oneshot::Sender<PeerAgentTaskResult>>,
    pub(super) delegated_capabilities: Option<BTreeMap<String, bool>>,
    pub(super) promotion_candidate: Option<TaskPromotionCandidate>,
    pub(super) linked_session: Option<LinkedSessionTarget>,
    pub(super) session_target: TaskSessionTarget,
}

pub(super) struct PeerTaskSubmission {
    pub(super) delegated_capabilities: Option<BTreeMap<String, bool>>,
    pub(super) promotion_candidate: Option<TaskPromotionCandidate>,
    pub(super) linked_session: Option<LinkedSessionTarget>,
    pub(super) session_target: TaskSessionTarget,
    pub(super) delegation_admission: Option<DelegationAdmission>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DelegationAdmission {
    pub(super) parent_session_id: i64,
    pub(super) persisted_direct_children: usize,
    pub(super) max_fan_out: usize,
    pub(super) max_concurrent_children: usize,
}

#[derive(Debug, Clone)]
pub(super) struct LinkedSessionTarget {
    pub(super) state_selector: StoreSelector,
    pub(super) default_store_selector: Option<StoreSelector>,
    pub(super) context: SessionContextOverrides,
    pub(super) link: LinkedSessionCreate,
}

/// A handle to a running peer agent.
pub struct AgentRuntimeHandle {
    /// Explicit queued envelopes awaiting execution.
    pub(super) queue: Arc<Mutex<VecDeque<PeerAgentTaskEnvelope>>>,
    /// Notification used to wake the background runtime when new work arrives.
    pub(super) notify: Arc<Notify>,
    /// Shared execution/session control state for the runtime.
    pub(super) control: Arc<RuntimeControl>,
    /// Cooperative stop signal for the runtime event loop.
    pub(super) shutdown_token: CancellationToken,
    /// The background task running the agent's event loop.
    pub(super) task: Option<JoinHandle<()>>,
    /// Approximate number of tasks currently queued in the runtime channel.
    pub(super) queued_tasks: Arc<AtomicUsize>,
    /// Number of tasks currently executing inside the runtime loop.
    pub(super) active_tasks: Arc<AtomicUsize>,
}

impl AgentRuntimeHandle {
    pub(super) fn is_running(&self) -> bool {
        self.task
            .as_ref()
            .map(|jh| !jh.is_finished())
            .unwrap_or(false)
    }
}
