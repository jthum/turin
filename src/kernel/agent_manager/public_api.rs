use std::collections::BTreeMap;
use std::sync::Arc;

use crate::kernel::config::InferenceOverrideConfig;
use crate::kernel::error::{KernelError, KernelErrorKind, KernelResult};
use crate::kernel::session::QueuedTask;
use crate::persistence::manager::StoreSelector;

use super::{
    AgentManager, LinkedSessionMode, LiveSessionSnapshot, PeerAgentTaskResult, PromotedTaskBranch,
    TaskStatusSnapshot,
};

fn classify<T>(result: anyhow::Result<T>, kind: KernelErrorKind) -> KernelResult<T> {
    result.map_err(|error| KernelError::new(kind, error))
}

impl AgentManager {
    /// Ensure an agent runtime is resident and wake its worker.
    pub async fn wake_agent(self: &Arc<Self>, agent_id: &str) -> KernelResult<()> {
        classify(
            self.wake_agent_inner(agent_id).await,
            KernelErrorKind::Agent,
        )
    }

    /// Resolve a persisted session reference to its owning agent and canonical reference.
    pub async fn resolve_session_target(&self, session_id: &str) -> KernelResult<(String, String)> {
        classify(
            self.resolve_session_target_inner(session_id).await,
            KernelErrorKind::Session,
        )
    }

    /// Wake the live runtime for a session, resuming it when necessary.
    pub async fn wake_session(self: &Arc<Self>, session_id: &str) -> KernelResult<()> {
        classify(
            self.wake_session_inner(session_id).await,
            KernelErrorKind::Session,
        )
    }

    /// Open a new managed session in a runtime slot.
    pub async fn open_session(
        self: &Arc<Self>,
        agent_id: &str,
        slot_id: Option<&str>,
        initial_state_selector: Option<StoreSelector>,
        initial_default_store_selector: Option<StoreSelector>,
        origin_id: Option<String>,
        initial_inference: InferenceOverrideConfig,
    ) -> KernelResult<LiveSessionSnapshot> {
        classify(
            self.open_session_inner(
                agent_id,
                slot_id,
                initial_state_selector,
                initial_default_store_selector,
                origin_id,
                initial_inference,
            )
            .await,
            KernelErrorKind::Session,
        )
    }

    /// Resume a persisted session in a managed runtime slot.
    pub async fn resume_session(
        self: &Arc<Self>,
        session_id: &str,
        slot_id: Option<&str>,
        origin_id: Option<String>,
        initial_inference: InferenceOverrideConfig,
    ) -> KernelResult<LiveSessionSnapshot> {
        classify(
            self.resume_session_inner(session_id, slot_id, origin_id, initial_inference)
                .await,
            KernelErrorKind::Session,
        )
    }

    /// Reload a live session from persistence while preserving its runtime slot.
    pub async fn reload_session(
        self: &Arc<Self>,
        session_id: &str,
        slot_id: Option<&str>,
    ) -> KernelResult<LiveSessionSnapshot> {
        classify(
            self.reload_session_inner(session_id, slot_id).await,
            KernelErrorKind::Session,
        )
    }

    /// Reload a session only when it is currently live.
    pub async fn reload_session_if_live(
        self: &Arc<Self>,
        session_id: &str,
        slot_id: Option<&str>,
    ) -> KernelResult<bool> {
        classify(
            self.reload_session_if_live_inner(session_id, slot_id).await,
            KernelErrorKind::Session,
        )
    }

    /// Submit a task to an agent's default runtime.
    pub async fn submit(
        self: &Arc<Self>,
        agent_id: &str,
        task: QueuedTask,
        delegated_capabilities: Option<BTreeMap<String, bool>>,
    ) -> KernelResult<String> {
        classify(
            self.submit_inner(agent_id, task, delegated_capabilities)
                .await,
            KernelErrorKind::Task,
        )
    }

    /// Submit a task into an agent-owned child session linked to an origin session.
    pub async fn submit_linked(
        self: &Arc<Self>,
        origin_session_id: &str,
        origin_turn_id: Option<i64>,
        agent_id: &str,
        mode: LinkedSessionMode,
        task: QueuedTask,
        delegated_capabilities: Option<BTreeMap<String, bool>>,
    ) -> KernelResult<String> {
        classify(
            self.submit_linked_inner(
                origin_session_id,
                origin_turn_id,
                agent_id,
                mode,
                task,
                delegated_capabilities,
            )
            .await,
            KernelErrorKind::Task,
        )
    }

    /// Submit a task to a specific live session runtime.
    pub async fn submit_to_session(
        self: &Arc<Self>,
        session_id: &str,
        slot_id: Option<&str>,
        task: QueuedTask,
        delegated_capabilities: Option<BTreeMap<String, bool>>,
    ) -> KernelResult<String> {
        classify(
            self.submit_to_session_inner(session_id, slot_id, task, delegated_capabilities)
                .await,
            KernelErrorKind::Task,
        )
    }

    /// Await a previously submitted task result.
    pub async fn await_result(
        &self,
        request_id: &str,
        timeout_ms: Option<u64>,
    ) -> KernelResult<PeerAgentTaskResult> {
        classify(
            self.await_result_inner(request_id, timeout_ms).await,
            KernelErrorKind::Task,
        )
    }

    /// Promote a completed linked task result into a parent-session branch.
    pub async fn promote_completed_task(
        &self,
        request_id: &str,
        branch_name: Option<&str>,
        source_turn_id: Option<i64>,
    ) -> KernelResult<PromotedTaskBranch> {
        classify(
            self.promote_completed_task_inner(request_id, branch_name, source_turn_id)
                .await,
            KernelErrorKind::Task,
        )
    }

    /// Cooperatively cancel one queued or running task.
    pub async fn cancel_task(&self, request_id: &str) -> KernelResult<TaskStatusSnapshot> {
        classify(
            self.cancel_task_inner(request_id).await,
            KernelErrorKind::Task,
        )
    }

    /// Cooperatively cancel work belonging to one live session.
    pub async fn cancel_session(
        &self,
        session_id: &str,
        slot_id: Option<&str>,
    ) -> KernelResult<(String, String, String)> {
        classify(
            self.cancel_session_inner(session_id, slot_id).await,
            KernelErrorKind::Session,
        )
    }

    /// Forcefully terminate the runtime work belonging to one live session.
    pub async fn kill_session(
        &self,
        session_id: &str,
        slot_id: Option<&str>,
    ) -> KernelResult<(String, String, String)> {
        classify(
            self.kill_session_inner(session_id, slot_id).await,
            KernelErrorKind::Session,
        )
    }

    /// Cooperatively cancel a persisted session and its linked descendants.
    pub async fn cancel_session_family(
        &self,
        session_id: &str,
    ) -> KernelResult<(String, String, usize)> {
        classify(
            self.cancel_session_family_inner(session_id).await,
            KernelErrorKind::Session,
        )
    }
}
