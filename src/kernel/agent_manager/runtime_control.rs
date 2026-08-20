use std::sync::{Mutex, RwLock};

use tokio_util::sync::CancellationToken;

use crate::kernel::config::InferenceOverrideConfig;
use crate::kernel::session::ExecutionConflictPolicy;

use super::{
    ExecutionStatusSnapshot, LiveSessionHistorySnapshot, SessionEventReceiver, SessionEventSender,
};

pub(crate) struct RuntimeControl {
    state: RwLock<RuntimeControlState>,
    session_reset_request: Mutex<Option<SessionResetRequest>>,
}

#[derive(Clone)]
pub(crate) struct RuntimeControlSnapshot {
    pub(super) session_id: Option<String>,
    pub(super) session_events: Option<SessionEventSender>,
    pub(super) session_context: SessionContextOverrides,
    pub(super) execution: Option<ExecutionStatusSnapshot>,
    pub(super) conflict_policy: ExecutionConflictPolicy,
    pub(super) history: Option<LiveSessionHistorySnapshot>,
    pub(super) request_id: Option<String>,
    pub(super) runtime_task_id: Option<String>,
    pub(super) cancel_token: Option<CancellationToken>,
    pub(super) generation: u64,
}

#[derive(Clone)]
struct RuntimeControlState {
    session_id: Option<String>,
    session_events: Option<SessionEventSender>,
    session_context: SessionContextOverrides,
    execution: Option<ExecutionStatusSnapshot>,
    conflict_policy: ExecutionConflictPolicy,
    history: Option<LiveSessionHistorySnapshot>,
    request_id: Option<String>,
    runtime_task_id: Option<String>,
    cancel_token: Option<CancellationToken>,
    generation: u64,
}

impl Default for RuntimeControlState {
    fn default() -> Self {
        Self {
            session_id: None,
            session_events: None,
            session_context: SessionContextOverrides::default(),
            execution: None,
            conflict_policy: ExecutionConflictPolicy::Reject,
            history: None,
            request_id: None,
            runtime_task_id: None,
            cancel_token: None,
            generation: 0,
        }
    }
}

impl Default for RuntimeControl {
    fn default() -> Self {
        Self {
            state: RwLock::new(RuntimeControlState::default()),
            session_reset_request: Mutex::new(None),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SessionContextOverrides {
    pub(crate) origin_id: Option<String>,
    pub(crate) inference: InferenceOverrideConfig,
}

#[derive(Debug, Clone)]
pub(crate) enum SessionResetRequest {
    Fresh(SessionContextOverrides),
    Resume {
        session_id: String,
        context: SessionContextOverrides,
    },
}

impl RuntimeControl {
    pub(super) fn set_current_session(
        &self,
        session_id: Option<String>,
        event_tx: Option<SessionEventSender>,
        context: SessionContextOverrides,
        execution: Option<ExecutionStatusSnapshot>,
        conflict_policy: ExecutionConflictPolicy,
        history: Option<LiveSessionHistorySnapshot>,
    ) {
        let mut state = self.write_state();
        state.session_id = session_id;
        state.session_events = event_tx;
        state.session_context = context;
        state.execution = execution;
        state.conflict_policy = conflict_policy;
        state.history = history;
        state.generation = state.generation.wrapping_add(1);
    }

    #[cfg(test)]
    pub(super) fn set_current_session_id(&self, session_id: Option<String>) {
        self.set_current_session(
            session_id,
            None,
            SessionContextOverrides::default(),
            None,
            ExecutionConflictPolicy::Reject,
            None,
        );
    }

    pub(super) fn current_session_id(&self) -> Option<String> {
        self.snapshot().session_id
    }

    pub(super) fn session_generation(&self) -> u64 {
        self.snapshot().generation
    }

    pub(super) fn current_session_context(&self) -> SessionContextOverrides {
        self.snapshot().session_context
    }

    pub(super) fn current_execution(&self) -> Option<ExecutionStatusSnapshot> {
        self.snapshot().execution
    }

    pub(super) fn set_current_conflict_policy(&self, conflict_policy: ExecutionConflictPolicy) {
        self.write_state().conflict_policy = conflict_policy;
    }

    pub(super) fn current_conflict_policy(&self) -> ExecutionConflictPolicy {
        self.snapshot().conflict_policy
    }

    pub(super) fn set_current_execution_snapshot(&self, execution: ExecutionStatusSnapshot) {
        self.write_state().execution = Some(execution);
    }

    pub(super) fn set_current_history_snapshot(&self, history: LiveSessionHistorySnapshot) {
        self.write_state().history = Some(history);
    }

    #[cfg(test)]
    pub(super) fn set_current_execution_conflict_policy(
        &self,
        conflict_policy: ExecutionConflictPolicy,
    ) {
        self.set_current_conflict_policy(conflict_policy);
    }

    pub(super) fn subscribe_current_session_events(&self) -> Option<SessionEventReceiver> {
        self.snapshot()
            .session_events
            .as_ref()
            .map(SessionEventSender::subscribe)
    }

    pub(super) fn activate_task(
        &self,
        request_id: Option<String>,
        runtime_task_id: String,
        cancel_token: CancellationToken,
    ) {
        let mut state = self.write_state();
        state.request_id = request_id;
        state.runtime_task_id = Some(runtime_task_id);
        state.cancel_token = Some(cancel_token);
    }

    pub(super) fn clear_active_task(&self) {
        let mut state = self.write_state();
        state.request_id = None;
        state.runtime_task_id = None;
        state.cancel_token = None;
    }

    pub(super) fn current_request_id(&self) -> Option<String> {
        self.snapshot().request_id
    }

    pub(super) fn current_runtime_task_id(&self) -> Option<String> {
        self.snapshot().runtime_task_id
    }

    pub(super) fn request_task_cancel(&self) -> bool {
        let token = self.snapshot().cancel_token;
        if let Some(token) = token {
            token.cancel();
            true
        } else {
            false
        }
    }

    pub(super) fn request_session_cancel(&self) -> bool {
        *self
            .session_reset_request
            .lock()
            .expect("runtime control session reset lock poisoned") =
            Some(SessionResetRequest::Fresh(self.current_session_context()));
        self.request_task_cancel()
    }

    pub(super) fn request_session_resume(
        &self,
        session_id: String,
        context: SessionContextOverrides,
    ) {
        *self
            .session_reset_request
            .lock()
            .expect("runtime control session reset lock poisoned") =
            Some(SessionResetRequest::Resume {
                session_id,
                context,
            });
    }

    pub(super) fn take_session_reset_request(&self) -> Option<SessionResetRequest> {
        self.session_reset_request
            .lock()
            .expect("runtime control session reset lock poisoned")
            .take()
    }

    pub(crate) fn snapshot(&self) -> RuntimeControlSnapshot {
        let state = self.read_state();
        RuntimeControlSnapshot {
            session_id: state.session_id.clone(),
            session_events: state.session_events.clone(),
            session_context: state.session_context.clone(),
            execution: state.execution.clone(),
            conflict_policy: state.conflict_policy,
            history: state.history.clone(),
            request_id: state.request_id.clone(),
            runtime_task_id: state.runtime_task_id.clone(),
            cancel_token: state.cancel_token.clone(),
            generation: state.generation,
        }
    }

    fn read_state(&self) -> std::sync::RwLockReadGuard<'_, RuntimeControlState> {
        self.state
            .read()
            .expect("runtime control state lock poisoned")
    }

    fn write_state(&self) -> std::sync::RwLockWriteGuard<'_, RuntimeControlState> {
        self.state
            .write()
            .expect("runtime control state lock poisoned")
    }
}
