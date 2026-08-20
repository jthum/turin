use std::collections::VecDeque;

use crate::persistence::manager::StoreSelector;

use super::PeerAgentTaskEnvelope;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum TaskSchedulingKey {
    Session(String),
    PendingLinked {
        store: Option<StoreSelector>,
        parent_session_id: i64,
        thread_key: String,
    },
    Runtime,
}

impl TaskSchedulingKey {
    fn from_envelope(envelope: &PeerAgentTaskEnvelope) -> Self {
        if let Some(session_id) = envelope.session_target.session_id.as_ref() {
            return Self::Session(session_id.clone());
        }
        match (
            envelope.session_target.linked_parent_session_id,
            envelope.session_target.thread_key.as_ref(),
        ) {
            (Some(parent_session_id), Some(thread_key)) => Self::PendingLinked {
                store: envelope.session_target.store_selector.clone(),
                parent_session_id,
                thread_key: thread_key.clone(),
            },
            _ => Self::Runtime,
        }
    }

    fn matches(&self, envelope: &PeerAgentTaskEnvelope) -> bool {
        match self {
            Self::Session(session_id) => {
                envelope.session_target.session_id.as_ref() == Some(session_id)
            }
            Self::PendingLinked {
                store,
                parent_session_id,
                thread_key,
            } => {
                envelope.session_target.session_id.is_none()
                    && envelope.session_target.store_selector.as_ref() == store.as_ref()
                    && envelope.session_target.linked_parent_session_id == Some(*parent_session_id)
                    && envelope.session_target.thread_key.as_ref() == Some(thread_key)
            }
            Self::Runtime => {
                envelope.session_target.session_id.is_none()
                    && (envelope.session_target.linked_parent_session_id.is_none()
                        || envelope.session_target.thread_key.is_none())
            }
        }
    }
}

pub(super) fn pop_fair_task(
    queue: &mut VecDeque<PeerAgentTaskEnvelope>,
    last_scheduled: &mut Option<TaskSchedulingKey>,
) -> Option<PeerAgentTaskEnvelope> {
    let index = last_scheduled
        .as_ref()
        .and_then(|last| queue.iter().position(|envelope| !last.matches(envelope)));
    let envelope = match index {
        Some(index) => queue.remove(index),
        None => queue.pop_front(),
    }?;
    *last_scheduled = Some(TaskSchedulingKey::from_envelope(&envelope));
    Some(envelope)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::agent_manager::TaskSessionTarget;
    use crate::kernel::session::QueuedTask;

    fn envelope(request_id: &str, session_id: &str) -> PeerAgentTaskEnvelope {
        PeerAgentTaskEnvelope {
            task: QueuedTask::ad_hoc(request_id),
            request_id: Some(request_id.to_string()),
            result_tx: None,
            delegated_capabilities: None,
            promotion_candidate: None,
            linked_session: None,
            session_target: TaskSessionTarget {
                session_id: Some(session_id.to_string()),
                ..TaskSessionTarget::default()
            },
        }
    }

    #[test]
    fn rotates_sessions_and_preserves_per_session_fifo() {
        let mut queue = VecDeque::from([
            envelope("a1", "session-a"),
            envelope("a2", "session-a"),
            envelope("b1", "session-b"),
            envelope("b2", "session-b"),
            envelope("a3", "session-a"),
        ]);
        let mut last = None;
        let mut order = Vec::new();
        while let Some(envelope) = pop_fair_task(&mut queue, &mut last) {
            order.push(envelope.request_id.expect("request id"));
        }
        assert_eq!(order, ["a1", "b1", "a2", "b2", "a3"]);
    }
}
