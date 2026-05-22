use crate::daemon::protocol::{
    DaemonRequest, EventEnvelope, RequestEnvelope, RuntimeEventsSubscribeParams,
};

#[derive(Debug, Clone, Default)]
pub(super) struct EventFilter {
    pub(super) agent_id: Option<String>,
    pub(super) session_id: Option<String>,
    pub(super) slot_id: Option<String>,
}

impl EventFilter {
    pub(super) fn from_request(request: &RequestEnvelope) -> Self {
        match &request.request {
            DaemonRequest::RuntimeEventsSubscribe(RuntimeEventsSubscribeParams {
                agent_id,
                session_id,
                slot_id,
            }) => Self {
                agent_id: agent_id.clone(),
                session_id: session_id.clone(),
                slot_id: slot_id.clone(),
            },
            _ => Self::default(),
        }
    }

    pub(super) fn matches(&self, event: &EventEnvelope) -> bool {
        if self.should_always_deliver(&event.event) {
            return true;
        }
        self.matches_agent(event) && self.matches_session(event) && self.matches_slot(event)
    }

    pub(super) fn has_scope(&self) -> bool {
        self.agent_id.is_some() || self.session_id.is_some() || self.slot_id.is_some()
    }

    fn should_always_deliver(&self, event_name: &str) -> bool {
        matches!(event_name, "runtime.rescan_failed" | "runtime.rescanned")
    }

    fn matches_agent(&self, event: &EventEnvelope) -> bool {
        let Some(expected) = self.agent_id.as_deref() else {
            return true;
        };

        event
            .data
            .get("agent_id")
            .and_then(|value| value.as_str())
            .or_else(|| event.data.get("id").and_then(|value| value.as_str()))
            == Some(expected)
    }

    fn matches_session(&self, event: &EventEnvelope) -> bool {
        let Some(expected) = self.session_id.as_deref() else {
            return true;
        };

        event
            .data
            .get("session_id")
            .and_then(|value| value.as_str())
            == Some(expected)
    }

    fn matches_slot(&self, event: &EventEnvelope) -> bool {
        let Some(expected) = self.slot_id.as_deref() else {
            return true;
        };

        event.data.get("slot_id").and_then(|value| value.as_str()) == Some(expected)
    }
}
