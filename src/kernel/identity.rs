use serde::{Deserialize, Serialize};

/// Routing identity envelope used across session/task/event boundaries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeIdentity {
    pub session_id: String,
    pub agent_id: Option<String>,
    pub user_id: Option<String>,
    pub channel_id: Option<String>,
    pub tenant_id: Option<String>,
    pub run_id: Option<String>,
}

impl RuntimeIdentity {
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            agent_id: None,
            user_id: None,
            channel_id: None,
            tenant_id: None,
            run_id: None,
        }
    }
}
