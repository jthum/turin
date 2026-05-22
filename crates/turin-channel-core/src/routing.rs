use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

use crate::ChannelConversationKey;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationBinding {
    pub agent_id: String,
    pub slot_id: String,
    pub session_id: String,
    pub updated_at_unix_seconds: u64,
}

impl ConversationBinding {
    pub fn new(
        agent_id: impl Into<String>,
        session_id: impl Into<String>,
        key: &ChannelConversationKey,
        now: SystemTime,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            slot_id: key.deterministic_slot_id(),
            session_id: session_id.into(),
            updated_at_unix_seconds: unix_seconds(now),
        }
    }

    pub fn touch(&mut self, now: SystemTime) {
        self.updated_at_unix_seconds = unix_seconds(now);
    }

    pub fn is_expired(&self, now: SystemTime, ttl: Duration) -> bool {
        unix_seconds(now).saturating_sub(self.updated_at_unix_seconds) > ttl.as_secs()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingDecision {
    Reuse { slot_id: String, session_id: String },
    StartFresh { slot_id: String },
}

pub fn decide_routing(
    key: &ChannelConversationKey,
    binding: Option<&ConversationBinding>,
    now: SystemTime,
    ttl: Option<Duration>,
    reset_requested: bool,
) -> RoutingDecision {
    let slot_id = key.deterministic_slot_id();
    if reset_requested {
        return RoutingDecision::StartFresh { slot_id };
    }

    match binding {
        Some(binding) => {
            if ttl.is_some_and(|ttl| binding.is_expired(now, ttl)) {
                RoutingDecision::StartFresh { slot_id }
            } else {
                RoutingDecision::Reuse {
                    slot_id,
                    session_id: binding.session_id.clone(),
                }
            }
        }
        None => RoutingDecision::StartFresh { slot_id },
    }
}

fn unix_seconds(time: SystemTime) -> u64 {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
