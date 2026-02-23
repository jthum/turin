use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Routing identity envelope used across session/task/event boundaries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeIdentity {
    pub session_id: String,
    pub agent_id: String,
    pub user_id: Option<String>,
    pub run_id: Option<String>,
    #[serde(default)]
    pub extra: std::collections::BTreeMap<String, String>,
}

impl RuntimeIdentity {
    pub fn new(session_id: impl Into<String>, agent_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            agent_id: agent_id.into(),
            user_id: None,
            run_id: None,
            extra: std::collections::BTreeMap::new(),
        }
    }

    pub fn verify_access(&self, selector: &ContextSelector) -> Result<()> {
        // v1 simplistic policy
        if selector.visibility == "private" {
            let agent_tag = format!("agent:{}", self.agent_id);
            if !selector.tags.contains(&agent_tag) {
                return Err(anyhow!("Policy denial: Agent {} lacks private access to contexts outside its own tags.", self.agent_id));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextSelector {
    pub tags: Vec<String>,
    pub namespace: String,
    pub visibility: String,
}

impl ContextSelector {
    /// Maps the context selector to a logical database alias string.
    pub fn to_alias(&self) -> String {
        // Extremely simple v1 translation: just join sorted tags + namespace
        let mut sorted_tags = self.tags.clone();
        sorted_tags.sort();
        format!("{}__{}", sorted_tags.join("_"), self.namespace)
    }
}
