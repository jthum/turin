use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Routing identity envelope used across session/task/event boundaries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeIdentity {
    session_id: String,
    agent_id: String,
    #[serde(default)]
    user_id: Option<String>,
    #[serde(default)]
    channel_id: Option<String>,
    #[serde(default)]
    tenant_id: Option<String>,
    #[serde(default)]
    run_id: Option<String>,
    #[serde(default)]
    extra: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdentityKey {
    SessionId,
    AgentId,
    UserId,
    ChannelId,
    TenantId,
    RunId,
    Custom(String),
}

impl RuntimeIdentity {
    pub fn new(session_id: impl Into<String>, agent_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            agent_id: agent_id.into(),
            user_id: None,
            channel_id: None,
            tenant_id: None,
            run_id: None,
            extra: BTreeMap::new(),
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    pub fn user_id(&self) -> Option<&str> {
        self.user_id.as_deref()
    }

    pub fn channel_id(&self) -> Option<&str> {
        self.channel_id.as_deref()
    }

    pub fn tenant_id(&self) -> Option<&str> {
        self.tenant_id.as_deref()
    }

    pub fn run_id(&self) -> Option<&str> {
        self.run_id.as_deref()
    }

    pub fn extra(&self) -> &BTreeMap<String, String> {
        &self.extra
    }

    pub fn set_session_id(&mut self, session_id: impl Into<String>) {
        self.session_id = session_id.into();
    }

    pub fn set_agent_id(&mut self, agent_id: impl Into<String>) {
        self.agent_id = agent_id.into();
    }

    pub fn set_user_id(&mut self, user_id: Option<String>) {
        self.user_id = user_id;
    }

    pub fn set_channel_id(&mut self, channel_id: Option<String>) {
        self.channel_id = channel_id;
    }

    pub fn set_tenant_id(&mut self, tenant_id: Option<String>) {
        self.tenant_id = tenant_id;
    }

    pub fn set_run_id(&mut self, run_id: Option<String>) {
        self.run_id = run_id;
    }

    pub fn insert_extra(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.extra.insert(key.into(), value.into());
    }

    pub fn remove_extra(&mut self, key: &str) {
        self.extra.remove(key);
    }

    pub fn get(&self, key: &IdentityKey) -> Option<&str> {
        match key {
            IdentityKey::SessionId => Some(self.session_id()),
            IdentityKey::AgentId => Some(self.agent_id()),
            IdentityKey::UserId => self.user_id(),
            IdentityKey::ChannelId => self.channel_id(),
            IdentityKey::TenantId => self.tenant_id(),
            IdentityKey::RunId => self.run_id(),
            IdentityKey::Custom(k) => self.extra.get(k).map(String::as_str),
        }
    }

    pub fn has(&self, key: &IdentityKey) -> bool {
        self.get(key).is_some()
    }

    pub fn require(&self, key: &IdentityKey) -> Result<&str> {
        self.get(key)
            .ok_or_else(|| anyhow!("Missing required identity field '{}'", key.as_str()))
    }

    pub fn set_key(&mut self, key: IdentityKey, value: Option<String>) -> Result<()> {
        match key {
            IdentityKey::SessionId => {
                let v = value.ok_or_else(|| anyhow!("session_id cannot be nil"))?;
                self.session_id = v;
            }
            IdentityKey::AgentId => {
                let v = value.ok_or_else(|| anyhow!("agent_id cannot be nil"))?;
                self.agent_id = v;
            }
            IdentityKey::UserId => self.user_id = value,
            IdentityKey::ChannelId => self.channel_id = value,
            IdentityKey::TenantId => self.tenant_id = value,
            IdentityKey::RunId => self.run_id = value,
            IdentityKey::Custom(k) => {
                if let Some(v) = value {
                    self.extra.insert(k, v);
                } else {
                    self.extra.remove(&k);
                }
            }
        }
        Ok(())
    }

    pub fn validate_for(&self, scope: &str) -> Result<()> {
        match scope {
            "session" => {
                self.require(&IdentityKey::SessionId)?;
            }
            "agent" => {
                self.require(&IdentityKey::AgentId)?;
            }
            "user" => {
                self.require(&IdentityKey::UserId)?;
            }
            "channel" => {
                self.require(&IdentityKey::ChannelId)?;
            }
            "tenant" => {
                self.require(&IdentityKey::TenantId)?;
            }
            "run" => {
                self.require(&IdentityKey::RunId)?;
            }
            _ => {}
        }
        Ok(())
    }

    pub fn verify_access(&self, selector: &ContextSelector) -> Result<()> {
        // v1 simplistic policy
        if selector.visibility == "private" {
            let agent_tag = format!("agent:{}", self.agent_id());
            if !selector.tags.contains(&agent_tag) {
                return Err(anyhow!(
                    "Policy denial: Agent {} lacks private access to contexts outside its own tags.",
                    self.agent_id()
                ));
            }
        }
        Ok(())
    }
}

impl IdentityKey {
    pub fn as_str(&self) -> &str {
        match self {
            IdentityKey::SessionId => "session_id",
            IdentityKey::AgentId => "agent_id",
            IdentityKey::UserId => "user_id",
            IdentityKey::ChannelId => "channel_id",
            IdentityKey::TenantId => "tenant_id",
            IdentityKey::RunId => "run_id",
            IdentityKey::Custom(k) => k.as_str(),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_key_access_and_custom_fields() {
        let mut id = RuntimeIdentity::new("s1", "a1");
        id.set_key(IdentityKey::UserId, Some("u1".to_string()))
            .unwrap();
        id.set_key(
            IdentityKey::Custom("project_id".to_string()),
            Some("p1".to_string()),
        )
        .unwrap();

        assert_eq!(id.get(&IdentityKey::SessionId), Some("s1"));
        assert_eq!(id.get(&IdentityKey::UserId), Some("u1"));
        assert_eq!(
            id.get(&IdentityKey::Custom("project_id".to_string())),
            Some("p1")
        );
        assert!(id.has(&IdentityKey::AgentId));
        assert!(!id.has(&IdentityKey::ChannelId));
    }

    #[test]
    fn identity_validate_for_scope() {
        let id = RuntimeIdentity::new("s1", "a1");
        assert!(id.validate_for("session").is_ok());
        assert!(id.validate_for("agent").is_ok());
        assert!(id.validate_for("user").is_err());
    }
}
