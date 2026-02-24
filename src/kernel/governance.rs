use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::kernel::config::{
    GovernanceAuditMode, GovernanceConfig, GovernanceImportMode, GovernanceProfile,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GovernanceRootSnapshot {
    pub name: String,
    pub path: String,
    pub writable_hint: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_profile: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GovernanceAgentSnapshot {
    pub agent_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capability_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_child_agents: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GovernanceSnapshot {
    pub profile: GovernanceProfile,
    pub enforcement_enabled: bool,
    pub audit_mode: GovernanceAuditMode,
    pub audit_persist_before_hooks: bool,
    pub audit_include_capability_context: bool,
    pub import_mode: GovernanceImportMode,
    pub import_allow_unscoped_in_open: bool,
    pub capabilities_observability_only: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject_agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub roots: Vec<GovernanceRootSnapshot>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub agents: Vec<GovernanceAgentSnapshot>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub preset_capabilities: BTreeMap<String, serde_json::Value>,
    pub grants_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grants_max_ttl_ms: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct GovernanceManager {
    config: GovernanceConfig,
}

impl GovernanceManager {
    pub fn new(config: GovernanceConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &GovernanceConfig {
        &self.config
    }

    pub fn snapshot(&self) -> GovernanceSnapshot {
        self.snapshot_for_agent(None)
    }

    pub fn snapshot_for_agent(&self, agent_id: Option<&str>) -> GovernanceSnapshot {
        let mut roots: Vec<GovernanceRootSnapshot> = self
            .config
            .roots
            .iter()
            .map(|(name, root)| GovernanceRootSnapshot {
                name: name.clone(),
                path: root.path.clone(),
                writable_hint: root.writable_hint,
                default_profile: root.default_profile.clone(),
            })
            .collect();
        roots.sort_by(|a, b| a.name.cmp(&b.name));

        let mut agents: Vec<GovernanceAgentSnapshot> = self
            .config
            .agents
            .iter()
            .map(|(id, cfg)| GovernanceAgentSnapshot {
                agent_id: id.clone(),
                capability_profile: cfg.capability_profile.clone(),
                allowed_child_agents: cfg.allowed_child_agents.clone(),
            })
            .collect();
        agents.sort_by(|a, b| a.agent_id.cmp(&b.agent_id));

        GovernanceSnapshot {
            profile: self.config.profile.clone(),
            enforcement_enabled: self.config.enforcement_enabled,
            audit_mode: self.config.audit.mode.clone(),
            audit_persist_before_hooks: self.config.audit.persist_before_hooks.unwrap_or(matches!(
                self.config.audit.mode,
                GovernanceAuditMode::Immutable
            )),
            audit_include_capability_context: self.config.audit.include_capability_context,
            import_mode: self.config.import.mode.clone(),
            import_allow_unscoped_in_open: self.config.import.allow_unscoped_in_open,
            capabilities_observability_only: true,
            subject_agent_id: agent_id.map(str::to_string),
            roots,
            agents,
            preset_capabilities: preset_capabilities_for_profile(&self.config.profile),
            grants_enabled: self.config.grants.enabled,
            grants_max_ttl_ms: self.config.grants.max_ttl_ms,
        }
    }
}

fn preset_capabilities_for_profile(
    profile: &GovernanceProfile,
) -> BTreeMap<String, serde_json::Value> {
    let mut caps = BTreeMap::new();
    match profile {
        GovernanceProfile::Open => {
            caps.insert("runtime.db.*".into(), serde_json::Value::Bool(true));
            caps.insert("runtime.agent.*".into(), serde_json::Value::Bool(true));
            caps.insert("runtime.policy.set".into(), serde_json::Value::Bool(true));
            caps.insert("fs.write".into(), serde_json::Value::Bool(true));
            caps.insert("shell.exec".into(), serde_json::Value::Bool(true));
        }
        GovernanceProfile::Balanced => {
            caps.insert("runtime.db.query".into(), serde_json::Value::Bool(true));
            caps.insert("runtime.db.exec".into(), serde_json::Value::Bool(true));
            caps.insert("runtime.agent.submit".into(), serde_json::Value::Bool(true));
            caps.insert("runtime.agent.await".into(), serde_json::Value::Bool(true));
            caps.insert("runtime.agent.spawn".into(), serde_json::Value::Bool(true));
            caps.insert("runtime.policy.set".into(), serde_json::Value::Bool(true));
            caps.insert("fs.write".into(), serde_json::Value::Bool(true));
            caps.insert("shell.exec".into(), serde_json::Value::Bool(false));
        }
        GovernanceProfile::Governed => {
            caps.insert("runtime.db.query".into(), serde_json::Value::Bool(true));
            caps.insert("runtime.db.exec".into(), serde_json::Value::Bool(false));
            caps.insert("runtime.agent.submit".into(), serde_json::Value::Bool(true));
            caps.insert("runtime.agent.spawn".into(), serde_json::Value::Bool(false));
            caps.insert("runtime.policy.set".into(), serde_json::Value::Bool(false));
            caps.insert("fs.write".into(), serde_json::Value::Bool(false));
            caps.insert("shell.exec".into(), serde_json::Value::Bool(false));
        }
        GovernanceProfile::Custom => {}
    }
    caps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::config::{
        GovernanceAgentCapabilitiesConfig, GovernanceAuditConfig, GovernanceGrantsConfig,
        GovernanceImportConfig, GovernanceRootConfig,
    };

    #[test]
    fn snapshot_includes_profile_defaults_and_subject() {
        let mut cfg = GovernanceConfig {
            profile: GovernanceProfile::Balanced,
            enforcement_enabled: false,
            audit: GovernanceAuditConfig {
                mode: GovernanceAuditMode::Observational,
                include_capability_context: true,
                persist_before_hooks: None,
            },
            import: GovernanceImportConfig {
                mode: GovernanceImportMode::Mixed,
                default_root: Some("core".into()),
                allow_unscoped_in_open: false,
            },
            roots: Default::default(),
            agents: Default::default(),
            grants: GovernanceGrantsConfig {
                enabled: true,
                max_ttl_ms: Some(1000),
                require_audit_reason: false,
            },
        };
        cfg.roots.insert(
            "core".into(),
            GovernanceRootConfig {
                path: "harness/core".into(),
                writable_hint: false,
                default_profile: Some("core_full".into()),
                max_capabilities: Default::default(),
            },
        );
        cfg.agents.insert(
            "reviewer".into(),
            GovernanceAgentCapabilitiesConfig {
                capability_profile: Some("reviewer_ro".into()),
                max_capabilities: Default::default(),
                allowed_child_agents: vec!["worker".into()],
            },
        );

        let mgr = GovernanceManager::new(cfg);
        let snapshot = mgr.snapshot_for_agent(Some("reviewer"));
        assert_eq!(snapshot.profile, GovernanceProfile::Balanced);
        assert_eq!(snapshot.subject_agent_id.as_deref(), Some("reviewer"));
        assert_eq!(snapshot.audit_mode, GovernanceAuditMode::Observational);
        assert!(!snapshot.audit_persist_before_hooks);
        assert!(
            snapshot
                .preset_capabilities
                .contains_key("runtime.db.query")
        );
        assert_eq!(snapshot.roots.len(), 1);
        assert_eq!(snapshot.agents.len(), 1);
    }
}
