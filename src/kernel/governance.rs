use std::collections::{BTreeMap, HashMap};

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CapabilityDecision {
    pub capability: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject_agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject_module_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject_root_name: Option<String>,
    pub profile: GovernanceProfile,
    pub enforcement_enabled: bool,
    pub matched_rule: Option<String>,
    pub matched_via_wildcard: bool,
    pub baseline_allowed: bool,
    pub allowed: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct GovernanceSubject {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub module_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grant_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub import_capabilities: Option<BTreeMap<String, bool>>,
}

impl GovernanceSubject {
    pub fn for_agent(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: Some(agent_id.into()),
            ..Self::default()
        }
    }
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

    pub fn capability_decision(
        &self,
        agent_id: Option<&str>,
        capability: &str,
    ) -> CapabilityDecision {
        let subject = GovernanceSubject {
            agent_id: agent_id.map(str::to_string),
            ..GovernanceSubject::default()
        };
        self.capability_decision_for_subject(&subject, capability)
    }

    pub fn capability_decision_for_subject(
        &self,
        subject: &GovernanceSubject,
        capability: &str,
    ) -> CapabilityDecision {
        let caps = preset_capabilities_for_profile(&self.config.profile);
        let (matched_rule, matched_via_wildcard, matched_value) =
            match_capability_rule(&caps, capability);

        let baseline_allowed = match matched_value {
            Some(v) => v,
            None => matches!(self.config.profile, GovernanceProfile::Open),
        };

        let mut ceiling_denial_reason = None;

        if baseline_allowed {
            if let Some(agent_id) = subject.agent_id.as_deref()
                && let Some(agent_cfg) = self.config.agents.get(agent_id)
                && let Some(reason) = capability_ceiling_denial_reason_json_map(
                    &agent_cfg.max_capabilities,
                    capability,
                    "agent max_capabilities",
                    agent_id,
                    false,
                )
            {
                ceiling_denial_reason = Some(reason);
            }

            if ceiling_denial_reason.is_none()
                && let Some(root_name) = subject.root_name.as_deref()
                && let Some(root_cfg) = self.config.roots.get(root_name)
                && let Some(reason) = capability_ceiling_denial_reason_json_map(
                    &root_cfg.max_capabilities,
                    capability,
                    "root max_capabilities",
                    root_name,
                    false,
                )
            {
                ceiling_denial_reason = Some(reason);
            }

            if ceiling_denial_reason.is_none()
                && let Some(import_caps) = subject.import_capabilities.as_ref()
                && let Some(reason) = capability_ceiling_denial_reason_bool_map(
                    import_caps,
                    capability,
                    "delegated capabilities",
                    subject.module_name.as_deref().unwrap_or("<unknown>"),
                    true,
                )
            {
                ceiling_denial_reason = Some(reason);
            }
        }

        let effective_allowed = baseline_allowed && ceiling_denial_reason.is_none();

        let allowed = if self.config.enforcement_enabled {
            effective_allowed
        } else {
            true
        };

        let reason = if allowed {
            None
        } else if baseline_allowed {
            ceiling_denial_reason
        } else {
            Some(match &matched_rule {
                Some(rule) => format!(
                    "Governance denial: capability '{}' denied by profile '{}' (rule '{}')",
                    capability,
                    profile_name(&self.config.profile),
                    rule
                ),
                None => format!(
                    "Governance denial: capability '{}' denied by profile '{}' (no matching allow rule)",
                    capability,
                    profile_name(&self.config.profile)
                ),
            })
        };

        CapabilityDecision {
            capability: capability.to_string(),
            subject_agent_id: subject.agent_id.clone(),
            subject_module_name: subject.module_name.clone(),
            subject_root_name: subject.root_name.clone(),
            profile: self.config.profile.clone(),
            enforcement_enabled: self.config.enforcement_enabled,
            matched_rule,
            matched_via_wildcard,
            baseline_allowed,
            allowed,
            reason,
        }
    }

    pub fn require_capability(
        &self,
        agent_id: Option<&str>,
        capability: &str,
    ) -> Result<(), String> {
        let subject = GovernanceSubject {
            agent_id: agent_id.map(str::to_string),
            ..GovernanceSubject::default()
        };
        self.require_capability_for_subject(&subject, capability)
    }

    pub fn require_capability_for_subject(
        &self,
        subject: &GovernanceSubject,
        capability: &str,
    ) -> Result<(), String> {
        let decision = self.capability_decision_for_subject(subject, capability);
        if decision.allowed {
            Ok(())
        } else {
            Err(decision
                .reason
                .unwrap_or_else(|| "Governance denial".to_string()))
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
            caps.insert("harness.import.*".into(), serde_json::Value::Bool(true));
            caps.insert("fs.read".into(), serde_json::Value::Bool(true));
            caps.insert("fs.write".into(), serde_json::Value::Bool(true));
            caps.insert("shell.exec".into(), serde_json::Value::Bool(true));
        }
        GovernanceProfile::Balanced => {
            caps.insert("runtime.db.query".into(), serde_json::Value::Bool(true));
            caps.insert("runtime.db.exec".into(), serde_json::Value::Bool(true));
            caps.insert("runtime.agent.submit".into(), serde_json::Value::Bool(true));
            caps.insert("runtime.agent.await".into(), serde_json::Value::Bool(true));
            caps.insert("runtime.agent.status".into(), serde_json::Value::Bool(true));
            caps.insert("runtime.agent.spawn".into(), serde_json::Value::Bool(true));
            caps.insert("runtime.policy.set".into(), serde_json::Value::Bool(true));
            caps.insert(
                "harness.import.unscoped".into(),
                serde_json::Value::Bool(true),
            );
            caps.insert(
                "harness.import.scoped".into(),
                serde_json::Value::Bool(true),
            );
            caps.insert("fs.read".into(), serde_json::Value::Bool(true));
            caps.insert("fs.write".into(), serde_json::Value::Bool(true));
            caps.insert("shell.exec".into(), serde_json::Value::Bool(false));
        }
        GovernanceProfile::Governed => {
            caps.insert("runtime.db.query".into(), serde_json::Value::Bool(true));
            caps.insert("runtime.db.exec".into(), serde_json::Value::Bool(false));
            caps.insert("runtime.agent.submit".into(), serde_json::Value::Bool(true));
            caps.insert("runtime.agent.await".into(), serde_json::Value::Bool(true));
            caps.insert("runtime.agent.status".into(), serde_json::Value::Bool(true));
            caps.insert("runtime.agent.spawn".into(), serde_json::Value::Bool(false));
            caps.insert("runtime.policy.set".into(), serde_json::Value::Bool(false));
            caps.insert(
                "harness.import.unscoped".into(),
                serde_json::Value::Bool(false),
            );
            caps.insert(
                "harness.import.scoped".into(),
                serde_json::Value::Bool(true),
            );
            caps.insert("fs.read".into(), serde_json::Value::Bool(true));
            caps.insert("fs.write".into(), serde_json::Value::Bool(false));
            caps.insert("shell.exec".into(), serde_json::Value::Bool(false));
        }
        GovernanceProfile::Custom => {}
    }
    caps
}

fn match_capability_rule(
    caps: &BTreeMap<String, serde_json::Value>,
    capability: &str,
) -> (Option<String>, bool, Option<bool>) {
    match_capability_rule_bool_iter(
        caps.iter()
            .filter_map(|(rule, value)| value.as_bool().map(|b| (rule.as_str(), b))),
        capability,
    )
}

fn match_capability_rule_json_map(
    caps: &HashMap<String, serde_json::Value>,
    capability: &str,
) -> (Option<String>, bool, Option<bool>) {
    match_capability_rule_bool_iter(
        caps.iter()
            .filter_map(|(rule, value)| value.as_bool().map(|b| (rule.as_str(), b))),
        capability,
    )
}

fn match_capability_rule_bool_map(
    caps: &BTreeMap<String, bool>,
    capability: &str,
) -> (Option<String>, bool, Option<bool>) {
    match_capability_rule_bool_iter(
        caps.iter().map(|(rule, value)| (rule.as_str(), *value)),
        capability,
    )
}

fn match_capability_rule_bool_iter<'a, I>(
    iter: I,
    capability: &str,
) -> (Option<String>, bool, Option<bool>)
where
    I: IntoIterator<Item = (&'a str, bool)>,
{
    let entries: Vec<(&'a str, bool)> = iter.into_iter().collect();
    if let Some((_, v)) = entries.iter().find(|(rule, _)| *rule == capability) {
        return (Some(capability.to_string()), false, Some(*v));
    }

    let mut best: Option<(&str, bool)> = None;
    for (rule, b) in entries {
        let Some(prefix) = rule.strip_suffix(".*") else {
            continue;
        };
        if capability == prefix || capability.starts_with(&format!("{prefix}.")) {
            match best {
                Some((best_rule, _)) if best_rule.len() >= rule.len() => {}
                _ => best = Some((rule, b)),
            }
        }
    }

    match best {
        Some((rule, b)) => (Some(rule.to_string()), true, Some(b)),
        None => (None, false, None),
    }
}

fn capability_ceiling_denial_reason_json_map(
    caps: &HashMap<String, serde_json::Value>,
    capability: &str,
    source_kind: &str,
    source_name: &str,
    default_deny_on_no_match: bool,
) -> Option<String> {
    if caps.is_empty() {
        return None;
    }
    let (matched_rule, _, matched_value) = match_capability_rule_json_map(caps, capability);
    let allowed = match matched_value {
        Some(v) => v,
        None => !default_deny_on_no_match,
    };
    if allowed {
        None
    } else {
        Some(match matched_rule {
            Some(rule) => format!(
                "Governance denial: capability '{}' denied by {} '{}' (rule '{}')",
                capability, source_kind, source_name, rule
            ),
            None => format!(
                "Governance denial: capability '{}' denied by {} '{}' (no matching allow rule)",
                capability, source_kind, source_name
            ),
        })
    }
}

fn capability_ceiling_denial_reason_bool_map(
    caps: &BTreeMap<String, bool>,
    capability: &str,
    source_kind: &str,
    source_name: &str,
    default_deny_on_no_match: bool,
) -> Option<String> {
    if caps.is_empty() {
        return if default_deny_on_no_match {
            Some(format!(
                "Governance denial: capability '{}' denied by {} '{}' (empty allowlist)",
                capability, source_kind, source_name
            ))
        } else {
            None
        };
    }
    let (matched_rule, _, matched_value) = match_capability_rule_bool_map(caps, capability);
    let allowed = match matched_value {
        Some(v) => v,
        None => !default_deny_on_no_match,
    };
    if allowed {
        None
    } else {
        Some(match matched_rule {
            Some(rule) => format!(
                "Governance denial: capability '{}' denied by {} '{}' (rule '{}')",
                capability, source_kind, source_name, rule
            ),
            None => format!(
                "Governance denial: capability '{}' denied by {} '{}' (no matching allow rule)",
                capability, source_kind, source_name
            ),
        })
    }
}

fn profile_name(profile: &GovernanceProfile) -> &'static str {
    match profile {
        GovernanceProfile::Open => "open",
        GovernanceProfile::Balanced => "balanced",
        GovernanceProfile::Governed => "governed",
        GovernanceProfile::Custom => "custom",
    }
}

pub(crate) fn tool_capability_name(tool_name: &str) -> Option<&'static str> {
    match tool_name {
        "read_file" => Some("fs.read"),
        "write_file" | "edit_file" => Some("fs.write"),
        "shell_exec" => Some("shell.exec"),
        _ => None,
    }
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

    #[test]
    fn capability_decision_respects_profile_and_enforcement() {
        let mut cfg = GovernanceConfig {
            profile: GovernanceProfile::Governed,
            enforcement_enabled: true,
            ..GovernanceConfig::default()
        };
        let mgr = GovernanceManager::new(cfg.clone());

        let deny_exec = mgr.capability_decision(Some("default"), "runtime.db.exec");
        assert!(!deny_exec.allowed);
        assert_eq!(deny_exec.matched_rule.as_deref(), Some("runtime.db.exec"));

        let allow_query = mgr.capability_decision(Some("default"), "runtime.db.query");
        assert!(allow_query.allowed);

        let deny_unknown = mgr.capability_decision(Some("default"), "runtime.db.list_handles");
        assert!(!deny_unknown.allowed);
        assert!(
            deny_unknown
                .reason
                .as_deref()
                .unwrap()
                .contains("no matching allow rule")
        );

        cfg.enforcement_enabled = false;
        let mgr_obs = GovernanceManager::new(cfg);
        let observed = mgr_obs.capability_decision(Some("default"), "runtime.db.exec");
        assert!(!observed.baseline_allowed);
        assert!(observed.allowed, "observability mode should not deny");
    }

    #[test]
    fn tool_capability_mapping_covers_high_risk_builtins() {
        assert_eq!(tool_capability_name("read_file"), Some("fs.read"));
        assert_eq!(tool_capability_name("write_file"), Some("fs.write"));
        assert_eq!(tool_capability_name("edit_file"), Some("fs.write"));
        assert_eq!(tool_capability_name("shell_exec"), Some("shell.exec"));
        assert_eq!(tool_capability_name("submit_plan"), None);
    }

    #[test]
    fn capability_decision_preserves_module_subject_context() {
        let mgr = GovernanceManager::new(GovernanceConfig {
            profile: GovernanceProfile::Balanced,
            enforcement_enabled: false,
            ..GovernanceConfig::default()
        });
        let subject = GovernanceSubject {
            agent_id: Some("default".into()),
            module_name: Some("planner".into()),
            root_name: None,
            grant_id: None,
            import_capabilities: None,
        };
        let decision = mgr.capability_decision_for_subject(&subject, "runtime.db.query");
        assert_eq!(decision.subject_agent_id.as_deref(), Some("default"));
        assert_eq!(decision.subject_module_name.as_deref(), Some("planner"));
    }
}
