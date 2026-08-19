mod capabilities;
mod grants;

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use turin_types::governance_templates;

use crate::kernel::config::{
    GovernanceAuditMode, GovernanceConfig, GovernanceImportMode, GovernanceUnmatchedCapability,
};

pub(crate) use capabilities::{
    capability_allowed_by_bool_rules, intersect_capability_bool_rules, tool_capability_name,
};
use capabilities::{
    capability_ceiling_denial_reason_bool_map, capability_ceiling_denial_reason_json_map,
    match_capability_rule,
};
pub use grants::GovernanceGrantSnapshot;
use grants::{
    ActiveGovernanceGrant, GrantChainValidationError, ensure_grant_subject_access, now_unix_ms,
    validate_grant_chain_locked,
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
    pub profile: String,
    pub enforcement_enabled: bool,
    pub audit_mode: GovernanceAuditMode,
    pub audit_persist_before_hooks: bool,
    pub audit_include_capability_context: bool,
    pub import_mode: GovernanceImportMode,
    pub import_allow_unscoped_in_open: bool,
    pub unmatched_capability: GovernanceUnmatchedCapability,
    pub capabilities_observability_only: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject_agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub roots: Vec<GovernanceRootSnapshot>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub agents: Vec<GovernanceAgentSnapshot>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub capabilities: BTreeMap<String, serde_json::Value>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject_grant_id: Option<String>,
    pub profile: String,
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
    pub session_reference: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub module_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grant_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub import_capabilities: Option<BTreeMap<String, bool>>,
}

#[derive(Debug, Clone)]
struct CapabilityBaseline {
    capabilities: BTreeMap<String, serde_json::Value>,
    unmatched_capability: GovernanceUnmatchedCapability,
}

#[derive(Debug, Deserialize)]
struct GovernanceTemplateDocument {
    governance: GovernanceConfig,
}

fn resolve_capability_baseline(config: &GovernanceConfig) -> CapabilityBaseline {
    if !config.capabilities.is_empty() {
        return CapabilityBaseline {
            capabilities: config.capabilities.clone(),
            unmatched_capability: config.unmatched_capability.clone(),
        };
    }

    let Some(template) = governance_templates::by_name(config.profile.as_str()) else {
        return CapabilityBaseline {
            capabilities: BTreeMap::new(),
            unmatched_capability: config.unmatched_capability.clone(),
        };
    };

    match toml::from_str::<GovernanceTemplateDocument>(template) {
        Ok(document) => CapabilityBaseline {
            capabilities: document.governance.capabilities,
            unmatched_capability: document.governance.unmatched_capability,
        },
        Err(_) => CapabilityBaseline {
            capabilities: BTreeMap::new(),
            unmatched_capability: config.unmatched_capability.clone(),
        },
    }
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
    baseline_capabilities: BTreeMap<String, serde_json::Value>,
    unmatched_capability: GovernanceUnmatchedCapability,
    grants: Arc<Mutex<HashMap<String, ActiveGovernanceGrant>>>,
}

impl GovernanceManager {
    pub fn new(config: GovernanceConfig) -> Self {
        let baseline = resolve_capability_baseline(&config);
        Self {
            config,
            baseline_capabilities: baseline.capabilities,
            unmatched_capability: baseline.unmatched_capability,
            grants: Arc::new(Mutex::new(HashMap::new())),
        }
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
            unmatched_capability: self.unmatched_capability.clone(),
            capabilities_observability_only: true,
            subject_agent_id: agent_id.map(str::to_string),
            roots,
            agents,
            capabilities: self.baseline_capabilities.clone(),
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
        let matched = match_capability_rule(&self.baseline_capabilities, capability);

        let baseline_allowed = match matched.allowed {
            Some(v) => v,
            None => matches!(
                self.unmatched_capability,
                GovernanceUnmatchedCapability::Allow
            ),
        };

        let mut ceiling_denial_reason = None;

        if baseline_allowed {
            if let Some(agent_id) = subject.agent_id.as_deref()
                && let Some(agent_cfg) = self.config.agents.get(agent_id)
                && let Some(profile_name) = agent_cfg.capability_profile.as_deref()
                && let Some(profile_caps) = self.config.capability_profiles.get(profile_name)
                && let Some(reason) = capability_ceiling_denial_reason_json_map(
                    profile_caps,
                    capability,
                    "agent capability_profile",
                    profile_name,
                    false,
                )
            {
                ceiling_denial_reason = Some(reason);
            }

            if ceiling_denial_reason.is_none()
                && let Some(agent_id) = subject.agent_id.as_deref()
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

            if ceiling_denial_reason.is_none()
                && let Some(grant_id) = subject.grant_id.as_deref()
                && let Some(reason) =
                    self.grant_ceiling_denial_reason(subject, grant_id, capability)
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
            Some(match &matched.rule {
                Some(rule) => format!(
                    "Governance denial: capability '{}' denied by profile '{}' (rule '{}')",
                    capability, self.config.profile, rule
                ),
                None => format!(
                    "Governance denial: capability '{}' denied by profile '{}' (no matching allow rule)",
                    capability, self.config.profile
                ),
            })
        };

        CapabilityDecision {
            capability: capability.to_string(),
            subject_agent_id: subject.agent_id.clone(),
            subject_module_name: subject.module_name.clone(),
            subject_root_name: subject.root_name.clone(),
            subject_grant_id: subject.grant_id.clone(),
            profile: self.config.profile.clone(),
            enforcement_enabled: self.config.enforcement_enabled,
            matched_rule: matched.rule,
            matched_via_wildcard: matched.via_wildcard,
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

    pub fn require_child_agent_for_subject(
        &self,
        subject: &GovernanceSubject,
        child_agent_id: &str,
    ) -> Result<(), String> {
        if !self.config.enforcement_enabled {
            return Ok(());
        }

        let Some(parent_agent_id) = subject.agent_id.as_deref() else {
            return Ok(());
        };

        let Some(agent_cfg) = self.config.agents.get(parent_agent_id) else {
            return Ok(());
        };

        if agent_cfg.allowed_child_agents.is_empty() {
            return Ok(());
        }

        if agent_cfg
            .allowed_child_agents
            .iter()
            .any(|id| id == child_agent_id)
        {
            Ok(())
        } else {
            Err(format!(
                "Governance denial: child agent '{}' is not allowed for agent '{}' (allowed_child_agents)",
                child_agent_id, parent_agent_id
            ))
        }
    }

    pub fn issue_grant_for_subject(
        &self,
        subject: &GovernanceSubject,
        capabilities: BTreeMap<String, bool>,
        ttl_ms: Option<u64>,
        max_uses: Option<u64>,
        reason: Option<String>,
    ) -> Result<GovernanceGrantSnapshot, String> {
        if !self.config.grants.enabled {
            return Err("Governance grants are disabled".to_string());
        }
        if capabilities.is_empty() {
            return Err("grant capabilities must not be empty".to_string());
        }
        if !capabilities.values().any(|v| *v) {
            return Err(
                "grant capabilities must include at least one allowed capability".to_string(),
            );
        }
        if let Some(ttl_ms) = ttl_ms {
            if ttl_ms == 0 {
                return Err("grant ttl_ms must be greater than 0".to_string());
            }
            if let Some(max_ttl_ms) = self.config.grants.max_ttl_ms
                && ttl_ms > max_ttl_ms
            {
                return Err(format!(
                    "grant ttl_ms {} exceeds governance.grants.max_ttl_ms {}",
                    ttl_ms, max_ttl_ms
                ));
            }
        }
        if let Some(max_uses) = max_uses
            && max_uses == 0
        {
            return Err("grant max_uses must be greater than 0".to_string());
        }
        if self.config.grants.require_audit_reason
            && reason.as_deref().is_none_or(|r| r.trim().is_empty())
        {
            return Err(
                "grant reason is required by governance.grants.require_audit_reason".to_string(),
            );
        }

        let issued_at_ms = now_unix_ms()?;
        let expires_at_ms = ttl_ms.map(|ttl| issued_at_ms.saturating_add(ttl));
        let grant_id = format!("g_{}", uuid::Uuid::new_v4().simple());
        let mut grants = self
            .grants
            .lock()
            .map_err(|_| "governance grants mutex poisoned")?;
        let issued_from_grant_id = if let Some(parent_grant_id) = subject.grant_id.as_deref() {
            let parent_snapshot =
                validate_grant_chain_locked(&mut grants, subject, parent_grant_id, issued_at_ms)
                    .map_err(GrantChainValidationError::into_message)?;
            for (capability, allowed) in &capabilities {
                if !*allowed {
                    continue;
                }
                if let Some(reason) = capability_ceiling_denial_reason_bool_map(
                    &parent_snapshot.capabilities,
                    capability,
                    "temporary grant",
                    parent_grant_id,
                    true,
                ) {
                    return Err(reason);
                }
            }
            Some(parent_grant_id.to_string())
        } else {
            None
        };
        let snapshot = GovernanceGrantSnapshot {
            grant_id: grant_id.clone(),
            issued_from_grant_id,
            issuer_session_reference: subject.session_reference.clone(),
            issuer_agent_id: subject.agent_id.clone(),
            issuer_module_name: subject.module_name.clone(),
            issuer_root_name: subject.root_name.clone(),
            reason: reason.filter(|r| !r.trim().is_empty()),
            capabilities,
            issued_at_ms,
            expires_at_ms,
            max_uses,
            uses_remaining: max_uses,
        };

        grants.insert(
            grant_id,
            ActiveGovernanceGrant {
                snapshot: snapshot.clone(),
            },
        );
        Ok(snapshot)
    }

    pub fn grant_snapshot_for_subject(
        &self,
        subject: &GovernanceSubject,
        grant_id: &str,
    ) -> Result<Option<GovernanceGrantSnapshot>, String> {
        let mut grants = self
            .grants
            .lock()
            .map_err(|_| "governance grants mutex poisoned")?;
        let now_ms = now_unix_ms()?;
        match validate_grant_chain_locked(&mut grants, subject, grant_id, now_ms) {
            Ok(snapshot) => Ok(Some(snapshot)),
            Err(
                GrantChainValidationError::NotActive(_)
                | GrantChainValidationError::Expired(_)
                | GrantChainValidationError::Invalid(_),
            ) => Ok(None),
            Err(GrantChainValidationError::Forbidden(message)) => Err(message),
        }
    }

    pub fn revoke_grant_for_subject(
        &self,
        subject: &GovernanceSubject,
        grant_id: &str,
    ) -> Result<Option<GovernanceGrantSnapshot>, String> {
        if !self.config.grants.enabled {
            return Err("Governance grants are disabled".to_string());
        }
        let mut grants = self
            .grants
            .lock()
            .map_err(|_| "governance grants mutex poisoned")?;
        let Some(entry) = grants.get(grant_id) else {
            return Ok(None);
        };
        ensure_grant_subject_access(subject, &entry.snapshot)?;
        let removed = grants.remove(grant_id).map(|e| e.snapshot);
        Ok(removed)
    }

    pub fn enter_grant_for_subject(
        &self,
        subject: &GovernanceSubject,
        grant_id: &str,
    ) -> Result<GovernanceGrantSnapshot, String> {
        if !self.config.grants.enabled {
            return Err("Governance grants are disabled".to_string());
        }
        let mut grants = self
            .grants
            .lock()
            .map_err(|_| "governance grants mutex poisoned")?;
        let now_ms = now_unix_ms()?;
        validate_grant_chain_locked(&mut grants, subject, grant_id, now_ms)
            .map_err(GrantChainValidationError::into_message)?;

        let entry = grants
            .get_mut(grant_id)
            .ok_or_else(|| format!("Governance grant '{}' not found", grant_id))?;

        if let Some(uses_remaining) = entry.snapshot.uses_remaining.as_mut() {
            if *uses_remaining == 0 {
                return Err(format!(
                    "Governance grant '{}' has no uses remaining",
                    grant_id
                ));
            }
            *uses_remaining -= 1;
        }

        let snapshot = entry.snapshot.clone();
        if snapshot.uses_remaining == Some(0) {
            grants.remove(grant_id);
        }
        Ok(snapshot)
    }

    fn grant_ceiling_denial_reason(
        &self,
        subject: &GovernanceSubject,
        grant_id: &str,
        capability: &str,
    ) -> Option<String> {
        let now_ms = now_unix_ms().ok()?;
        let mut grants = self.grants.lock().ok()?;
        let entry = match validate_grant_chain_locked(&mut grants, subject, grant_id, now_ms) {
            Ok(snapshot) => snapshot,
            Err(err) => return Some(format!("Governance denial: {}", err.into_message())),
        };

        capability_ceiling_denial_reason_bool_map(
            &entry.capabilities,
            capability,
            "temporary grant",
            grant_id,
            true,
        )
    }
}

#[cfg(test)]
#[path = "tests/governance.rs"]
mod tests;
