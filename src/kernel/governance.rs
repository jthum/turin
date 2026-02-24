use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject_grant_id: Option<String>,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GovernanceGrantSnapshot {
    pub grant_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issuer_agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issuer_module_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issuer_root_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub capabilities: BTreeMap<String, bool>,
    pub issued_at_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uses_remaining: Option<u64>,
}

#[derive(Debug, Clone)]
struct ActiveGovernanceGrant {
    snapshot: GovernanceGrantSnapshot,
}

#[derive(Debug, Clone)]
pub struct GovernanceManager {
    config: GovernanceConfig,
    grants: Arc<Mutex<HashMap<String, ActiveGovernanceGrant>>>,
}

impl GovernanceManager {
    pub fn new(config: GovernanceConfig) -> Self {
        Self {
            config,
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
                && let Some(reason) = self.grant_ceiling_denial_reason(subject, grant_id, capability)
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
            subject_grant_id: subject.grant_id.clone(),
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
            return Err("grant capabilities must include at least one allowed capability".to_string());
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
            return Err("grant reason is required by governance.grants.require_audit_reason".to_string());
        }

        let issued_at_ms = now_unix_ms()?;
        let expires_at_ms = ttl_ms.map(|ttl| issued_at_ms.saturating_add(ttl));
        let grant_id = format!("g_{}", uuid::Uuid::now_v7().simple());
        let snapshot = GovernanceGrantSnapshot {
            grant_id: grant_id.clone(),
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

        let mut grants = self.grants.lock().map_err(|_| "governance grants mutex poisoned")?;
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
        let mut grants = self.grants.lock().map_err(|_| "governance grants mutex poisoned")?;
        let now_ms = now_unix_ms()?;
        if let Some(entry) = grants.get(grant_id)
            && grant_expired(&entry.snapshot, now_ms)
        {
            grants.remove(grant_id);
            return Ok(None);
        }
        let Some(entry) = grants.get(grant_id) else {
            return Ok(None);
        };
        ensure_grant_subject_access(subject, &entry.snapshot)?;
        Ok(Some(entry.snapshot.clone()))
    }

    pub fn revoke_grant_for_subject(
        &self,
        subject: &GovernanceSubject,
        grant_id: &str,
    ) -> Result<bool, String> {
        if !self.config.grants.enabled {
            return Err("Governance grants are disabled".to_string());
        }
        let mut grants = self.grants.lock().map_err(|_| "governance grants mutex poisoned")?;
        let Some(entry) = grants.get(grant_id) else {
            return Ok(false);
        };
        ensure_grant_subject_access(subject, &entry.snapshot)?;
        grants.remove(grant_id);
        Ok(true)
    }

    pub fn enter_grant_for_subject(
        &self,
        subject: &GovernanceSubject,
        grant_id: &str,
    ) -> Result<GovernanceGrantSnapshot, String> {
        if !self.config.grants.enabled {
            return Err("Governance grants are disabled".to_string());
        }
        let mut grants = self.grants.lock().map_err(|_| "governance grants mutex poisoned")?;
        let now_ms = now_unix_ms()?;
        if let Some(entry) = grants.get(grant_id)
            && grant_expired(&entry.snapshot, now_ms)
        {
            grants.remove(grant_id);
            return Err(format!(
                "Governance grant '{}' has expired",
                grant_id
            ));
        }

        let entry = grants
            .get_mut(grant_id)
            .ok_or_else(|| format!("Governance grant '{}' not found", grant_id))?;
        ensure_grant_subject_access(subject, &entry.snapshot)?;

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
        let Some(entry) = grants.get(grant_id) else {
            return Some(format!(
                "Governance denial: grant '{}' is not active",
                grant_id
            ));
        };

        if grant_expired(&entry.snapshot, now_ms) {
            grants.remove(grant_id);
            return Some(format!(
                "Governance denial: grant '{}' has expired",
                grant_id
            ));
        }

        if let Err(err) = ensure_grant_subject_access(subject, &entry.snapshot) {
            return Some(format!("Governance denial: {}", err));
        }

        capability_ceiling_denial_reason_bool_map(
            &entry.snapshot.capabilities,
            capability,
            "temporary grant",
            grant_id,
            true,
        )
    }
}

fn now_unix_ms() -> Result<u64, String> {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("system clock error: {}", e))?;
    Ok(dur.as_millis().try_into().unwrap_or(u64::MAX))
}

fn grant_expired(grant: &GovernanceGrantSnapshot, now_ms: u64) -> bool {
    grant
        .expires_at_ms
        .is_some_and(|expires_at_ms| now_ms >= expires_at_ms)
}

fn ensure_grant_subject_access(
    subject: &GovernanceSubject,
    grant: &GovernanceGrantSnapshot,
) -> Result<(), String> {
    if let Some(grant_agent_id) = grant.issuer_agent_id.as_deref() {
        let subject_agent_id = subject.agent_id.as_deref().unwrap_or("<unknown>");
        if subject_agent_id != grant_agent_id {
            return Err(format!(
                "grant '{}' was issued for agent '{}' and cannot be used by agent '{}'",
                grant.grant_id, grant_agent_id, subject_agent_id
            ));
        }
    }
    Ok(())
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
            caps.insert(
                "runtime.governance.grant.*".into(),
                serde_json::Value::Bool(true),
            );
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
                "runtime.governance.grant.*".into(),
                serde_json::Value::Bool(true),
            );
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
                "runtime.governance.grant.*".into(),
                serde_json::Value::Bool(true),
            );
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
            capability_profiles: Default::default(),
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

    #[test]
    fn allowed_child_agents_is_opt_in_and_enforced_when_configured() {
        let mut cfg = GovernanceConfig {
            profile: GovernanceProfile::Balanced,
            enforcement_enabled: true,
            ..GovernanceConfig::default()
        };
        cfg.agents.insert(
            "orchestrator".into(),
            crate::kernel::config::GovernanceAgentCapabilitiesConfig {
                capability_profile: None,
                max_capabilities: Default::default(),
                allowed_child_agents: vec!["worker_allowed".into()],
            },
        );
        let mgr = GovernanceManager::new(cfg);
        let subject = GovernanceSubject {
            agent_id: Some("orchestrator".into()),
            ..GovernanceSubject::default()
        };

        assert!(
            mgr.require_child_agent_for_subject(&subject, "worker_allowed")
                .is_ok()
        );
        let err = mgr
            .require_child_agent_for_subject(&subject, "worker_blocked")
            .unwrap_err();
        assert!(err.contains("allowed_child_agents"));
    }

    #[test]
    fn agent_capability_profile_applies_named_capability_ceiling() {
        let mut cfg = GovernanceConfig {
            profile: GovernanceProfile::Balanced,
            enforcement_enabled: true,
            ..GovernanceConfig::default()
        };
        cfg.capability_profiles.insert(
            "reviewer_ro".into(),
            HashMap::from([
                ("runtime.db.query".to_string(), serde_json::Value::Bool(true)),
                ("runtime.policy.set".to_string(), serde_json::Value::Bool(false)),
            ]),
        );
        cfg.agents.insert(
            "reviewer".into(),
            crate::kernel::config::GovernanceAgentCapabilitiesConfig {
                capability_profile: Some("reviewer_ro".into()),
                max_capabilities: Default::default(),
                allowed_child_agents: vec![],
            },
        );

        let mgr = GovernanceManager::new(cfg);
        let reviewer = GovernanceSubject {
            agent_id: Some("reviewer".into()),
            ..GovernanceSubject::default()
        };
        let default_agent = GovernanceSubject {
            agent_id: Some("default".into()),
            ..GovernanceSubject::default()
        };

        let reviewer_policy = mgr.capability_decision_for_subject(&reviewer, "runtime.policy.set");
        assert!(!reviewer_policy.allowed);
        assert!(
            reviewer_policy
                .reason
                .as_deref()
                .unwrap()
                .contains("agent capability_profile")
        );

        let reviewer_query = mgr.capability_decision_for_subject(&reviewer, "runtime.db.query");
        assert!(reviewer_query.allowed);

        let default_policy = mgr.capability_decision_for_subject(&default_agent, "runtime.policy.set");
        assert!(default_policy.allowed);
    }

    #[test]
    fn temporary_grants_apply_ceiling_and_consume_max_uses() {
        let cfg = GovernanceConfig {
            profile: GovernanceProfile::Balanced,
            enforcement_enabled: true,
            grants: GovernanceGrantsConfig {
                enabled: true,
                max_ttl_ms: Some(5_000),
                require_audit_reason: true,
            },
            ..GovernanceConfig::default()
        };
        let mgr = GovernanceManager::new(cfg);
        let subject = GovernanceSubject {
            agent_id: Some("default".into()),
            ..GovernanceSubject::default()
        };

        let grant = mgr
            .issue_grant_for_subject(
                &subject,
                BTreeMap::from([("runtime.db.query".into(), true)]),
                Some(1_000),
                Some(2),
                Some("one-shot test".into()),
            )
            .unwrap();

        let entered = mgr.enter_grant_for_subject(&subject, &grant.grant_id).unwrap();
        assert_eq!(entered.max_uses, Some(2));
        assert_eq!(entered.uses_remaining, Some(1));

        let granted_subject = GovernanceSubject {
            grant_id: Some(grant.grant_id.clone()),
            ..subject.clone()
        };
        let deny_policy = mgr.capability_decision_for_subject(&granted_subject, "runtime.policy.set");
        assert!(!deny_policy.allowed);
        assert!(
            deny_policy
                .reason
                .as_deref()
                .unwrap()
                .contains("temporary grant")
        );
        let allow_query = mgr.capability_decision_for_subject(&granted_subject, "runtime.db.query");
        assert!(allow_query.allowed);
        assert_eq!(allow_query.subject_grant_id.as_deref(), Some(grant.grant_id.as_str()));

        let second_enter = mgr.enter_grant_for_subject(&subject, &grant.grant_id);
        assert!(second_enter.is_ok());
        let third_enter = mgr.enter_grant_for_subject(&subject, &grant.grant_id);
        assert!(third_enter.is_err());
    }
}
