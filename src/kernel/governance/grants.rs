use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::GovernanceSubject;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GovernanceGrantSnapshot {
    pub grant_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issued_from_grant_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issuer_session_reference: Option<String>,
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
pub(super) struct ActiveGovernanceGrant {
    pub(super) snapshot: GovernanceGrantSnapshot,
}

pub(super) fn now_unix_ms() -> Result<u64, String> {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum GrantChainValidationError {
    NotActive(String),
    Expired(String),
    Forbidden(String),
    Invalid(String),
}

impl GrantChainValidationError {
    pub(super) fn into_message(self) -> String {
        match self {
            Self::NotActive(message)
            | Self::Expired(message)
            | Self::Forbidden(message)
            | Self::Invalid(message) => message,
        }
    }
}

pub(super) fn validate_grant_chain_locked(
    grants: &mut HashMap<String, ActiveGovernanceGrant>,
    subject: &GovernanceSubject,
    grant_id: &str,
    now_ms: u64,
) -> Result<GovernanceGrantSnapshot, GrantChainValidationError> {
    let mut current_id = grant_id.to_string();
    let mut seen = HashSet::new();
    let mut leaf_snapshot = None;

    loop {
        if !seen.insert(current_id.clone()) {
            grants.remove(grant_id);
            return Err(GrantChainValidationError::Invalid(format!(
                "Governance grant '{}' has cyclic delegation ancestry",
                grant_id
            )));
        }

        let Some(snapshot) = grants.get(&current_id).map(|entry| entry.snapshot.clone()) else {
            if current_id != grant_id {
                grants.remove(grant_id);
            }
            return Err(GrantChainValidationError::NotActive(format!(
                "Governance grant '{}' is not active",
                current_id
            )));
        };

        if grant_expired(&snapshot, now_ms) {
            grants.remove(&current_id);
            if current_id != grant_id {
                grants.remove(grant_id);
            }
            return Err(GrantChainValidationError::Expired(format!(
                "Governance grant '{}' has expired",
                current_id
            )));
        }

        if let Err(message) = ensure_grant_subject_access(subject, &snapshot) {
            if current_id != grant_id {
                grants.remove(grant_id);
            }
            return Err(GrantChainValidationError::Forbidden(message));
        }

        if leaf_snapshot.is_none() {
            leaf_snapshot = Some(snapshot.clone());
        }

        match snapshot.issued_from_grant_id {
            Some(parent_id) => current_id = parent_id,
            None => return Ok(leaf_snapshot.expect("leaf snapshot set")),
        }
    }
}

pub(super) fn ensure_grant_subject_access(
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
    if let Some(grant_session_reference) = grant.issuer_session_reference.as_deref() {
        let subject_session_reference = subject.session_reference.as_deref().unwrap_or("<unknown>");
        if subject_session_reference != grant_session_reference {
            return Err(format!(
                "grant '{}' was issued for session '{}' and cannot be used by session '{}'",
                grant.grant_id, grant_session_reference, subject_session_reference
            ));
        }
    }
    Ok(())
}
