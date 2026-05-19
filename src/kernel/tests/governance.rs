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
    assert_eq!(tool_capability_name("apply_patch"), Some("fs.write"));
    assert_eq!(tool_capability_name("shell_exec"), Some("shell.exec"));
    assert_eq!(
        tool_capability_name("bridge_mcp"),
        Some("integration.mcp.bridge")
    );
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
        session_reference: None,
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
            (
                "runtime.db.query".to_string(),
                serde_json::Value::Bool(true),
            ),
            (
                "runtime.policy.set".to_string(),
                serde_json::Value::Bool(false),
            ),
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

    let entered = mgr
        .enter_grant_for_subject(&subject, &grant.grant_id)
        .unwrap();
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
    assert_eq!(
        allow_query.subject_grant_id.as_deref(),
        Some(grant.grant_id.as_str())
    );

    let second_enter = mgr.enter_grant_for_subject(&subject, &grant.grant_id);
    assert!(second_enter.is_ok());
    let third_enter = mgr.enter_grant_for_subject(&subject, &grant.grant_id);
    assert!(third_enter.is_err());
}

#[test]
fn delegated_grants_record_parent_and_invalidate_when_parent_is_revoked() {
    let mgr = GovernanceManager::new(GovernanceConfig {
        grants: GovernanceGrantsConfig {
            enabled: true,
            max_ttl_ms: Some(10_000),
            require_audit_reason: false,
        },
        ..GovernanceConfig::default()
    });
    let subject = GovernanceSubject {
        agent_id: Some("default".into()),
        ..GovernanceSubject::default()
    };

    let parent = mgr
        .issue_grant_for_subject(
            &subject,
            BTreeMap::from([("runtime.db.query".into(), true)]),
            Some(10_000),
            Some(2),
            Some("parent".into()),
        )
        .unwrap();
    let delegated_subject = GovernanceSubject {
        grant_id: Some(parent.grant_id.clone()),
        ..subject.clone()
    };
    let child = mgr
        .issue_grant_for_subject(
            &delegated_subject,
            BTreeMap::from([("runtime.db.query".into(), true)]),
            Some(10_000),
            Some(1),
            Some("child".into()),
        )
        .unwrap();

    assert_eq!(
        child.issued_from_grant_id.as_deref(),
        Some(parent.grant_id.as_str())
    );
    assert!(
        mgr.grant_snapshot_for_subject(&subject, &child.grant_id)
            .unwrap()
            .is_some()
    );

    mgr.revoke_grant_for_subject(&subject, &parent.grant_id)
        .unwrap();

    assert!(
        mgr.grant_snapshot_for_subject(&subject, &child.grant_id)
            .unwrap()
            .is_none()
    );
    assert!(
        mgr.enter_grant_for_subject(&subject, &child.grant_id)
            .is_err()
    );
}

#[test]
fn delegated_grants_cannot_exceed_parent_capability_ceiling() {
    let mgr = GovernanceManager::new(GovernanceConfig {
        grants: GovernanceGrantsConfig {
            enabled: true,
            max_ttl_ms: Some(10_000),
            require_audit_reason: false,
        },
        ..GovernanceConfig::default()
    });
    let subject = GovernanceSubject {
        agent_id: Some("default".into()),
        ..GovernanceSubject::default()
    };

    let parent = mgr
        .issue_grant_for_subject(
            &subject,
            BTreeMap::from([("runtime.db.query".into(), true)]),
            Some(10_000),
            Some(2),
            Some("parent".into()),
        )
        .unwrap();
    let delegated_subject = GovernanceSubject {
        grant_id: Some(parent.grant_id.clone()),
        ..subject
    };

    let err = mgr
        .issue_grant_for_subject(
            &delegated_subject,
            BTreeMap::from([("runtime.policy.set".into(), true)]),
            Some(10_000),
            Some(1),
            Some("child".into()),
        )
        .unwrap_err();
    assert!(err.contains("temporary grant"));
}

#[test]
fn grants_bound_to_store_qualified_session_reference() {
    let mgr = GovernanceManager::new(GovernanceConfig {
        grants: GovernanceGrantsConfig {
            enabled: true,
            max_ttl_ms: Some(10_000),
            require_audit_reason: false,
        },
        ..GovernanceConfig::default()
    });
    let issuing_subject = GovernanceSubject {
        agent_id: Some("default".into()),
        session_reference: Some("018f1f4f1f4f4f4f8f8f8f8f8f8f8f8f@telegram".into()),
        ..GovernanceSubject::default()
    };

    let grant = mgr
        .issue_grant_for_subject(
            &issuing_subject,
            BTreeMap::from([("runtime.db.query".into(), true)]),
            Some(10_000),
            Some(1),
            Some("bound".into()),
        )
        .unwrap();
    assert_eq!(
        grant.issuer_session_reference.as_deref(),
        issuing_subject.session_reference.as_deref()
    );

    let same_session = GovernanceSubject {
        session_reference: issuing_subject.session_reference.clone(),
        ..issuing_subject.clone()
    };
    assert!(
        mgr.grant_snapshot_for_subject(&same_session, &grant.grant_id)
            .unwrap()
            .is_some()
    );

    let different_session = GovernanceSubject {
        session_reference: Some("018f1f4f1f4f4f4f8f8f8f8f8f8f8f8f@rocketchat".into()),
        ..issuing_subject
    };
    let err = mgr
        .grant_snapshot_for_subject(&different_session, &grant.grant_id)
        .unwrap_err();
    assert!(err.contains("issued for session"));
}
