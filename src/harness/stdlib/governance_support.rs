use crate::harness::globals::HarnessAppData;
use crate::kernel::event::{AuditEvent, KernelEvent};
use crate::kernel::governance::{
    CapabilityDecision, GovernanceSubject, capability_allowed_by_bool_rules,
    intersect_capability_bool_rules,
};
use crate::kernel::session::{PersistedKernelEvent, PersistedKernelRecord};
use crate::kernel::session_refs::format_session_reference;
use mlua::{Result as LuaResult, Table, Value};
use std::collections::BTreeMap;

pub(crate) fn current_agent_id(app_data: &HarnessAppData) -> String {
    if let Ok(lock) = app_data.execution_ctx.lock()
        && let Some(agent_id) = lock.agent_id.as_deref()
    {
        return agent_id.to_string();
    }
    app_data.config.agent.id.clone()
}

pub(crate) fn capability_decision(
    app_data: &HarnessAppData,
    capability: &str,
) -> CapabilityDecision {
    let subject = current_subject(app_data);
    app_data
        .governance_manager
        .capability_decision_for_subject(&subject, capability)
}

pub(crate) fn require_capability(
    app_data: &HarnessAppData,
    capability: &str,
) -> Result<(), String> {
    let decision = capability_decision(app_data, capability);
    if decision.allowed {
        return Ok(());
    }

    emit_governance_audit_event(
        app_data,
        AuditEvent::GovernanceDenial {
            decision: decision.clone(),
        },
    );
    Err(decision
        .reason
        .unwrap_or_else(|| format!("Governance denial for capability '{capability}'")))
}

pub(crate) fn require_child_agent(
    app_data: &HarnessAppData,
    child_agent_id: &str,
) -> Result<(), String> {
    let subject = current_subject(app_data);
    app_data
        .governance_manager
        .require_child_agent_for_subject(&subject, child_agent_id)
}

pub(crate) fn parse_delegated_capabilities(
    app_data: &HarnessAppData,
    opts: Option<&Table>,
    field_name: &str,
    caller_label: &str,
) -> LuaResult<Option<BTreeMap<String, bool>>> {
    let Some(opts) = opts else {
        return Ok(None);
    };

    let caps_value = opts.get::<Value>(field_name).unwrap_or(Value::Nil);
    match caps_value {
        Value::Nil => Ok(None),
        Value::Table(t) => {
            let mut caps = BTreeMap::new();
            for pair in t.pairs::<String, Value>() {
                let (key, value) = pair?;
                let allowed = match value {
                    Value::Boolean(b) => b,
                    _ => {
                        return Err(mlua::Error::runtime(format!(
                            "{} opts.{} values must be booleans (key '{}')",
                            caller_label, field_name, key
                        )));
                    }
                };
                if allowed {
                    require_capability(app_data, &key).map_err(mlua::Error::runtime)?;
                }
                caps.insert(key, allowed);
            }
            Ok(Some(caps))
        }
        _ => Err(mlua::Error::runtime(format!(
            "{} opts.{} must be a table",
            caller_label, field_name
        ))),
    }
}

pub(crate) fn apply_active_grant_ceiling_to_peer_delegation(
    app_data: &HarnessAppData,
    requested_capabilities: Option<BTreeMap<String, bool>>,
    caller_label: &str,
) -> LuaResult<Option<BTreeMap<String, bool>>> {
    let subject = current_subject(app_data);
    let delegated_capabilities =
        match (subject.import_capabilities.as_ref(), requested_capabilities) {
            (Some(inherited), Some(requested)) => {
                Some(intersect_capability_bool_rules(inherited, &requested))
            }
            (Some(inherited), None) => Some(inherited.clone()),
            (None, requested) => requested,
        };
    let Some(grant_id) = subject.grant_id.as_deref() else {
        return Ok(delegated_capabilities);
    };

    let grant = app_data
        .governance_manager
        .grant_snapshot_for_subject(&subject, grant_id)
        .map_err(mlua::Error::runtime)?
        .ok_or_else(|| {
            mlua::Error::runtime(format!(
                "{} cannot use active governance grant '{}' for peer delegation: grant not found",
                caller_label, grant_id
            ))
        })?;

    if let Some(requested) = delegated_capabilities.as_ref() {
        for (capability, allowed) in requested {
            if !*allowed {
                continue;
            }
            if !capability_allowed_by_bool_rules(&grant.capabilities, capability) {
                return Err(mlua::Error::runtime(format!(
                    "{} cannot grant '{}' beyond active governance grant '{}'",
                    caller_label, capability, grant.grant_id
                )));
            }
        }
        Ok(delegated_capabilities)
    } else {
        Ok(Some(grant.capabilities))
    }
}

pub(crate) fn current_subject(app_data: &HarnessAppData) -> GovernanceSubject {
    let (session_reference, module_name, root_name, import_capabilities, grant_id) = app_data
        .execution_ctx
        .lock()
        .ok()
        .map(|lock| {
            let session_reference = lock.session_id.as_deref().map(|session_id| {
                match lock.session_store_selector.as_ref() {
                    Some(store_selector) => format_session_reference(session_id, store_selector),
                    None => session_id.to_string(),
                }
            });
            (
                session_reference,
                lock.harness_module.clone(),
                lock.harness_root.clone(),
                lock.import_capabilities.clone(),
                lock.governance_grant.clone(),
            )
        })
        .unwrap_or((None, None, None, None, None));
    GovernanceSubject {
        agent_id: Some(current_agent_id(app_data)),
        session_reference,
        module_name,
        root_name,
        grant_id,
        import_capabilities,
    }
}

pub(crate) fn emit_governance_audit_event(app_data: &HarnessAppData, audit_event: AuditEvent) {
    let Some(ctx) = app_data
        .execution_ctx
        .lock()
        .ok()
        .and_then(|lock| lock.event_context.clone())
    else {
        return;
    };

    let event = KernelEvent::Audit(audit_event);
    if ctx.json {
        println!("{}", serde_json::to_string(&event).unwrap_or_default());
    }
    let _ = ctx.event_tx.send((ctx.internal_id, event.clone()));
    if let Some(durability_tx) = ctx.durability_tx {
        let _ = durability_tx.send(PersistedKernelRecord::Event(Box::new(
            PersistedKernelEvent {
                internal_id: ctx.internal_id,
                turn_target: None,
                event,
            },
        )));
    }
}
