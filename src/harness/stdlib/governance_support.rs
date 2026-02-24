use crate::harness::globals::HarnessAppData;
use crate::kernel::governance::{CapabilityDecision, GovernanceSubject};
use mlua::{Result as LuaResult, Table, Value};
use std::collections::BTreeMap;

pub(crate) fn current_agent_id(app_data: &HarnessAppData) -> &str {
    app_data.config.agent.id.as_str()
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
    let subject = current_subject(app_data);
    app_data
        .governance_manager
        .require_capability_for_subject(&subject, capability)
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
            let subject = current_subject(app_data);
            for pair in t.pairs::<String, Value>() {
                let (key, value) = pair?;
                if key.ends_with(".*") {
                    return Err(mlua::Error::runtime(format!(
                        "{} opts.{} wildcard rules are not yet supported (key '{}')",
                        caller_label, field_name, key
                    )));
                }
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
                    app_data
                        .governance_manager
                        .require_capability_for_subject(&subject, &key)
                        .map_err(mlua::Error::runtime)?;
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

pub(crate) fn current_subject(app_data: &HarnessAppData) -> GovernanceSubject {
    let module_name = app_data
        .active_harness_module
        .lock()
        .ok()
        .and_then(|lock| lock.clone());
    let root_name = app_data
        .active_harness_root
        .lock()
        .ok()
        .and_then(|lock| lock.clone());
    let import_capabilities: Option<BTreeMap<String, bool>> = app_data
        .active_import_capabilities
        .lock()
        .ok()
        .and_then(|lock| lock.clone());
    GovernanceSubject {
        agent_id: Some(current_agent_id(app_data).to_string()),
        module_name,
        root_name,
        grant_id: None,
        import_capabilities,
    }
}
