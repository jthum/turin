use std::collections::HashMap;

use mlua::{Result as LuaResult, Value};

use crate::harness::globals::{HarnessAppData, block_on_current};
use crate::harness::stdlib::identity_support::get_active_identity;
use crate::kernel::policy::PolicyScope;

pub(crate) fn runtime_policy_snapshot(
    app_data: &HarnessAppData,
) -> anyhow::Result<HashMap<String, serde_json::Value>> {
    let mut scope = PolicyScope::default();
    if let Ok(identity) = get_active_identity(app_data) {
        scope.agent_id = Some(identity.agent_id().to_string());
        scope.session_id = Some(identity.session_id().to_string());
        scope.run_id = identity.run_id().map(ToString::to_string);
    }

    let policy_manager = app_data.policy_manager.clone();
    Ok(block_on_current(async move {
        policy_manager.snapshot(&scope).await
    }))
}

pub(crate) fn policy_bool(
    snapshot: &HashMap<String, serde_json::Value>,
    key: &str,
    default: bool,
) -> bool {
    snapshot
        .get(key)
        .and_then(|v| v.as_bool())
        .unwrap_or(default)
}

pub(crate) fn policy_u64(
    snapshot: &HashMap<String, serde_json::Value>,
    key: &str,
    default: u64,
) -> u64 {
    snapshot
        .get(key)
        .and_then(|v| v.as_u64())
        .unwrap_or(default)
}

pub(crate) fn policy_string<'a>(
    snapshot: &'a HashMap<String, serde_json::Value>,
    key: &str,
    default: &'a str,
) -> &'a str {
    snapshot
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or(default)
}

pub(crate) fn policy_scope_from_value(
    app_data: &HarnessAppData,
    scope: Option<Value>,
) -> LuaResult<PolicyScope> {
    let mut out = PolicyScope::default();

    match scope {
        None | Some(Value::Nil) => {
            out.scope = Some("global".to_string());
            return Ok(out);
        }
        Some(Value::String(s)) => {
            out.scope = Some(s.to_str()?.to_string());
        }
        Some(Value::Table(t)) => {
            if let Ok(s) = t.get::<String>("scope") {
                out.scope = Some(s);
            }
            if let Ok(agent_id) = t.get::<String>("agent_id") {
                out.agent_id = Some(agent_id);
            }
            if let Ok(session_id) = t.get::<String>("session_id") {
                out.session_id = Some(session_id);
            }
            if let Ok(run_id) = t.get::<String>("run_id") {
                out.run_id = Some(run_id);
            }
        }
        _ => {
            return Err(mlua::Error::runtime(
                "invalid policy scope; expected nil, string, or table",
            ));
        }
    }

    if out.scope.is_none() {
        out.scope = Some("global".to_string());
    }

    if (out.agent_id.is_none() || out.session_id.is_none() || out.run_id.is_none())
        && let Ok(identity) = get_active_identity(app_data)
    {
        if out.agent_id.is_none() {
            out.agent_id = Some(identity.agent_id().to_string());
        }
        if out.session_id.is_none() {
            out.session_id = Some(identity.session_id().to_string());
        }
        if out.run_id.is_none() {
            out.run_id = identity.run_id().map(ToString::to_string);
        }
    }

    Ok(out)
}
