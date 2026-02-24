use mlua::{Function, Lua, MultiValue, Result as LuaResult, Table, Value};

use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::binding_common::{bool_value_ok, json_ok, nil_err, string_value};
use crate::harness::stdlib::governance_support::{
    capability_decision as governance_capability_decision, current_subject,
    parse_delegated_capabilities, require_capability as require_governance_capability,
};

pub fn register_runtime_governance_namespace(
    lua: &Lua,
    runtime_table: &Table,
    app_data: &HarnessAppData,
) -> LuaResult<()> {
    let governance_table = lua.create_table()?;

    {
        let governance_manager = app_data.governance_manager.clone();
        governance_table.set(
            "profile",
            lua.create_function(move |lua, ()| {
                let snapshot = governance_manager.snapshot();
                match serde_json::to_string(&snapshot.profile) {
                    Ok(serialized) => {
                        let profile = serialized.trim_matches('"').to_string();
                        string_value(lua, &profile)
                    }
                    Err(e) => Err(mlua::Error::runtime(format!(
                        "failed to serialize governance profile: {}",
                        e
                    ))),
                }
            })?,
        )?;
    }

    {
        let governance_manager = app_data.governance_manager.clone();
        governance_table.set(
            "snapshot",
            lua.create_function(move |lua, agent_id: Option<String>| {
                let snapshot = governance_manager.snapshot_for_agent(agent_id.as_deref());
                json_ok(lua, &snapshot)
            })?,
        )?;
    }

    {
        let app_data_snapshot = app_data.clone();
        governance_table.set(
            "check",
            lua.create_function(
                move |lua, (capability, agent_id): (String, Option<String>)| {
                    if capability.trim().is_empty() {
                        return nil_err(lua, "capability must not be empty");
                    }
                    let decision = if let Some(agent_id) = agent_id.as_deref() {
                        app_data_snapshot
                            .governance_manager
                            .capability_decision(Some(agent_id), &capability)
                    } else {
                        governance_capability_decision(&app_data_snapshot, &capability)
                    };
                    json_ok(lua, &decision)
                },
            )?,
        )?;
    }

    {
        let governance_manager = app_data.governance_manager.clone();
        governance_table.set(
            "agent",
            lua.create_function(move |lua, agent_id: String| {
                if agent_id.trim().is_empty() {
                    return nil_err(lua, "agent_id must not be empty");
                }
                let snapshot = governance_manager.snapshot_for_agent(Some(agent_id.as_str()));
                json_ok(lua, &snapshot)
            })?,
        )?;
    }

    {
        let app_data_snapshot = app_data.clone();
        governance_table.set(
            "grant_issue",
            lua.create_function(move |lua, opts: Table| {
                require_governance_capability(&app_data_snapshot, "runtime.governance.grant.issue")
                    .map_err(mlua::Error::runtime)?;
                let caps = parse_delegated_capabilities(
                    &app_data_snapshot,
                    Some(&opts),
                    "capabilities",
                    "runtime.governance.grant_issue",
                )?
                .ok_or_else(|| {
                    mlua::Error::runtime(
                        "runtime.governance.grant_issue opts.capabilities is required",
                    )
                })?;
                let ttl_ms = opts.get::<Option<u64>>("ttl_ms")?;
                let max_uses = opts.get::<Option<u64>>("max_uses")?;
                let reason = opts.get::<Option<String>>("reason")?;
                let subject = current_subject(&app_data_snapshot);
                let grant = app_data_snapshot
                    .governance_manager
                    .issue_grant_for_subject(&subject, caps, ttl_ms, max_uses, reason)
                    .map_err(mlua::Error::runtime)?;
                json_ok(lua, &grant)
            })?,
        )?;
    }

    {
        let app_data_snapshot = app_data.clone();
        governance_table.set(
            "grant_get",
            lua.create_function(move |lua, grant_id: String| {
                if grant_id.trim().is_empty() {
                    return nil_err(lua, "grant_id must not be empty");
                }
                require_governance_capability(&app_data_snapshot, "runtime.governance.grant.get")
                    .map_err(mlua::Error::runtime)?;
                let subject = current_subject(&app_data_snapshot);
                match app_data_snapshot
                    .governance_manager
                    .grant_snapshot_for_subject(&subject, &grant_id)
                    .map_err(mlua::Error::runtime)?
                {
                    Some(grant) => json_ok(lua, &grant),
                    None => nil_err(lua, "grant not found"),
                }
            })?,
        )?;
    }

    {
        let app_data_snapshot = app_data.clone();
        governance_table.set(
            "grant_revoke",
            lua.create_function(move |lua, grant_id: String| {
                if grant_id.trim().is_empty() {
                    return nil_err(lua, "grant_id must not be empty");
                }
                require_governance_capability(
                    &app_data_snapshot,
                    "runtime.governance.grant.revoke",
                )
                .map_err(mlua::Error::runtime)?;
                let subject = current_subject(&app_data_snapshot);
                let revoked = app_data_snapshot
                    .governance_manager
                    .revoke_grant_for_subject(&subject, &grant_id)
                    .map_err(mlua::Error::runtime)?;
                if revoked {
                    Ok(bool_value_ok(true))
                } else {
                    Ok((
                        Value::Boolean(false),
                        Value::String(lua.create_string("grant not found")?),
                    ))
                }
            })?,
        )?;
    }

    {
        let app_data_snapshot = app_data.clone();
        governance_table.set(
            "with_grant",
            lua.create_function(move |_, (grant_id, func): (String, Function)| {
                if grant_id.trim().is_empty() {
                    return Err(mlua::Error::runtime("grant_id must not be empty"));
                }
                require_governance_capability(&app_data_snapshot, "runtime.governance.grant.use")
                    .map_err(mlua::Error::runtime)?;
                let subject = current_subject(&app_data_snapshot);
                app_data_snapshot
                    .governance_manager
                    .enter_grant_for_subject(&subject, &grant_id)
                    .map_err(mlua::Error::runtime)?;

                let previous_grant = {
                    let mut lock = app_data_snapshot
                        .active_governance_grant
                        .lock()
                        .map_err(|_| mlua::Error::runtime("active governance grant mutex poisoned"))?;
                    let previous = lock.clone();
                    *lock = Some(grant_id.clone());
                    previous
                };

                let call_result = func.call::<MultiValue>(());

                {
                    let mut lock = app_data_snapshot
                        .active_governance_grant
                        .lock()
                        .map_err(|_| mlua::Error::runtime("active governance grant mutex poisoned"))?;
                    *lock = previous_grant;
                }

                call_result
            })?,
        )?;
    }

    runtime_table.set("governance", governance_table)?;
    Ok(())
}
