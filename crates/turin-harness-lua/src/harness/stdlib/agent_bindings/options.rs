use mlua::{Lua, LuaSerdeExt, Table, Value};

use crate::harness::globals::HarnessAppData;
use crate::kernel::agent_manager::LinkedSessionMode;
use crate::kernel::session::{
    ExecutionConflictPolicy, ExecutionContextTarget, QueuedTask, SidestepMode,
    TaskExecutionOverrides,
};

use super::active_trace_id;

pub(super) fn opt_session_id(opts: Option<&Table>) -> Option<String> {
    opts.and_then(|table| table.get::<String>("session_id").ok())
}

pub(super) fn opt_slot_id(opts: Option<&Table>) -> Option<String> {
    opts.and_then(|table| table.get::<String>("slot_id").ok())
}

pub(super) fn opt_from_turn_index(opts: Option<&Table>) -> Option<u32> {
    opts.and_then(|table| table.get::<u32>("from_turn_index").ok())
}

pub(super) fn opt_activate(opts: Option<&Table>, default: bool) -> bool {
    opts.and_then(|table| table.get::<bool>("activate").ok())
        .unwrap_or(default)
}

pub(super) fn opt_conflict_policy(
    opts: Option<&Table>,
) -> std::result::Result<Option<ExecutionConflictPolicy>, String> {
    let Some(opts) = opts else {
        return Ok(None);
    };
    let Ok(raw) = opts.get::<String>("conflict_policy") else {
        return Ok(None);
    };
    raw.parse().map(Some)
}

pub(super) fn opt_execution_overrides(
    lua: &Lua,
    opts: Option<&Table>,
) -> std::result::Result<Option<TaskExecutionOverrides>, String> {
    let Some(opts) = opts else {
        return Ok(None);
    };
    let Ok(value) = opts.get::<Value>("execution") else {
        return Ok(None);
    };
    if matches!(value, Value::Nil) {
        return Ok(None);
    }
    let overrides = lua
        .from_value::<TaskExecutionOverrides>(value)
        .map_err(|err| err.to_string())?;
    if overrides.is_empty() {
        return Err("execution overrides must not be an empty table".to_string());
    }
    Ok(Some(overrides))
}

pub(super) fn opt_peer_agent_id(opts: Option<&Table>, default_agent: &str) -> String {
    opts.and_then(|table| table.get::<String>("agent_id").ok())
        .unwrap_or_else(|| default_agent.to_string())
}

pub(super) fn opt_linked_session_mode(
    opts: Option<&Table>,
) -> std::result::Result<LinkedSessionMode, String> {
    let mode = opts
        .and_then(|table| table.get::<String>("mode").ok())
        .unwrap_or_else(|| "thread".to_string());
    let thread = opts.and_then(|table| table.get::<String>("thread").ok());
    match mode.as_str() {
        "thread" => Ok(LinkedSessionMode::Thread(
            thread.unwrap_or_else(|| "default".to_string()),
        )),
        "fresh" if thread.is_none() => Ok(LinkedSessionMode::Fresh),
        "fresh" => Err("peer mode='fresh' cannot be combined with thread".to_string()),
        other => Err(format!(
            "invalid peer mode '{other}'; expected thread|fresh"
        )),
    }
}

pub(super) fn peer_prompt_task(
    lua: &Lua,
    app_data: &HarnessAppData,
    prompt: String,
    opts: Option<&Table>,
) -> std::result::Result<QueuedTask, String> {
    let trace_id = active_trace_id(app_data);
    let execution = opt_execution_overrides(lua, opts)?;
    Ok(QueuedTask::ad_hoc(prompt)
        .with_inherited_trace(trace_id.as_deref())
        .with_execution(execution))
}

pub(super) fn opt_sidestep_mode(opts: Option<&Table>) -> std::result::Result<SidestepMode, String> {
    let Some(opts) = opts else {
        return Ok(SidestepMode::Ephemeral);
    };
    let Ok(raw) = opts.get::<String>("mode") else {
        return Ok(SidestepMode::Ephemeral);
    };
    raw.parse()
}

pub(super) fn sidestep_opts_table_from_value(
    lua: &Lua,
    value: Option<Value>,
) -> std::result::Result<Option<Table>, String> {
    match value {
        None | Some(Value::Nil) => Ok(None),
        Some(Value::Table(table)) => Ok(Some(table)),
        Some(Value::String(mode)) => {
            let table = lua.create_table().map_err(|err| err.to_string())?;
            table
                .set(
                    "mode",
                    mode.to_str().map_err(|err| err.to_string())?.to_string(),
                )
                .map_err(|err| err.to_string())?;
            Ok(Some(table))
        }
        Some(_) => {
            Err("invalid sidestep opts; expected nil, mode string, or options table".to_string())
        }
    }
}

pub(super) fn opt_sidestep_context_target(
    lua: &Lua,
    opts: Option<&Table>,
) -> std::result::Result<Option<ExecutionContextTarget>, String> {
    let Some(opts) = opts else {
        return Ok(None);
    };
    let value = match opts.get::<Value>("target") {
        Ok(Value::Nil) | Err(_) => match opts.get::<Value>("context_target") {
            Ok(value) => value,
            Err(_) => Value::Nil,
        },
        Ok(value) => value,
    };
    if matches!(value, Value::Nil) {
        return Ok(None);
    }
    lua.from_value::<ExecutionContextTarget>(value)
        .map(Some)
        .map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_session_modes_are_explicit_and_unambiguous() -> mlua::Result<()> {
        let lua = Lua::new();
        assert_eq!(
            opt_linked_session_mode(None).unwrap(),
            LinkedSessionMode::Thread("default".to_string())
        );

        let named = lua.create_table()?;
        named.set("thread", "review")?;
        assert_eq!(
            opt_linked_session_mode(Some(&named)).unwrap(),
            LinkedSessionMode::Thread("review".to_string())
        );

        let fresh = lua.create_table()?;
        fresh.set("mode", "fresh")?;
        assert_eq!(
            opt_linked_session_mode(Some(&fresh)).unwrap(),
            LinkedSessionMode::Fresh
        );
        fresh.set("thread", "ambiguous")?;
        assert!(opt_linked_session_mode(Some(&fresh)).is_err());
        Ok(())
    }
}
