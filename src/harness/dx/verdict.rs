use mlua::{Lua, Result as LuaResult, Table, Value};

fn verdict_table(
    lua: &Lua,
    code: i64,
    reason: Option<String>,
    patch: Option<Value>,
) -> LuaResult<Table> {
    let table = lua.create_table()?;
    table.set("code", code)?;
    if let Some(reason) = reason {
        table.set("reason", reason)?;
    }
    if let Some(patch) = patch {
        table.set("patch", patch)?;
    }
    Ok(table)
}

pub fn register_verdict_globals(lua: &Lua) -> LuaResult<()> {
    let verdict = lua.create_table()?;

    verdict.set(
        "allow",
        lua.create_function(|lua, ()| Ok(Value::Table(verdict_table(lua, 1, None, None)?)))?,
    )?;

    verdict.set(
        "reject",
        lua.create_function(|lua, reason: Option<String>| {
            Ok(Value::Table(verdict_table(
                lua,
                2,
                Some(reason.unwrap_or_default()),
                None,
            )?))
        })?,
    )?;

    verdict.set(
        "escalate",
        lua.create_function(|lua, reason: Option<String>| {
            Ok(Value::Table(verdict_table(
                lua,
                3,
                Some(reason.unwrap_or_default()),
                None,
            )?))
        })?,
    )?;

    verdict.set(
        "patch",
        lua.create_function(|lua, patch: Value| {
            Ok(Value::Table(verdict_table(lua, 4, None, Some(patch))?))
        })?,
    )?;

    verdict.set(
        "reject_if",
        lua.create_function(|lua, (condition, reason): (bool, String)| {
            if condition {
                Ok(Value::Table(verdict_table(lua, 2, Some(reason), None)?))
            } else {
                Ok(Value::Nil)
            }
        })?,
    )?;

    verdict.set(
        "escalate_if",
        lua.create_function(|lua, (condition, reason): (bool, String)| {
            if condition {
                Ok(Value::Table(verdict_table(lua, 3, Some(reason), None)?))
            } else {
                Ok(Value::Nil)
            }
        })?,
    )?;

    lua.globals().set("verdict", verdict)?;
    Ok(())
}
