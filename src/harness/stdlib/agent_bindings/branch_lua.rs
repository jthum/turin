use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};

use crate::persistence::schema::BranchHeadRow;

fn bytes_to_simple_uuid(bytes: &[u8]) -> String {
    uuid::Uuid::from_slice(bytes)
        .map(|uuid| uuid.simple().to_string())
        .unwrap_or_else(|_| {
            let mut out = String::with_capacity(bytes.len() * 2);
            for byte in bytes {
                use std::fmt::Write as _;
                let _ = write!(&mut out, "{:02x}", byte);
            }
            out
        })
}

pub(super) fn branch_row_to_lua_table(
    lua: &Lua,
    row: &BranchHeadRow,
    deferred: bool,
) -> LuaResult<Table> {
    let table = lua.create_table()?;
    table.set("branch_id", bytes_to_simple_uuid(&row.public_id))?;
    table.set("name", row.name.clone())?;
    match row.head_turn_depth {
        Some(depth) => table.set("head_turn_index", depth)?,
        None => table.set("head_turn_index", Value::Nil)?,
    }
    match row.created_from_turn_id {
        Some(turn_id) => table.set("source_turn_id", turn_id)?,
        None => table.set("source_turn_id", Value::Nil)?,
    }
    table.set("origin_kind", row.origin_kind.clone())?;
    match row.origin_task_id.as_deref() {
        Some(task_id) => table.set("origin_task_id", task_id)?,
        None => table.set("origin_task_id", Value::Nil)?,
    }
    match row.origin_execution_id.as_deref() {
        Some(execution_id) => table.set("origin_execution_id", execution_id)?,
        None => table.set("origin_execution_id", Value::Nil)?,
    }
    let metadata = row
        .origin_metadata
        .as_deref()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok());
    match metadata {
        Some(metadata) => table.set("origin_metadata", lua.to_value(&metadata)?)?,
        None => table.set("origin_metadata", Value::Nil)?,
    }
    table.set("active", row.is_active)?;
    table.set("deferred", deferred)?;
    table.set("created_at", row.created_at.clone())?;
    Ok(table)
}

pub(super) fn branch_rows_to_lua_table(lua: &Lua, rows: &[BranchHeadRow]) -> LuaResult<Table> {
    let out = lua.create_table()?;
    for (i, row) in rows.iter().enumerate() {
        out.set(i + 1, branch_row_to_lua_table(lua, row, false)?)?;
    }
    Ok(out)
}
