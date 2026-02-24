use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};

use crate::harness::globals::HarnessAppData;
use crate::kernel::identity::RuntimeIdentity;

pub(crate) fn get_active_identity(app_data: &HarnessAppData) -> anyhow::Result<RuntimeIdentity> {
    let session_id = app_data
        .execution_ctx
        .lock()
        .unwrap()
        .session_id
        .clone()
        .ok_or_else(|| anyhow::anyhow!("No active session context"))?;

    Ok(RuntimeIdentity::new(
        session_id,
        app_data.config.agent.id.clone(),
    ))
}

pub(crate) fn identity_to_lua_table(lua: &Lua, identity: &RuntimeIdentity) -> LuaResult<Table> {
    let tbl = lua.create_table()?;
    tbl.set("session_id", identity.session_id())?;
    tbl.set("agent_id", identity.agent_id())?;
    match identity.user_id() {
        Some(v) => tbl.set("user_id", v)?,
        None => tbl.set("user_id", Value::Nil)?,
    }
    match identity.channel_id() {
        Some(v) => tbl.set("channel_id", v)?,
        None => tbl.set("channel_id", Value::Nil)?,
    }
    match identity.tenant_id() {
        Some(v) => tbl.set("tenant_id", v)?,
        None => tbl.set("tenant_id", Value::Nil)?,
    }
    match identity.run_id() {
        Some(v) => tbl.set("run_id", v)?,
        None => tbl.set("run_id", Value::Nil)?,
    }
    let extra = lua.create_table()?;
    for (k, v) in identity.extra() {
        extra.set(k.as_str(), v.as_str())?;
    }
    tbl.set("extra", extra)?;
    Ok(tbl)
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{:02x}", b);
    }
    out
}

fn format_uuid_bytes_simple(bytes: &[u8]) -> Option<String> {
    if bytes.len() != 16 {
        return None;
    }
    let uuid = uuid::Uuid::from_slice(bytes).ok()?;
    Some(uuid.simple().to_string())
}

pub(crate) fn session_row_to_lua_table(
    lua: &Lua,
    row: &crate::persistence::schema::SessionRow,
) -> LuaResult<Table> {
    let t = lua.create_table()?;
    t.set("internal_id", row.id)?;
    t.set(
        "session_id",
        format_uuid_bytes_simple(&row.public_id).unwrap_or_else(|| bytes_to_hex(&row.public_id)),
    )?;
    t.set("agent_id", row.agent_id.clone())?;
    if let Some(m) = &row.metadata {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(m) {
            t.set("metadata", lua.to_value(&json)?)?;
        } else {
            t.set("metadata", m.clone())?;
        }
    } else {
        t.set("metadata", Value::Nil)?;
    }
    t.set("created_at", row.created_at.clone())?;
    Ok(t)
}
