use mlua::{Lua, Result as LuaResult, Table, Value};

fn parse_seconds(value: Value, arg_name: &str) -> LuaResult<f64> {
    match value {
        Value::Integer(i) => Ok(i as f64),
        Value::Number(n) if n.is_finite() => Ok(n),
        Value::String(s) => s.to_str()?.trim().parse::<f64>().map_err(|_| {
            mlua::Error::runtime(format!("[time] {} is not a valid number", arg_name))
        }),
        other => Err(mlua::Error::runtime(format!(
            "[time] {} must be number or numeric string, got {:?}",
            arg_name, other
        ))),
    }
}

fn now_epoch_seconds() -> LuaResult<f64> {
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| mlua::Error::runtime(format!("[time] system clock error: {}", e)))?;
    Ok(duration.as_secs_f64())
}

pub fn register_time_dx(lua: &Lua) -> LuaResult<()> {
    let globals = lua.globals();
    let time_table: Table = globals.get("time")?;

    time_table.set(
        "since",
        lua.create_function(move |_lua, ts: Value| {
            let ts_seconds = parse_seconds(ts, "ts")?;
            let now_seconds = now_epoch_seconds()?;
            Ok(now_seconds - ts_seconds)
        })?,
    )?;

    time_table.set(
        "after",
        lua.create_function(move |_lua, (ts, threshold): (Value, Value)| {
            let ts_seconds = parse_seconds(ts, "ts")?;
            let threshold_seconds = parse_seconds(threshold, "threshold")?;
            let now_seconds = now_epoch_seconds()?;
            Ok((now_seconds - ts_seconds) >= threshold_seconds)
        })?,
    )?;

    Ok(())
}
