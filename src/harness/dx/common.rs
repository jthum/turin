use mlua::{Function, IntoLuaMulti, Lua, MultiValue, Result as LuaResult, Table, Value};

pub fn tuple_first_and_err(values: MultiValue) -> (Value, Option<String>) {
    let mut iter = values.into_iter();
    let first = iter.next().unwrap_or(Value::Nil);
    let err = iter.next().and_then(value_to_error_string);
    (first, err)
}

pub fn call_and_raise_on_err<A>(
    _lua: &Lua,
    func: &Function,
    args: A,
    op_name: &str,
) -> LuaResult<Value>
where
    A: IntoLuaMulti,
{
    let values = func.call::<MultiValue>(args)?;
    let (value, err) = tuple_first_and_err(values);
    if let Some(err) = err {
        Err(mlua::Error::runtime(format!("[{}] {}", op_name, err)))
    } else {
        Ok(value)
    }
}

pub fn value_to_error_string(value: Value) -> Option<String> {
    match value {
        Value::Nil => None,
        Value::String(s) => Some(match s.to_str() {
            Ok(v) => v.to_string(),
            Err(_) => "<invalid utf-8 error string>".to_string(),
        }),
        Value::Boolean(false) => Some("operation returned false".to_string()),
        other => Some(format!("{:?}", other)),
    }
}

pub fn normalize_helper_capability_name(capability: &str) -> String {
    if capability.starts_with("runtime.")
        || capability.starts_with("fs.")
        || capability.starts_with("tool.")
        || capability.starts_with("shell.")
    {
        return capability.to_string();
    }

    let runtime_roots = [
        "agent",
        "code",
        "context",
        "db",
        "governance",
        "graph",
        "kv",
        "memory",
        "policy",
    ];

    if let Some((root, _)) = capability.split_once('.')
        && runtime_roots.contains(&root)
    {
        return format!("runtime.{capability}");
    }

    capability.to_string()
}

pub fn normalize_capability_table_in_place(lua: &Lua, table: &Table) -> LuaResult<()> {
    let mut entries = Vec::new();
    for pair in table.pairs::<Value, Value>() {
        let (key, value) = pair?;
        let normalized_key = match key {
            Value::String(s) => Value::String(
                lua.create_string(normalize_helper_capability_name(s.to_str()?.as_ref()))?,
            ),
            other => other,
        };
        entries.push((normalized_key, value));
    }
    table.clear()?;
    for (key, value) in entries {
        table.set(key, value)?;
    }
    Ok(())
}

pub fn normalize_capabilities_field(lua: &Lua, table: &Table) -> LuaResult<()> {
    if let Value::Table(capabilities) = table.get::<Value>("capabilities")? {
        normalize_capability_table_in_place(lua, &capabilities)?;
    }
    Ok(())
}
