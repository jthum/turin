use mlua::{Function, IntoLuaMulti, Lua, MultiValue, Result as LuaResult, Value};

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
