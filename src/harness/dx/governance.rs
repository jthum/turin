use mlua::{Function, Lua, MultiValue, Result as LuaResult, Table, Value};

use crate::harness::dx::common::{call_and_raise_on_err, tuple_first_and_err};

fn revoke_grant_checked(_lua: &Lua, grant_revoke_fn: &Function, grant_id: &str) -> LuaResult<()> {
    let values = grant_revoke_fn.call::<MultiValue>(grant_id.to_string())?;
    let (value, err) = tuple_first_and_err(values);
    if let Some(err) = err {
        return Err(mlua::Error::runtime(format!(
            "[runtime.governance.grant_revoke] {}",
            err
        )));
    }
    if matches!(value, Value::Boolean(false)) {
        return Err(mlua::Error::runtime(
            "[runtime.governance.grant_revoke] grant revoke returned false".to_string(),
        ));
    }
    Ok(())
}

pub fn register_governance_dx(lua: &Lua) -> LuaResult<()> {
    let globals = lua.globals();
    let runtime: Table = globals.get("runtime")?;
    let runtime_governance: Table = runtime.get("governance")?;

    let grant_issue_fn: Function = runtime_governance.get("grant_issue")?;
    let with_grant_fn: Function = runtime_governance.get("with_grant")?;
    let grant_revoke_fn: Function = runtime_governance.get("grant_revoke")?;

    runtime_governance.set(
        "grant",
        lua.create_function(move |lua, (spec, callback): (Table, Function)| {
            let issued = call_and_raise_on_err(
                lua,
                &grant_issue_fn,
                spec,
                "runtime.governance.grant_issue",
            )?;
            let issued_table = match issued {
                Value::Table(t) => t,
                other => {
                    return Err(mlua::Error::runtime(format!(
                        "[runtime.governance.grant] expected grant table from grant_issue, got {:?}",
                        other
                    )))
                }
            };
            let grant_id = issued_table.get::<String>("grant_id").map_err(|_| {
                mlua::Error::runtime(
                    "[runtime.governance.grant] grant_issue result missing grant_id".to_string(),
                )
            })?;
            if grant_id.trim().is_empty() {
                return Err(mlua::Error::runtime(
                    "[runtime.governance.grant] grant_id must not be empty".to_string(),
                ));
            }

            let callback_result = with_grant_fn.call::<MultiValue>((grant_id.clone(), callback));
            let revoke_result = revoke_grant_checked(lua, &grant_revoke_fn, &grant_id);

            match (callback_result, revoke_result) {
                (Err(callback_err), _) => Err(callback_err),
                (Ok(_), Err(revoke_err)) => Err(revoke_err),
                (Ok(values), Ok(())) => Ok(values),
            }
        })?,
    )?;

    Ok(())
}
