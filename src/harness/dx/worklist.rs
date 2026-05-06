use mlua::{Function, Lua, Result as LuaResult, Table, Value};

use crate::harness::dx::common::call_and_raise_on_err;

pub fn register_worklist_dx(lua: &Lua) -> LuaResult<()> {
    let globals = lua.globals();
    let runtime: Table = globals.get("runtime")?;
    let runtime_worklist: Table = runtime.get("worklist")?;
    let open_fn: Function = runtime_worklist.get("open")?;

    globals.set(
        "worklist",
        lua.create_function(move |lua, (name, opts): (String, Option<Table>)| {
            let merged = lua.create_table()?;
            if let Some(opts) = opts {
                for pair in opts.pairs::<Value, Value>() {
                    let (key, value) = pair?;
                    merged.set(key, value)?;
                }
            }
            merged.set("name", name)?;
            call_and_raise_on_err(lua, &open_fn, merged, "runtime.worklist.open")
        })?,
    )?;

    Ok(())
}
