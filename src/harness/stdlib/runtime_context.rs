use mlua::{Lua, MultiValue, Result as LuaResult, Table, Value};

use crate::harness::globals::{HarnessAppData, block_on_current};
use crate::harness::stdlib::binding_common::ok_value;
use crate::harness::stdlib::context_selectors::{parse_context_args, selector_to_lua_table};

fn wildcard_match(pattern: &str, text: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    let p = pattern.as_bytes();
    let s = text.as_bytes();
    let (mut pi, mut si, mut star_idx, mut match_idx) = (0usize, 0usize, None, 0usize);
    while si < s.len() {
        if pi < p.len() && p[pi] == s[si] {
            pi += 1;
            si += 1;
        } else if pi < p.len() && p[pi] == b'*' {
            star_idx = Some(pi);
            match_idx = si;
            pi += 1;
        } else if let Some(star) = star_idx {
            pi = star + 1;
            match_idx += 1;
            si = match_idx;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    pi == p.len()
}

pub fn register_runtime_context_namespace(
    lua: &Lua,
    runtime_table: &Table,
    app_data: &HarnessAppData,
) -> LuaResult<()> {
    let context_ns = lua.create_table()?;
    {
        let manager = app_data.store_manager.clone();
        context_ns.set(
            "glob",
            lua.create_function(move |lua, pattern: String| {
                let manager = manager.clone();
                let aliases = block_on_current(async move { manager.list_aliases().await });
                let out = lua.create_table()?;
                let mut idx = 1;
                for alias in aliases {
                    if wildcard_match(&pattern, &alias) {
                        out.set(idx, alias)?;
                        idx += 1;
                    }
                }
                Ok(ok_value(Value::Table(out)))
            })?,
        )?;
    }
    let context_meta = lua.create_table()?;
    context_meta.set(
        "__call",
        lua.create_function(|lua, (_self, args): (Value, MultiValue)| {
            let selector = parse_context_args(lua, args)?;
            Ok(Value::Table(selector_to_lua_table(lua, &selector)?))
        })?,
    )?;
    let _ = context_ns.set_metatable(Some(context_meta));
    runtime_table.set("context", context_ns)?;
    Ok(())
}
