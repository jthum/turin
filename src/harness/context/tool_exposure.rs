use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use mlua::{Lua, Table, UserData, UserDataMethods, Value};

use super::ContextState;

#[derive(Clone)]
pub(super) struct ToolExposureProxy {
    pub(super) state: Arc<Mutex<ContextState>>,
    pub(super) available_tools: Arc<BTreeSet<String>>,
}

fn tool_names_from_value(value: Value) -> mlua::Result<BTreeSet<String>> {
    let names = match value {
        Value::String(name) => vec![name.to_str()?.to_string()],
        Value::Table(table) => table
            .sequence_values::<String>()
            .collect::<mlua::Result<Vec<_>>>()?,
        _ => {
            return Err(mlua::Error::runtime(
                "tool selection expects a tool name or an array of tool names",
            ));
        }
    };

    let mut normalized = BTreeSet::new();
    for name in names {
        let name = name.trim();
        if name.is_empty() {
            return Err(mlua::Error::runtime("tool names must not be empty"));
        }
        normalized.insert(name.to_string());
    }
    Ok(normalized)
}

fn validate_tool_names(
    names: &BTreeSet<String>,
    available_tools: &BTreeSet<String>,
) -> mlua::Result<()> {
    let unknown = names
        .difference(available_tools)
        .cloned()
        .collect::<Vec<_>>();
    if unknown.is_empty() {
        Ok(())
    } else {
        Err(mlua::Error::runtime(format!(
            "unknown or unavailable tool(s): {}",
            unknown.join(", ")
        )))
    }
}

fn names_to_lua_table(lua: &Lua, names: impl IntoIterator<Item = String>) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    for (index, name) in names.into_iter().enumerate() {
        table.set(index + 1, name)?;
    }
    Ok(table)
}

impl UserData for ToolExposureProxy {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("only", |_, this, value: Value| {
            let names = tool_names_from_value(value)?;
            validate_tool_names(&names, &this.available_tools)?;
            this.state
                .lock()
                .expect("context state mutex poisoned")
                .tool_exposure
                .only(names);
            Ok(())
        });

        methods.add_method("include", |_, this, value: Value| {
            let names = tool_names_from_value(value)?;
            validate_tool_names(&names, &this.available_tools)?;
            this.state
                .lock()
                .expect("context state mutex poisoned")
                .tool_exposure
                .include(names);
            Ok(())
        });

        methods.add_method("exclude", |_, this, value: Value| {
            let names = tool_names_from_value(value)?;
            validate_tool_names(&names, &this.available_tools)?;
            this.state
                .lock()
                .expect("context state mutex poisoned")
                .tool_exposure
                .exclude(names);
            Ok(())
        });

        methods.add_method("all", |_, this, ()| {
            this.state
                .lock()
                .expect("context state mutex poisoned")
                .tool_exposure
                .expose_all();
            Ok(())
        });

        methods.add_method("available", |lua, this, ()| {
            names_to_lua_table(lua, this.available_tools.iter().cloned())
        });

        methods.add_method("exposed", |lua, this, ()| {
            let state = this.state.lock().expect("context state mutex poisoned");
            names_to_lua_table(
                lua,
                this.available_tools
                    .iter()
                    .filter(|name| state.tool_exposure.exposes(name))
                    .cloned(),
            )
        });
    }
}
