use mlua::{Lua, Result as LuaResult, Table, Value};
use turin_daemon_protocol::{UI_INTENT_VERSION, UiIntentMessage};

use crate::harness::stdlib::object_refs;

const UI_TABLE_KEY: &str = "ui";
const UI_INTENT_REGISTRY_KEY: &str = "__harness_ui_intents";

pub fn register_ui_globals(lua: &Lua) -> LuaResult<()> {
    let ui = lua.create_table()?;

    ui.set(
        "app",
        lua.create_function(|lua, (title, opts): (String, Option<Table>)| {
            let intent = lua.create_table()?;
            intent.set("version", UI_INTENT_VERSION)?;
            intent.set("type", "app")?;
            intent.set("title", title)?;
            copy_optional_fields(&intent, opts.as_ref(), &["subtitle", "icon"])?;
            push_ui_intent(lua, intent)
        })?,
    )?;

    ui.set(
        "home",
        lua.create_function(|lua, (title, nodes): (String, Option<Table>)| {
            let intent = lua.create_table()?;
            intent.set("version", UI_INTENT_VERSION)?;
            intent.set("type", "home")?;
            intent.set("title", title)?;
            intent.set("nodes", nodes.unwrap_or(lua.create_table()?))?;
            push_ui_intent(lua, intent)
        })?,
    )?;

    ui.set(
        "show",
        lua.create_function(|lua, (area, node): (String, Table)| {
            let intent = lua.create_table()?;
            intent.set("version", UI_INTENT_VERSION)?;
            intent.set("type", "show")?;
            intent.set("area", area)?;
            intent.set("node", node)?;
            push_ui_intent(lua, intent)
        })?,
    )?;

    ui.set(
        "notice",
        lua.create_function(|lua, (title, opts): (String, Option<Table>)| {
            let intent = lua.create_table()?;
            intent.set("version", UI_INTENT_VERSION)?;
            intent.set("type", "notify")?;
            intent.set("title", title)?;
            copy_optional_fields(&intent, opts.as_ref(), &["body", "level"])?;
            push_ui_intent(lua, intent)
        })?,
    )?;

    ui.set(
        "focus",
        lua.create_function(|lua, target: String| {
            let intent = lua.create_table()?;
            intent.set("version", UI_INTENT_VERSION)?;
            intent.set("type", "focus")?;
            intent.set("target", target)?;
            push_ui_intent(lua, intent)
        })?,
    )?;

    ui.set(
        "refresh",
        lua.create_function(|lua, binding: String| {
            let intent = lua.create_table()?;
            intent.set("version", UI_INTENT_VERSION)?;
            intent.set("type", "refresh")?;
            intent.set("binding", binding)?;
            push_ui_intent(lua, intent)
        })?,
    )?;

    ui.set(
        "section",
        lua.create_function(|lua, (title, nodes): (String, Option<Table>)| {
            let node = lua.create_table()?;
            node.set("kind", "section")?;
            node.set("title", title)?;
            node.set("nodes", nodes.unwrap_or(lua.create_table()?))?;
            Ok(node)
        })?,
    )?;

    ui.set(
        "text",
        lua.create_function(|lua, (text, opts): (String, Option<Table>)| {
            let node = lua.create_table()?;
            node.set("kind", "text")?;
            node.set("text", text)?;
            copy_optional_fields(&node, opts.as_ref(), &["id"])?;
            Ok(node)
        })?,
    )?;

    ui.set(
        "action",
        lua.create_function(
            |lua, (label, action, opts): (String, String, Option<Table>)| {
                let node = lua.create_table()?;
                node.set("kind", "action")?;
                node.set("label", label)?;
                node.set("action", action)?;
                copy_optional_fields(&node, opts.as_ref(), &["id", "params", "confirm"])?;
                Ok(node)
            },
        )?,
    )?;

    ui.set(
        "worklist",
        lua.create_function(|lua, (title, opts): (String, Option<Table>)| {
            let node = source_node(lua, "worklist", title, opts.as_ref())?;
            copy_optional_fields(&node, opts.as_ref(), &["filters"])?;
            Ok(node)
        })?,
    )?;

    ui.set(
        "activity",
        lua.create_function(|lua, (title, opts): (String, Option<Table>)| {
            source_node(lua, "activity", title, opts.as_ref())
        })?,
    )?;

    ui.set(
        "detail",
        lua.create_function(|lua, (title, opts): (String, Option<Table>)| {
            source_node(lua, "detail", title, opts.as_ref())
        })?,
    )?;

    ui.set(
        "approval_queue",
        lua.create_function(|lua, (title, opts): (String, Option<Table>)| {
            let node = source_node(lua, "approval_queue", title, opts.as_ref())?;
            copy_optional_fields(&node, opts.as_ref(), &["filters"])?;
            Ok(node)
        })?,
    )?;

    lua.globals().set(UI_TABLE_KEY, ui)?;
    Ok(())
}

pub(crate) fn ui_intents(lua: &Lua) -> LuaResult<Vec<UiIntentMessage>> {
    let registry = ensure_ui_intent_registry(lua)?;
    let mut out = Vec::new();
    for index in 1..=registry.raw_len() {
        let value: Value = registry.raw_get(index)?;
        let json = object_refs::encode_lua_payload(lua, value)?;
        let message = serde_json::from_value(json).map_err(mlua::Error::external)?;
        out.push(message);
    }
    Ok(out)
}

fn source_node(lua: &Lua, kind: &str, title: String, opts: Option<&Table>) -> LuaResult<Table> {
    let node = lua.create_table()?;
    node.set("kind", kind)?;
    node.set("title", title)?;
    if let Some(opts) = opts {
        copy_optional_fields(&node, Some(opts), &["id"])?;
        match opts.get::<Value>("source")? {
            Value::Nil => match opts.get::<Value>("from")? {
                Value::Nil => {}
                value => node.set("source", value)?,
            },
            value => node.set("source", value)?,
        }
    }
    Ok(node)
}

fn push_ui_intent(lua: &Lua, intent: Table) -> LuaResult<Table> {
    let registry = ensure_ui_intent_registry(lua)?;
    let next_index = registry.raw_len() + 1;
    registry.raw_set(next_index, intent.clone())?;
    Ok(intent)
}

fn ensure_ui_intent_registry(lua: &Lua) -> LuaResult<Table> {
    let globals = lua.globals();
    if !globals.contains_key(UI_INTENT_REGISTRY_KEY)? {
        globals.set(UI_INTENT_REGISTRY_KEY, lua.create_table()?)?;
    }
    globals.get(UI_INTENT_REGISTRY_KEY)
}

fn copy_optional_fields(target: &Table, source: Option<&Table>, fields: &[&str]) -> LuaResult<()> {
    let Some(source) = source else {
        return Ok(());
    };

    for field in fields {
        let value: Value = source.get(*field)?;
        if !matches!(value, Value::Nil) {
            target.set(*field, value)?;
        }
    }

    Ok(())
}
