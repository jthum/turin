use mlua::{Function, Lua, MultiValue, Result as LuaResult, Table, Value};
use turin_daemon_protocol::{UI_INTENT_VERSION, UiIntentMessage};

use crate::harness::stdlib::object_refs;

const UI_TABLE_KEY: &str = "ui";
const UI_INTENT_REGISTRY_KEY: &str = "__harness_ui_intents";
const UI_DEFAULT_APP_KEY: &str = "__harness_ui_default_app";
const APP_ID_KEY: &str = "__ui_app_id";
const NODE_REGISTRY_KEY: &str = "__ui_nodes";
const MENU_ITEM_REGISTRY_KEY: &str = "__ui_menu_items";

pub fn register_ui_globals(lua: &Lua) -> LuaResult<()> {
    let ui = lua.create_table()?;

    ui.set(
        "app",
        lua.create_function(|lua, (title, opts): (String, Option<Table>)| {
            let app_id = option_string(opts.as_ref(), "id")?.unwrap_or_else(|| "main".to_string());

            let intent = lua.create_table()?;
            intent.set("version", UI_INTENT_VERSION)?;
            intent.set("type", "app")?;
            intent.set("id", app_id.clone())?;
            intent.set("title", title)?;
            copy_optional_fields(&intent, opts.as_ref(), &["about", "icon"])?;
            push_ui_intent(lua, intent)?;

            if lua.globals().get::<Value>(UI_DEFAULT_APP_KEY)?.is_nil() {
                lua.globals().set(UI_DEFAULT_APP_KEY, app_id.clone())?;
            }

            app_proxy(lua, app_id)
        })?,
    )?;

    ui.set(
        "notice",
        lua.create_function(|lua, (title, opts): (String, Option<Table>)| {
            let app_id = default_app_id(lua)?;
            push_notice(lua, &app_id, title, opts.as_ref())
        })?,
    )?;

    ui.set(
        "open",
        lua.create_function(|lua, (target, opts): (String, Option<Table>)| {
            let app_id = default_app_id(lua)?;
            push_open(lua, &app_id, target, opts.as_ref())
        })?,
    )?;

    ui.set(
        "show",
        lua.create_function(|lua, (target, opts): (String, Option<Table>)| {
            let app_id = default_app_id(lua)?;
            push_show(lua, &app_id, target, opts.as_ref())
        })?,
    )?;

    ui.set(
        "badge",
        lua.create_function(|lua, (target, opts): (String, Option<Table>)| {
            let app_id = default_app_id(lua)?;
            push_badge(lua, &app_id, target, opts.as_ref())
        })?,
    )?;

    ui.set(
        "focus",
        lua.create_function(|lua, target: String| {
            let app_id = default_app_id(lua)?;
            push_focus(lua, &app_id, target)
        })?,
    )?;

    ui.set(
        "refresh",
        lua.create_function(|lua, binding: String| {
            let app_id = default_app_id(lua)?;
            push_refresh(lua, &app_id, binding)
        })?,
    )?;

    lua.globals().set(UI_TABLE_KEY, ui)?;
    Ok(())
}

pub(crate) fn ui_intents(lua: &Lua) -> LuaResult<Vec<UiIntentMessage>> {
    ui_intents_from(lua, 0)
}

pub(crate) fn ui_intent_count(lua: &Lua) -> LuaResult<usize> {
    Ok(ensure_ui_intent_registry(lua)?.raw_len())
}

pub(crate) fn ui_intents_from(lua: &Lua, start_index: usize) -> LuaResult<Vec<UiIntentMessage>> {
    let registry = ensure_ui_intent_registry(lua)?;
    let mut out = Vec::new();
    for index in (start_index + 1)..=registry.raw_len() {
        let value: Value = registry.raw_get(index)?;
        let json = object_refs::encode_lua_payload(lua, value)?;
        let message = serde_json::from_value(json).map_err(mlua::Error::external)?;
        out.push(message);
    }
    Ok(out)
}

fn app_proxy(lua: &Lua, app_id: String) -> LuaResult<Table> {
    let app = lua.create_table()?;
    app.set(APP_ID_KEY, app_id.clone())?;

    app.set(
        "screen",
        lua.create_function(
            |lua, (app, id, title, callback, opts): (Table, String, String, Function, Option<Table>)| {
                let app_id = app_id_from_proxy(&app)?;
                push_screen(lua, &app_id, id, title, callback, opts.as_ref(), false)?;
                Ok(app)
            },
        )?,
    )?;

    app.set(
        "home",
        lua.create_function(
            |lua, (app, title, callback, opts): (Table, String, Function, Option<Table>)| {
                let app_id = app_id_from_proxy(&app)?;
                push_screen(
                    lua,
                    &app_id,
                    "home".to_string(),
                    title,
                    callback,
                    opts.as_ref(),
                    true,
                )?;
                Ok(app)
            },
        )?,
    )?;

    app.set(
        "opens_with",
        lua.create_function(|lua, (app, screen_id): (Table, String)| {
            let app_id = app_id_from_proxy(&app)?;
            push_opens_with(lua, &app_id, screen_id)?;
            Ok(app)
        })?,
    )?;

    app.set(
        "pane",
        lua.create_function(
            |lua, (app, id, title, callback, opts): (Table, String, String, Function, Option<Table>)| {
                let app_id = app_id_from_proxy(&app)?;
                push_pane(lua, &app_id, id, title, callback, opts.as_ref())?;
                Ok(app)
            },
        )?,
    )?;

    app.set(
        "menu",
        lua.create_function(|lua, (app, title, callback): (Table, String, Function)| {
            let app_id = app_id_from_proxy(&app)?;
            push_menu(lua, &app_id, title, callback)?;
            Ok(app)
        })?,
    )?;

    app.set(
        "notice",
        lua.create_function(|lua, (app, title, opts): (Table, String, Option<Table>)| {
            let app_id = app_id_from_proxy(&app)?;
            push_notice(lua, &app_id, title, opts.as_ref())?;
            Ok(app)
        })?,
    )?;

    app.set(
        "open",
        lua.create_function(|lua, (app, target, opts): (Table, String, Option<Table>)| {
            let app_id = app_id_from_proxy(&app)?;
            push_open(lua, &app_id, target, opts.as_ref())?;
            Ok(app)
        })?,
    )?;

    app.set(
        "show",
        lua.create_function(|lua, (app, target, opts): (Table, String, Option<Table>)| {
            let app_id = app_id_from_proxy(&app)?;
            push_show(lua, &app_id, target, opts.as_ref())?;
            Ok(app)
        })?,
    )?;

    app.set(
        "badge",
        lua.create_function(|lua, (app, target, opts): (Table, String, Option<Table>)| {
            let app_id = app_id_from_proxy(&app)?;
            push_badge(lua, &app_id, target, opts.as_ref())?;
            Ok(app)
        })?,
    )?;

    app.set(
        "focus",
        lua.create_function(|lua, (app, target): (Table, String)| {
            let app_id = app_id_from_proxy(&app)?;
            push_focus(lua, &app_id, target)?;
            Ok(app)
        })?,
    )?;

    app.set(
        "refresh",
        lua.create_function(|lua, (app, binding): (Table, String)| {
            let app_id = app_id_from_proxy(&app)?;
            push_refresh(lua, &app_id, binding)?;
            Ok(app)
        })?,
    )?;

    Ok(app)
}

fn push_screen(
    lua: &Lua,
    app_id: &str,
    id: String,
    title: String,
    callback: Function,
    opts: Option<&Table>,
    opens_with: bool,
) -> LuaResult<()> {
    let screen = node_container_proxy(lua)?;
    callback.call::<()>(screen.clone())?;

    let intent = lua.create_table()?;
    intent.set("version", UI_INTENT_VERSION)?;
    intent.set("type", "screen")?;
    intent.set("app_id", app_id)?;
    intent.set("id", id.clone())?;
    intent.set("title", title)?;
    copy_optional_fields(&intent, opts, &["presentation"])?;
    intent.set("nodes", nodes_from_proxy(&screen)?)?;
    push_ui_intent(lua, intent)?;

    if opens_with {
        push_opens_with(lua, app_id, id)?;
    }

    Ok(())
}

fn push_pane(
    lua: &Lua,
    app_id: &str,
    id: String,
    title: String,
    callback: Function,
    opts: Option<&Table>,
) -> LuaResult<()> {
    let pane = node_container_proxy(lua)?;
    callback.call::<()>(pane.clone())?;

    let intent = lua.create_table()?;
    intent.set("version", UI_INTENT_VERSION)?;
    intent.set("type", "pane")?;
    intent.set("app_id", app_id)?;
    intent.set("id", id)?;
    intent.set("title", title)?;
    copy_optional_fields(&intent, opts, &["presentation"])?;
    intent.set("nodes", nodes_from_proxy(&pane)?)?;
    push_ui_intent(lua, intent)?;
    Ok(())
}

fn push_menu(lua: &Lua, app_id: &str, title: String, callback: Function) -> LuaResult<()> {
    let menu = menu_proxy(lua)?;
    callback.call::<()>(menu.clone())?;

    let intent = lua.create_table()?;
    intent.set("version", UI_INTENT_VERSION)?;
    intent.set("type", "menu")?;
    intent.set("app_id", app_id)?;
    intent.set("title", title)?;
    intent.set("items", menu_items_from_proxy(&menu)?)?;
    push_ui_intent(lua, intent)?;
    Ok(())
}

fn node_container_proxy(lua: &Lua) -> LuaResult<Table> {
    let proxy = lua.create_table()?;
    proxy.set(NODE_REGISTRY_KEY, lua.create_table()?)?;
    install_node_methods(lua, &proxy)?;
    Ok(proxy)
}

fn install_node_methods(lua: &Lua, proxy: &Table) -> LuaResult<()> {
    proxy.set(
        "section",
        lua.create_function(
            |lua, (container, title, callback): (Table, String, Function)| {
                let section = node_container_proxy(lua)?;
                callback.call::<()>(section.clone())?;
                let node = lua.create_table()?;
                node.set("kind", "section")?;
                node.set("title", title)?;
                node.set("nodes", nodes_from_proxy(&section)?)?;
                push_node(&container, node)?;
                Ok(container)
            },
        )?,
    )?;

    proxy.set(
        "text",
        lua.create_function(
            |lua, (container, text, opts): (Table, String, Option<Table>)| {
                let node = lua.create_table()?;
                node.set("kind", "text")?;
                node.set("text", text)?;
                copy_optional_fields(&node, opts.as_ref(), &["id"])?;
                push_node(&container, node)?;
                Ok(container)
            },
        )?,
    )?;

    proxy.set(
        "action",
        lua.create_function(
            |lua, (container, label, action, opts): (Table, String, String, Option<Table>)| {
                let node = lua.create_table()?;
                node.set("kind", "action")?;
                node.set("label", label)?;
                node.set("action", action)?;
                copy_optional_fields(&node, opts.as_ref(), &["id", "params", "confirm"])?;
                push_node(&container, node)?;
                Ok(container)
            },
        )?,
    )?;

    proxy.set(
        "list",
        lua.create_function(
            |lua, (container, title, opts): (Table, String, Option<Table>)| {
                let node = source_node(lua, "list", title, opts.as_ref())?;
                copy_optional_fields(
                    &node,
                    opts.as_ref(),
                    &["where", "fields", "sort", "limit", "intent", "as"],
                )?;
                push_node(&container, node)?;
                Ok(container)
            },
        )?,
    )?;

    proxy.set(
        "worklist",
        lua.create_function(
            |lua, (container, title, opts): (Table, String, Option<Table>)| {
                let node = source_node(lua, "list", title, opts.as_ref())?;
                if option_string(opts.as_ref(), "source")?.is_none()
                    && let Some(from) = option_string(opts.as_ref(), "from")?
                {
                    node.set("source", format!("worklists.{from}"))?;
                }
                if node.get::<Value>("intent")?.is_nil() {
                    node.set("intent", "tasks")?;
                }
                if node.get::<Value>("as")?.is_nil() {
                    node.set("as", "table")?;
                }
                copy_optional_fields(
                    &node,
                    opts.as_ref(),
                    &["where", "fields", "sort", "limit", "intent", "as"],
                )?;
                push_node(&container, node)?;
                Ok(container)
            },
        )?,
    )?;

    proxy.set(
        "activity",
        lua.create_function(
            |lua, (container, title, opts): (Table, String, Option<Table>)| {
                let node = source_node(lua, "activity", title, opts.as_ref())?;
                push_node(&container, node)?;
                Ok(container)
            },
        )?,
    )?;

    proxy.set(
        "detail",
        lua.create_function(
            |lua, (container, title, opts): (Table, String, Option<Table>)| {
                let node = source_node(lua, "detail", title, opts.as_ref())?;
                copy_optional_fields(&node, opts.as_ref(), &["item_id"])?;
                push_node(&container, node)?;
                Ok(container)
            },
        )?,
    )?;

    proxy.set(
        "form",
        lua.create_function(|lua, (container, title, opts): (Table, String, Table)| {
            let node = lua.create_table()?;
            node.set("kind", "form")?;
            node.set("title", title)?;
            copy_optional_fields(&node, Some(&opts), &["id", "action", "fields", "params"])?;
            push_node(&container, node)?;
            Ok(container)
        })?,
    )?;

    proxy.set(
        "report",
        lua.create_function(
            |lua, (container, title, opts): (Table, String, Option<Table>)| {
                let node = source_node(lua, "report", title, opts.as_ref())?;
                copy_optional_fields(&node, opts.as_ref(), &["prompt"])?;
                push_node(&container, node)?;
                Ok(container)
            },
        )?,
    )?;

    proxy.set(
        "chart",
        lua.create_function(
            |lua, (container, title, opts): (Table, String, Option<Table>)| {
                let node = source_node(lua, "chart", title, opts.as_ref())?;
                copy_optional_fields(&node, opts.as_ref(), &["intent", "as"])?;
                push_node(&container, node)?;
                Ok(container)
            },
        )?,
    )?;

    Ok(())
}

fn menu_proxy(lua: &Lua) -> LuaResult<Table> {
    let proxy = lua.create_table()?;
    proxy.set(MENU_ITEM_REGISTRY_KEY, lua.create_table()?)?;

    proxy.set(
        "item",
        lua.create_function(|lua, args: MultiValue| {
            let mut args = args.into_iter();
            let menu = expect_table(args.next(), "menu")?;
            let label = expect_string(args.next(), "menu item label")?;
            let opens = expect_string(args.next(), "menu item target")?;
            let third = args.next().unwrap_or(Value::Nil);
            let fourth = args.next().unwrap_or(Value::Nil);

            let (opts, callback) = match (third, fourth) {
                (Value::Table(table), Value::Function(function)) => (Some(table), Some(function)),
                (Value::Function(function), Value::Nil) => (None, Some(function)),
                (Value::Table(table), Value::Nil) => (Some(table), None),
                (Value::Nil, Value::Nil) => (None, None),
                (other, _) => {
                    return Err(mlua::Error::runtime(format!(
                        "menu:item third argument must be opts table or callback, got {other:?}"
                    )));
                }
            };

            let item = lua.create_table()?;
            item.set("label", label)?;
            item.set("opens", opens)?;
            copy_optional_fields(&item, opts.as_ref(), &["id", "icon", "badge"])?;

            if let Some(callback) = callback {
                let child = menu_proxy(lua)?;
                callback.call::<()>(child.clone())?;
                item.set("items", menu_items_from_proxy(&child)?)?;
            }

            push_menu_item(&menu, item)?;
            Ok(menu)
        })?,
    )?;

    Ok(proxy)
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

fn push_opens_with(lua: &Lua, app_id: &str, screen_id: String) -> LuaResult<()> {
    let intent = lua.create_table()?;
    intent.set("version", UI_INTENT_VERSION)?;
    intent.set("type", "opens_with")?;
    intent.set("app_id", app_id)?;
    intent.set("screen_id", screen_id)?;
    push_ui_intent(lua, intent)?;
    Ok(())
}

fn push_notice(lua: &Lua, app_id: &str, title: String, opts: Option<&Table>) -> LuaResult<()> {
    let intent = lua.create_table()?;
    intent.set("version", UI_INTENT_VERSION)?;
    intent.set("type", "notify")?;
    intent.set("app_id", app_id)?;
    intent.set("title", title)?;
    copy_optional_fields(&intent, opts, &["body", "level"])?;
    push_ui_intent(lua, intent)?;
    Ok(())
}

fn push_open(lua: &Lua, app_id: &str, target: String, opts: Option<&Table>) -> LuaResult<()> {
    let intent = lua.create_table()?;
    intent.set("version", UI_INTENT_VERSION)?;
    intent.set("type", "open")?;
    intent.set("app_id", app_id)?;
    intent.set("target", target)?;
    copy_optional_fields(&intent, opts, &["presentation"])?;
    push_ui_intent(lua, intent)?;
    Ok(())
}

fn push_show(lua: &Lua, app_id: &str, target: String, opts: Option<&Table>) -> LuaResult<()> {
    let intent = lua.create_table()?;
    intent.set("version", UI_INTENT_VERSION)?;
    intent.set("type", "show")?;
    intent.set("app_id", app_id)?;
    intent.set("target", target)?;
    copy_optional_fields(&intent, opts, &["area", "presentation"])?;
    push_ui_intent(lua, intent)?;
    Ok(())
}

fn push_badge(lua: &Lua, app_id: &str, target: String, opts: Option<&Table>) -> LuaResult<()> {
    let intent = lua.create_table()?;
    intent.set("version", UI_INTENT_VERSION)?;
    intent.set("type", "badge")?;
    intent.set("app_id", app_id)?;
    intent.set("target", target)?;
    copy_optional_fields(&intent, opts, &["count", "label", "level", "data"])?;
    push_ui_intent(lua, intent)?;
    Ok(())
}

fn push_focus(lua: &Lua, app_id: &str, target: String) -> LuaResult<()> {
    let intent = lua.create_table()?;
    intent.set("version", UI_INTENT_VERSION)?;
    intent.set("type", "focus")?;
    intent.set("app_id", app_id)?;
    intent.set("target", target)?;
    push_ui_intent(lua, intent)?;
    Ok(())
}

fn push_refresh(lua: &Lua, app_id: &str, binding: String) -> LuaResult<()> {
    let intent = lua.create_table()?;
    intent.set("version", UI_INTENT_VERSION)?;
    intent.set("type", "refresh")?;
    intent.set("app_id", app_id)?;
    intent.set("binding", binding)?;
    push_ui_intent(lua, intent)?;
    Ok(())
}

fn push_ui_intent(lua: &Lua, intent: Table) -> LuaResult<Table> {
    let registry = ensure_ui_intent_registry(lua)?;
    let next_index = registry.raw_len() + 1;
    registry.raw_set(next_index, intent.clone())?;
    Ok(intent)
}

fn push_node(container: &Table, node: Table) -> LuaResult<()> {
    let nodes: Table = container.get(NODE_REGISTRY_KEY)?;
    let next_index = nodes.raw_len() + 1;
    nodes.raw_set(next_index, node)?;
    Ok(())
}

fn push_menu_item(menu: &Table, item: Table) -> LuaResult<()> {
    let items: Table = menu.get(MENU_ITEM_REGISTRY_KEY)?;
    let next_index = items.raw_len() + 1;
    items.raw_set(next_index, item)?;
    Ok(())
}

fn nodes_from_proxy(proxy: &Table) -> LuaResult<Table> {
    proxy.get(NODE_REGISTRY_KEY)
}

fn menu_items_from_proxy(proxy: &Table) -> LuaResult<Table> {
    proxy.get(MENU_ITEM_REGISTRY_KEY)
}

fn ensure_ui_intent_registry(lua: &Lua) -> LuaResult<Table> {
    let globals = lua.globals();
    if !globals.contains_key(UI_INTENT_REGISTRY_KEY)? {
        globals.set(UI_INTENT_REGISTRY_KEY, lua.create_table()?)?;
    }
    globals.get(UI_INTENT_REGISTRY_KEY)
}

fn default_app_id(lua: &Lua) -> LuaResult<String> {
    match lua.globals().get::<Value>(UI_DEFAULT_APP_KEY)? {
        Value::String(value) => Ok(value.to_str()?.to_string()),
        Value::Nil => Err(mlua::Error::runtime(
            "ui intent requires ui.app(...) to be declared first",
        )),
        other => Err(mlua::Error::runtime(format!(
            "default ui app registry has invalid type {other:?}"
        ))),
    }
}

fn app_id_from_proxy(app: &Table) -> LuaResult<String> {
    app.get(APP_ID_KEY)
}

fn option_string(opts: Option<&Table>, field: &str) -> LuaResult<Option<String>> {
    let Some(opts) = opts else {
        return Ok(None);
    };
    match opts.get::<Value>(field)? {
        Value::Nil => Ok(None),
        Value::String(value) => Ok(Some(value.to_str()?.to_string())),
        other => Err(mlua::Error::runtime(format!(
            "ui option '{field}' must be a string, got {other:?}"
        ))),
    }
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

fn expect_table(value: Option<Value>, label: &str) -> LuaResult<Table> {
    match value {
        Some(Value::Table(value)) => Ok(value),
        Some(other) => Err(mlua::Error::runtime(format!(
            "{label} must be a table, got {other:?}"
        ))),
        None => Err(mlua::Error::runtime(format!("{label} is required"))),
    }
}

fn expect_string(value: Option<Value>, label: &str) -> LuaResult<String> {
    match value {
        Some(Value::String(value)) => Ok(value.to_str()?.to_string()),
        Some(other) => Err(mlua::Error::runtime(format!(
            "{label} must be a string, got {other:?}"
        ))),
        None => Err(mlua::Error::runtime(format!("{label} is required"))),
    }
}
