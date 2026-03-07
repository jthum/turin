use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};
use serde::Deserialize;

use crate::code_index_reader::{
    CodeSearchMode, CodeSearchRequest, CodebaseSelector, search as code_search,
    status as code_status,
};
use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::binding_common::{bridge_async_result, nil_err, ok_value};
use crate::harness::stdlib::governance_support::require_capability as require_governance_capability;

#[derive(Debug, Default, Deserialize)]
struct LuaCodeSearchOpts {
    limit: Option<i64>,
    languages: Option<Vec<String>>,
    kinds: Option<Vec<String>>,
    min_score: Option<f64>,
    strict: Option<bool>,
    trace: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct LuaCodeStatusOpts {
    trace: Option<bool>,
}

pub fn register_runtime_code_namespace(
    lua: &Lua,
    runtime_table: &Table,
    app_data: &HarnessAppData,
) -> LuaResult<()> {
    let runtime_code = lua.create_table()?;
    let runtime_search = lua.create_table()?;

    {
        let workspace_root = app_data.workspace_root.clone();
        let app_data_snapshot = app_data.clone();
        runtime_search.set(
            "status",
            lua.create_function(move |lua, (codebase, opts): (Value, Option<Table>)| {
                if let Err(err) =
                    require_governance_capability(&app_data_snapshot, "runtime.code.search.status")
                {
                    return nil_err(lua, &err);
                }
                let selector = codebase_selector_from_value(codebase)?;
                let parsed = parse_status_opts(lua, opts)?;
                let _ = parsed;
                let workspace_root = workspace_root.clone();
                let result = bridge_async_result(async move {
                    code_status(&workspace_root, selector)
                        .await
                        .map_err(|e| e.to_string())
                });
                match result {
                    Ok(status) => Ok(ok_value(lua.to_value(&status)?)),
                    Err(err) => nil_err(lua, &format!("runtime.code.search.status: {err}")),
                }
            })?,
        )?;
    }

    {
        let workspace_root = app_data.workspace_root.clone();
        let app_data_snapshot = app_data.clone();
        runtime_search.set(
            "lexical",
            lua.create_function(
                move |lua, (codebase, query, opts): (Value, String, Option<Table>)| {
                    if let Err(err) = require_governance_capability(
                        &app_data_snapshot,
                        "runtime.code.search.lexical",
                    ) {
                        return nil_err(lua, &err);
                    }
                    run_code_search(
                        lua,
                        &workspace_root,
                        codebase,
                        query,
                        opts,
                        CodeSearchMode::Lexical,
                        "runtime.code.search.lexical",
                    )
                },
            )?,
        )?;
    }

    {
        let workspace_root = app_data.workspace_root.clone();
        let app_data_snapshot = app_data.clone();
        runtime_search.set(
            "semantic",
            lua.create_function(
                move |lua, (codebase, query, opts): (Value, String, Option<Table>)| {
                    if let Err(err) = require_governance_capability(
                        &app_data_snapshot,
                        "runtime.code.search.semantic",
                    ) {
                        return nil_err(lua, &err);
                    }
                    run_code_search(
                        lua,
                        &workspace_root,
                        codebase,
                        query,
                        opts,
                        CodeSearchMode::Semantic,
                        "runtime.code.search.semantic",
                    )
                },
            )?,
        )?;
    }

    {
        let workspace_root = app_data.workspace_root.clone();
        let app_data_snapshot = app_data.clone();
        runtime_search.set(
            "hybrid",
            lua.create_function(
                move |lua, (codebase, query, opts): (Value, String, Option<Table>)| {
                    if let Err(err) = require_governance_capability(
                        &app_data_snapshot,
                        "runtime.code.search.hybrid",
                    ) {
                        return nil_err(lua, &err);
                    }
                    run_code_search(
                        lua,
                        &workspace_root,
                        codebase,
                        query,
                        opts,
                        CodeSearchMode::Hybrid,
                        "runtime.code.search.hybrid",
                    )
                },
            )?,
        )?;
    }

    runtime_code.set("search", runtime_search)?;
    runtime_table.set("code", runtime_code)?;
    Ok(())
}

fn run_code_search(
    lua: &Lua,
    workspace_root: &std::path::Path,
    codebase: Value,
    query: String,
    opts: Option<Table>,
    mode: CodeSearchMode,
    label: &str,
) -> LuaResult<(Value, Value)> {
    let selector = codebase_selector_from_value(codebase)?;
    let request = code_search_request_from_opts(lua, opts)?;
    let workspace_root = workspace_root.to_path_buf();
    let result = bridge_async_result(async move {
        code_search(&workspace_root, selector, mode, &query, &request)
            .await
            .map_err(|e| e.to_string())
    });
    match result {
        Ok(rows) => Ok(ok_value(lua.to_value(&rows)?)),
        Err(err) => nil_err(lua, &format!("{label}: {err}")),
    }
}

fn codebase_selector_from_value(value: Value) -> LuaResult<CodebaseSelector> {
    match value {
        Value::String(s) => Ok(CodebaseSelector {
            root: s.to_str()?.to_string(),
            index_path: None,
        }),
        Value::Table(t) => {
            let root = t.get::<String>("root").map_err(|_| {
                mlua::Error::runtime("invalid codebase selector; expected string or { root = ... }")
            })?;
            let index_path = t.get::<Option<String>>("index_path")?;
            Ok(CodebaseSelector { root, index_path })
        }
        _ => Err(mlua::Error::runtime(
            "invalid codebase selector; expected string or { root = ... }",
        )),
    }
}

fn code_search_request_from_opts(lua: &Lua, opts: Option<Table>) -> LuaResult<CodeSearchRequest> {
    let parsed = match opts {
        None => LuaCodeSearchOpts::default(),
        Some(table) => lua
            .from_value::<LuaCodeSearchOpts>(Value::Table(table))
            .map_err(|e| mlua::Error::runtime(format!("invalid code search opts: {e}")))?,
    };
    let _ = parsed.trace;
    Ok(CodeSearchRequest {
        limit: parsed.limit.unwrap_or(10).max(1) as usize,
        languages: parsed.languages.unwrap_or_default(),
        kinds: parsed.kinds.unwrap_or_default(),
        min_score: parsed.min_score.unwrap_or(0.0),
        strict: parsed.strict.unwrap_or(false),
    })
}

fn parse_status_opts(lua: &Lua, opts: Option<Table>) -> LuaResult<LuaCodeStatusOpts> {
    let parsed = match opts {
        None => LuaCodeStatusOpts::default(),
        Some(table) => lua
            .from_value::<LuaCodeStatusOpts>(Value::Table(table))
            .map_err(|e| mlua::Error::runtime(format!("invalid code search status opts: {e}")))?,
    };
    let _ = parsed.trace;
    Ok(parsed)
}
