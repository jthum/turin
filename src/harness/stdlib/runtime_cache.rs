use std::path::Path;
use std::sync::Arc;

use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};
use serde::Deserialize;
use uuid::Uuid;

use crate::harness::globals::{ActiveHarnessExecutionContext, HarnessAppData};
use crate::harness::stdlib::binding_common::{
    bool_value_ok, bridge_async_result, nil_err, ok_value,
};
use crate::harness::stdlib::governance_support::require_capability as require_governance_capability;
use crate::harness::stdlib::system_globals::{require_capability_for_lua, resolve_safe_path};
use crate::kernel::session_refs::parse_session_reference;
use crate::persistence::manager::StoreManager;
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::{
    CacheReadResult, CacheResetReport, CacheSessionStats, CacheStatsReport,
};

#[derive(Debug, Default, Deserialize)]
struct LuaCacheReadOpts {
    session_id: Option<String>,
    include_content: Option<bool>,
    include_previous: Option<bool>,
    max_diff_lines: Option<usize>,
    token_estimate: Option<bool>,
    trace: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct LuaCacheInvalidateOpts {
    scope: Option<String>,
    session_id: Option<String>,
    trace: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct LuaCacheStatsOpts {
    scope: Option<String>,
    session_id: Option<String>,
    trace: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct LuaCacheResetOpts {
    scope: Option<String>,
    session_id: Option<String>,
    dry_run: Option<bool>,
    trace: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheScope {
    Session,
    Global,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheStatsScope {
    Session,
    Global,
    Both,
}

pub fn register_runtime_cache_namespace(
    lua: &Lua,
    runtime_table: &Table,
    app_data: &HarnessAppData,
) -> LuaResult<()> {
    let runtime_cache = lua.create_table()?;

    {
        let manager = app_data.store_manager.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        let fs_root = app_data.fs_root.clone();
        runtime_cache.set(
            "read",
            lua.create_function(move |lua, (path, opts): (String, Option<Table>)| {
                if let Err(err) = require_capability_for_lua(lua, "fs.read") {
                    return nil_err(lua, &err.to_string());
                }

                let parsed = cache_read_opts(lua, opts)?;
                let manager = manager.clone();
                let execution_ctx = execution_ctx.clone();
                let fs_root = fs_root.clone();
                let result = bridge_async_result(async move {
                    let (session_id, _, store_selector) = resolve_cache_session(
                        manager.clone(),
                        execution_ctx,
                        parsed.session_id.clone(),
                    )
                    .await?;
                    let (cache_path, resolved) =
                        normalize_cache_path(&fs_root, &path).map_err(|e| e.to_string())?;
                    let content = std::fs::read_to_string(&resolved).map_err(|e| e.to_string())?;
                    let store = manager
                        .open(&store_selector)
                        .await
                        .map_err(|e| e.to_string())?;
                    store
                        .cache_read_file(
                            session_id,
                            &cache_path,
                            &content,
                            parsed.include_content.unwrap_or(false),
                            parsed.include_previous.unwrap_or(false),
                            parsed.max_diff_lines.unwrap_or(200),
                            parsed.token_estimate.unwrap_or(true),
                        )
                        .await
                        .map_err(|e| e.to_string())
                });

                match result {
                    Ok(report) => Ok(ok_value(cache_read_result_to_lua(lua, &report)?)),
                    Err(err) => nil_err(lua, &err),
                }
            })?,
        )?;
    }

    {
        let manager = app_data.store_manager.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        let fs_root = app_data.fs_root.clone();
        let app_data_snapshot = app_data.clone();
        runtime_cache.set(
            "invalidate",
            lua.create_function(move |lua, (path, opts): (String, Option<Table>)| {
                if let Err(err) =
                    require_governance_capability(&app_data_snapshot, "runtime.cache.invalidate")
                {
                    return crate::harness::stdlib::binding_common::bool_err(lua, &err);
                }

                let parsed = cache_invalidate_opts(lua, opts)?;
                let manager = manager.clone();
                let execution_ctx = execution_ctx.clone();
                let fs_root = fs_root.clone();
                let result = bridge_async_result(async move {
                    let scope = parse_cache_scope(parsed.scope.as_deref())?;
                    let session_id = if matches!(scope, CacheScope::Session) {
                        Some(
                            resolve_cache_session(
                                manager.clone(),
                                execution_ctx.clone(),
                                parsed.session_id.clone(),
                            )
                            .await?
                            .0,
                        )
                    } else {
                        None
                    };
                    let store_selector =
                        resolve_cache_store_selector(execution_ctx, parsed.session_id.as_deref())?;
                    let (cache_path, _) =
                        normalize_cache_path(&fs_root, &path).map_err(|e| e.to_string())?;
                    let store = manager
                        .open(&store_selector)
                        .await
                        .map_err(|e| e.to_string())?;
                    store
                        .cache_invalidate_file(
                            &cache_path,
                            session_id,
                            matches!(scope, CacheScope::Global),
                        )
                        .await
                        .map_err(|e| e.to_string())
                });

                match result {
                    Ok(changed) => Ok(bool_value_ok(changed)),
                    Err(err) => crate::harness::stdlib::binding_common::bool_err(lua, &err),
                }
            })?,
        )?;
    }

    {
        let manager = app_data.store_manager.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        let app_data_snapshot = app_data.clone();
        runtime_cache.set(
            "stats",
            lua.create_function(move |lua, opts: Option<Table>| {
                if let Err(err) =
                    require_governance_capability(&app_data_snapshot, "runtime.cache.stats")
                {
                    return nil_err(lua, &err);
                }

                let parsed = cache_stats_opts(lua, opts)?;
                let manager = manager.clone();
                let execution_ctx = execution_ctx.clone();
                let result = bridge_async_result(async move {
                    let scope = parse_cache_stats_scope(parsed.scope.as_deref())?;
                    let include_global =
                        matches!(scope, CacheStatsScope::Global | CacheStatsScope::Both);
                    let include_session =
                        matches!(scope, CacheStatsScope::Session | CacheStatsScope::Both);
                    let session_id = if include_session {
                        Some(
                            resolve_cache_session(
                                manager.clone(),
                                execution_ctx.clone(),
                                parsed.session_id.clone(),
                            )
                            .await?
                            .0,
                        )
                    } else {
                        None
                    };
                    let store_selector =
                        resolve_cache_store_selector(execution_ctx, parsed.session_id.as_deref())?;
                    let store = manager
                        .open(&store_selector)
                        .await
                        .map_err(|e| e.to_string())?;
                    store
                        .cache_stats(session_id, include_global, include_session)
                        .await
                        .map_err(|e| e.to_string())
                });

                match result {
                    Ok(report) => Ok(ok_value(cache_stats_to_lua(lua, &report)?)),
                    Err(err) => nil_err(lua, &err),
                }
            })?,
        )?;
    }

    {
        let manager = app_data.store_manager.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        let app_data_snapshot = app_data.clone();
        runtime_cache.set(
            "reset",
            lua.create_function(move |lua, opts: Option<Table>| {
                if let Err(err) =
                    require_governance_capability(&app_data_snapshot, "runtime.cache.reset")
                {
                    return nil_err(lua, &err);
                }

                let parsed = cache_reset_opts(lua, opts)?;
                let manager = manager.clone();
                let execution_ctx = execution_ctx.clone();
                let result = bridge_async_result(async move {
                    let scope = parse_cache_scope(parsed.scope.as_deref())?;
                    let dry_run = parsed.dry_run.unwrap_or(true);
                    let session_id = if matches!(scope, CacheScope::Session) {
                        Some(
                            resolve_cache_session(
                                manager.clone(),
                                execution_ctx.clone(),
                                parsed.session_id.clone(),
                            )
                            .await?
                            .0,
                        )
                    } else {
                        None
                    };
                    let store_selector =
                        resolve_cache_store_selector(execution_ctx, parsed.session_id.as_deref())?;
                    let store = manager
                        .open(&store_selector)
                        .await
                        .map_err(|e| e.to_string())?;
                    store
                        .cache_reset(session_id, matches!(scope, CacheScope::Global), dry_run)
                        .await
                        .map_err(|e| e.to_string())
                });

                match result {
                    Ok(report) => Ok(ok_value(cache_reset_to_lua(lua, &report)?)),
                    Err(err) => nil_err(lua, &err),
                }
            })?,
        )?;
    }

    runtime_table.set("cache", runtime_cache)?;
    Ok(())
}

fn cache_read_opts(lua: &Lua, opts: Option<Table>) -> LuaResult<LuaCacheReadOpts> {
    let parsed: LuaCacheReadOpts = parse_opts(lua, opts, "invalid cache read opts")?;
    let _ = parsed.trace;
    Ok(parsed)
}

fn cache_invalidate_opts(lua: &Lua, opts: Option<Table>) -> LuaResult<LuaCacheInvalidateOpts> {
    let parsed: LuaCacheInvalidateOpts = parse_opts(lua, opts, "invalid cache invalidate opts")?;
    let _ = parsed.trace;
    Ok(parsed)
}

fn cache_stats_opts(lua: &Lua, opts: Option<Table>) -> LuaResult<LuaCacheStatsOpts> {
    let parsed: LuaCacheStatsOpts = parse_opts(lua, opts, "invalid cache stats opts")?;
    let _ = parsed.trace;
    Ok(parsed)
}

fn cache_reset_opts(lua: &Lua, opts: Option<Table>) -> LuaResult<LuaCacheResetOpts> {
    let parsed: LuaCacheResetOpts = parse_opts(lua, opts, "invalid cache reset opts")?;
    let _ = parsed.trace;
    Ok(parsed)
}

fn parse_opts<T>(lua: &Lua, opts: Option<Table>, label: &str) -> LuaResult<T>
where
    T: for<'de> Deserialize<'de> + Default,
{
    match opts {
        None => Ok(T::default()),
        Some(table) => {
            let parsed = lua
                .from_value::<T>(Value::Table(table))
                .map_err(|e| mlua::Error::runtime(format!("{label}: {e}")))?;
            Ok(parsed)
        }
    }
}

async fn resolve_cache_session(
    manager: Arc<StoreManager>,
    execution_ctx: ActiveHarnessExecutionContext,
    requested_session_id: Option<String>,
) -> Result<(i64, String, StoreSelector), String> {
    let (public_id, selector) = if let Some(requested) = requested_session_id {
        let session_ref = parse_session_reference(&requested).map_err(|e| e.to_string())?;
        (
            session_ref.public_id,
            session_ref
                .store_selector
                .unwrap_or_else(|| StoreSelector::Alias("state".to_string())),
        )
    } else {
        let lock = execution_ctx
            .lock()
            .map_err(|_| "execution context mutex poisoned".to_string())?;
        (
            lock.session_id
                .clone()
                .ok_or_else(|| "No active session context".to_string())?,
            lock.session_store_selector
                .clone()
                .unwrap_or_else(|| StoreSelector::Alias("state".to_string())),
        )
    };
    let uuid = Uuid::parse_str(&public_id)
        .map_err(|e| format!("invalid session id '{}': {}", public_id, e))?;
    let store = manager.open(&selector).await.map_err(|e| e.to_string())?;
    let session_id = store
        .get_session_by_public_id(uuid)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("unknown session id '{}'", public_id))?;
    Ok((session_id, uuid.simple().to_string(), selector))
}

fn resolve_cache_store_selector(
    execution_ctx: ActiveHarnessExecutionContext,
    requested_session_id: Option<&str>,
) -> Result<StoreSelector, String> {
    if let Some(requested) = requested_session_id {
        let session_ref = parse_session_reference(requested).map_err(|e| e.to_string())?;
        return Ok(session_ref
            .store_selector
            .unwrap_or_else(|| StoreSelector::Alias("state".to_string())));
    }

    let lock = execution_ctx
        .lock()
        .map_err(|_| "execution context mutex poisoned".to_string())?;
    Ok(lock
        .session_store_selector
        .clone()
        .unwrap_or_else(|| StoreSelector::Alias("state".to_string())))
}

fn normalize_cache_path(
    root: &Path,
    requested_path: &str,
) -> Result<(String, std::path::PathBuf), String> {
    let resolved = resolve_safe_path(root, requested_path)
        .ok_or_else(|| "Unsafe path traversal".to_string())?;
    let display_path = resolved
        .strip_prefix(root)
        .unwrap_or(&resolved)
        .to_string_lossy()
        .replace('\\', "/");
    Ok((display_path, resolved))
}

fn parse_cache_scope(value: Option<&str>) -> Result<CacheScope, String> {
    match value.unwrap_or("session") {
        "session" => Ok(CacheScope::Session),
        "global" => Ok(CacheScope::Global),
        other => Err(format!("invalid cache scope '{}'", other)),
    }
}

fn parse_cache_stats_scope(value: Option<&str>) -> Result<CacheStatsScope, String> {
    match value.unwrap_or("both") {
        "session" => Ok(CacheStatsScope::Session),
        "global" => Ok(CacheStatsScope::Global),
        "both" => Ok(CacheStatsScope::Both),
        other => Err(format!("invalid cache stats scope '{}'", other)),
    }
}

fn cache_read_result_to_lua(lua: &Lua, report: &CacheReadResult) -> LuaResult<Value> {
    let table = lua.create_table()?;
    table.set("status", report.status.as_str())?;
    table.set("path", report.path.clone())?;
    table.set("hash", report.hash.clone())?;
    if let Some(previous_hash) = &report.previous_hash {
        table.set("previous_hash", previous_hash.clone())?;
    }
    if let Some(content) = &report.content {
        table.set("content", content.clone())?;
    }
    if let Some(previous_content) = &report.previous_content {
        table.set("previous_content", previous_content.clone())?;
    }
    if let Some(diff) = &report.diff {
        table.set("diff", diff.clone())?;
    }
    table.set("diff_truncated", report.diff_truncated)?;
    table.set("estimated_tokens_saved", report.estimated_tokens_saved)?;
    table.set("read_at", report.read_at.clone())?;
    Ok(Value::Table(table))
}

fn cache_stats_to_lua(lua: &Lua, report: &CacheStatsReport) -> LuaResult<Value> {
    let table = lua.create_table()?;
    if let Some(global) = &report.global {
        let global_table = lua.create_table()?;
        global_table.set("cached_files", global.cached_files)?;
        global_table.set("cached_versions", global.cached_versions)?;
        global_table.set("tokens_saved", global.tokens_saved)?;
        table.set("global", global_table)?;
    }
    if let Some(session) = &report.session {
        table.set("session", cache_session_stats_to_lua(lua, session)?)?;
    }
    Ok(Value::Table(table))
}

fn cache_session_stats_to_lua(lua: &Lua, stats: &CacheSessionStats) -> LuaResult<Table> {
    let table = lua.create_table()?;
    table.set("id", public_id_to_simple_string(&stats.public_id)?)?;
    table.set("files_seen", stats.files_seen)?;
    table.set("tokens_saved", stats.tokens_saved)?;
    Ok(table)
}

fn cache_reset_to_lua(lua: &Lua, report: &CacheResetReport) -> LuaResult<Value> {
    let table = lua.create_table()?;
    table.set("scope", report.scope.clone())?;
    table.set("removed_versions", report.removed_versions)?;
    table.set("removed_reads", report.removed_reads)?;
    table.set("reset_stats", report.reset_stats)?;
    table.set("dry_run", report.dry_run)?;
    Ok(Value::Table(table))
}

fn public_id_to_simple_string(bytes: &[u8]) -> LuaResult<String> {
    Uuid::from_slice(bytes)
        .map(|id| id.simple().to_string())
        .map_err(|e| mlua::Error::runtime(format!("invalid session public id: {}", e)))
}
