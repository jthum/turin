use std::path::Path;

use mlua::{Lua, Result as LuaResult, Value};

use crate::harness::globals::HarnessAppData;
use crate::harness::globals::block_on_current;
use crate::harness::stdlib::binding_common::{bool_err, nil_err, ok_bool, string_ok};
use crate::harness::stdlib::scoped_data_backend::encode_scope_key;
use crate::persistence::manager::{StorePathScope, StoreSelector};

use super::{hash_sha256_hex, require_capability_for_lua, resolve_safe_path};

const FS_STAT_HASH_KEY_PREFIX: &str = "_turin:fs_stat_hash:";

fn normalize_tracking_path(root: &Path, resolved: &Path) -> String {
    resolved
        .strip_prefix(root)
        .unwrap_or(resolved)
        .to_string_lossy()
        .replace('\\', "/")
}

fn current_session_tracking_context(
    lua: &Lua,
) -> Result<
    Option<(
        std::sync::Arc<crate::persistence::state::StateStore>,
        String,
    )>,
    String,
> {
    let Some(app_data) = lua.app_data_ref::<HarnessAppData>() else {
        return Ok(None);
    };

    let store_manager = app_data.store_manager.clone();
    let execution_ctx = app_data.execution_ctx.clone();
    let _ = app_data;

    let (session_id, store_selector) = execution_ctx
        .lock()
        .map_err(|_| "execution context mutex poisoned".to_string())
        .map(|lock| {
            (
                lock.session_id.clone(),
                lock.session_store_selector
                    .clone()
                    .unwrap_or_else(|| StoreSelector::Alias("state".to_string())),
            )
        })?;

    let Some(session_id) = session_id else {
        return Ok(None);
    };

    let store = block_on_current(async move {
        store_manager
            .open_with_path_scope(&store_selector, StorePathScope::AllowAny)
            .await
            .map_err(|err| err.to_string())
    })?;

    Ok(Some((store, encode_scope_key(&session_id, "default"))))
}

fn load_previous_session_hash(lua: &Lua, tracking_key: &str) -> Result<Option<String>, String> {
    let Some((store, session_scope_key)) = current_session_tracking_context(lua)? else {
        return Ok(None);
    };
    block_on_current(async move {
        store
            .kv_get("session", &session_scope_key, tracking_key)
            .await
            .map_err(|err| err.to_string())
    })
}

fn store_current_session_hash(lua: &Lua, tracking_key: &str, hash: &str) -> Result<(), String> {
    let Some((store, session_scope_key)) = current_session_tracking_context(lua)? else {
        return Ok(());
    };
    let tracking_key = tracking_key.to_string();
    let hash = hash.to_string();
    block_on_current(async move {
        store
            .kv_set("session", &session_scope_key, &tracking_key, &hash)
            .await
            .map_err(|err| err.to_string())
    })
}

pub(super) fn register_fs_module(lua: &Lua, fs_root: &Path, max_file_size: usize) -> LuaResult<()> {
    let fs_table = lua.create_table()?;
    let root = fs_root.to_path_buf();

    let r1 = root.clone();
    fs_table.set(
        "read",
        lua.create_function(move |lua, path: String| {
            if let Err(err) = require_capability_for_lua(lua, "fs.read") {
                return nil_err(lua, &err.to_string());
            }
            match resolve_safe_path(&r1, &path) {
                Some(p) => match std::fs::read_to_string(&p) {
                    Ok(c) => string_ok(lua, &c),
                    Err(e) => nil_err(lua, &e.to_string()),
                },
                None => nil_err(lua, "Unsafe path traversal"),
            }
        })?,
    )?;

    let r2 = root.clone();
    fs_table.set(
        "write",
        lua.create_function(move |lua, (path, content): (String, String)| {
            if let Err(err) = require_capability_for_lua(lua, "fs.write") {
                return bool_err(lua, &err.to_string());
            }
            if content.len() > max_file_size {
                return bool_err(lua, "File exceeds max size");
            }
            match resolve_safe_path(&r2, &path) {
                Some(p) => {
                    if let Some(parent) = p.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    match std::fs::write(&p, content) {
                        Ok(_) => Ok(ok_bool()),
                        Err(e) => bool_err(lua, &e.to_string()),
                    }
                }
                None => bool_err(lua, "Unsafe path traversal"),
            }
        })?,
    )?;

    let r3 = root.clone();
    fs_table.set(
        "exists",
        lua.create_function(
            move |_lua, path: String| match resolve_safe_path(&r3, &path) {
                Some(p) => Ok(p.exists()),
                None => Ok(false),
            },
        )?,
    )?;

    let r4 = root.clone();
    fs_table.set(
        "is_safe_path",
        lua.create_function(move |_lua, path: String| Ok(resolve_safe_path(&r4, &path).is_some()))?,
    )?;

    let r5 = root.clone();
    fs_table.set(
        "stat",
        lua.create_function(move |lua, path: String| {
            require_capability_for_lua(lua, "fs.read")?;

            let resolved = resolve_safe_path(&r5, &path)
                .ok_or_else(|| mlua::Error::runtime("Unsafe path traversal".to_string()))?;
            let bytes =
                std::fs::read(&resolved).map_err(|err| mlua::Error::runtime(err.to_string()))?;
            let metadata = std::fs::metadata(&resolved)
                .map_err(|err| mlua::Error::runtime(err.to_string()))?;
            let hash = hash_sha256_hex(&bytes);
            let tracking_path = normalize_tracking_path(&r5, &resolved);
            let tracking_key = format!("{FS_STAT_HASH_KEY_PREFIX}{tracking_path}");
            let previous_hash =
                load_previous_session_hash(lua, &tracking_key).map_err(mlua::Error::runtime)?;
            store_current_session_hash(lua, &tracking_key, &hash).map_err(mlua::Error::runtime)?;

            let stat = lua.create_table()?;
            stat.set("path", tracking_path)?;
            stat.set("bytes", bytes.len() as i64)?;
            stat.set("hash", hash.clone())?;
            stat.set("previous_hash", previous_hash.clone())?;
            stat.set("seen_before", previous_hash.is_some())?;
            stat.set(
                "changed",
                previous_hash
                    .as_ref()
                    .map(|previous| previous != &hash)
                    .unwrap_or(true),
            )?;

            if let Ok(modified) = metadata.modified()
                && let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH)
            {
                stat.set("modified_at", duration.as_secs() as i64)?;
            } else {
                stat.set("modified_at", Value::Nil)?;
            }

            Ok(Value::Table(stat))
        })?,
    )?;

    lua.globals().set("fs", fs_table)?;
    Ok(())
}
