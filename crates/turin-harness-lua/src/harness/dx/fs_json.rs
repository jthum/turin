use mlua::{
    AnyUserData, Function, Lua, LuaSerdeExt, ObjectLike, Result as LuaResult, Table, Value,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::harness::dx::common::call_and_raise_on_err;

const FS_SUMMARY_HASH_KEY_PREFIX: &str = "_turin:fs_summary_hash:";
const FS_SUMMARY_VALUE_KEY_PREFIX: &str = "_turin:fs_summary_value:";

#[derive(Debug, Default, Deserialize)]
struct FsSummaryOpts {
    prompt: Option<String>,
    force: Option<bool>,
}

fn sha256_hex(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn current_hook_context(lua: &Lua) -> LuaResult<AnyUserData> {
    match lua.globals().get::<Value>("__current_hook_context")? {
        Value::UserData(ud) => Ok(ud),
        _ => Err(mlua::Error::runtime(
            "[fs.summary] only available during on_turn_prepare(ctx)".to_string(),
        )),
    }
}

fn build_summary_messages(lua: &Lua, path: &str, content: &str, prompt: &str) -> LuaResult<Value> {
    let message = lua.create_table()?;
    message.set("role", "user")?;

    let part = lua.create_table()?;
    part.set("type", "text")?;
    part.set(
        "text",
        format!("File: {path}\n\nInstruction: {prompt}\n\n{content}"),
    )?;

    let parts = lua.create_table()?;
    parts.set(1, part)?;
    message.set("content", parts)?;

    let messages = lua.create_table()?;
    messages.set(1, message)?;
    Ok(Value::Table(messages))
}

pub fn register_fs_json_globals(lua: &Lua) -> LuaResult<()> {
    let globals = lua.globals();
    let fs: Table = globals.get("fs")?;
    let session: Table = globals.get("session")?;
    let json: Table = globals.get("json")?;

    let fs_read: Function = fs.get("read")?;
    let fs_stat: Function = fs.get("stat")?;
    let fs_write: Function = fs.get("write")?;
    let session_get: Function = session.get("get")?;
    let session_set: Function = session.get("set")?;
    let json_decode: Function = json.get("decode")?;
    let json_encode: Function = json.get("encode")?;

    {
        let fs_read = fs_read.clone();
        let json_decode = json_decode.clone();
        fs.set(
            "read_json",
            lua.create_function(move |lua, (path, _opts): (String, Option<Table>)| {
                let raw = call_and_raise_on_err(lua, &fs_read, path, "fs.read")?;
                let text = match raw {
                    Value::String(s) => s.to_str()?.to_string(),
                    other => {
                        return Err(mlua::Error::runtime(format!(
                            "[fs.read_json] expected string from fs.read, got {:?}",
                            other
                        )));
                    }
                };
                call_and_raise_on_err(lua, &json_decode, text, "json.decode")
            })?,
        )?;
    }

    {
        let fs_write = fs_write.clone();
        let json_encode = json_encode.clone();
        fs.set(
            "write_json",
            lua.create_function(
                move |lua, (path, value, opts): (String, Value, Option<Table>)| {
                    let pretty = opts
                        .as_ref()
                        .and_then(|t| t.get::<bool>("pretty").ok())
                        .unwrap_or(false);

                    let encoded = if pretty {
                        let as_json: serde_json::Value = lua.from_value(value)?;
                        serde_json::to_string_pretty(&as_json).map_err(|e| {
                            mlua::Error::runtime(format!(
                                "[fs.write_json] failed to encode JSON: {}",
                                e
                            ))
                        })?
                    } else {
                        let encoded_val =
                            call_and_raise_on_err(lua, &json_encode, value, "json.encode")?;
                        match encoded_val {
                            Value::String(s) => s.to_str()?.to_string(),
                            other => {
                                return Err(mlua::Error::runtime(format!(
                                    "[fs.write_json] expected string from json.encode, got {:?}",
                                    other
                                )));
                            }
                        }
                    };

                    call_and_raise_on_err(lua, &fs_write, (path, encoded), "fs.write")
                },
            )?,
        )?;
    }

    {
        let fs_read = fs_read.clone();
        let fs_stat = fs_stat.clone();
        let session_get = session_get.clone();
        let session_set = session_set.clone();
        fs.set(
            "summary",
            lua.create_function(move |lua, (path, opts): (String, Option<Table>)| {
                let parsed = match opts {
                    Some(table) => lua
                        .from_value::<FsSummaryOpts>(Value::Table(table))
                        .map_err(|e| {
                            mlua::Error::runtime(format!("invalid fs.summary opts: {}", e))
                        })?,
                    None => FsSummaryOpts::default(),
                };

                let stat_value = fs_stat.call::<Value>(path.clone())?;
                let stat = match stat_value {
                    Value::Table(table) => table,
                    other => {
                        return Err(mlua::Error::runtime(format!(
                            "[fs.summary] expected table from fs.stat, got {:?}",
                            other
                        )));
                    }
                };

                let normalized_path = stat.get::<String>("path")?;
                let current_hash = stat.get::<String>("hash")?;
                let prompt = parsed.prompt.unwrap_or_else(|| {
                    "Summarize this file concisely for future prompt context.".to_string()
                });
                let prompt_hash = sha256_hex(&prompt);
                let key_suffix = format!("{prompt_hash}:{normalized_path}");
                let hash_key = format!("{FS_SUMMARY_HASH_KEY_PREFIX}{key_suffix}");
                let value_key = format!("{FS_SUMMARY_VALUE_KEY_PREFIX}{key_suffix}");

                if !parsed.force.unwrap_or(false) {
                    let cached_hash =
                        call_and_raise_on_err(lua, &session_get, hash_key.clone(), "session.get")?;
                    let cached_value =
                        call_and_raise_on_err(lua, &session_get, value_key.clone(), "session.get")?;

                    if let (Value::String(ch), Value::String(cv)) = (cached_hash, cached_value)
                        && ch.to_str()? == current_hash
                    {
                        return Ok(Value::String(cv));
                    }
                }

                let content_value = call_and_raise_on_err(lua, &fs_read, path.clone(), "fs.read")?;
                let content = match content_value {
                    Value::String(text) => text.to_str()?.to_string(),
                    other => {
                        return Err(mlua::Error::runtime(format!(
                            "[fs.summary] expected string from fs.read, got {:?}",
                            other
                        )));
                    }
                };

                let context = current_hook_context(lua)?;
                let messages = build_summary_messages(lua, &normalized_path, &content, &prompt)?;
                let summary_value = context.call_method::<Value>("summarize", messages)?;
                let summary = match summary_value {
                    Value::String(text) => text.to_str()?.to_string(),
                    Value::Nil => {
                        return Err(mlua::Error::runtime(
                            "[fs.summary] context summarizer returned nil".to_string(),
                        ));
                    }
                    other => {
                        return Err(mlua::Error::runtime(format!(
                            "[fs.summary] expected string from ctx:summarize(...), got {:?}",
                            other
                        )));
                    }
                };

                let _ = call_and_raise_on_err(
                    lua,
                    &session_set,
                    (hash_key, current_hash),
                    "session.set",
                )?;
                let _ = call_and_raise_on_err(
                    lua,
                    &session_set,
                    (value_key, summary.clone()),
                    "session.set",
                )?;

                Ok(Value::String(lua.create_string(&summary)?))
            })?,
        )?;
    }

    Ok(())
}
