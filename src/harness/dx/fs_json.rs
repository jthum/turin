use mlua::{Function, Lua, LuaSerdeExt, Result as LuaResult, Table, Value};

use crate::harness::dx::common::call_and_raise_on_err;

pub fn register_fs_json_globals(lua: &Lua) -> LuaResult<()> {
    let globals = lua.globals();
    let fs: Table = globals.get("fs")?;
    let json: Table = globals.get("json")?;

    let fs_read: Function = fs.get("read")?;
    let fs_write: Function = fs.get("write")?;
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

    Ok(())
}
