use mlua::{Lua, Result as LuaResult, Table};

use crate::harness::globals::HarnessAppData;

pub fn register_runtime_inference_namespace(
    lua: &Lua,
    runtime_table: &Table,
    app_data: &HarnessAppData,
) -> LuaResult<()> {
    let runtime_inference = lua.create_table()?;

    {
        let app_data = app_data.clone();
        runtime_inference.set(
            "available",
            lua.create_function(move |_, context_name: String| {
                let requested = context_name.trim();
                if requested.is_empty() {
                    return Ok(false);
                }

                let agent_id = app_data
                    .execution_ctx
                    .lock()
                    .ok()
                    .and_then(|ctx| ctx.agent_id.clone())
                    .unwrap_or_else(|| app_data.config.agent.id.clone());

                let effective = app_data
                    .config
                    .effective_inference_config_for_agent(&agent_id, None)
                    .map_err(mlua::Error::runtime)?;

                Ok(effective.contexts.contains_key(requested))
            })?,
        )?;
    }

    runtime_table.set("inference", runtime_inference)?;
    Ok(())
}
