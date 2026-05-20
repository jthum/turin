use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};
use serde::Deserialize;
use std::sync::Arc;

use crate::code_index_reader::{
    CodeSearchMode, CodeSearchRequest, CodeSearchRow, CodeSearchTrace, CodebaseSelector,
    search as code_search, status as code_status,
};
use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::binding_common::{bridge_async_result, nil_err, ok_value};
use crate::harness::stdlib::governance_support::require_capability as require_governance_capability;
use crate::inference::embeddings::EmbeddingProvider;
use turin_code_index::metadata::CodeIndexSemanticStatus;

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

    register_search_mode(
        lua,
        &runtime_search,
        app_data,
        "lexical",
        CodeSearchMode::Lexical,
    )?;
    register_search_mode(
        lua,
        &runtime_search,
        app_data,
        "semantic",
        CodeSearchMode::Semantic,
    )?;
    register_search_mode(
        lua,
        &runtime_search,
        app_data,
        "hybrid",
        CodeSearchMode::Hybrid,
    )?;

    runtime_code.set("search", runtime_search)?;
    runtime_table.set("code", runtime_code)?;
    Ok(())
}

fn register_search_mode(
    lua: &Lua,
    runtime_search: &Table,
    app_data: &HarnessAppData,
    name: &'static str,
    mode: CodeSearchMode,
) -> LuaResult<()> {
    let workspace_root = app_data.workspace_root.clone();
    let embedding = app_data.embedding_provider.clone();
    let app_data_snapshot = app_data.clone();
    let capability = search_capability(mode);
    runtime_search.set(
        name,
        lua.create_function(
            move |lua, (codebase, query, opts): (Value, String, Option<Table>)| {
                if let Err(err) = require_governance_capability(&app_data_snapshot, capability) {
                    return nil_err(lua, &err);
                }
                run_code_search(
                    lua,
                    &workspace_root,
                    codebase,
                    query,
                    opts,
                    embedding.clone(),
                    mode,
                    capability,
                )
            },
        )?,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_code_search(
    lua: &Lua,
    workspace_root: &std::path::Path,
    codebase: Value,
    query: String,
    opts: Option<Table>,
    embedding: Option<Arc<dyn EmbeddingProvider>>,
    mode: CodeSearchMode,
    label: &str,
) -> LuaResult<(Value, Value)> {
    let selector = codebase_selector_from_value(codebase)?;
    let request = code_search_request_from_opts(lua, opts)?;
    let workspace_root = workspace_root.to_path_buf();
    let result = bridge_async_result(async move {
        let embedding = resolve_query_embedding(
            &workspace_root,
            selector.clone(),
            &request,
            embedding,
            mode,
            &query,
        )
        .await?;

        let mut rows = code_search(
            &workspace_root,
            selector,
            embedding.effective_mode,
            &query,
            &request,
            embedding.query_vector.as_deref(),
        )
        .await
        .map_err(|e| e.to_string())?;
        if request.trace {
            annotate_runtime_trace(&mut rows, mode, &embedding);
        }
        Ok(rows)
    });
    match result {
        Ok(rows) => Ok(ok_value(lua.to_value(&rows)?)),
        Err(err) => nil_err(lua, &format!("{label}: {err}")),
    }
}

struct RuntimeCodeEmbedding {
    effective_mode: CodeSearchMode,
    query_vector: Option<Vec<f32>>,
    fallback_reason: Option<&'static str>,
}

async fn resolve_query_embedding(
    workspace_root: &std::path::Path,
    selector: CodebaseSelector,
    request: &CodeSearchRequest,
    embedding: Option<Arc<dyn EmbeddingProvider>>,
    mode: CodeSearchMode,
    query: &str,
) -> Result<RuntimeCodeEmbedding, String> {
    match mode {
        CodeSearchMode::Lexical => Ok(RuntimeCodeEmbedding {
            effective_mode: CodeSearchMode::Lexical,
            query_vector: None,
            fallback_reason: None,
        }),
        CodeSearchMode::Semantic | CodeSearchMode::Hybrid => {
            let Some(provider) = embedding else {
                if request.strict {
                    return Err(format!(
                        "{} search requires an embedding provider",
                        mode.as_str()
                    ));
                }
                return Ok(RuntimeCodeEmbedding {
                    effective_mode: CodeSearchMode::Lexical,
                    query_vector: None,
                    fallback_reason: Some("missing_embedding_provider"),
                });
            };

            let status = code_status(workspace_root, selector)
                .await
                .map_err(|e| e.to_string())?;
            let provider_key = provider.config_key();
            let provider_dimensions = provider.dimensions();
            if embedding_profile_mismatch(&status.semantic, &provider_key, provider_dimensions) {
                if request.strict {
                    return Err(format!(
                        "embedding configuration does not match code index (index_key={:?}, index_dimensions={:?}, provider_key={}, provider_dimensions={})",
                        status.semantic.embedding_key,
                        status.semantic.embedding_dimensions,
                        provider_key,
                        provider_dimensions
                    ));
                }
                return Ok(RuntimeCodeEmbedding {
                    effective_mode: CodeSearchMode::Lexical,
                    query_vector: None,
                    fallback_reason: Some("embedding_profile_mismatch"),
                });
            }

            Ok(RuntimeCodeEmbedding {
                effective_mode: mode,
                query_vector: Some(
                    provider
                        .embed(query)
                        .await
                        .map_err(|e| e.to_string())?
                        .vector,
                ),
                fallback_reason: None,
            })
        }
    }
}

fn embedding_profile_mismatch(
    status: &CodeIndexSemanticStatus,
    provider_key: &str,
    provider_dimensions: usize,
) -> bool {
    status
        .embedding_key
        .as_deref()
        .is_some_and(|key| key != provider_key)
        || status
            .embedding_dimensions
            .is_some_and(|dimensions| dimensions != provider_dimensions)
}

fn annotate_runtime_trace(
    rows: &mut [CodeSearchRow],
    requested_mode: CodeSearchMode,
    embedding: &RuntimeCodeEmbedding,
) {
    for row in rows {
        let trace = row.trace.get_or_insert_with(|| CodeSearchTrace {
            requested_mode: None,
            effective_mode: embedding.effective_mode.as_str().to_string(),
            fallback_reason: None,
            lexical_rank: None,
            semantic_rank: None,
            lexical_rrf: None,
            semantic_rrf: None,
            fusion: None,
        });
        trace.requested_mode = Some(requested_mode.as_str().to_string());
        trace.effective_mode = embedding.effective_mode.as_str().to_string();
        if let Some(reason) = embedding.fallback_reason {
            trace.fallback_reason = Some(reason.to_string());
        }
    }
}

fn search_capability(mode: CodeSearchMode) -> &'static str {
    match mode {
        CodeSearchMode::Lexical => "runtime.code.search.lexical",
        CodeSearchMode::Semantic => "runtime.code.search.semantic",
        CodeSearchMode::Hybrid => "runtime.code.search.hybrid",
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
        trace: parsed.trace.unwrap_or(false),
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
