use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;

use crate::harness::stdlib::scoped_data_backend::{
    MemorySearchMode, MemorySearchRequest, MemoryStoreMode, MemoryStoreRequest,
    memory_search_backend_with_request, memory_store_backend_with_request, selector_scope_ref,
};
use crate::kernel::identity::ContextSelector;
use crate::persistence::manager::StoreSelector;
use crate::tools::{Tool, ToolContext, ToolEffect, ToolError, ToolOutput, parse_args};

pub struct RememberTool;
pub struct RecallTool;

#[derive(Deserialize)]
struct RememberArgs {
    content: String,
    #[serde(default)]
    metadata: Option<Value>,
    #[serde(default)]
    storage: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    source_task: Option<String>,
}

#[derive(Deserialize)]
struct RecallArgs {
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    include_metadata: bool,
    #[serde(default)]
    include_superseded: bool,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    min_score: Option<f64>,
    #[serde(default)]
    strict: bool,
}

fn default_limit() -> usize {
    5
}

fn default_agent_selector(agent_id: &str) -> ContextSelector {
    ContextSelector {
        tags: vec![format!("agent:{agent_id}")],
        namespace: "default".to_string(),
        visibility: "private".to_string(),
    }
}

fn parse_memory_store_mode(value: Option<&str>) -> Result<MemoryStoreMode, ToolError> {
    match value.unwrap_or("auto") {
        "auto" => Ok(MemoryStoreMode::Auto),
        "lexical" | "lexical_only" => Ok(MemoryStoreMode::LexicalOnly),
        "embedded" => Ok(MemoryStoreMode::Embedded),
        other => Err(ToolError::InvalidParams(format!(
            "Invalid storage mode '{}'; expected auto, lexical_only, or embedded",
            other
        ))),
    }
}

fn parse_memory_search_mode(value: Option<&str>) -> Result<MemorySearchMode, ToolError> {
    match value.unwrap_or("auto") {
        "auto" => Ok(MemorySearchMode::Auto),
        "lexical" => Ok(MemorySearchMode::Lexical),
        "semantic" => Ok(MemorySearchMode::Semantic),
        "hybrid" => Ok(MemorySearchMode::Hybrid),
        other => Err(ToolError::InvalidParams(format!(
            "Invalid search mode '{}'; expected auto, lexical, semantic, or hybrid",
            other
        ))),
    }
}

fn memory_public_id_string(bytes: &[u8]) -> Result<String, ToolError> {
    uuid::Uuid::from_slice(bytes)
        .map(|id| id.simple().to_string())
        .map_err(|e| ToolError::ExecutionError(format!("Invalid stored memory id: {e}")))
}

fn scoped_store_selector(
    ctx: &ToolContext,
    selector: &ContextSelector,
) -> Result<Option<StoreSelector>, ToolError> {
    let Some(config) = ctx.config.as_ref() else {
        return Ok(None);
    };
    let scope = selector_scope_ref(selector)
        .map_err(|e| ToolError::ExecutionError(format!("Invalid memory selector: {e}")))?;
    Ok(config
        .persistence
        .resolve_store_alias_for_scope(
            &scope.scope_kind,
            scope.raw_scope_key.as_deref(),
            &scope.namespace,
        )
        .map(|alias| StoreSelector::Alias(alias.to_string())))
}

#[async_trait]
impl Tool for RememberTool {
    fn name(&self) -> &str {
        "remember"
    }

    fn description(&self) -> &str {
        "Store durable agent memory that can be recalled later. Use this for user preferences, important facts, and session-spanning notes."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Memory content to store"
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional structured metadata to store with the memory"
                },
                "storage": {
                    "type": "string",
                    "description": "Storage mode: auto, lexical_only, or embedded",
                    "enum": ["auto", "lexical_only", "embedded"]
                },
                "tags": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional tags for downstream filtering or auditing"
                },
                "source_task": {
                    "type": "string",
                    "description": "Optional originating task identifier"
                }
            },
            "required": ["content"]
        })
    }

    async fn execute(&self, params: Value, ctx: &ToolContext) -> Result<ToolEffect, ToolError> {
        let args: RememberArgs = parse_args(params)?;
        let store_manager = ctx
            .store_manager
            .as_deref()
            .ok_or_else(|| ToolError::ExecutionError("Memory store manager unavailable".into()))?;
        let selector = default_agent_selector(&ctx.agent_id);
        let request = MemoryStoreRequest {
            source_task: args.source_task,
            tags: args.tags,
            storage: parse_memory_store_mode(args.storage.as_deref())?,
            store_selector: scoped_store_selector(ctx, &selector)?,
        };
        let metadata = args.metadata.unwrap_or_else(|| serde_json::json!({}));
        let stored = memory_store_backend_with_request(
            store_manager,
            ctx.embedding_provider.as_ref(),
            &selector,
            &args.content,
            &metadata,
            &request,
            crate::persistence::manager::StorePathScope::WorkspaceOnly,
        )
        .await
        .map_err(|e| ToolError::ExecutionError(e.to_string()))?;
        let memory_id = memory_public_id_string(&stored.public_id)?;
        let storage = stored.storage.as_str();

        Ok(ToolEffect::Output(ToolOutput {
            content: format!(
                "Remembered entry {} using {} storage at {}.",
                memory_id, storage, stored.stored_at
            ),
            metadata: serde_json::json!({
                "memory_id": memory_id,
                "stored_at": stored.stored_at,
                "storage": storage,
                "scope": "agent",
                "agent_id": ctx.agent_id,
            }),
        }))
    }
}

#[async_trait]
impl Tool for RecallTool {
    fn name(&self) -> &str {
        "recall"
    }

    fn description(&self) -> &str {
        "Search durable agent memory for relevant facts, preferences, and notes stored with remember."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for matching memories"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of memories to return",
                    "default": 5
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Include stored metadata in the returned memory hits"
                },
                "include_superseded": {
                    "type": "boolean",
                    "description": "Include memories that were superseded by later corrections"
                },
                "mode": {
                    "type": "string",
                    "description": "Search mode: auto, lexical, semantic, or hybrid",
                    "enum": ["auto", "lexical", "semantic", "hybrid"]
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum fused score for returned memory hits"
                },
                "strict": {
                    "type": "boolean",
                    "description": "If true, semantic/hybrid mode fails when embeddings are unavailable"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, params: Value, ctx: &ToolContext) -> Result<ToolEffect, ToolError> {
        let args: RecallArgs = parse_args(params)?;
        let store_manager = ctx
            .store_manager
            .as_deref()
            .ok_or_else(|| ToolError::ExecutionError("Memory store manager unavailable".into()))?;
        let selector = default_agent_selector(&ctx.agent_id);
        let request = MemorySearchRequest {
            limit: args.limit.clamp(1, 20),
            mode: parse_memory_search_mode(args.mode.as_deref())?,
            min_score: args.min_score.unwrap_or(0.0),
            include_metadata: args.include_metadata,
            include_superseded: args.include_superseded,
            strict: args.strict,
            store_selector: scoped_store_selector(ctx, &selector)?,
            sources: Vec::new(),
        };
        let rows = memory_search_backend_with_request(
            store_manager,
            ctx.embedding_provider.as_ref(),
            &selector,
            &args.query,
            &request,
            crate::persistence::manager::StorePathScope::WorkspaceOnly,
        )
        .await
        .map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        let hits = rows
            .iter()
            .map(|row| {
                let id = memory_public_id_string(&row.public_id)?;
                Ok(serde_json::json!({
                    "memory_id": id,
                    "content": row.content,
                    "score": row.score,
                    "lexical_score": row.lexical_score,
                    "semantic_score": row.semantic_score,
                    "created_at": row.created_at,
                    "metadata": row.metadata,
                }))
            })
            .collect::<Result<Vec<_>, ToolError>>()?;

        let mut content = if hits.is_empty() {
            format!("No memories matched '{}'.", args.query)
        } else {
            format!("Found {} memory hit(s) for '{}':", hits.len(), args.query)
        };
        for (index, row) in rows.iter().enumerate() {
            let id = memory_public_id_string(&row.public_id)?;
            content.push_str(&format!("\n{}. [{}] {}", index + 1, id, row.content));
            if args.include_metadata
                && let Some(metadata) = &row.metadata
            {
                content.push_str(&format!("\n   metadata: {}", metadata));
            }
        }

        Ok(ToolEffect::Output(ToolOutput {
            content,
            metadata: serde_json::json!({
                "query": args.query,
                "hits": hits,
                "scope": "agent",
                "agent_id": ctx.agent_id,
            }),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::manager::StoreManager;
    use tempfile::TempDir;

    fn tool_ctx(root: &std::path::Path) -> ToolContext {
        ToolContext {
            workspace_root: root.to_path_buf(),
            session_id: uuid::Uuid::now_v7().simple().to_string(),
            agent_id: "default".to_string(),
            store_manager: Some(std::sync::Arc::new(StoreManager::new(
                root,
                turin_types::layout::default_stores_dir_for_workspace(root),
            ))),
            embedding_provider: None,
            config: None,
            allowed_native_tools: std::sync::Arc::new(crate::tools::policy::full_native_tool_set()),
            tools: std::sync::Arc::new(turin_types::ToolsConfig::default()),
        }
    }

    #[tokio::test]
    async fn remember_and_recall_round_trip() {
        let dir = TempDir::new().unwrap();
        let ctx = tool_ctx(dir.path());

        let remember = RememberTool;
        let recall = RecallTool;

        remember
            .execute(
                serde_json::json!({
                    "content": "User prefers concise answers",
                    "metadata": { "kind": "preference" },
                    "storage": "lexical_only"
                }),
                &ctx,
            )
            .await
            .unwrap();

        let result = recall
            .execute(
                serde_json::json!({
                    "query": "concise answers",
                    "limit": 3,
                    "include_metadata": true
                }),
                &ctx,
            )
            .await
            .unwrap();

        let ToolEffect::Output(output) = result else {
            panic!("expected output effect");
        };
        assert!(output.content.contains("User prefers concise answers"));
        assert_eq!(output.metadata["hits"].as_array().unwrap().len(), 1);
    }
}
