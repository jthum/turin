use std::sync::Arc;

use crate::inference::embeddings::EmbeddingProvider;
use crate::kernel::identity::ContextSelector;
use crate::persistence::manager::{StoreManager, StoreSelector};
use crate::persistence::schema::{MemoryRow, StoredMemoryRow};

#[derive(Debug, Clone, Copy, Default)]
pub(crate) enum MemoryStoreMode {
    #[default]
    Auto,
    LexicalOnly,
    Embedded,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct MemoryStoreRequest {
    pub source_task: Option<String>,
    pub tags: Vec<String>,
    pub storage: MemoryStoreMode,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) enum MemorySearchMode {
    #[default]
    Auto,
    Lexical,
    Semantic,
    Hybrid,
}

#[derive(Debug, Clone)]
pub(crate) struct MemorySearchRequest {
    pub limit: usize,
    pub mode: MemorySearchMode,
    pub min_score: f64,
    pub include_metadata: bool,
    pub include_superseded: bool,
    pub strict: bool,
}

impl Default for MemorySearchRequest {
    fn default() -> Self {
        Self {
            limit: 5,
            mode: MemorySearchMode::Auto,
            min_score: 0.0,
            include_metadata: false,
            include_superseded: false,
            strict: false,
        }
    }
}

fn visibility_allowed(selector: &ContextSelector) -> anyhow::Result<()> {
    match selector.visibility.as_str() {
        "private" => Ok(()),
        "children" | "agent_group" | "all_agents" => {
            anyhow::bail!(
                "Policy denial: visibility '{}' not enabled",
                selector.visibility
            )
        }
        other => anyhow::bail!("Invalid visibility: {}", other),
    }
}

async fn open_selector_store(
    manager: &StoreManager,
    selector: &ContextSelector,
) -> anyhow::Result<Arc<crate::persistence::state::StateStore>> {
    visibility_allowed(selector)?;
    manager
        .open(&StoreSelector::Alias(selector.to_alias()))
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))
}

async fn ensure_context_memory_session(
    store: &crate::persistence::state::StateStore,
    selector: &ContextSelector,
) -> anyhow::Result<i64> {
    const KEY: &str = "__turin_context_session_public_id";

    let public_id = if let Some(existing) = store.kv_get(KEY).await? {
        uuid::Uuid::parse_str(&existing)
            .map_err(|e| anyhow::anyhow!("Invalid stored context session UUID: {}", e))?
    } else {
        let new_id = uuid::Uuid::now_v7();
        store.kv_set(KEY, &new_id.simple().to_string()).await?;
        new_id
    };

    if let Some(id) = store.get_session_by_public_id(public_id).await? {
        return Ok(id);
    }

    let agent_id = selector
        .tags
        .iter()
        .find_map(|t| t.strip_prefix("agent:").map(ToOwned::to_owned))
        .unwrap_or_else(|| "context".to_string());
    let metadata = serde_json::to_string(selector).ok();
    store
        .create_session(public_id, &agent_id, metadata.as_deref())
        .await
}

pub(crate) async fn kv_get_backend(
    manager: &StoreManager,
    selector: &ContextSelector,
    key: &str,
) -> anyhow::Result<Option<String>> {
    let store = open_selector_store(manager, selector).await?;
    store.kv_get(key).await
}

pub(crate) async fn kv_set_backend(
    manager: &StoreManager,
    selector: &ContextSelector,
    key: &str,
    value: &str,
) -> anyhow::Result<()> {
    let store = open_selector_store(manager, selector).await?;
    store.kv_set(key, value).await
}

pub(crate) async fn kv_delete_backend(
    manager: &StoreManager,
    selector: &ContextSelector,
    key: &str,
) -> anyhow::Result<()> {
    let store = open_selector_store(manager, selector).await?;
    store.kv_delete(key).await
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) async fn memory_store_backend(
    manager: &StoreManager,
    embedding_provider: Option<&Arc<dyn EmbeddingProvider>>,
    selector: &ContextSelector,
    content: &str,
    metadata: &serde_json::Value,
) -> anyhow::Result<StoredMemoryRow> {
    let request = MemoryStoreRequest::default();
    memory_store_backend_with_request(
        manager,
        embedding_provider,
        selector,
        content,
        metadata,
        &request,
    )
    .await
}

pub(crate) async fn memory_store_backend_with_request(
    manager: &StoreManager,
    embedding_provider: Option<&Arc<dyn EmbeddingProvider>>,
    selector: &ContextSelector,
    content: &str,
    metadata: &serde_json::Value,
    request: &MemoryStoreRequest,
) -> anyhow::Result<StoredMemoryRow> {
    let store = open_selector_store(manager, selector).await?;
    let session_id = ensure_context_memory_session(&store, selector).await?;
    let vector = match request.storage {
        MemoryStoreMode::Auto => {
            if let Some(provider) = embedding_provider {
                Some(provider.embed(content).await?.vector)
            } else {
                None
            }
        }
        MemoryStoreMode::LexicalOnly => None,
        MemoryStoreMode::Embedded => {
            let provider = embedding_provider.ok_or_else(|| {
                anyhow::anyhow!(
                    "runtime.memory.store: storage='embedded' requires an embedding provider"
                )
            })?;
            Some(provider.embed(content).await?.vector)
        }
    };
    let metadata = augment_memory_metadata(metadata, request);
    store
        .insert_memory(session_id, content, vector.as_deref(), &metadata)
        .await
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) async fn memory_search_backend(
    manager: &StoreManager,
    embedding_provider: Option<&Arc<dyn EmbeddingProvider>>,
    selector: &ContextSelector,
    query: &str,
    limit: usize,
) -> anyhow::Result<Vec<MemoryRow>> {
    let request = MemorySearchRequest {
        limit,
        ..MemorySearchRequest::default()
    };
    memory_search_backend_with_request(manager, embedding_provider, selector, query, &request).await
}

pub(crate) async fn memory_search_backend_with_request(
    manager: &StoreManager,
    embedding_provider: Option<&Arc<dyn EmbeddingProvider>>,
    selector: &ContextSelector,
    query: &str,
    request: &MemorySearchRequest,
) -> anyhow::Result<Vec<MemoryRow>> {
    let store = open_selector_store(manager, selector).await?;
    let session_id = ensure_context_memory_session(&store, selector).await?;
    let query = query.trim();
    if query.is_empty() {
        return Ok(Vec::new());
    }

    let effective_mode = match request.mode {
        MemorySearchMode::Auto => {
            if embedding_provider.is_some() {
                MemorySearchMode::Hybrid
            } else {
                MemorySearchMode::Lexical
            }
        }
        MemorySearchMode::Lexical => MemorySearchMode::Lexical,
        MemorySearchMode::Semantic => {
            if embedding_provider.is_some() {
                MemorySearchMode::Semantic
            } else if request.strict {
                anyhow::bail!("runtime.memory.search: semantic mode requires an embedding provider");
            } else {
                MemorySearchMode::Lexical
            }
        }
        MemorySearchMode::Hybrid => {
            if embedding_provider.is_some() {
                MemorySearchMode::Hybrid
            } else if request.strict {
                anyhow::bail!("runtime.memory.search: hybrid mode requires an embedding provider");
            } else {
                MemorySearchMode::Lexical
            }
        }
    };

    let vector = match effective_mode {
        MemorySearchMode::Semantic | MemorySearchMode::Hybrid => {
            let provider = embedding_provider.ok_or_else(|| {
                anyhow::anyhow!("runtime.memory.search: semantic mode requires an embedding provider")
            })?;
            Some(provider.embed(query).await?.vector)
        }
        MemorySearchMode::Auto | MemorySearchMode::Lexical => None,
    };
    let lexical_query = match effective_mode {
        MemorySearchMode::Lexical | MemorySearchMode::Hybrid | MemorySearchMode::Auto => Some(query),
        MemorySearchMode::Semantic => None,
    };

    store
        .search_memories(
            session_id,
            vector.as_deref(),
            lexical_query,
            request.limit,
            request.min_score,
            request.include_metadata,
            request.include_superseded,
        )
        .await
}

fn augment_memory_metadata(
    metadata: &serde_json::Value,
    request: &MemoryStoreRequest,
) -> serde_json::Value {
    if request.source_task.is_none() && request.tags.is_empty() {
        return metadata.clone();
    }

    let mut object = match metadata {
        serde_json::Value::Object(map) => map.clone(),
        serde_json::Value::Null => serde_json::Map::new(),
        other => {
            let mut wrapped = serde_json::Map::new();
            wrapped.insert("value".to_string(), other.clone());
            wrapped
        }
    };

    let mut turin = serde_json::Map::new();
    if let Some(source_task) = &request.source_task {
        turin.insert(
            "source_task".to_string(),
            serde_json::Value::String(source_task.clone()),
        );
    }
    if !request.tags.is_empty() {
        turin.insert(
            "tags".to_string(),
            serde_json::Value::Array(
                request
                    .tags
                    .iter()
                    .cloned()
                    .map(serde_json::Value::String)
                    .collect(),
            ),
        );
    }
    object.insert("_turin".to_string(), serde_json::Value::Object(turin));
    serde_json::Value::Object(object)
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use tempfile::tempdir;

    use super::{
        memory_search_backend, memory_search_backend_with_request, memory_store_backend,
        memory_store_backend_with_request, MemorySearchMode, MemorySearchRequest,
        MemoryStoreMode, MemoryStoreRequest,
    };
    use crate::kernel::identity::ContextSelector;
    use crate::persistence::manager::StoreManager;

    fn test_selector() -> ContextSelector {
        ContextSelector {
            tags: vec!["agent:test".to_string()],
            namespace: "default".to_string(),
            visibility: "private".to_string(),
        }
    }

    #[tokio::test]
    async fn memory_backend_works_without_embedding_provider() {
        let tmp = tempdir().expect("tempdir");
        let manager = StoreManager::new(tmp.path());
        let selector = test_selector();

        memory_store_backend(
            &manager,
            None,
            &selector,
            "alpha beta lexical memory",
            &json!({ "kind": "note" }),
        )
        .await
        .expect("lexical-only memory store should succeed");

        let rows = memory_search_backend(&manager, None, &selector, "alpha", 5)
            .await
            .expect("lexical-only memory search should succeed");

        assert_eq!(rows.len(), 1);
        assert!(rows[0].content.contains("alpha beta lexical memory"));
    }

    #[tokio::test]
    async fn memory_store_embedded_mode_requires_embedding_provider() {
        let tmp = tempdir().expect("tempdir");
        let manager = StoreManager::new(tmp.path());
        let selector = test_selector();

        let err = memory_store_backend_with_request(
            &manager,
            None,
            &selector,
            "alpha beta lexical memory",
            &json!({ "kind": "note" }),
            &MemoryStoreRequest {
                storage: MemoryStoreMode::Embedded,
                ..MemoryStoreRequest::default()
            },
        )
        .await
        .expect_err("embedded-only store should fail without provider");

        assert!(err
            .to_string()
            .contains("storage='embedded' requires an embedding provider"));
    }

    #[tokio::test]
    async fn memory_search_semantic_mode_falls_back_or_errors_without_embeddings() {
        let tmp = tempdir().expect("tempdir");
        let manager = StoreManager::new(tmp.path());
        let selector = test_selector();

        memory_store_backend(
            &manager,
            None,
            &selector,
            "alpha beta lexical memory",
            &json!({ "kind": "note" }),
        )
        .await
        .expect("lexical-only memory store should succeed");

        let relaxed = memory_search_backend_with_request(
            &manager,
            None,
            &selector,
            "alpha",
            &MemorySearchRequest {
                mode: MemorySearchMode::Semantic,
                ..MemorySearchRequest::default()
            },
        )
        .await
        .expect("semantic search should fall back to lexical when strict is false");
        assert_eq!(relaxed.len(), 1);

        let err = memory_search_backend_with_request(
            &manager,
            None,
            &selector,
            "alpha",
            &MemorySearchRequest {
                mode: MemorySearchMode::Semantic,
                strict: true,
                ..MemorySearchRequest::default()
            },
        )
        .await
        .expect_err("strict semantic search should fail without embeddings");
        assert!(err
            .to_string()
            .contains("semantic mode requires an embedding provider"));
    }
}
