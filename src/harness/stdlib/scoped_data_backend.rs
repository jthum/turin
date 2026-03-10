use std::sync::Arc;

use crate::inference::embeddings::EmbeddingProvider;
use crate::kernel::identity::ContextSelector;
use crate::persistence::manager::{StoreManager, StoreSelector};
use crate::persistence::schema::{
    MemoryCorrectionRow, MemoryFeedbackState, MemoryPurgeReport, MemoryRow, StoredMemoryRow,
};

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

#[derive(Debug, Clone)]
pub(crate) enum MemoryFeedbackSignal {
    Up,
    Down,
    Delta(f64),
}

#[derive(Debug, Clone)]
pub(crate) struct MemoryFeedbackRequest {
    pub reason: Option<String>,
    pub task_id: Option<String>,
    pub step: f64,
    pub clamp_min: f64,
    pub clamp_max: f64,
}

impl Default for MemoryFeedbackRequest {
    fn default() -> Self {
        Self {
            reason: None,
            task_id: None,
            step: 0.1,
            clamp_min: 0.1,
            clamp_max: 5.0,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MemoryPurgeRequest {
    pub older_than_days: Option<u64>,
    pub min_weight: Option<f64>,
    pub max_retrieval_count: Option<u64>,
    pub only_superseded: bool,
    pub all: bool,
    pub dry_run: bool,
}

impl Default for MemoryPurgeRequest {
    fn default() -> Self {
        Self {
            older_than_days: None,
            min_weight: None,
            max_retrieval_count: None,
            only_superseded: false,
            all: false,
            dry_run: true,
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

async fn resolve_context_memory_session(
    store: &crate::persistence::state::StateStore,
    selector: &ContextSelector,
) -> anyhow::Result<Option<i64>> {
    const KEY: &str = "__turin_context_session_public_id";

    let public_id = if let Some(existing) = store.kv_get(KEY).await? {
        Some(
            uuid::Uuid::parse_str(&existing)
                .map_err(|e| anyhow::anyhow!("Invalid stored context session UUID: {}", e))?,
        )
    } else {
        None
    };

    let Some(public_id) = public_id else {
        return Ok(None);
    };

    if let Some(id) = store.get_session_by_public_id(public_id).await? {
        return Ok(Some(id));
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
        .map(Some)
}

async fn ensure_context_memory_session(
    store: &crate::persistence::state::StateStore,
    selector: &ContextSelector,
) -> anyhow::Result<i64> {
    if let Some(id) = resolve_context_memory_session(store, selector).await? {
        return Ok(id);
    }

    let new_id = uuid::Uuid::now_v7();
    store
        .kv_set(
            "__turin_context_session_public_id",
            &new_id.simple().to_string(),
        )
        .await?;
    let agent_id = selector
        .tags
        .iter()
        .find_map(|t| t.strip_prefix("agent:").map(ToOwned::to_owned))
        .unwrap_or_else(|| "context".to_string());
    let metadata = serde_json::to_string(selector).ok();
    store
        .create_session(new_id, &agent_id, metadata.as_deref())
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
    let embedding_key = vector
        .as_ref()
        .and_then(|_| embedding_provider.map(|provider| provider.config_key()));
    let embedding_dimensions = vector
        .as_ref()
        .and_then(|_| embedding_provider.map(|provider| provider.dimensions()));
    let metadata = augment_memory_metadata(metadata, request);
    store
        .insert_memory(
            session_id,
            content,
            vector.as_deref(),
            embedding_key.as_deref(),
            embedding_dimensions,
            &metadata,
        )
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
                anyhow::bail!(
                    "runtime.memory.search: semantic mode requires an embedding provider"
                );
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
                anyhow::anyhow!(
                    "runtime.memory.search: semantic mode requires an embedding provider"
                )
            })?;
            Some(provider.embed(query).await?.vector)
        }
        MemorySearchMode::Auto | MemorySearchMode::Lexical => None,
    };
    let query_embedding_key = vector
        .as_ref()
        .and_then(|_| embedding_provider.map(|provider| provider.config_key()));
    let query_embedding_dimensions = vector
        .as_ref()
        .and_then(|_| embedding_provider.map(|provider| provider.dimensions()));
    let lexical_query = match effective_mode {
        MemorySearchMode::Lexical | MemorySearchMode::Hybrid | MemorySearchMode::Auto => {
            Some(query)
        }
        MemorySearchMode::Semantic => None,
    };

    store
        .search_memories(
            session_id,
            vector.as_deref(),
            query_embedding_key.as_deref(),
            query_embedding_dimensions,
            lexical_query,
            request.limit,
            request.min_score,
            request.include_metadata,
            request.include_superseded,
        )
        .await
}

pub(crate) async fn memory_feedback_backend_with_request(
    manager: &StoreManager,
    selector: &ContextSelector,
    memory_id: &str,
    signal: MemoryFeedbackSignal,
    request: &MemoryFeedbackRequest,
) -> anyhow::Result<MemoryFeedbackState> {
    let store = open_selector_store(manager, selector).await?;
    let session_id = resolve_context_memory_session(&store, selector)
        .await?
        .ok_or_else(|| anyhow::anyhow!("runtime.memory.feedback: no memory session exists"))?;
    let public_id = uuid::Uuid::parse_str(memory_id)
        .map_err(|e| anyhow::anyhow!("runtime.memory.feedback: invalid memory id: {}", e))?;
    let delta = match signal {
        MemoryFeedbackSignal::Up => request.step,
        MemoryFeedbackSignal::Down => -request.step,
        MemoryFeedbackSignal::Delta(delta) => delta,
    };
    store
        .apply_memory_feedback(
            session_id,
            public_id,
            delta,
            request.clamp_min,
            request.clamp_max,
            request.reason.as_deref(),
            request.task_id.as_deref(),
        )
        .await
}

pub(crate) async fn memory_correct_backend_with_request(
    manager: &StoreManager,
    embedding_provider: Option<&Arc<dyn EmbeddingProvider>>,
    selector: &ContextSelector,
    memory_id: &str,
    content: &str,
    metadata: &serde_json::Value,
    request: &MemoryStoreRequest,
) -> anyhow::Result<MemoryCorrectionRow> {
    let store = open_selector_store(manager, selector).await?;
    let session_id = resolve_context_memory_session(&store, selector)
        .await?
        .ok_or_else(|| anyhow::anyhow!("runtime.memory.correct: no memory session exists"))?;
    let public_id = uuid::Uuid::parse_str(memory_id)
        .map_err(|e| anyhow::anyhow!("runtime.memory.correct: invalid memory id: {}", e))?;
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
                    "runtime.memory.correct: storage='embedded' requires an embedding provider"
                )
            })?;
            Some(provider.embed(content).await?.vector)
        }
    };
    let embedding_key = vector
        .as_ref()
        .and_then(|_| embedding_provider.map(|provider| provider.config_key()));
    let embedding_dimensions = vector
        .as_ref()
        .and_then(|_| embedding_provider.map(|provider| provider.dimensions()));
    let metadata = augment_memory_metadata(metadata, request);
    store
        .correct_memory(
            session_id,
            public_id,
            content,
            vector.as_deref(),
            embedding_key.as_deref(),
            embedding_dimensions,
            &metadata,
        )
        .await
}

pub(crate) async fn memory_purge_backend_with_request(
    manager: &StoreManager,
    selector: &ContextSelector,
    request: &MemoryPurgeRequest,
) -> anyhow::Result<MemoryPurgeReport> {
    let store = open_selector_store(manager, selector).await?;
    let Some(session_id) = resolve_context_memory_session(&store, selector).await? else {
        return Ok(MemoryPurgeReport {
            matched: 0,
            deleted: 0,
            dry_run: request.dry_run,
        });
    };
    store
        .purge_memories(
            session_id,
            request.older_than_days,
            request.min_weight,
            request.max_retrieval_count,
            request.only_superseded,
            request.all,
            request.dry_run,
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
    use uuid::Uuid;

    use super::{
        MemoryFeedbackRequest, MemoryFeedbackSignal, MemoryPurgeRequest, MemorySearchMode,
        MemorySearchRequest, MemoryStoreMode, MemoryStoreRequest,
        memory_correct_backend_with_request, memory_feedback_backend_with_request,
        memory_purge_backend_with_request, memory_search_backend,
        memory_search_backend_with_request, memory_store_backend,
        memory_store_backend_with_request,
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

        assert!(
            err.to_string()
                .contains("storage='embedded' requires an embedding provider")
        );
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
        assert!(
            err.to_string()
                .contains("semantic mode requires an embedding provider")
        );
    }

    #[tokio::test]
    async fn memory_lifecycle_feedback_correct_and_purge_work() {
        let tmp = tempdir().expect("tempdir");
        let manager = StoreManager::new(tmp.path());
        let selector = test_selector();

        let stored = memory_store_backend(
            &manager,
            None,
            &selector,
            "stale alpha memory",
            &json!({ "kind": "note", "source": "initial" }),
        )
        .await
        .expect("initial memory store should succeed");
        let memory_id = Uuid::from_slice(&stored.public_id)
            .expect("uuid bytes")
            .simple()
            .to_string();

        let feedback = memory_feedback_backend_with_request(
            &manager,
            &selector,
            &memory_id,
            MemoryFeedbackSignal::Up,
            &MemoryFeedbackRequest {
                step: 0.25,
                ..MemoryFeedbackRequest::default()
            },
        )
        .await
        .expect("feedback should succeed");
        assert!(feedback.weight > 1.0);

        let correction = memory_correct_backend_with_request(
            &manager,
            None,
            &selector,
            &memory_id,
            "fresh beta memory",
            &json!({ "kind": "note", "source": "corrected" }),
            &MemoryStoreRequest {
                storage: MemoryStoreMode::LexicalOnly,
                ..MemoryStoreRequest::default()
            },
        )
        .await
        .expect("correction should succeed");

        let visible = memory_search_backend_with_request(
            &manager,
            None,
            &selector,
            "fresh",
            &MemorySearchRequest {
                include_metadata: true,
                ..MemorySearchRequest::default()
            },
        )
        .await
        .expect("corrected memory should be searchable");
        assert_eq!(visible.len(), 1);
        assert_eq!(
            Uuid::from_slice(&visible[0].public_id)
                .expect("uuid bytes")
                .simple()
                .to_string(),
            Uuid::from_slice(&correction.replacement_public_id)
                .expect("uuid bytes")
                .simple()
                .to_string()
        );

        let hidden_old = memory_search_backend_with_request(
            &manager,
            None,
            &selector,
            "stale",
            &MemorySearchRequest::default(),
        )
        .await
        .expect("superseded search should succeed");
        assert!(
            hidden_old.is_empty(),
            "superseded memory should be hidden by default"
        );

        let old_visible = memory_search_backend_with_request(
            &manager,
            None,
            &selector,
            "stale",
            &MemorySearchRequest {
                include_superseded: true,
                ..MemorySearchRequest::default()
            },
        )
        .await
        .expect("superseded-inclusive search should succeed");
        assert_eq!(old_visible.len(), 1);

        let dry_run = memory_purge_backend_with_request(
            &manager,
            &selector,
            &MemoryPurgeRequest {
                only_superseded: true,
                ..MemoryPurgeRequest::default()
            },
        )
        .await
        .expect("purge dry-run should succeed");
        assert_eq!(dry_run.matched, 1);
        assert_eq!(dry_run.deleted, 0);
        assert!(dry_run.dry_run);

        let purge = memory_purge_backend_with_request(
            &manager,
            &selector,
            &MemoryPurgeRequest {
                only_superseded: true,
                dry_run: false,
                ..MemoryPurgeRequest::default()
            },
        )
        .await
        .expect("purge should succeed");
        assert_eq!(purge.deleted, 1);

        let after_purge = memory_search_backend_with_request(
            &manager,
            None,
            &selector,
            "stale",
            &MemorySearchRequest {
                include_superseded: true,
                ..MemorySearchRequest::default()
            },
        )
        .await
        .expect("post-purge search should succeed");
        assert!(after_purge.is_empty(), "purged memory should be gone");
    }
}
