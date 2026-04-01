use std::sync::Arc;

use crate::inference::embeddings::EmbeddingProvider;
use crate::kernel::identity::ContextSelector;
use crate::persistence::manager::{StoreManager, StorePathScope};
use crate::persistence::schema::{
    MemoryCorrectionRow, MemoryFeedbackState, MemoryPurgeReport, MemoryRow, StoredMemoryRow,
};

use super::{
    MemoryFeedbackRequest, MemoryFeedbackSignal, MemoryPurgeRequest, MemorySearchMode,
    MemorySearchRequest, MemorySearchSource, MemoryStoreMode, MemoryStoreRequest,
    augment_memory_metadata, open_state_store, selector_scope_ref,
};

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) async fn memory_store_backend(
    manager: &StoreManager,
    embedding_provider: Option<&Arc<dyn EmbeddingProvider>>,
    selector: &ContextSelector,
    content: &str,
    metadata: &serde_json::Value,
    path_scope: StorePathScope,
) -> anyhow::Result<StoredMemoryRow> {
    let request = MemoryStoreRequest::default();
    memory_store_backend_with_request(
        manager,
        embedding_provider,
        selector,
        content,
        metadata,
        &request,
        path_scope,
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
    path_scope: StorePathScope,
) -> anyhow::Result<StoredMemoryRow> {
    let store = open_state_store(manager, request.store_selector.as_ref(), path_scope).await?;
    let scope = selector_scope_ref(selector)?;
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
            &scope.scope_kind,
            &scope.scope_key,
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
    path_scope: StorePathScope,
) -> anyhow::Result<Vec<MemoryRow>> {
    let request = MemorySearchRequest {
        limit,
        ..MemorySearchRequest::default()
    };
    memory_search_backend_with_request(
        manager,
        embedding_provider,
        selector,
        query,
        &request,
        path_scope,
    )
    .await
}

pub(crate) async fn memory_search_backend_with_request(
    manager: &StoreManager,
    embedding_provider: Option<&Arc<dyn EmbeddingProvider>>,
    selector: &ContextSelector,
    query: &str,
    request: &MemorySearchRequest,
    path_scope: StorePathScope,
) -> anyhow::Result<Vec<MemoryRow>> {
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

    let default_scope = selector_scope_ref(selector)?;
    let sources = if request.sources.is_empty() {
        vec![MemorySearchSource {
            scope_kind: default_scope.scope_kind,
            scope_key: default_scope.scope_key,
            raw_scope_key: default_scope.raw_scope_key.unwrap_or_default(),
            namespace: default_scope.namespace,
            store_selector: request.store_selector.clone(),
        }]
    } else {
        request.sources.clone()
    };

    let mut combined = Vec::new();
    for source in &sources {
        let store = open_state_store(manager, source.store_selector.as_ref(), path_scope).await?;
        let mut rows = store
            .search_memories(
                &source.scope_kind,
                &source.scope_key,
                vector.as_deref(),
                query_embedding_key.as_deref(),
                query_embedding_dimensions,
                lexical_query,
                request.limit,
                request.min_score,
                request.include_metadata,
                request.include_superseded,
            )
            .await?;
        combined.append(&mut rows);
    }

    combined.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.created_at.cmp(&left.created_at))
    });
    combined.truncate(request.limit);
    Ok(combined)
}

pub(crate) async fn memory_feedback_backend_with_request(
    manager: &StoreManager,
    selector: &ContextSelector,
    memory_id: &str,
    signal: MemoryFeedbackSignal,
    request: &MemoryFeedbackRequest,
    path_scope: StorePathScope,
) -> anyhow::Result<MemoryFeedbackState> {
    let store = open_state_store(manager, request.store_selector.as_ref(), path_scope).await?;
    let scope = selector_scope_ref(selector)?;
    let public_id = uuid::Uuid::parse_str(memory_id)
        .map_err(|e| anyhow::anyhow!("runtime.memory.feedback: invalid memory id: {}", e))?;
    let delta = match signal {
        MemoryFeedbackSignal::Up => request.step,
        MemoryFeedbackSignal::Down => -request.step,
        MemoryFeedbackSignal::Delta(delta) => delta,
    };
    store
        .apply_memory_feedback(
            &scope.scope_kind,
            &scope.scope_key,
            public_id,
            delta,
            request.clamp_min,
            request.clamp_max,
            request.reason.as_deref(),
            request.task_id.as_deref(),
        )
        .await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn memory_correct_backend_with_request(
    manager: &StoreManager,
    embedding_provider: Option<&Arc<dyn EmbeddingProvider>>,
    selector: &ContextSelector,
    memory_id: &str,
    content: &str,
    metadata: &serde_json::Value,
    request: &MemoryStoreRequest,
    path_scope: StorePathScope,
) -> anyhow::Result<MemoryCorrectionRow> {
    let store = open_state_store(manager, request.store_selector.as_ref(), path_scope).await?;
    let scope = selector_scope_ref(selector)?;
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
            &scope.scope_kind,
            &scope.scope_key,
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
    path_scope: StorePathScope,
) -> anyhow::Result<MemoryPurgeReport> {
    let store = open_state_store(manager, request.store_selector.as_ref(), path_scope).await?;
    let scope = selector_scope_ref(selector)?;
    store
        .purge_memories(
            &scope.scope_kind,
            &scope.scope_key,
            request.older_than_days,
            request.min_weight,
            request.max_retrieval_count,
            request.only_superseded,
            request.all,
            request.dry_run,
        )
        .await
}
