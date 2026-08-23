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

struct MemoryEmbedding {
    vector: Option<Vec<f32>>,
    config_key: Option<String>,
    dimensions: Option<usize>,
}

impl MemoryEmbedding {
    fn lexical_only() -> Self {
        Self {
            vector: None,
            config_key: None,
            dimensions: None,
        }
    }

    async fn from_provider(
        provider: &Arc<dyn EmbeddingProvider>,
        content: &str,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            vector: Some(provider.embed(content).await?.vector),
            config_key: Some(provider.config_key()),
            dimensions: Some(provider.dimensions()),
        })
    }

    fn vector(&self) -> Option<&[f32]> {
        self.vector.as_deref()
    }

    fn config_key(&self) -> Option<&str> {
        self.config_key.as_deref()
    }
}

async fn resolve_store_embedding(
    operation: &str,
    embedding_provider: Option<&Arc<dyn EmbeddingProvider>>,
    content: &str,
    mode: MemoryStoreMode,
) -> anyhow::Result<MemoryEmbedding> {
    match mode {
        MemoryStoreMode::Auto => match embedding_provider {
            Some(provider) => MemoryEmbedding::from_provider(provider, content).await,
            None => Ok(MemoryEmbedding::lexical_only()),
        },
        MemoryStoreMode::LexicalOnly => Ok(MemoryEmbedding::lexical_only()),
        MemoryStoreMode::Embedded => {
            let provider = embedding_provider.ok_or_else(|| {
                anyhow::anyhow!(
                    "runtime.memory.{}: storage='embedded' requires an embedding provider",
                    operation
                )
            })?;
            MemoryEmbedding::from_provider(provider, content).await
        }
    }
}

fn resolve_search_mode(
    request: &MemorySearchRequest,
    has_embedding_provider: bool,
) -> anyhow::Result<MemorySearchMode> {
    match request.mode {
        MemorySearchMode::Auto if has_embedding_provider => Ok(MemorySearchMode::Hybrid),
        MemorySearchMode::Auto => Ok(MemorySearchMode::Lexical),
        MemorySearchMode::Lexical => Ok(MemorySearchMode::Lexical),
        MemorySearchMode::Semantic if has_embedding_provider => Ok(MemorySearchMode::Semantic),
        MemorySearchMode::Semantic if request.strict => {
            anyhow::bail!("runtime.memory.search: semantic mode requires an embedding provider")
        }
        MemorySearchMode::Semantic => Ok(MemorySearchMode::Lexical),
        MemorySearchMode::Hybrid if has_embedding_provider => Ok(MemorySearchMode::Hybrid),
        MemorySearchMode::Hybrid if request.strict => {
            anyhow::bail!("runtime.memory.search: hybrid mode requires an embedding provider")
        }
        MemorySearchMode::Hybrid => Ok(MemorySearchMode::Lexical),
    }
}

async fn resolve_search_embedding(
    embedding_provider: Option<&Arc<dyn EmbeddingProvider>>,
    query: &str,
    mode: MemorySearchMode,
) -> anyhow::Result<MemoryEmbedding> {
    match mode {
        MemorySearchMode::Semantic | MemorySearchMode::Hybrid => {
            let provider = embedding_provider.ok_or_else(|| {
                anyhow::anyhow!(
                    "runtime.memory.search: semantic mode requires an embedding provider"
                )
            })?;
            MemoryEmbedding::from_provider(provider, query).await
        }
        MemorySearchMode::Auto | MemorySearchMode::Lexical => Ok(MemoryEmbedding::lexical_only()),
    }
}

fn lexical_query_for(mode: MemorySearchMode, query: &str) -> Option<&str> {
    match mode {
        MemorySearchMode::Lexical | MemorySearchMode::Hybrid | MemorySearchMode::Auto => {
            Some(query)
        }
        MemorySearchMode::Semantic => None,
    }
}

fn search_sources(
    selector: &ContextSelector,
    request: &MemorySearchRequest,
) -> anyhow::Result<Vec<MemorySearchSource>> {
    if !request.sources.is_empty() {
        return Ok(request.sources.clone());
    }

    let scope = selector_scope_ref(selector)?;
    Ok(vec![MemorySearchSource {
        scope_kind: scope.scope_kind,
        scope_key: scope.scope_key,
        raw_scope_key: scope.raw_scope_key.unwrap_or_default(),
        namespace: scope.namespace,
        store_selector: request.store_selector.clone(),
    }])
}

fn parse_public_memory_id(memory_id: &str, operation: &str) -> anyhow::Result<uuid::Uuid> {
    uuid::Uuid::parse_str(memory_id)
        .map_err(|e| anyhow::anyhow!("runtime.memory.{}: invalid memory id: {}", operation, e))
}

fn feedback_delta(signal: MemoryFeedbackSignal, request: &MemoryFeedbackRequest) -> f64 {
    match signal {
        MemoryFeedbackSignal::Up => request.step,
        MemoryFeedbackSignal::Down => -request.step,
        MemoryFeedbackSignal::Delta(delta) => delta,
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub async fn memory_store_backend(
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

pub async fn memory_store_backend_with_request(
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
    let embedding =
        resolve_store_embedding("store", embedding_provider, content, request.storage).await?;
    let metadata = augment_memory_metadata(metadata, request);
    store
        .insert_memory(
            &scope.scope_kind,
            &scope.scope_key,
            content,
            embedding.vector(),
            embedding.config_key(),
            embedding.dimensions,
            &metadata,
        )
        .await
}

#[cfg_attr(not(test), allow(dead_code))]
pub async fn memory_search_backend(
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

pub async fn memory_search_backend_with_request(
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

    let effective_mode = resolve_search_mode(request, embedding_provider.is_some())?;
    let embedding = resolve_search_embedding(embedding_provider, query, effective_mode).await?;
    let lexical_query = lexical_query_for(effective_mode, query);
    let sources = search_sources(selector, request)?;

    let mut combined = Vec::new();
    for source in &sources {
        let store = open_state_store(manager, source.store_selector.as_ref(), path_scope).await?;
        let mut rows = store
            .search_memories(
                &source.scope_kind,
                &source.scope_key,
                embedding.vector(),
                embedding.config_key(),
                embedding.dimensions,
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

pub async fn memory_feedback_backend_with_request(
    manager: &StoreManager,
    selector: &ContextSelector,
    memory_id: &str,
    signal: MemoryFeedbackSignal,
    request: &MemoryFeedbackRequest,
    path_scope: StorePathScope,
) -> anyhow::Result<MemoryFeedbackState> {
    let store = open_state_store(manager, request.store_selector.as_ref(), path_scope).await?;
    let scope = selector_scope_ref(selector)?;
    let public_id = parse_public_memory_id(memory_id, "feedback")?;
    let delta = feedback_delta(signal, request);
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
pub async fn memory_correct_backend_with_request(
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
    let public_id = parse_public_memory_id(memory_id, "correct")?;
    let embedding =
        resolve_store_embedding("correct", embedding_provider, content, request.storage).await?;
    let metadata = augment_memory_metadata(metadata, request);
    store
        .correct_memory(
            &scope.scope_kind,
            &scope.scope_key,
            public_id,
            content,
            embedding.vector(),
            embedding.config_key(),
            embedding.dimensions,
            &metadata,
        )
        .await
}

pub async fn memory_purge_backend_with_request(
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
