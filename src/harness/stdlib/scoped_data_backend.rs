use std::sync::Arc;

use crate::inference::embeddings::EmbeddingProvider;
use crate::kernel::identity::ContextSelector;
use crate::persistence::manager::{StoreManager, StoreSelector};

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

pub(crate) async fn memory_store_backend(
    manager: &StoreManager,
    embedding_provider: Option<&Arc<dyn EmbeddingProvider>>,
    selector: &ContextSelector,
    content: &str,
    metadata: &serde_json::Value,
) -> anyhow::Result<()> {
    let store = open_selector_store(manager, selector).await?;
    let session_id = ensure_context_memory_session(&store, selector).await?;

    let vector = if let Some(provider) = embedding_provider {
        Some(provider.embed(content).await?.vector)
    } else {
        None
    };
    store
        .insert_memory(session_id, content, vector.as_deref(), metadata)
        .await
}

pub(crate) async fn memory_search_backend(
    manager: &StoreManager,
    embedding_provider: Option<&Arc<dyn EmbeddingProvider>>,
    selector: &ContextSelector,
    query: &str,
    limit: usize,
) -> anyhow::Result<Vec<crate::persistence::schema::MemoryRow>> {
    let store = open_selector_store(manager, selector).await?;
    let session_id = ensure_context_memory_session(&store, selector).await?;

    let vector = if let Some(provider) = embedding_provider {
        provider.embed(query).await.ok().map(|emb| emb.vector)
    } else {
        None
    };

    store
        .search_memories(session_id, vector.as_deref(), Some(query), limit)
        .await
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use tempfile::tempdir;

    use super::{memory_search_backend, memory_store_backend};
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
}
