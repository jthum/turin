use std::sync::Arc;

use crate::kernel::identity::ContextSelector;
use crate::persistence::manager::{StoreManager, StorePathScope, StoreSelector};

mod kv;
mod memory;
#[cfg(test)]
mod tests;

pub(crate) use kv::{kv_delete_backend, kv_get_backend, kv_set_backend};
pub(crate) use memory::{
    memory_correct_backend_with_request, memory_feedback_backend_with_request,
    memory_purge_backend_with_request, memory_search_backend_with_request,
    memory_store_backend_with_request,
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
    pub store_selector: Option<StoreSelector>,
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
    pub store_selector: Option<StoreSelector>,
    pub sources: Vec<MemorySearchSource>,
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
            store_selector: None,
            sources: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MemorySearchSource {
    pub scope_kind: String,
    pub scope_key: String,
    pub raw_scope_key: String,
    pub namespace: String,
    pub store_selector: Option<StoreSelector>,
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
    pub store_selector: Option<StoreSelector>,
}

impl Default for MemoryFeedbackRequest {
    fn default() -> Self {
        Self {
            reason: None,
            task_id: None,
            step: 0.1,
            clamp_min: 0.1,
            clamp_max: 5.0,
            store_selector: None,
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
    pub store_selector: Option<StoreSelector>,
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
            store_selector: None,
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

#[derive(Debug, Clone)]
pub(crate) struct ScopedStateRef {
    pub scope_kind: String,
    pub scope_key: String,
    pub raw_scope_key: Option<String>,
    pub namespace: String,
}

pub(crate) fn encode_scope_key(raw_key: &str, namespace: &str) -> String {
    if namespace == "default" {
        raw_key.to_string()
    } else {
        serde_json::json!({
            "namespace": namespace,
            "key": raw_key,
        })
        .to_string()
    }
}

pub(crate) fn selector_scope_ref(selector: &ContextSelector) -> anyhow::Result<ScopedStateRef> {
    visibility_allowed(selector)?;
    if selector.tags.len() == 1
        && let Some((kind, key)) = selector.tags[0].split_once(':')
    {
        return Ok(ScopedStateRef {
            scope_kind: kind.to_string(),
            scope_key: encode_scope_key(key, &selector.namespace),
            raw_scope_key: Some(key.to_string()),
            namespace: selector.namespace.clone(),
        });
    }

    let mut tags = selector.tags.clone();
    tags.sort();
    Ok(ScopedStateRef {
        scope_kind: "selector".to_string(),
        scope_key: serde_json::json!({
            "tags": tags,
            "namespace": selector.namespace,
            "visibility": selector.visibility,
        })
        .to_string(),
        raw_scope_key: None,
        namespace: selector.namespace.clone(),
    })
}

pub(crate) async fn open_state_store(
    manager: &StoreManager,
    store_selector: Option<&StoreSelector>,
    path_scope: StorePathScope,
) -> anyhow::Result<Arc<crate::persistence::state::StateStore>> {
    match store_selector {
        Some(selector) => manager
            .open_with_path_scope(selector, path_scope)
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string())),
        None => manager
            .get_default()
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string())),
    }
}

pub(crate) fn augment_memory_metadata(
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
