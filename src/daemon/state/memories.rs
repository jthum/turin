use std::collections::HashMap;

use anyhow::{Result, bail};
use turin_daemon_protocol::{
    MemoryCorrectParams, MemoryDeleteResult, MemoryDetail, MemoryList, MemoryListParams,
    MemoryScopeKindDetail, MemoryTargetParams,
};

use super::DaemonState;
use crate::kernel::session_refs::parse_session_reference;

const DEFAULT_MEMORY_LIST_LIMIT: u32 = 100;
const MAX_MEMORY_LIST_LIMIT: u32 = 250;

impl DaemonState {
    pub async fn list_memories(&self, params: &MemoryListParams) -> Result<MemoryList> {
        let limit = params.limit.unwrap_or(DEFAULT_MEMORY_LIST_LIMIT);
        if limit == 0 || limit > MAX_MEMORY_LIST_LIMIT {
            bail!("memory list limit must be between 1 and {MAX_MEMORY_LIST_LIMIT}");
        }
        let offset = params.offset.unwrap_or_default();
        let selector = super::helpers::context_store_selector_from_params(
            &self.bootstrap_config,
            params.persistence.as_ref(),
        )?;
        let store = self.kernel.store_manager().open(&selector).await?;
        let page = store
            .inspect_memories(
                params.scope_kind.as_deref(),
                params.scope_key.as_deref(),
                params.query.as_deref(),
                params.include_superseded,
                limit,
                offset,
            )
            .await?;

        let mut session_ids = page
            .rows
            .iter()
            .filter(|row| row.scope_kind == "session")
            .filter_map(|row| session_scope_public_id(&row.scope_key))
            .collect::<Vec<_>>();
        session_ids.sort_unstable();
        session_ids.dedup();
        let session_titles = store
            .get_session_rows_by_public_ids(&session_ids)
            .await?
            .into_iter()
            .filter_map(|row| {
                let public_id = uuid::Uuid::from_slice(&row.public_id).ok()?;
                let title = super::helpers::session_title_from_metadata(row.metadata.as_deref())?;
                Some((public_id, title))
            })
            .collect::<HashMap<_, _>>();

        Ok(MemoryList {
            memories: page
                .rows
                .into_iter()
                .map(|row| {
                    let display_name = (row.scope_kind == "session")
                        .then(|| session_scope_public_id(&row.scope_key))
                        .flatten()
                        .and_then(|public_id| session_titles.get(&public_id).cloned());
                    memory_detail(row, display_name)
                })
                .collect(),
            scope_kinds: page
                .scope_kinds
                .into_iter()
                .map(|scope| MemoryScopeKindDetail {
                    scope_kind: scope.scope_kind,
                    count: scope.count,
                })
                .collect(),
            total: page.total,
            offset,
            limit,
        })
    }

    pub async fn memory_detail(&self, params: &MemoryTargetParams) -> Result<Option<MemoryDetail>> {
        let store = self.memory_store(params.persistence.as_ref()).await?;
        let public_id = uuid::Uuid::parse_str(&params.id)
            .map_err(|error| anyhow::anyhow!("invalid memory id: {error}"))?;
        let Some(row) = store.inspect_memory(public_id).await? else {
            return Ok(None);
        };
        let display_name = memory_scope_display_name(&store, &row).await?;
        Ok(Some(memory_detail(row, display_name)))
    }

    pub async fn correct_memory(&self, params: &MemoryCorrectParams) -> Result<MemoryDetail> {
        let content = params.content.trim();
        if content.is_empty() {
            bail!("memory content must not be empty");
        }
        let store = self.memory_store(params.persistence.as_ref()).await?;
        let public_id = uuid::Uuid::parse_str(&params.id)
            .map_err(|error| anyhow::anyhow!("invalid memory id: {error}"))?;
        let current = store
            .inspect_memory(public_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("memory '{}' not found", params.id))?;
        let metadata = current
            .metadata
            .as_deref()
            .map(serde_json::from_str)
            .transpose()?
            .unwrap_or_else(|| serde_json::json!({}));
        let (embedding, embedding_key, embedding_dimensions) = if current.embedded {
            let provider = self.kernel.embedding_provider().ok_or_else(|| {
                anyhow::anyhow!("embedded memory correction requires an embedding provider")
            })?;
            (
                Some(provider.embed(content).await?),
                Some(provider.config_key()),
                Some(provider.dimensions()),
            )
        } else {
            (None, None, None)
        };
        let correction = store
            .correct_memory(
                &current.scope_kind,
                &current.scope_key,
                public_id,
                content,
                embedding
                    .as_ref()
                    .map(|embedding| embedding.vector.as_slice()),
                embedding_key.as_deref(),
                embedding_dimensions,
                &metadata,
            )
            .await?;
        let replacement_id = uuid::Uuid::from_slice(&correction.replacement_public_id)?;
        let row = store
            .inspect_memory(replacement_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("corrected memory was not visible"))?;
        let display_name = memory_scope_display_name(&store, &row).await?;
        Ok(memory_detail(row, display_name))
    }

    pub async fn delete_memory(&self, params: &MemoryTargetParams) -> Result<MemoryDeleteResult> {
        let store = self.memory_store(params.persistence.as_ref()).await?;
        let public_id = uuid::Uuid::parse_str(&params.id)
            .map_err(|error| anyhow::anyhow!("invalid memory id: {error}"))?;
        Ok(MemoryDeleteResult {
            id: params.id.clone(),
            deleted: store.delete_memory(public_id).await?,
        })
    }

    async fn memory_store(
        &self,
        persistence: Option<&turin_daemon_protocol::ContextPersistenceParams>,
    ) -> Result<std::sync::Arc<crate::persistence::state::StateStore>> {
        let selector = super::helpers::context_store_selector_from_params(
            &self.bootstrap_config,
            persistence,
        )?;
        self.kernel.store_manager().open(&selector).await
    }
}

fn memory_detail(
    row: crate::persistence::schema::MemoryInspectionRow,
    scope_display_name: Option<String>,
) -> MemoryDetail {
    MemoryDetail {
        public_id: super::helpers::format_uuid_bytes_simple(&row.public_id),
        scope_kind: row.scope_kind,
        scope_key: row.scope_key,
        scope_display_name,
        content: row.content,
        metadata: row
            .metadata
            .as_deref()
            .map(super::helpers::parse_json_or_string),
        storage: if row.embedded {
            "embedded".to_string()
        } else {
            "lexical_only".to_string()
        },
        embedding_key: row.embedding_key,
        embedding_dimensions: row.embedding_dimensions,
        weight: row.weight,
        retrieval_count: row.retrieval_count,
        last_retrieved_at: row.last_retrieved_at,
        superseded_at: row.superseded_at,
        superseded_by_id: row
            .superseded_by_public_id
            .as_deref()
            .map(super::helpers::format_uuid_bytes_simple),
        created_at: row.created_at,
    }
}

fn session_scope_public_id(scope_key: &str) -> Option<uuid::Uuid> {
    let raw_key = serde_json::from_str::<serde_json::Value>(scope_key)
        .ok()
        .and_then(|value| {
            value
                .get("key")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| scope_key.to_string());
    let session_ref = parse_session_reference(&raw_key).ok()?;
    uuid::Uuid::parse_str(&session_ref.public_id).ok()
}

async fn memory_scope_display_name(
    store: &crate::persistence::state::StateStore,
    row: &crate::persistence::schema::MemoryInspectionRow,
) -> Result<Option<String>> {
    let Some(public_id) = (row.scope_kind == "session")
        .then(|| session_scope_public_id(&row.scope_key))
        .flatten()
    else {
        return Ok(None);
    };
    Ok(store
        .get_session_row_by_public_id(public_id)
        .await?
        .and_then(|session| {
            super::helpers::session_title_from_metadata(session.metadata.as_deref())
        }))
}
