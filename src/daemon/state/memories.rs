use anyhow::{Result, bail};
use turin_daemon_protocol::{MemoryDetail, MemoryList, MemoryListParams, MemoryScopeDetail};

use super::DaemonState;

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
                params.include_superseded,
                limit,
                offset,
            )
            .await?;

        Ok(MemoryList {
            memories: page
                .rows
                .into_iter()
                .map(|row| MemoryDetail {
                    public_id: super::helpers::format_uuid_bytes_simple(&row.public_id),
                    scope_kind: row.scope_kind,
                    scope_key: row.scope_key,
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
                    created_at: row.created_at,
                })
                .collect(),
            scopes: page
                .scopes
                .into_iter()
                .map(|scope| MemoryScopeDetail {
                    scope_kind: scope.scope_kind,
                    scope_key: scope.scope_key,
                    count: scope.count,
                })
                .collect(),
            total: page.total,
            offset,
            limit,
        })
    }
}
