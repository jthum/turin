use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use serde_json::{Map as JsonMap, Value as JsonValue};
use turin_daemon_protocol::{
    ContextPersistenceParams, ScheduleActionParams, StoreTargetParams, WorkItemDetail,
    WorkItemList, WorklistDetail,
};

use super::DaemonState;
use crate::kernel::config::ContextPersistenceConfig;
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::{WorkItemRow, WorklistRow};
use crate::persistence::state::StateStore;

pub(crate) struct WorklistItemsQuery<'a> {
    pub public_id: &'a str,
    pub persistence: Option<&'a ContextPersistenceParams>,
    pub status: Option<&'a str>,
    pub parent_public_id: Option<&'a str>,
    pub where_filter: Option<&'a JsonMap<String, JsonValue>>,
    pub claimed_only: bool,
    pub limit: Option<u32>,
}

impl DaemonState {
    pub async fn list_worklists(
        &self,
        persistence: Option<&ContextPersistenceParams>,
        name: Option<&str>,
        scope: Option<&str>,
    ) -> Result<Vec<WorklistDetail>> {
        let store = self.resolve_worklist_store(persistence).await?;
        let rows = store.list_worklists().await?;
        let resolved = persistence.cloned();
        Ok(rows
            .into_iter()
            .filter(|row| name.is_none_or(|value| row.name == value))
            .filter(|row| scope.is_none_or(|value| row.scope_ref == value))
            .map(|row| map_worklist_detail(row, resolved.clone()))
            .collect())
    }

    pub async fn worklist_detail(
        &self,
        public_id: &str,
        persistence: Option<&ContextPersistenceParams>,
    ) -> Result<Option<WorklistDetail>> {
        let store = self.resolve_worklist_store(persistence).await?;
        let public_id = uuid::Uuid::parse_str(public_id)
            .map_err(|err| anyhow!("invalid worklist id: {}", err))?;
        let Some(row) = store.get_worklist_by_public_id(public_id).await? else {
            return Ok(None);
        };
        Ok(Some(map_worklist_detail(row, persistence.cloned())))
    }

    pub(crate) async fn worklist_items(
        &self,
        query: WorklistItemsQuery<'_>,
    ) -> Result<Option<WorkItemList>> {
        let store = self.resolve_worklist_store(query.persistence).await?;
        let public_id = uuid::Uuid::parse_str(query.public_id)
            .map_err(|err| anyhow!("invalid worklist id: {}", err))?;
        let Some(worklist) = store.get_worklist_by_public_id(public_id).await? else {
            return Ok(None);
        };
        let rows = store.list_work_items(worklist.id).await?;
        let parent_row_id = if let Some(parent_public_id) = query.parent_public_id {
            let parent_uuid = uuid::Uuid::parse_str(parent_public_id)
                .map_err(|err| anyhow!("invalid parent item id: {}", err))?;
            rows.iter()
                .find(|row| row.public_id == parent_uuid.into_bytes().to_vec())
                .map(|row| row.id)
        } else {
            None
        };
        let public_ids = rows
            .iter()
            .map(|row| (row.id, format_public_id(&row.public_id)))
            .collect::<HashMap<_, _>>();
        let items = rows
            .into_iter()
            .filter(|row| query.status.is_none_or(|value| row.status == value))
            .filter(|row| !query.claimed_only || row.claim_execution_id.is_some())
            .filter(|row| match (query.parent_public_id, parent_row_id) {
                (Some(_), Some(parent_row_id)) => row.parent_item_id == Some(parent_row_id),
                (Some(_), None) => false,
                (None, _) => true,
            })
            .filter(|row| work_item_matches_where(row, &public_ids, query.where_filter))
            .take(query.limit.unwrap_or(u32::MAX) as usize)
            .map(|row| map_work_item_detail(row, &public_ids, &worklist))
            .collect();
        Ok(Some(WorkItemList {
            worklist_id: format_public_id(&worklist.public_id),
            items,
        }))
    }

    pub async fn work_item_detail(
        &self,
        public_id: &str,
        persistence: Option<&ContextPersistenceParams>,
    ) -> Result<Option<WorkItemDetail>> {
        let store = self.resolve_worklist_store(persistence).await?;
        let public_id = uuid::Uuid::parse_str(public_id)
            .map_err(|err| anyhow!("invalid work item id: {}", err))?;
        let Some(row) = store.get_work_item_by_public_id(public_id).await? else {
            return Ok(None);
        };
        let Some(worklist) = store.get_worklist_by_id(row.worklist_id).await? else {
            return Err(anyhow!(
                "work item {} references missing worklist {}",
                format_public_id(&row.public_id),
                row.worklist_id
            ));
        };
        let rows = store.list_work_items(worklist.id).await?;
        let public_ids = rows
            .iter()
            .map(|row| (row.id, format_public_id(&row.public_id)))
            .collect::<HashMap<_, _>>();
        Ok(Some(map_work_item_detail(row, &public_ids, &worklist)))
    }

    async fn resolve_worklist_store(
        &self,
        persistence: Option<&ContextPersistenceParams>,
    ) -> Result<Arc<StateStore>> {
        let selector = resolve_worklist_store_selector(&self.bootstrap_config, persistence)?;
        self.kernel.store_manager().open(&selector).await
    }
}

fn resolve_worklist_store_selector(
    config: &crate::kernel::config::TurinConfig,
    persistence: Option<&ContextPersistenceParams>,
) -> Result<StoreSelector> {
    let Some(persistence) = persistence else {
        return config.persistence.top_level_state_selector();
    };
    let context = ContextPersistenceConfig {
        state: persistence
            .state
            .as_ref()
            .map(store_target_config_from_params),
        store: persistence
            .store
            .as_ref()
            .map(store_target_config_from_params),
    };
    if persistence.store.is_some() {
        config
            .persistence
            .resolve_context_store_selector(Some(&context))
    } else if persistence.state.is_some() {
        config
            .persistence
            .resolve_context_state_selector(Some(&context))
    } else {
        config.persistence.top_level_state_selector()
    }
}

fn store_target_config_from_params(
    value: &StoreTargetParams,
) -> crate::kernel::config::StoreTargetConfig {
    crate::kernel::config::StoreTargetConfig {
        path: value.path.clone(),
        alias: value.alias.clone(),
    }
}

fn map_worklist_detail(
    row: WorklistRow,
    persistence: Option<ContextPersistenceParams>,
) -> WorklistDetail {
    WorklistDetail {
        id: row.id,
        public_id: format_public_id(&row.public_id),
        name: row.name,
        scope_ref: row.scope_ref,
        metadata: parse_json(row.metadata.as_deref()).ok().flatten(),
        persistence,
        created_at: row.created_at,
        updated_at: row.updated_at,
    }
}

fn map_work_item_detail(
    row: WorkItemRow,
    public_ids: &HashMap<i64, String>,
    worklist: &WorklistRow,
) -> WorkItemDetail {
    WorkItemDetail {
        id: row.id,
        public_id: format_public_id(&row.public_id),
        worklist_id: format_public_id(&worklist.public_id),
        parent_id: row
            .parent_item_id
            .and_then(|id| public_ids.get(&id).cloned()),
        title: row.title,
        kind: row.item_kind,
        prompt: row.prompt,
        content: parse_json(row.content.as_deref()).ok().flatten(),
        tools: parse_json(row.tools.as_deref()).ok().flatten(),
        conflict_policy: row.conflict_policy,
        action: scheduled_action(row.action_name, row.action_params),
        status: row.status,
        priority: row.priority,
        after: parse_json(row.after_ids.as_deref()).ok().flatten(),
        metadata: parse_json(row.metadata.as_deref()).ok().flatten(),
        claim_agent_id: row.claim_agent_id,
        claim_session_id: row.claim_session_id,
        claim_execution_id: row.claim_execution_id,
        claim_heartbeat_unix_ms: row.claim_heartbeat_unix_ms,
        claimed_at: row.claimed_at,
        completed_at: row.completed_at,
        failure_reason: row.failure_reason,
        created_at: row.created_at,
        updated_at: row.updated_at,
    }
}

fn scheduled_action(name: Option<String>, params: Option<String>) -> Option<ScheduleActionParams> {
    name.map(|name| ScheduleActionParams {
        name,
        params: parse_json(params.as_deref()).ok().flatten(),
    })
}

fn format_public_id(bytes: &[u8]) -> String {
    uuid::Uuid::from_slice(bytes)
        .map(|id| id.to_string())
        .unwrap_or_else(|_| super::helpers::format_uuid_bytes_simple(bytes))
}

fn parse_json<T: serde::de::DeserializeOwned>(value: Option<&str>) -> Result<Option<T>> {
    match value {
        Some(value) => Ok(Some(serde_json::from_str::<T>(value)?)),
        None => Ok(None),
    }
}

fn work_item_matches_where(
    row: &WorkItemRow,
    public_ids: &HashMap<i64, String>,
    where_filter: Option<&JsonMap<String, JsonValue>>,
) -> bool {
    let Some(where_filter) = where_filter else {
        return true;
    };
    let metadata = row
        .metadata
        .as_deref()
        .and_then(|raw| serde_json::from_str::<JsonValue>(raw).ok())
        .unwrap_or(JsonValue::Null);
    where_filter.iter().all(|(key, expected)| {
        work_item_filter_value(row, public_ids, &metadata, key).as_ref() == Some(expected)
    })
}

fn work_item_filter_value(
    row: &WorkItemRow,
    public_ids: &HashMap<i64, String>,
    metadata: &JsonValue,
    key: &str,
) -> Option<JsonValue> {
    match key {
        "id" | "public_id" => Some(JsonValue::String(format_public_id(&row.public_id))),
        "title" => Some(JsonValue::String(row.title.clone())),
        "kind" => Some(JsonValue::String(row.item_kind.clone())),
        "status" => Some(JsonValue::String(row.status.clone())),
        "priority" => Some(JsonValue::Number(row.priority.into())),
        "parent_id" => Some(
            row.parent_item_id
                .and_then(|id| public_ids.get(&id).cloned())
                .map(JsonValue::String)
                .unwrap_or(JsonValue::Null),
        ),
        _ => metadata.get(key).cloned(),
    }
}
