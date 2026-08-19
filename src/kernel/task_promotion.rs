use std::sync::Arc;

use anyhow::{Context, Result, anyhow};

use crate::inference::content::{encode_content_json, task_content_from_parts};
use crate::kernel::session_refs::parse_session_reference;
use crate::persistence::manager::{StoreManager, StoreSelector};
use crate::persistence::schema::BranchProvenance;
use crate::persistence::state::TurnWriteTarget;
use turin_types::TaskInputContent;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaskPromotionCandidate {
    pub session_id: String,
    pub source_turn_id: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_session_id: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PromotedTaskBranch {
    pub session_id: String,
    pub branch_id: String,
    pub name: String,
    pub head_turn_index: Option<u32>,
    pub source_turn_id: Option<i64>,
    pub origin_kind: String,
    pub origin_task_id: Option<String>,
    pub origin_execution_id: Option<String>,
    pub origin_metadata: Option<serde_json::Value>,
    pub active: bool,
    pub created_at: String,
}

pub(crate) async fn promote_task_result(
    store_manager: &Arc<StoreManager>,
    promotion: &TaskPromotionCandidate,
    input_content: &[TaskInputContent],
    assistant_content: &[TaskInputContent],
    origin_task_id: Option<&str>,
    branch_name: Option<&str>,
) -> Result<PromotedTaskBranch> {
    if input_content.is_empty() {
        anyhow::bail!("Task is missing promotable task input");
    }
    if assistant_content.is_empty() {
        anyhow::bail!("Task has no promotable assistant output");
    }

    let session_ref = parse_session_reference(&promotion.session_id)?;
    let public_id = uuid::Uuid::parse_str(&session_ref.public_id)
        .with_context(|| format!("Invalid session id '{}'", promotion.session_id))?;
    let store_selector = session_ref
        .store_selector
        .unwrap_or_else(|| StoreSelector::Alias("state".to_string()));
    let store = store_manager.open(&store_selector).await?;
    let row = store
        .get_session_row_by_public_id(public_id)
        .await?
        .ok_or_else(|| anyhow!("Session '{}' not found", promotion.session_id))?;
    let source_turn = store
        .get_turn_row(promotion.source_turn_id)
        .await?
        .ok_or_else(|| anyhow!("Source turn '{}' not found", promotion.source_turn_id))?;
    if source_turn.session_id != row.id {
        anyhow::bail!(
            "Source turn '{}' does not belong to promoted session '{}'",
            promotion.source_turn_id,
            promotion.session_id
        );
    }

    let branch_name = branch_name
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| format!("promoted-{}", uuid::Uuid::now_v7().simple()));
    let branch = store
        .create_branch_head_from_turn_id_with_provenance(
            row.id,
            &branch_name,
            promotion.source_turn_id,
            false,
            BranchProvenance::promotion(
                origin_task_id.map(str::to_string),
                promotion.source_session_id.clone(),
            ),
        )
        .await?;
    let turn_target = store
        .prepare_turn_write_target(
            row.id,
            TurnWriteTarget::branch_head_with_expectation(
                Some(branch.id),
                Some(promotion.source_turn_id),
                source_turn.branch_depth + 1,
            ),
        )
        .await?
        .ok_or_else(|| anyhow!("Failed to allocate promoted turn target"))?;

    store
        .insert_message(
            row.id,
            turn_target,
            "user",
            &encode_content_json(&task_content_from_parts(input_content)),
            None,
        )
        .await?;
    store
        .insert_message(
            row.id,
            turn_target,
            "assistant",
            &encode_content_json(&task_content_from_parts(assistant_content)),
            None,
        )
        .await?;

    let branch = store
        .get_branch_head(row.id, branch.id)
        .await?
        .ok_or_else(|| anyhow!("Promoted branch '{}' was not readable", branch_name))?;

    Ok(PromotedTaskBranch {
        session_id: promotion.session_id.clone(),
        branch_id: uuid::Uuid::from_slice(&branch.public_id)
            .map(|value| value.to_string())
            .map_err(anyhow::Error::from)?,
        name: branch.name,
        head_turn_index: branch.head_turn_depth,
        source_turn_id: branch.created_from_turn_id,
        origin_kind: branch.origin_kind,
        origin_task_id: branch.origin_task_id,
        origin_execution_id: branch.origin_execution_id,
        origin_metadata: branch
            .origin_metadata
            .as_deref()
            .and_then(|raw| serde_json::from_str(raw).ok()),
        active: branch.is_active,
        created_at: branch.created_at,
    })
}
