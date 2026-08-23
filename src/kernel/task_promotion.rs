use std::sync::Arc;

use anyhow::{Context, Result, anyhow};

use crate::inference::content::{
    decode_content_json, encode_content_json, task_content_from_parts,
    task_output_content_from_inference,
};
use crate::kernel::session_refs::parse_session_reference;
use crate::persistence::manager::{StoreManager, StoreSelector};
use crate::persistence::schema::BranchProvenance;
use crate::persistence::state::SessionReadTarget;
use turin_types::TaskInputContent;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaskPromotionCandidate {
    pub session_id: String,
    pub source_turn_id: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_session_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskPromotionSelection {
    Result,
    LinkedTurn(i64),
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

#[doc(hidden)]
pub async fn promote_task_result(
    store_manager: &Arc<StoreManager>,
    promotion: &TaskPromotionCandidate,
    input_content: &[TaskInputContent],
    assistant_content: &[TaskInputContent],
    origin_task_id: Option<&str>,
    branch_name: Option<&str>,
    selection: TaskPromotionSelection,
) -> Result<PromotedTaskBranch> {
    let selected_source_turn_id = match selection {
        TaskPromotionSelection::Result => None,
        TaskPromotionSelection::LinkedTurn(turn_id) => Some(turn_id),
    };
    let selected_content = match selection {
        TaskPromotionSelection::Result => None,
        TaskPromotionSelection::LinkedTurn(turn_id) => {
            Some(load_linked_turn_content(store_manager, promotion, turn_id).await?)
        }
    };
    let (input_content, assistant_content) = selected_content
        .as_ref()
        .map(|(input, assistant)| (input.as_slice(), assistant.as_slice()))
        .unwrap_or((input_content, assistant_content));
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
    let branch_name = branch_name
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| format!("promoted-{}", uuid::Uuid::now_v7().simple()));
    let branch = store
        .create_promoted_branch_from_turn(
            row.id,
            &branch_name,
            promotion.source_turn_id,
            BranchProvenance::promotion(
                origin_task_id.map(str::to_string),
                promotion.source_session_id.clone(),
                selected_source_turn_id,
            ),
            &encode_content_json(&task_content_from_parts(input_content)),
            &encode_content_json(&task_content_from_parts(assistant_content)),
        )
        .await?;

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

async fn load_linked_turn_content(
    store_manager: &Arc<StoreManager>,
    promotion: &TaskPromotionCandidate,
    turn_id: i64,
) -> Result<(Vec<TaskInputContent>, Vec<TaskInputContent>)> {
    let source_session_id = promotion
        .source_session_id
        .as_deref()
        .ok_or_else(|| anyhow!("Selected-turn promotion requires a linked source session"))?;
    let source_ref = parse_session_reference(source_session_id)?;
    let source_public_id = uuid::Uuid::parse_str(&source_ref.public_id)
        .with_context(|| format!("Invalid source session id '{source_session_id}'"))?;
    let source_selector = source_ref
        .store_selector
        .unwrap_or_else(|| StoreSelector::Alias("state".to_string()));
    let source_store = store_manager.open(&source_selector).await?;
    let source_session = source_store
        .get_session_row_by_public_id(source_public_id)
        .await?
        .ok_or_else(|| anyhow!("Source session '{}' not found", source_session_id))?;
    let messages = source_store
        .get_messages(
            source_session.id,
            &SessionReadTarget::SelectedPath(vec![turn_id]),
        )
        .await?;

    let mut input = Vec::new();
    let mut assistant = Vec::new();
    for message in messages {
        let value: serde_json::Value = serde_json::from_str(&message.content)
            .with_context(|| format!("Invalid content for source message '{}'", message.id))?;
        let parts = task_output_content_from_inference(&decode_content_json(value)?);
        match message.role.as_str() {
            "user" => input.extend(parts),
            "assistant" => assistant.extend(parts),
            _ => {}
        }
    }
    if input.is_empty() || assistant.is_empty() {
        anyhow::bail!(
            "Linked source turn '{}' must contain user input and assistant output",
            turn_id
        );
    }
    Ok((input, assistant))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::session_refs::format_session_reference;
    use crate::persistence::state::TurnWriteTarget;
    use serde_json::json;

    #[tokio::test]
    async fn selected_linked_turn_promotes_only_its_message_boundary() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let state_path = temp.path().join("state.db");
        let manager = Arc::new(StoreManager::new(temp.path(), temp.path().join("stores")));
        manager.register_alias("state", &state_path).await?;
        let store = manager.get_default().await?;

        let parent_public_id = uuid::Uuid::now_v7();
        let parent_id = store
            .create_session(parent_public_id, "parent", None)
            .await?;
        store
            .insert_message(
                parent_id,
                TurnWriteTarget::active_branch(0),
                "user",
                &json!([{"type": "text", "text": "origin"}]),
                None,
            )
            .await?;
        let parent_turn_id = store
            .get_active_branch_head(parent_id)
            .await?
            .and_then(|branch| branch.head_turn_id)
            .expect("parent source turn");

        let child_public_id = uuid::Uuid::now_v7();
        let child_id = store
            .create_session(child_public_id, "worker", None)
            .await?;
        for (role, text) in [("user", "selected input"), ("assistant", "selected result")] {
            store
                .insert_message(
                    child_id,
                    TurnWriteTarget::active_branch(0),
                    role,
                    &json!([{"type": "text", "text": text}]),
                    None,
                )
                .await?;
        }
        let child_turn_id = store
            .get_active_branch_head(child_id)
            .await?
            .and_then(|branch| branch.head_turn_id)
            .expect("child source turn");

        let promoted = promote_task_result(
            &manager,
            &TaskPromotionCandidate {
                session_id: format_session_reference(
                    &parent_public_id.simple().to_string(),
                    &StoreSelector::Alias("state".to_string()),
                ),
                source_turn_id: parent_turn_id,
                source_session_id: Some(format_session_reference(
                    &child_public_id.simple().to_string(),
                    &StoreSelector::Alias("state".to_string()),
                )),
            },
            &[],
            &[],
            Some("request-1"),
            Some("selected-child-turn"),
            TaskPromotionSelection::LinkedTurn(child_turn_id),
        )
        .await?;

        let branch = store
            .get_branch_head_by_name(parent_id, &promoted.name)
            .await?
            .expect("promoted branch");
        let messages = store
            .get_messages(parent_id, &SessionReadTarget::BranchHead(branch.id))
            .await?;
        assert!(
            messages
                .iter()
                .any(|message| message.content.contains("selected input"))
        );
        assert!(
            messages
                .iter()
                .any(|message| message.content.contains("selected result"))
        );
        assert_eq!(
            promoted
                .origin_metadata
                .as_ref()
                .and_then(|metadata| metadata.get("source_turn_id"))
                .and_then(serde_json::Value::as_i64),
            Some(child_turn_id)
        );
        Ok(())
    }
}
