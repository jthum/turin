use anyhow::{Context, Result, anyhow};

use crate::inference::content::decode_content_json;
use crate::inference::provider::{InferenceMessage, InferenceRole};
use crate::kernel::event::{AuditEvent, KernelEvent};
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::session::{ContextCompactionCheckpoint, ExecutionContextTarget, HistoryOrigin};
use crate::kernel::session_refs::parse_session_reference;
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::{MessageRow, SessionRow};
use crate::persistence::state::TokenBoundedMessages;
use crate::persistence::state::{SessionReadTarget, StateStore};

pub(super) struct MaterializedExecutionTarget {
    pub(super) messages: Vec<MessageRow>,
    pub(super) has_prior_history: bool,
    pub(super) context_checkpoint: Option<ContextCompactionCheckpoint>,
    pub(super) branch_head_turn_id: Option<i64>,
    pub(super) branch_head_turn_index: Option<u32>,
}

pub(super) struct TokenContextBounds {
    pub(super) token_budget: u64,
    pub(super) minimum_messages: usize,
    pub(super) max_turns: usize,
}

pub(super) type MaterializedHistory = Vec<(InferenceMessage, Option<HistoryOrigin>)>;

pub(super) async fn materialize_execution_target(
    host: &ExecutionHost,
    store: &StateStore,
    current_store_selector: &StoreSelector,
    row: &SessionRow,
    target: &ExecutionContextTarget,
) -> Result<MaterializedExecutionTarget> {
    let max_messages = host
        .config
        .inference
        .hot_history
        .effective_max_messages()
        .unwrap_or(4_096)
        .max(1);
    let max_turns = max_messages.saturating_mul(2).max(64);
    match target {
        ExecutionContextTarget::BranchHead { branch_head_id } => {
            let branch_head_id = branch_head_id.or(row.active_branch_head_id);
            let target = SessionReadTarget::branch_head(branch_head_id);
            let branch_head_turn_id = match branch_head_id {
                Some(branch_head_id) => store
                    .get_branch_head(row.id, branch_head_id)
                    .await?
                    .and_then(|branch| branch.head_turn_id),
                None => store
                    .get_active_branch_head(row.id)
                    .await?
                    .and_then(|branch| branch.head_turn_id),
            };
            let (messages, has_prior_history) = store
                .get_bounded_context_messages(row.id, &target, max_turns, max_messages)
                .await?;
            Ok(MaterializedExecutionTarget {
                messages,
                has_prior_history,
                context_checkpoint: latest_context_checkpoint(store, row.id).await?,
                branch_head_turn_id,
                branch_head_turn_index: match branch_head_turn_id {
                    Some(turn_id) => store
                        .get_turn_row(turn_id)
                        .await?
                        .map(|turn| turn.branch_depth),
                    None => None,
                },
            })
        }
        ExecutionContextTarget::TurnId { turn_id } => {
            let target = SessionReadTarget::TurnId(*turn_id);
            let (messages, has_prior_history) = store
                .get_bounded_context_messages(row.id, &target, max_turns, max_messages)
                .await?;
            Ok(MaterializedExecutionTarget {
                messages,
                has_prior_history,
                context_checkpoint: latest_context_checkpoint(store, row.id).await?,
                branch_head_turn_id: None,
                branch_head_turn_index: None,
            })
        }
        ExecutionContextTarget::SelectedPath { turn_ids } => {
            let target = SessionReadTarget::SelectedPath(turn_ids.clone());
            let (messages, has_prior_history) = store
                .get_bounded_context_messages(row.id, &target, max_turns, max_messages)
                .await?;
            Ok(MaterializedExecutionTarget {
                messages,
                has_prior_history,
                context_checkpoint: latest_context_checkpoint(store, row.id).await?,
                branch_head_turn_id: None,
                branch_head_turn_index: None,
            })
        }
        ExecutionContextTarget::SummarySource { source_turn_id } => {
            let target = SessionReadTarget::TurnId(*source_turn_id);
            let (messages, has_prior_history) = store
                .get_bounded_context_messages(row.id, &target, max_turns, max_messages)
                .await?;
            Ok(MaterializedExecutionTarget {
                messages,
                has_prior_history,
                context_checkpoint: latest_context_checkpoint(store, row.id).await?,
                branch_head_turn_id: None,
                branch_head_turn_index: None,
            })
        }
        ExecutionContextTarget::ExternalReference { reference } => {
            let session_ref = parse_session_reference(reference)?;
            let target_selector = session_ref
                .store_selector
                .clone()
                .unwrap_or_else(|| current_store_selector.clone());
            let target_store = host.store_manager.open(&target_selector).await.with_context(|| {
                format!(
                    "Execution context target '{}' requires a configured persistent state store",
                    reference
                )
            })?;
            let public_id = uuid::Uuid::parse_str(&session_ref.public_id)
                .with_context(|| format!("Invalid session id '{}'", session_ref.public_id))?;
            let referenced_row = target_store
                .get_session_row_by_public_id(public_id)
                .await?
                .ok_or_else(|| anyhow!("Execution context target '{}' was not found", reference))?;
            let target = SessionReadTarget::branch_head(referenced_row.active_branch_head_id);
            let (messages, has_prior_history) = target_store
                .get_bounded_context_messages(referenced_row.id, &target, max_turns, max_messages)
                .await?;
            Ok(MaterializedExecutionTarget {
                messages,
                has_prior_history,
                context_checkpoint: latest_context_checkpoint(&target_store, referenced_row.id)
                    .await?,
                branch_head_turn_id: None,
                branch_head_turn_index: None,
            })
        }
    }
}

async fn latest_context_checkpoint(
    store: &StateStore,
    session_id: i64,
) -> Result<Option<ContextCompactionCheckpoint>> {
    let Some(event) = store
        .get_latest_session_event_by_type(session_id, "context_compaction")
        .await?
    else {
        return Ok(None);
    };
    let Ok(KernelEvent::Audit(AuditEvent::ContextCompaction { checkpoint })) =
        serde_json::from_str::<KernelEvent>(&event.payload)
    else {
        return Ok(None);
    };
    Ok(Some(checkpoint))
}

pub(super) async fn materialize_token_bounded_messages(
    host: &ExecutionHost,
    store: &StateStore,
    current_store_selector: &StoreSelector,
    row: &SessionRow,
    target: &ExecutionContextTarget,
    bounds: TokenContextBounds,
) -> Result<TokenBoundedMessages> {
    let TokenContextBounds {
        token_budget,
        minimum_messages,
        max_turns,
    } = bounds;
    match target {
        ExecutionContextTarget::BranchHead { branch_head_id } => {
            let target =
                SessionReadTarget::branch_head(branch_head_id.or(row.active_branch_head_id));
            store
                .get_token_bounded_context_messages(
                    row.id,
                    &target,
                    token_budget,
                    minimum_messages,
                    max_turns,
                )
                .await
        }
        ExecutionContextTarget::TurnId { turn_id } => {
            store
                .get_token_bounded_context_messages(
                    row.id,
                    &SessionReadTarget::TurnId(*turn_id),
                    token_budget,
                    minimum_messages,
                    max_turns,
                )
                .await
        }
        ExecutionContextTarget::SelectedPath { turn_ids } => {
            store
                .get_token_bounded_context_messages(
                    row.id,
                    &SessionReadTarget::SelectedPath(turn_ids.clone()),
                    token_budget,
                    minimum_messages,
                    max_turns,
                )
                .await
        }
        ExecutionContextTarget::SummarySource { source_turn_id } => {
            store
                .get_token_bounded_context_messages(
                    row.id,
                    &SessionReadTarget::TurnId(*source_turn_id),
                    token_budget,
                    minimum_messages,
                    max_turns,
                )
                .await
        }
        ExecutionContextTarget::ExternalReference { reference } => {
            let session_ref = parse_session_reference(reference)?;
            let target_selector = session_ref
                .store_selector
                .clone()
                .unwrap_or_else(|| current_store_selector.clone());
            let target_store = host.store_manager.open(&target_selector).await.with_context(|| {
                format!(
                    "Execution context target '{}' requires a configured persistent state store",
                    reference
                )
            })?;
            let public_id = uuid::Uuid::parse_str(&session_ref.public_id)
                .with_context(|| format!("Invalid session id '{}'", session_ref.public_id))?;
            let referenced_row = target_store
                .get_session_row_by_public_id(public_id)
                .await?
                .ok_or_else(|| anyhow!("Execution context target '{}' was not found", reference))?;
            target_store
                .get_token_bounded_context_messages(
                    referenced_row.id,
                    &SessionReadTarget::branch_head(referenced_row.active_branch_head_id),
                    token_budget,
                    minimum_messages,
                    max_turns,
                )
                .await
        }
    }
}

pub(super) fn rebuild_history(messages: &[MessageRow]) -> Result<(MaterializedHistory, u32)> {
    let mut history = Vec::new();
    let mut max_turn_index = None;

    for message in messages {
        max_turn_index =
            Some(max_turn_index.map_or(message.turn_index, |max: u32| max.max(message.turn_index)));
        let content_json: serde_json::Value = serde_json::from_str(&message.content)
            .with_context(|| format!("Failed to parse persisted message {}", message.id))?;
        let content = decode_content_json(content_json)
            .with_context(|| format!("Failed to rebuild persisted message {}", message.id))?;
        history.push((
            InferenceMessage {
                role: decode_role(&message.role)?,
                content,
                tool_call_id: None,
            },
            Some(HistoryOrigin {
                turn_id: message.turn_id,
                turn_index: message.turn_index,
            }),
        ));
    }

    Ok((history, max_turn_index.map_or(0, |idx| idx + 1)))
}

fn decode_role(role: &str) -> Result<InferenceRole> {
    match role {
        "user" => Ok(InferenceRole::User),
        "assistant" => Ok(InferenceRole::Assistant),
        "tool_result" => Ok(InferenceRole::Tool),
        other => anyhow::bail!("Unsupported persisted role '{}'", other),
    }
}
