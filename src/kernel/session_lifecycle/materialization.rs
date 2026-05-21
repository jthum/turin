use anyhow::{Context, Result, anyhow};

use crate::inference::content::decode_content_json;
use crate::inference::provider::{InferenceMessage, InferenceRole};
use crate::kernel::event::{AuditEvent, KernelEvent};
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::session::{ContextCompactionCheckpoint, ExecutionContextTarget};
use crate::kernel::session_refs::parse_session_reference;
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::{EventRow, MessageRow, SessionRow};
use crate::persistence::state::{SessionReadTarget, StateStore};

pub(super) struct MaterializedExecutionTarget {
    pub(super) messages: Vec<MessageRow>,
    pub(super) active_events: Vec<EventRow>,
    pub(super) branch_head_turn_id: Option<i64>,
    pub(super) branch_head_turn_index: Option<u32>,
}

pub(super) async fn materialize_execution_target(
    host: &ExecutionHost,
    store: &StateStore,
    current_store_selector: &StoreSelector,
    row: &SessionRow,
    target: &ExecutionContextTarget,
) -> Result<MaterializedExecutionTarget> {
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
            Ok(MaterializedExecutionTarget {
                messages: store.get_messages(row.id, &target).await?,
                active_events: store.get_events(row.id, &target).await?,
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
            Ok(MaterializedExecutionTarget {
                messages: store.get_messages(row.id, &target).await?,
                active_events: store.get_events(row.id, &target).await?,
                branch_head_turn_id: None,
                branch_head_turn_index: None,
            })
        }
        ExecutionContextTarget::SelectedPath { turn_ids } => {
            let target = SessionReadTarget::SelectedPath(turn_ids.clone());
            Ok(MaterializedExecutionTarget {
                messages: store.get_messages(row.id, &target).await?,
                active_events: store.get_events(row.id, &target).await?,
                branch_head_turn_id: None,
                branch_head_turn_index: None,
            })
        }
        ExecutionContextTarget::SummarySource { source_turn_id } => {
            let target = SessionReadTarget::TurnId(*source_turn_id);
            Ok(MaterializedExecutionTarget {
                messages: store.get_messages(row.id, &target).await?,
                active_events: store.get_events(row.id, &target).await?,
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
            Ok(MaterializedExecutionTarget {
                messages: target_store
                    .get_messages(referenced_row.id, &target)
                    .await?,
                active_events: target_store.get_events(referenced_row.id, &target).await?,
                branch_head_turn_id: None,
                branch_head_turn_index: None,
            })
        }
    }
}

pub(super) fn rebuild_history(messages: &[MessageRow]) -> Result<(Vec<InferenceMessage>, u32)> {
    let mut history = Vec::new();
    let mut max_turn_index = None;

    for message in messages {
        max_turn_index =
            Some(max_turn_index.map_or(message.turn_index, |max: u32| max.max(message.turn_index)));
        let content_json: serde_json::Value = serde_json::from_str(&message.content)
            .with_context(|| format!("Failed to parse persisted message {}", message.id))?;
        let content = decode_content_json(content_json)
            .with_context(|| format!("Failed to rebuild persisted message {}", message.id))?;
        history.push(InferenceMessage {
            role: decode_role(&message.role)?,
            content,
            tool_call_id: None,
        });
    }

    Ok((history, max_turn_index.map_or(0, |idx| idx + 1)))
}

pub(super) fn rebuild_session_counters(events: &[EventRow]) -> (u32, u32, u64, u64) {
    let mut next_task_id = 1;
    let mut next_plan_id = 1;
    let mut total_input_tokens = 0;
    let mut total_output_tokens = 0;

    for event in events {
        let Ok(payload) = serde_json::from_str::<serde_json::Value>(&event.payload) else {
            continue;
        };
        if let Some(task_id) = payload.get("task_id").and_then(|value| value.as_str()) {
            next_task_id = next_task_id.max(next_numeric_suffix(task_id, "t_"));
        }
        if let Some(plan_id) = payload.get("plan_id").and_then(|value| value.as_str()) {
            next_plan_id = next_plan_id.max(next_numeric_suffix(plan_id, "p_"));
        }
        match event.event_type.as_str() {
            "message_end" => {
                total_input_tokens += payload
                    .get("input_tokens")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0);
                total_output_tokens += payload
                    .get("output_tokens")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0);
            }
            "session_end" => {
                total_input_tokens = payload
                    .get("total_input_tokens")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(total_input_tokens);
                total_output_tokens = payload
                    .get("total_output_tokens")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(total_output_tokens);
            }
            _ => {}
        }
    }

    (
        next_task_id,
        next_plan_id,
        total_input_tokens,
        total_output_tokens,
    )
}

pub(super) fn rebuild_context_checkpoint(
    events: &[EventRow],
) -> Option<ContextCompactionCheckpoint> {
    let mut checkpoint = None;

    for event in events {
        if event.event_type != "context_compaction" {
            continue;
        }

        let Ok(KernelEvent::Audit(AuditEvent::ContextCompaction {
            checkpoint: persisted,
        })) = serde_json::from_str::<KernelEvent>(&event.payload)
        else {
            continue;
        };

        checkpoint = Some(persisted);
    }

    checkpoint
}

fn decode_role(role: &str) -> Result<InferenceRole> {
    match role {
        "user" => Ok(InferenceRole::User),
        "assistant" => Ok(InferenceRole::Assistant),
        "tool_result" => Ok(InferenceRole::Tool),
        other => anyhow::bail!("Unsupported persisted role '{}'", other),
    }
}

fn next_numeric_suffix(value: &str, prefix: &str) -> u32 {
    value
        .strip_prefix(prefix)
        .and_then(|suffix| suffix.parse::<u32>().ok())
        .map(|value| value + 1)
        .unwrap_or(1)
}
