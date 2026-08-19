use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use tracing::{debug, info, instrument};
use uuid::Uuid;

mod branches;
mod discovery;
mod family;
mod projection;

use projection::{session_efficiency_from_events, session_execution_from_events};

use super::{
    DaemonState, SessionBranchDetail, SessionDetail, SessionEventDetail, SessionMessageDetail,
    SessionMessageWindow, SessionSummary, SessionToolExecutionDetail,
};
use crate::kernel::agent_manager::LiveSessionSnapshot;
use crate::kernel::session_refs::{
    describe_store_selector, format_session_reference, parse_session_reference,
};
use crate::perf_diagnostics::{perf_stage, perf_stage_finish};
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::{BranchHeadRow, SessionRow};
use crate::persistence::state::{SessionReadTarget, StateStore};

const SESSION_EXECUTION_EVENT_LIMIT: usize = 400;

#[derive(Debug, thiserror::Error)]
#[error("Session '{session_id}' still has {work_count} active or queued runtime target(s)")]
pub(crate) struct SessionDeleteBusy {
    session_id: String,
    work_count: usize,
}

impl DaemonState {
    #[instrument(skip(self), fields(session_id = %session_id))]
    pub async fn get_session(&self, session_id: &str) -> Result<Option<SessionDetail>> {
        self.get_session_projection(session_id, None, None, None, true, true)
            .await
    }

    #[instrument(
        skip(self),
        fields(
            session_id = %session_id,
            message_limit = ?message_limit,
            include_events = include_events
        )
    )]
    pub async fn get_session_projection(
        &self,
        session_id: &str,
        target_turn_id: Option<i64>,
        message_limit: Option<usize>,
        message_offset: Option<usize>,
        include_events: bool,
        include_efficiency: bool,
    ) -> Result<Option<SessionDetail>> {
        anyhow::ensure!(
            message_offset.is_none() || message_limit.is_some(),
            "message_offset requires message_limit"
        );
        perf_stage!(
            projection_stage,
            "session.projection",
            Some(session_id),
            serde_json::json!({
                "message_limit": message_limit,
                "message_offset": message_offset,
                "target_turn_id": target_turn_id,
                "include_events": include_events,
                "include_efficiency": include_efficiency,
            })
        );
        perf_stage!(
            resolve_stage,
            "session.resolve",
            Some(session_id),
            serde_json::json!({})
        );
        let resolved = self.resolve_persisted_session(session_id).await?;
        perf_stage_finish!(
            resolve_stage,
            if resolved.is_some() {
                "ok"
            } else {
                "not_found"
            },
            serde_json::json!({})
        );
        let Some((store_selector, store, row)) = resolved else {
            perf_stage_finish!(projection_stage, "not_found", serde_json::json!({}));
            return Ok(None);
        };
        debug!(
            store = %describe_store_selector(&store_selector),
            "Resolved persisted session detail target"
        );
        let read_target =
            target_turn_id.map_or(SessionReadTarget::ActiveBranch, SessionReadTarget::TurnId);

        let events = if include_events {
            perf_stage!(
                events_stage,
                "session.events",
                Some(session_id),
                serde_json::json!({ "internal_session_id": row.id })
            );
            let persisted_events = store.get_events(row.id, &read_target).await?;
            let _payload_bytes = persisted_events
                .iter()
                .map(|event| event.payload.len())
                .sum::<usize>();
            let events = persisted_events
                .into_iter()
                .map(|event| SessionEventDetail {
                    id: event.id,
                    event_type: event.event_type,
                    payload: super::helpers::parse_json_or_string(&event.payload),
                    created_at: event.created_at,
                })
                .collect::<Vec<_>>();
            perf_stage_finish!(
                events_stage,
                "ok",
                serde_json::json!({
                    "rows": events.len(),
                    "payload_bytes": _payload_bytes,
                })
            );
            events
        } else {
            Vec::new()
        };

        perf_stage!(
            messages_stage,
            "session.messages",
            Some(session_id),
            serde_json::json!({
                "internal_session_id": row.id,
                "message_limit": message_limit,
                "message_offset": message_offset,
            })
        );
        let (persisted_messages, total_messages, message_offset) =
            match (message_limit, message_offset) {
                (Some(limit), Some(offset)) => {
                    store
                        .get_message_window(row.id, &read_target, offset, limit)
                        .await?
                }
                (Some(limit), None) => {
                    let (messages, total) = store
                        .get_recent_messages(row.id, &read_target, limit)
                        .await?;
                    let offset = total.saturating_sub(messages.len());
                    (messages, total, offset)
                }
                (None, None) => {
                    let messages = store.get_messages(row.id, &read_target).await?;
                    let total = messages.len();
                    (messages, total, 0)
                }
                (None, Some(_)) => unreachable!("message offset was validated above"),
            };
        let _message_payload_bytes = persisted_messages
            .iter()
            .map(|message| message.content.len())
            .sum::<usize>();
        let _loaded_message_count = persisted_messages.len();
        perf_stage_finish!(
            messages_stage,
            "ok",
            serde_json::json!({
                "rows": _loaded_message_count,
                "total_rows": total_messages,
                "resolved_offset": message_offset,
                "payload_bytes": _message_payload_bytes,
            })
        );
        perf_stage!(
            message_projection_stage,
            "session.messages.project",
            Some(session_id),
            serde_json::json!({ "rows": _loaded_message_count })
        );
        let messages = persisted_messages
            .into_iter()
            .map(|message| {
                let content = super::helpers::parse_json_or_string(&message.content);
                let estimated_token_count =
                    crate::kernel::estimate_persisted_message_input_tokens(&message.role, &content);
                SessionMessageDetail {
                    id: message.id,
                    turn_id: message.turn_id,
                    turn_index: message.turn_index,
                    role: message.role,
                    content,
                    token_count: message.token_count,
                    estimated_token_count,
                    created_at: message.created_at,
                }
            })
            .collect::<Vec<_>>();
        perf_stage_finish!(
            message_projection_stage,
            "ok",
            serde_json::json!({ "rows": messages.len() })
        );

        let visible_turn_indexes = message_limit.map(|_| {
            messages
                .iter()
                .map(|message| message.turn_index)
                .collect::<HashSet<_>>()
        });

        perf_stage!(
            tools_stage,
            "session.tools",
            Some(session_id),
            serde_json::json!({
                "visible_turns": visible_turn_indexes.as_ref().map(HashSet::len),
            })
        );
        let persisted_tool_executions = store
            .get_tool_executions_for_turn_indexes(
                row.id,
                &read_target,
                visible_turn_indexes.as_ref(),
            )
            .await?;
        let _tool_payload_bytes = persisted_tool_executions
            .iter()
            .map(|execution| execution.args.len() + execution.output.as_deref().map_or(0, str::len))
            .sum::<usize>();
        let tool_executions = persisted_tool_executions
            .into_iter()
            .map(|execution| SessionToolExecutionDetail {
                id: execution.id,
                turn_index: execution.turn_index,
                tool_call_id: execution.tool_call_id,
                tool_name: execution.tool_name,
                args: super::helpers::parse_json_or_string(&execution.args),
                output: execution
                    .output
                    .as_deref()
                    .map(super::helpers::parse_json_or_string),
                is_error: execution.is_error,
                duration_ms: execution.duration_ms,
                verdict: execution.verdict,
                created_at: execution.created_at,
            })
            .collect::<Vec<_>>();
        perf_stage_finish!(
            tools_stage,
            "ok",
            serde_json::json!({
                "rows": tool_executions.len(),
                "payload_bytes": _tool_payload_bytes,
                "visible_turns": visible_turn_indexes.as_ref().map(HashSet::len),
            })
        );

        perf_stage!(
            branches_stage,
            "session.branches",
            Some(session_id),
            serde_json::json!({})
        );
        let branches = store
            .list_branch_heads(row.id)
            .await?
            .into_iter()
            .map(branch_detail_from_row)
            .collect::<Vec<_>>();
        perf_stage_finish!(
            branches_stage,
            "ok",
            serde_json::json!({ "rows": branches.len() })
        );

        let efficiency = if include_efficiency {
            perf_stage!(
                efficiency_stage,
                "session.efficiency",
                Some(session_id),
                serde_json::json!({})
            );
            let events = store
                .get_events_by_types(
                    row.id,
                    &read_target,
                    &["inference_request", "message_end", "context_compaction"],
                )
                .await?;
            let _event_count = events.len();
            let efficiency = session_efficiency_from_events(events, visible_turn_indexes.as_ref());
            perf_stage_finish!(
                efficiency_stage,
                "ok",
                serde_json::json!({
                    "event_rows": _event_count,
                    "request_rows": efficiency.total_request_count,
                })
            );
            Some(efficiency)
        } else {
            None
        };

        perf_stage!(
            execution_stage,
            "session.execution",
            Some(session_id),
            serde_json::json!({ "event_limit": SESSION_EXECUTION_EVENT_LIMIT })
        );
        let mut execution_events = store
            .get_recent_events_by_types(
                row.id,
                &read_target,
                &[
                    "task_start",
                    "task_complete",
                    "plan_complete",
                    "turn_start",
                    "turn_end",
                ],
                SESSION_EXECUTION_EVENT_LIMIT + 1,
            )
            .await?;
        let _execution_event_count = execution_events.len();
        let execution_truncated = execution_events.len() > SESSION_EXECUTION_EVENT_LIMIT;
        if execution_truncated {
            execution_events.remove(0);
        }
        let execution = session_execution_from_events(execution_events, execution_truncated);
        perf_stage_finish!(
            execution_stage,
            "ok",
            serde_json::json!({
                "event_rows": _execution_event_count,
                "truncated": execution_truncated,
                "tasks": execution.tasks.len(),
                "plans": execution.plans.len(),
            })
        );

        let detail = SessionDetail {
            session: session_summary_from_row_and_selector(&row, &store_selector),
            branches,
            events,
            messages,
            tool_executions,
            efficiency,
            execution,
            message_window: message_limit.map(|_| SessionMessageWindow {
                offset: message_offset,
                total: total_messages,
            }),
        };
        perf_stage_finish!(
            projection_stage,
            "ok",
            serde_json::json!({
                "message_rows": detail.messages.len(),
                "total_message_rows": total_messages,
                "event_rows": detail.events.len(),
                "tool_rows": detail.tool_executions.len(),
                "branch_rows": detail.branches.len(),
            })
        );
        Ok(Some(detail))
    }

    pub async fn set_session_title(
        &self,
        session_id: &str,
        title: Option<&str>,
    ) -> Result<Option<SessionSummary>> {
        let (store_selector, public_id) = persisted_session_target(session_id)?;
        debug!(
            session_id = %session_id,
            store = %describe_store_selector(&store_selector),
            "Updating persisted session title"
        );
        let store = self.kernel.store_manager().open(&store_selector).await?;
        let updated = store.update_session_title(public_id, title).await?;
        Ok(updated
            .as_ref()
            .map(|row| session_summary_from_row_and_selector(row, &store_selector)))
    }

    pub async fn delete_session(&self, session_id: &str) -> Result<bool> {
        let (store_selector, public_id) = persisted_session_target(session_id)?;
        let store = self.kernel.store_manager().open(&store_selector).await?;
        let Some(session) = store.get_session_row_by_public_id(public_id).await? else {
            return Ok(false);
        };
        let mut family = store.list_linked_session_descendants(session.id).await?;
        family.push(session);
        let family_session_ids = family
            .iter()
            .map(|row| {
                let public_id = uuid::Uuid::from_slice(&row.public_id)?.simple().to_string();
                Ok(format_session_reference(&public_id, &store_selector))
            })
            .collect::<Result<Vec<_>>>()?;
        let family_persisted_ids = family
            .iter()
            .map(|row| (store_selector.clone(), row.id))
            .collect::<HashSet<_>>();
        let work_count = self
            .kernel
            .agent_manager()
            .session_family_work_count(&family_session_ids, &family_persisted_ids)
            .await;
        if work_count > 0 {
            return Err(SessionDeleteBusy {
                session_id: session_id.to_string(),
                work_count,
            }
            .into());
        }
        let deleted = store.delete_session_by_public_id(public_id).await?;
        if deleted {
            info!(
                session_id = %session_id,
                store = %describe_store_selector(&store_selector),
                "Deleted persisted session"
            );
        }
        Ok(deleted)
    }

    #[instrument(skip(self), fields(session_id = %session_id))]

    pub(super) async fn resolve_persisted_session(
        &self,
        session_id: &str,
    ) -> Result<Option<(StoreSelector, Arc<StateStore>, SessionRow)>> {
        let (store_selector, public_id) = persisted_session_target(session_id)?;
        let store = self.kernel.store_manager().open(&store_selector).await?;
        let row = store.get_session_row_by_public_id(public_id).await?;
        Ok(row.map(|row| (store_selector, store, row)))
    }

    async fn resolve_live_branch_target(
        &self,
        session_id: &str,
        public_id: &[u8],
        slot_id: Option<&str>,
        action: &str,
    ) -> Result<Option<crate::kernel::agent_manager::LiveSessionSnapshot>> {
        let live = self.live_session_snapshots(public_id).await;
        if let Some(slot_id) = slot_id {
            let Some(snapshot) = live
                .into_iter()
                .find(|snapshot| snapshot.slot_id == slot_id)
            else {
                anyhow::bail!(
                    "Session '{}' is not live in runtime slot '{}'",
                    session_id,
                    slot_id
                );
            };
            ensure_live_session_idle(&snapshot, session_id, Some(slot_id), action)?;
            return Ok(Some(snapshot));
        }

        match live.as_slice() {
            [] => Ok(None),
            [snapshot] => {
                ensure_live_session_idle(snapshot, session_id, None, action)?;
                Ok(Some(snapshot.clone()))
            }
            _ => {
                anyhow::bail!(
                    "Cannot {} for session '{}' while multiple runtime slots are attached; specify slot_id",
                    action,
                    session_id
                );
            }
        }
    }
}

pub(super) fn persisted_session_target(session_id: &str) -> Result<(StoreSelector, Uuid)> {
    let session_ref = parse_session_reference(session_id)?;
    let public_id = Uuid::parse_str(&session_ref.public_id)
        .map_err(|_| anyhow!("Invalid session id '{}'", session_ref.public_id))?;
    // Bare references resolve against the primary state store; cross-store access is explicit.
    let store_selector = session_ref
        .store_selector
        .unwrap_or_else(|| StoreSelector::Alias("state".to_string()));
    Ok((store_selector, public_id))
}

fn ensure_live_session_idle(
    snapshot: &LiveSessionSnapshot,
    session_id: &str,
    slot_id: Option<&str>,
    action: &str,
) -> Result<()> {
    if snapshot.active_tasks == 0 && snapshot.queued_tasks == 0 {
        return Ok(());
    }
    if let Some(slot_id) = slot_id {
        anyhow::bail!(
            "Cannot {} for busy live session '{}' in slot '{}'",
            action,
            session_id,
            slot_id
        );
    }
    anyhow::bail!("Cannot {} for busy live session '{}'", action, session_id);
}

pub(crate) fn session_store_selector_from_filters(
    store: Option<&str>,
    path: Option<&str>,
) -> Result<Option<StoreSelector>> {
    if store.is_some() && path.is_some() {
        anyhow::bail!("Only one of 'store' or 'path' may be supplied");
    }
    if let Some(store) = store {
        let trimmed = store.trim();
        anyhow::ensure!(!trimmed.is_empty(), "'store' must not be empty");
        return Ok(Some(StoreSelector::Alias(trimmed.to_string())));
    }
    if let Some(path) = path {
        let trimmed = path.trim();
        anyhow::ensure!(!trimmed.is_empty(), "'path' must not be empty");
        return Ok(Some(StoreSelector::Path(trimmed.to_string())));
    }
    Ok(None)
}

pub(super) fn session_summary_from_row_and_selector(
    row: &SessionRow,
    selector: &StoreSelector,
) -> SessionSummary {
    let mut summary = super::helpers::session_summary_from_row(row);
    summary.session_id = format_session_reference(&summary.session_id, selector);
    summary
}

fn branch_detail_from_row(row: BranchHeadRow) -> SessionBranchDetail {
    SessionBranchDetail {
        branch_id: super::helpers::format_uuid_bytes_simple(&row.public_id),
        name: row.name,
        head_turn_id: row.head_turn_id,
        head_turn_index: row.head_turn_depth,
        source_turn_id: row.created_from_turn_id,
        origin_kind: row.origin_kind,
        origin_task_id: row.origin_task_id,
        origin_execution_id: row.origin_execution_id,
        origin_metadata: row
            .origin_metadata
            .as_deref()
            .and_then(|raw| serde_json::from_str(raw).ok()),
        active: row.is_active,
        created_at: row.created_at,
    }
}

fn graph_turn_preview(raw: &str) -> Option<String> {
    let value = super::helpers::parse_json_or_string(raw);
    let text = match value {
        serde_json::Value::String(text) => text,
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|part| {
                part.as_object()
                    .and_then(|part| part.get("text").or_else(|| part.get("content")))
                    .and_then(serde_json::Value::as_str)
            })
            .collect::<Vec<_>>()
            .join(" "),
        serde_json::Value::Object(object) => object
            .get("text")
            .or_else(|| object.get("content"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    };
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return None;
    }
    let mut preview = collapsed.chars().take(180).collect::<String>();
    if collapsed.chars().count() > 180 {
        preview.push_str("...");
    }
    Some(preview)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::session_efficiency_from_events;
    use crate::kernel::event::InferenceRequestMetrics;
    use crate::persistence::schema::EventRow;

    #[test]
    fn efficiency_projection_preserves_each_request_in_a_tool_turn() {
        let request = InferenceRequestMetrics {
            provider: "test".to_string(),
            model: "test-model".to_string(),
            requested_context: "default".to_string(),
            resolved_context: "default".to_string(),
            compaction_mode: "hybrid".to_string(),
            estimated_input_tokens_before_compaction: 120,
            estimated_input_tokens: 100,
            system_prompt_tokens: 20,
            message_tokens: 70,
            tool_definition_tokens: 10,
            reusable_prefix_tokens: 65,
            context_window_tokens: 128_000,
            context_window_configured: false,
            input_budget_tokens: 123_904,
            max_output_tokens: Some(4_096),
            thinking_budget_tokens: None,
            available_message_count: 4,
            sent_message_count: 4,
            has_prior_history: false,
            checkpoint_covered_through_turn_id: None,
            truncated_tool_results: 0,
            dropped_messages: 0,
            estimated_payload_bytes: 400,
        };
        let events = vec![
            efficiency_event(
                1,
                "inference_request",
                json!({ "metrics": request.clone() }),
            ),
            efficiency_event(
                2,
                "message_end",
                json!({
                    "input_tokens": 101,
                    "output_tokens": 12,
                    "cache_read_input_tokens": 80
                }),
            ),
            efficiency_event(
                3,
                "inference_request",
                json!({ "metrics": InferenceRequestMetrics {
                    estimated_input_tokens: 140,
                    ..request.clone()
                } }),
            ),
            efficiency_event(
                4,
                "message_end",
                json!({
                    "input_tokens": 143,
                    "output_tokens": 29,
                    "cache_read_input_tokens": 60,
                    "cache_creation_input_tokens": 20
                }),
            ),
        ];

        let efficiency = session_efficiency_from_events(events, None);
        assert_eq!(efficiency.turns.len(), 1);
        assert_eq!(efficiency.total_request_count, 2);
        assert_eq!(efficiency.turns[0].requests.len(), 2);
        assert_eq!(efficiency.turns[0].requests[0].input_tokens, Some(101));
        assert_eq!(efficiency.turns[0].requests[1].input_tokens, Some(143));
        assert_eq!(efficiency.total_input_tokens, 244);
        assert_eq!(efficiency.total_output_tokens, 41);
        assert_eq!(efficiency.total_cache_read_input_tokens, 140);
        assert_eq!(efficiency.total_cache_creation_input_tokens, 20);
        assert!(efficiency.provider_cache_metrics_available);
        assert_eq!(
            efficiency.turns[0].requests[0].cache_read_input_tokens,
            Some(80)
        );
        assert_eq!(
            efficiency.turns[0].requests[1].cache_creation_input_tokens,
            Some(20)
        );
    }

    fn efficiency_event(id: i64, event_type: &str, payload: serde_json::Value) -> EventRow {
        EventRow {
            id,
            session_id: 1,
            turn_id: Some(1),
            event_type: event_type.to_string(),
            payload: payload.to_string(),
            turn_index: Some(1),
            created_at: format!("2026-08-10T00:00:0{id}Z"),
        }
    }
}
