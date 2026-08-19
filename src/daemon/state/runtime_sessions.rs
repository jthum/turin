use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Result, anyhow};
use tracing::{debug, info, instrument};
use turin_daemon_protocol::SessionSearchScope;
use uuid::Uuid;

use super::{
    DaemonState, SessionBranchDetail, SessionCompactionDetail, SessionDetail,
    SessionEfficiencyDetail, SessionEventDetail, SessionExecutionContextDetail,
    SessionExecutionDetail, SessionGraphDetail, SessionGraphTurnDetail, SessionMessageDetail,
    SessionMessageWindow, SessionPlanExecutionDetail, SessionRequestEfficiencyDetail,
    SessionSearchHit, SessionSummary, SessionTaskExecutionDetail, SessionTaskTurnDetail,
    SessionToolExecutionDetail, SessionTurnEfficiencyDetail,
};
use crate::kernel::agent_manager::LiveSessionSnapshot;
use crate::kernel::event::{
    InferenceRequestMetrics, KernelEvent, LifecycleEvent, TaskTerminalStatus,
};
use crate::kernel::session::ExecutionStatusSnapshot;
use crate::kernel::session_refs::{
    describe_store_selector, format_session_reference, parse_session_reference,
};
use crate::perf_diagnostics::{perf_stage, perf_stage_finish};
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::{BranchHeadRow, SessionRow};
use crate::persistence::state::{SessionReadTarget, StateStore};

const SESSION_EXECUTION_EVENT_LIMIT: usize = 400;

#[derive(Debug, thiserror::Error)]
#[error("Session '{session_id}' is still open in {runtime_count} runtime slot(s)")]
pub(crate) struct SessionDeleteBusy {
    session_id: String,
    runtime_count: usize,
}

impl DaemonState {
    #[instrument(skip(self), fields(store = %store_selector_label_opt(store_selector.as_ref())))]
    pub async fn list_sessions(
        &self,
        limit: usize,
        offset: usize,
        store_selector: Option<StoreSelector>,
    ) -> Result<Vec<SessionSummary>> {
        // Session listing is intentionally single-store scoped by default. When no explicit
        // selector is supplied, that store is the primary `state` target.
        let store = match &store_selector {
            Some(selector) => self.kernel.store_manager().open(selector).await?,
            None => self.kernel.store_manager().get_default().await?,
        };
        let rows = store.list_session_rows(limit, offset).await?;
        debug!(count = rows.len(), "Listed persisted sessions");
        Ok(rows
            .iter()
            .map(|row| match &store_selector {
                Some(selector) => session_summary_from_row_and_selector(row, selector),
                None => super::helpers::session_summary_from_row(row),
            })
            .collect())
    }

    #[instrument(
        skip(self, query),
        fields(
            scope = ?scope,
            store = %store_selector_label_opt(store_selector.as_ref())
        )
    )]
    pub async fn search_sessions(
        &self,
        query: &str,
        scope: SessionSearchScope,
        limit: usize,
        offset: usize,
        store_selector: Option<StoreSelector>,
    ) -> Result<Vec<SessionSearchHit>> {
        // Session history search is intentionally single-store scoped by default. When no
        // explicit selector is supplied, that store is the primary `state` target.
        let store = match &store_selector {
            Some(selector) => self.kernel.store_manager().open(selector).await?,
            None => self.kernel.store_manager().get_default().await?,
        };
        let rows = store
            .search_session_history(query, scope, limit, offset)
            .await?;
        debug!(count = rows.len(), "Searched persisted session history");
        Ok(rows
            .into_iter()
            .map(|row| {
                let title = super::helpers::session_title_from_metadata(row.metadata.as_deref());
                let session_id = super::helpers::format_uuid_bytes_simple(&row.public_id);
                SessionSearchHit {
                    kind: row.kind,
                    score: row.score,
                    session_id: match &store_selector {
                        Some(selector) => format_session_reference(&session_id, selector),
                        None => session_id,
                    },
                    agent_id: row.agent_id,
                    title,
                    created_at: row.created_at,
                    turn_index: row.turn_index,
                    role: row.role,
                    tool_name: row.tool_name,
                    event_type: row.event_type,
                    summary: summarize_search_hit(&row.match_text, query),
                    snippet: excerpt_search_text(&row.match_text, query, 220),
                }
            })
            .collect())
    }

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
        let public_id_bytes = public_id.into_bytes().to_vec();
        let live = self.live_session_snapshots(&public_id_bytes).await;
        if !live.is_empty() {
            return Err(SessionDeleteBusy {
                session_id: session_id.to_string(),
                runtime_count: live.len(),
            }
            .into());
        }
        let store = self.kernel.store_manager().open(&store_selector).await?;
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
    pub async fn list_session_branches(
        &self,
        session_id: &str,
    ) -> Result<Option<Vec<SessionBranchDetail>>> {
        let Some((store_selector, store, row)) = self.resolve_persisted_session(session_id).await?
        else {
            return Ok(None);
        };
        debug!(
            store = %describe_store_selector(&store_selector),
            "Listing session branches"
        );
        let branches = store
            .list_branch_heads(row.id)
            .await?
            .into_iter()
            .map(branch_detail_from_row)
            .collect();
        Ok(Some(branches))
    }

    #[instrument(skip(self), fields(session_id = %session_id))]
    pub async fn get_session_graph(&self, session_id: &str) -> Result<Option<SessionGraphDetail>> {
        let Some((store_selector, store, row)) = self.resolve_persisted_session(session_id).await?
        else {
            return Ok(None);
        };
        let turns = store
            .list_session_graph_turns(row.id)
            .await?
            .into_iter()
            .map(|graph_turn| SessionGraphTurnDetail {
                turn_id: graph_turn.turn.id,
                turn_public_id: super::helpers::format_uuid_bytes_simple(
                    &graph_turn.turn.public_id,
                ),
                parent_turn_id: graph_turn.turn.parent_turn_id,
                turn_index: graph_turn.turn.branch_depth,
                message_count: graph_turn.message_count,
                tool_execution_count: graph_turn.tool_execution_count,
                preview: graph_turn.preview.as_deref().and_then(graph_turn_preview),
                created_at: graph_turn.turn.created_at,
            })
            .collect();
        let branches = store
            .list_branch_heads(row.id)
            .await?
            .into_iter()
            .map(branch_detail_from_row)
            .collect();
        Ok(Some(SessionGraphDetail {
            session: session_summary_from_row_and_selector(&row, &store_selector),
            turns,
            branches,
        }))
    }

    #[instrument(skip(self), fields(session_id = %session_id, source_turn_id = source_turn_id))]
    pub async fn list_session_branch_siblings(
        &self,
        session_id: &str,
        source_turn_id: i64,
    ) -> Result<Option<Vec<SessionBranchDetail>>> {
        let Some((store_selector, store, row)) = self.resolve_persisted_session(session_id).await?
        else {
            return Ok(None);
        };
        debug!(
            store = %describe_store_selector(&store_selector),
            "Listing session branch siblings"
        );
        let branches = store
            .list_branch_heads_from_source_turn(row.id, source_turn_id)
            .await?
            .into_iter()
            .map(branch_detail_from_row)
            .collect();
        Ok(Some(branches))
    }

    #[instrument(
        skip(self),
        fields(
            session_id = %session_id,
            branch = %name,
            slot_id = ?slot_id,
            from_turn_index = ?from_turn_index,
            activate = activate
        )
    )]
    pub async fn create_session_branch(
        &self,
        session_id: &str,
        name: &str,
        slot_id: Option<&str>,
        from_turn_index: Option<u32>,
        activate: bool,
    ) -> Result<Option<SessionBranchDetail>> {
        let Some((store_selector, store, row)) = self.resolve_persisted_session(session_id).await?
        else {
            return Ok(None);
        };
        let live_snapshot = if activate {
            self.resolve_live_branch_target(session_id, &row.public_id, slot_id, "activate branch")
                .await?
        } else {
            None
        };
        debug!(
            store = %describe_store_selector(&store_selector),
            live_session = live_snapshot.is_some(),
            "Creating session branch"
        );
        let branch = store
            .create_branch_head_from_turn_index(row.id, name, from_turn_index, activate)
            .await?;
        if activate && let Some(live_snapshot) = live_snapshot.as_ref() {
            self.kernel
                .agent_manager()
                .reload_session(session_id, Some(&live_snapshot.slot_id))
                .await?;
        }
        info!(
            session_id = %session_id,
            store = %describe_store_selector(&store_selector),
            branch = %branch.name,
            activate = activate,
            reloaded_live_session = live_snapshot.is_some(),
            "Created session branch"
        );
        Ok(Some(branch_detail_from_row(branch)))
    }

    #[instrument(
        skip(self),
        fields(
            session_id = %session_id,
            branch = %name,
            slot_id = ?slot_id,
            from_turn_id = from_turn_id,
            activate = activate
        )
    )]
    pub async fn create_session_branch_from_turn_id(
        &self,
        session_id: &str,
        name: &str,
        slot_id: Option<&str>,
        from_turn_id: i64,
        activate: bool,
    ) -> Result<Option<SessionBranchDetail>> {
        let Some((store_selector, store, row)) = self.resolve_persisted_session(session_id).await?
        else {
            return Ok(None);
        };
        let live_snapshot = if activate {
            self.resolve_live_branch_target(session_id, &row.public_id, slot_id, "activate branch")
                .await?
        } else {
            None
        };
        let branch = store
            .create_branch_head_from_turn_id(row.id, name, from_turn_id, activate)
            .await?;
        if activate && let Some(live_snapshot) = live_snapshot.as_ref() {
            self.kernel
                .agent_manager()
                .reload_session(session_id, Some(&live_snapshot.slot_id))
                .await?;
        }
        info!(
            session_id = %session_id,
            store = %describe_store_selector(&store_selector),
            branch = %branch.name,
            from_turn_id = from_turn_id,
            activate = activate,
            reloaded_live_session = live_snapshot.is_some(),
            "Created session branch from exact turn"
        );
        Ok(Some(branch_detail_from_row(branch)))
    }

    #[instrument(skip(self), fields(session_id = %session_id, branch = %branch, slot_id = ?slot_id))]
    pub async fn checkout_session_branch(
        &self,
        session_id: &str,
        branch: &str,
        slot_id: Option<&str>,
    ) -> Result<Option<SessionBranchDetail>> {
        let Some((store_selector, store, row)) = self.resolve_persisted_session(session_id).await?
        else {
            return Ok(None);
        };
        let live_snapshot = self
            .resolve_live_branch_target(session_id, &row.public_id, slot_id, "check out branch")
            .await?;
        debug!(
            store = %describe_store_selector(&store_selector),
            live_session = live_snapshot.is_some(),
            "Checking out session branch"
        );
        let branch = if let Ok(branch_id) = Uuid::parse_str(branch) {
            store
                .checkout_branch_head_by_public_id(row.id, branch_id)
                .await?
        } else {
            store.checkout_branch_head_by_name(row.id, branch).await?
        };
        if branch.is_some()
            && let Some(live_snapshot) = live_snapshot.as_ref()
        {
            self.kernel
                .agent_manager()
                .reload_session(session_id, Some(&live_snapshot.slot_id))
                .await?;
        }
        if let Some(branch) = &branch {
            info!(
                session_id = %session_id,
                store = %describe_store_selector(&store_selector),
                branch = %branch.name,
                reloaded_live_session = live_snapshot.is_some(),
                "Checked out session branch"
            );
        }
        Ok(branch.map(branch_detail_from_row))
    }

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

fn session_execution_from_events(
    events: Vec<crate::persistence::schema::EventRow>,
    truncated: bool,
) -> SessionExecutionDetail {
    let mut tasks = Vec::<SessionTaskExecutionDetail>::new();
    let mut task_indexes = HashMap::<String, usize>::new();
    let mut plan_completions = BTreeMap::<String, SessionPlanExecutionDetail>::new();

    for event in events {
        let Ok(KernelEvent::Lifecycle(lifecycle)) = serde_json::from_str(&event.payload) else {
            continue;
        };
        match lifecycle {
            LifecycleEvent::TaskStart {
                identity,
                task_id,
                trace_id,
                plan_id,
                title,
                prompt,
                queue_depth,
                execution,
            } => {
                let index = tasks.len();
                task_indexes.insert(task_id.clone(), index);
                tasks.push(SessionTaskExecutionDetail {
                    task_id,
                    trace_id,
                    plan_id,
                    run_id: identity.run_id().map(str::to_string),
                    agent_id: identity.agent_id().to_string(),
                    title,
                    prompt,
                    status: "running".to_string(),
                    queue_depth,
                    task_turn_count: 0,
                    execution: execution_context_detail(execution),
                    turns: Vec::new(),
                    branch_outcome: None,
                    error: None,
                    started_at: event.created_at,
                    completed_at: None,
                });
            }
            LifecycleEvent::TaskComplete {
                identity,
                task_id,
                trace_id,
                plan_id,
                status,
                task_turn_count,
                execution,
                branch_outcome,
                error,
            } => {
                let index = if let Some(index) = task_indexes.get(&task_id).copied() {
                    index
                } else {
                    let index = tasks.len();
                    task_indexes.insert(task_id.clone(), index);
                    tasks.push(SessionTaskExecutionDetail {
                        task_id: task_id.clone(),
                        trace_id: trace_id.clone(),
                        plan_id: plan_id.clone(),
                        run_id: identity.run_id().map(str::to_string),
                        agent_id: identity.agent_id().to_string(),
                        title: None,
                        prompt: String::new(),
                        status: task_terminal_status(status).to_string(),
                        queue_depth: 0,
                        task_turn_count,
                        execution: execution_context_detail(execution.clone()),
                        turns: Vec::new(),
                        branch_outcome: None,
                        error: None,
                        started_at: event.created_at.clone(),
                        completed_at: None,
                    });
                    index
                };
                let task = &mut tasks[index];
                task.trace_id = trace_id;
                task.plan_id = plan_id;
                task.status = task_terminal_status(status).to_string();
                task.task_turn_count = task_turn_count;
                task.execution = execution_context_detail(execution);
                task.branch_outcome =
                    branch_outcome.and_then(|outcome| serde_json::to_value(outcome).ok());
                task.error = error;
                task.completed_at = Some(event.created_at);
            }
            LifecycleEvent::TurnStart {
                turn_index,
                task_id,
                task_turn_index,
                ..
            } => {
                let Some(index) = task_indexes.get(&task_id).copied() else {
                    continue;
                };
                tasks[index].turns.push(SessionTaskTurnDetail {
                    turn_index,
                    task_turn_index,
                    has_tool_calls: None,
                    started_at: event.created_at,
                    completed_at: None,
                });
            }
            LifecycleEvent::TurnEnd {
                turn_index,
                task_id,
                task_turn_index,
                has_tool_calls,
                ..
            } => {
                let Some(index) = task_indexes.get(&task_id).copied() else {
                    continue;
                };
                let task = &mut tasks[index];
                if let Some(turn) = task
                    .turns
                    .iter_mut()
                    .rev()
                    .find(|turn| turn.turn_index == turn_index)
                {
                    turn.has_tool_calls = Some(has_tool_calls);
                    turn.completed_at = Some(event.created_at);
                } else {
                    task.turns.push(SessionTaskTurnDetail {
                        turn_index,
                        task_turn_index,
                        has_tool_calls: Some(has_tool_calls),
                        started_at: event.created_at.clone(),
                        completed_at: Some(event.created_at),
                    });
                }
            }
            LifecycleEvent::PlanComplete {
                plan_id,
                title,
                total_tasks,
                completed_tasks,
                ..
            } => {
                plan_completions.insert(
                    plan_id.clone(),
                    SessionPlanExecutionDetail {
                        plan_id,
                        title: Some(title),
                        status: "complete".to_string(),
                        total_tasks,
                        completed_tasks,
                        started_at: event.created_at.clone(),
                        completed_at: Some(event.created_at),
                    },
                );
            }
            _ => {}
        }
    }

    for task in &tasks {
        let Some(plan_id) = task.plan_id.as_ref() else {
            continue;
        };
        let plan =
            plan_completions
                .entry(plan_id.clone())
                .or_insert_with(|| SessionPlanExecutionDetail {
                    plan_id: plan_id.clone(),
                    title: None,
                    status: "running".to_string(),
                    total_tasks: 0,
                    completed_tasks: 0,
                    started_at: task.started_at.clone(),
                    completed_at: None,
                });
        if plan.completed_at.is_none() {
            plan.total_tasks = plan.total_tasks.saturating_add(1);
            if task.status != "running" {
                plan.completed_tasks = plan.completed_tasks.saturating_add(1);
            }
            if task.started_at < plan.started_at {
                plan.started_at.clone_from(&task.started_at);
            }
        }
    }

    tasks.sort_by(|left, right| {
        let left_date = left.completed_at.as_ref().unwrap_or(&left.started_at);
        let right_date = right.completed_at.as_ref().unwrap_or(&right.started_at);
        right_date.cmp(left_date)
    });
    let mut plans = plan_completions.into_values().collect::<Vec<_>>();
    plans.sort_by(|left, right| right.started_at.cmp(&left.started_at));

    SessionExecutionDetail {
        tasks,
        plans,
        event_limit: SESSION_EXECUTION_EVENT_LIMIT,
        truncated,
    }
}

fn execution_context_detail(execution: ExecutionStatusSnapshot) -> SessionExecutionContextDetail {
    SessionExecutionContextDetail {
        execution_id: execution.execution_id,
        context_target: serde_json::to_value(execution.context_target)
            .unwrap_or(serde_json::Value::Null),
        visibility: serialized_enum_name(&execution.visibility),
        durability: serialized_enum_name(&execution.durability),
        write_policy: serialized_enum_name(&execution.write_policy),
    }
}

fn serialized_enum_name(value: &impl serde::Serialize) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
        .unwrap_or_else(|| "unknown".to_string())
}

fn task_terminal_status(status: TaskTerminalStatus) -> &'static str {
    match status {
        TaskTerminalStatus::Success => "success",
        TaskTerminalStatus::Rejected => "rejected",
        TaskTerminalStatus::Conflict => "conflict",
        TaskTerminalStatus::MaxTurns => "max_turns",
        TaskTerminalStatus::Error => "error",
        TaskTerminalStatus::Cancelled => "cancelled",
        TaskTerminalStatus::TimedOut => "timed_out",
        TaskTerminalStatus::Killed => "killed",
    }
}

fn session_efficiency_from_events(
    events: Vec<crate::persistence::schema::EventRow>,
    visible_turn_indexes: Option<&HashSet<u32>>,
) -> SessionEfficiencyDetail {
    let mut turns = BTreeMap::<u32, SessionTurnEfficiencyDetail>::new();
    let mut latest_compaction = None;
    let mut total_input_tokens = 0_u64;
    let mut total_output_tokens = 0_u64;
    let mut total_cache_read_input_tokens = 0_u64;
    let mut total_cache_creation_input_tokens = 0_u64;
    let mut total_request_count = 0_usize;
    let mut provider_cache_metrics_available = false;

    for event in events {
        let Ok(payload) = serde_json::from_str::<serde_json::Value>(&event.payload) else {
            continue;
        };
        match event.event_type.as_str() {
            "inference_request" => {
                total_request_count = total_request_count.saturating_add(1);
                let (Some(turn_index), Some(metrics)) = (
                    event.turn_index,
                    payload.get("metrics").cloned().and_then(|value| {
                        serde_json::from_value::<InferenceRequestMetrics>(value).ok()
                    }),
                ) else {
                    continue;
                };
                if visible_turn_indexes.is_some_and(|visible| !visible.contains(&turn_index)) {
                    continue;
                }
                let turn = turns
                    .entry(turn_index)
                    .or_insert_with(|| empty_turn_efficiency(turn_index, event.created_at.clone()));
                turn.requests.push(SessionRequestEfficiencyDetail {
                    metrics: Some(metrics),
                    input_tokens: None,
                    output_tokens: None,
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                    created_at: event.created_at.clone(),
                });
                turn.created_at = event.created_at;
            }
            "message_end" => {
                let Some(turn_index) = event.turn_index else {
                    continue;
                };
                let input_tokens = payload
                    .get("input_tokens")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let output_tokens = payload
                    .get("output_tokens")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let cache_read_input_tokens = payload
                    .get("cache_read_input_tokens")
                    .and_then(serde_json::Value::as_u64);
                let cache_creation_input_tokens = payload
                    .get("cache_creation_input_tokens")
                    .and_then(serde_json::Value::as_u64);
                total_input_tokens = total_input_tokens.saturating_add(input_tokens);
                total_output_tokens = total_output_tokens.saturating_add(output_tokens);
                if let Some(tokens) = cache_read_input_tokens {
                    provider_cache_metrics_available = true;
                    total_cache_read_input_tokens =
                        total_cache_read_input_tokens.saturating_add(tokens);
                }
                if let Some(tokens) = cache_creation_input_tokens {
                    provider_cache_metrics_available = true;
                    total_cache_creation_input_tokens =
                        total_cache_creation_input_tokens.saturating_add(tokens);
                }
                if visible_turn_indexes.is_some_and(|visible| !visible.contains(&turn_index)) {
                    continue;
                }
                let turn = turns
                    .entry(turn_index)
                    .or_insert_with(|| empty_turn_efficiency(turn_index, event.created_at.clone()));
                turn.input_tokens = turn.input_tokens.saturating_add(input_tokens);
                turn.output_tokens = turn.output_tokens.saturating_add(output_tokens);
                if let Some(request) = turn
                    .requests
                    .iter_mut()
                    .rev()
                    .find(|request| request.input_tokens.is_none())
                {
                    request.input_tokens = Some(input_tokens);
                    request.output_tokens = Some(output_tokens);
                    request.cache_read_input_tokens = cache_read_input_tokens;
                    request.cache_creation_input_tokens = cache_creation_input_tokens;
                } else {
                    turn.requests.push(SessionRequestEfficiencyDetail {
                        metrics: None,
                        input_tokens: Some(input_tokens),
                        output_tokens: Some(output_tokens),
                        cache_read_input_tokens,
                        cache_creation_input_tokens,
                        created_at: event.created_at.clone(),
                    });
                }
                turn.created_at = event.created_at;
            }
            "context_compaction" => {
                let Some(checkpoint) = payload.get("checkpoint") else {
                    continue;
                };
                latest_compaction = Some(SessionCompactionDetail {
                    covered_through_turn_id: checkpoint
                        .get("covered_through_turn_id")
                        .and_then(serde_json::Value::as_i64)
                        .unwrap_or(0),
                    covered_through_turn_index: checkpoint
                        .get("covered_through_turn_index")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0) as u32,
                    generated_at_turn_index: checkpoint
                        .get("generated_at_turn_index")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0) as u32,
                    provider: checkpoint
                        .get("provider_name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("unknown")
                        .to_string(),
                    model: checkpoint
                        .get("model")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("unknown")
                        .to_string(),
                    created_at: event.created_at,
                });
            }
            _ => {}
        }
    }

    let turns = turns.into_values().collect::<Vec<_>>();
    SessionEfficiencyDetail {
        total_input_tokens,
        total_output_tokens,
        total_cache_read_input_tokens,
        total_cache_creation_input_tokens,
        total_request_count,
        turns,
        latest_compaction,
        provider_cache_metrics_available,
    }
}

fn empty_turn_efficiency(turn_index: u32, created_at: String) -> SessionTurnEfficiencyDetail {
    SessionTurnEfficiencyDetail {
        turn_index,
        requests: Vec::new(),
        input_tokens: 0,
        output_tokens: 0,
        created_at,
    }
}

pub(super) fn persisted_session_target(session_id: &str) -> Result<(StoreSelector, Uuid)> {
    let session_ref = parse_session_reference(session_id)?;
    let public_id = Uuid::parse_str(&session_ref.public_id)
        .map_err(|_| anyhow!("Invalid session id '{}'", session_ref.public_id))?;
    // A bare persisted session reference is interpreted against the primary `state` store.
    // Cross-state access must qualify the session id explicitly, e.g. `<session>@telegram`.
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

fn store_selector_label_opt(selector: Option<&StoreSelector>) -> String {
    selector
        .map(describe_store_selector)
        .unwrap_or_else(|| "state".to_string())
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

fn session_summary_from_row_and_selector(
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

fn summarize_search_hit(text: &str, query: &str) -> String {
    excerpt_search_text(text, query, 72)
}

fn excerpt_search_text(text: &str, query: &str, max_chars: usize) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = collapsed.trim();
    if trimmed.chars().count() <= max_chars {
        return trimmed.to_string();
    }

    let normalized_query = query.trim().to_ascii_lowercase();
    if normalized_query.is_empty() {
        let excerpt = trimmed.chars().take(max_chars).collect::<String>();
        return format!("{excerpt}…");
    }

    let lower_trimmed = trimmed.to_ascii_lowercase();
    if let Some(byte_index) = lower_trimmed.find(&normalized_query) {
        let match_start = trimmed[..byte_index].chars().count();
        let match_len = normalized_query.chars().count();
        let context_before = max_chars / 3;
        let context_after = max_chars.saturating_sub(context_before + match_len);
        let start = match_start.saturating_sub(context_before);
        let end = (match_start + match_len + context_after).min(trimmed.chars().count());
        let mut excerpt = slice_chars(trimmed, start, end).trim().to_string();
        if start > 0 {
            excerpt = format!("…{excerpt}");
        }
        if end < trimmed.chars().count() {
            excerpt.push('…');
        }
        return excerpt;
    }

    let excerpt = trimmed.chars().take(max_chars).collect::<String>();
    format!("{excerpt}…")
}

fn slice_chars(value: &str, start: usize, end: usize) -> String {
    value
        .chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{excerpt_search_text, session_efficiency_from_events};
    use crate::kernel::event::InferenceRequestMetrics;
    use crate::persistence::schema::EventRow;

    #[test]
    fn search_excerpt_centers_query_when_possible() {
        let text = "prefix context before the actual match lands on compiler panic in src/main.rs";
        let excerpt = excerpt_search_text(text, "compiler", 28);
        assert!(excerpt.contains("compiler"));
        assert!(excerpt.starts_with('…'));
        assert!(excerpt.ends_with('…'));
    }

    #[test]
    fn search_excerpt_falls_back_to_prefix_without_query() {
        let text = "alpha beta gamma delta epsilon";
        let excerpt = excerpt_search_text(text, "", 12);
        assert_eq!(excerpt, "alpha beta g…");
    }

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
