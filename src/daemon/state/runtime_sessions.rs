use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use tracing::{debug, info, instrument};
use turin_daemon_protocol::SessionSearchScope;
use uuid::Uuid;

use super::{
    DaemonState, SessionBranchDetail, SessionDetail, SessionEventDetail, SessionMessageDetail,
    SessionMessageWindow, SessionSearchHit, SessionSummary, SessionToolExecutionDetail,
};
use crate::kernel::agent_manager::LiveSessionSnapshot;
use crate::kernel::session_refs::{
    describe_store_selector, format_session_reference, parse_session_reference,
};
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::{BranchHeadRow, SessionRow};
use crate::persistence::state::{SessionReadTarget, StateStore};

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
        self.get_session_projection(session_id, None, true).await
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
        message_limit: Option<usize>,
        include_events: bool,
    ) -> Result<Option<SessionDetail>> {
        let Some((store_selector, store, row)) = self.resolve_persisted_session(session_id).await?
        else {
            return Ok(None);
        };
        debug!(
            store = %describe_store_selector(&store_selector),
            "Resolved persisted session detail target"
        );
        let active_branch = SessionReadTarget::ActiveBranch;

        let events = if include_events {
            store
                .get_events(row.id, &active_branch)
                .await?
                .into_iter()
                .map(|event| SessionEventDetail {
                    id: event.id,
                    event_type: event.event_type,
                    payload: super::helpers::parse_json_or_string(&event.payload),
                    created_at: event.created_at,
                })
                .collect()
        } else {
            Vec::new()
        };

        let (persisted_messages, total_messages) = match message_limit {
            Some(limit) => {
                store
                    .get_recent_messages(row.id, &active_branch, limit)
                    .await?
            }
            None => {
                let messages = store.get_messages(row.id, &active_branch).await?;
                let total = messages.len();
                (messages, total)
            }
        };
        let message_offset = total_messages.saturating_sub(persisted_messages.len());
        let messages = persisted_messages
            .into_iter()
            .map(|message| SessionMessageDetail {
                id: message.id,
                turn_index: message.turn_index,
                role: message.role,
                content: super::helpers::parse_json_or_string(&message.content),
                token_count: message.token_count,
                created_at: message.created_at,
            })
            .collect::<Vec<_>>();

        let visible_turn_indexes = message_limit.map(|_| {
            messages
                .iter()
                .map(|message| message.turn_index)
                .collect::<HashSet<_>>()
        });

        let tool_executions = store
            .get_tool_executions_for_turn_indexes(
                row.id,
                &active_branch,
                visible_turn_indexes.as_ref(),
            )
            .await?
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
            .collect();

        let branches = store
            .list_branch_heads(row.id)
            .await?
            .into_iter()
            .map(branch_detail_from_row)
            .collect();

        Ok(Some(SessionDetail {
            session: session_summary_from_row_and_selector(&row, &store_selector),
            branches,
            events,
            messages,
            tool_executions,
            message_window: message_limit.map(|_| SessionMessageWindow {
                offset: message_offset,
                total: total_messages,
            }),
        }))
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
    use super::excerpt_search_text;

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
}
