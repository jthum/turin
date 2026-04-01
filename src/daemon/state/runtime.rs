use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Result, anyhow};
use turin_daemon_protocol::SessionSearchScope;
use uuid::Uuid;

use super::{
    DaemonState, SessionBranchDetail, SessionDetail, SessionEventDetail, SessionMessageDetail,
    SessionSearchHit, SessionSummary, SessionToolExecutionDetail,
};
use crate::kernel::agent_manager::{AgentStatusSnapshot, TaskStatusSnapshot};
use crate::kernel::event::KernelEvent;
use crate::kernel::session::QueuedTask;
use crate::kernel::session_refs::{format_session_reference, parse_session_reference};
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::{BranchHeadRow, SessionRow};
use turin_types::ToolsConfig;

impl DaemonState {
    pub async fn agent_runtime_status(
        &self,
        agent_id: &str,
    ) -> Result<Option<AgentStatusSnapshot>> {
        Ok(self
            .list_agent_runtime_statuses()
            .await
            .into_iter()
            .find(|status| status.agent_id == agent_id))
    }

    pub async fn submit_task(
        &self,
        agent_id: Option<&str>,
        session_id: Option<&str>,
        prompt: String,
        tools: Option<ToolsConfig>,
    ) -> Result<TaskStatusSnapshot> {
        let mut task = QueuedTask::ad_hoc(prompt);
        if let Some(tools) = tools
            && !tools.is_empty()
        {
            task.tools = Some(tools);
        }
        let request_id = if let Some(session_id) = session_id {
            self.kernel
                .agent_manager()
                .submit_to_session(session_id, task, None)
                .await?
        } else {
            let agent_id =
                agent_id.ok_or_else(|| anyhow!("task.submit requires agent_id or session_id"))?;
            self.ensure_enabled_agent(agent_id)?;
            self.kernel
                .agent_manager()
                .submit(agent_id, task, None)
                .await?
        };
        self.kernel
            .agent_manager()
            .get_task(&request_id)
            .await
            .ok_or_else(|| anyhow!("Task '{}' was submitted but is not visible", request_id))
    }

    pub async fn list_tasks(&self) -> Vec<TaskStatusSnapshot> {
        self.kernel.agent_manager().list_tasks().await
    }

    pub async fn get_task(&self, request_id: &str) -> Option<TaskStatusSnapshot> {
        self.kernel.agent_manager().get_task(request_id).await
    }

    pub async fn cancel_task(&self, request_id: &str) -> Result<TaskStatusSnapshot> {
        self.kernel.agent_manager().cancel_task(request_id).await
    }

    pub async fn wait_for_task(
        &self,
        request_id: &str,
        timeout_ms: Option<u64>,
    ) -> Result<TaskStatusSnapshot> {
        let Some(initial) = self.get_task(request_id).await else {
            anyhow::bail!("Task '{}' not found", request_id);
        };
        if !matches!(initial.state.as_str(), "queued" | "running" | "cancelling") {
            return Ok(initial);
        }

        let deadline = timeout_ms.map(|ms| tokio::time::Instant::now() + Duration::from_millis(ms));
        loop {
            if let Some(snapshot) = self.get_task(request_id).await {
                if !matches!(snapshot.state.as_str(), "queued" | "running" | "cancelling") {
                    return Ok(snapshot);
                }
            } else {
                anyhow::bail!("Task '{}' disappeared while waiting", request_id);
            }

            if let Some(deadline) = deadline
                && tokio::time::Instant::now() >= deadline
            {
                anyhow::bail!("Timed out waiting for task '{}'", request_id);
            }

            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    pub async fn list_sessions(&self, limit: usize, offset: usize) -> Result<Vec<SessionSummary>> {
        // Session listing is intentionally primary-state scoped by default.
        // Cross-state aggregation is a higher-level UI concern rather than a core daemon default.
        let store = self.kernel.store_manager().get_default().await?;
        let rows = store.list_session_rows(limit, offset).await?;
        Ok(rows
            .iter()
            .map(super::helpers::session_summary_from_row)
            .collect())
    }

    pub async fn search_sessions(
        &self,
        query: &str,
        scope: SessionSearchScope,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<SessionSearchHit>> {
        // Session history search is intentionally primary-state scoped by default.
        // Store-qualified session references are supported elsewhere, but global multi-state
        // aggregation should remain explicit rather than implicit here.
        let store = self.kernel.store_manager().get_default().await?;
        let rows = store
            .search_session_history(query, scope, limit, offset)
            .await?;
        Ok(rows
            .into_iter()
            .map(|row| {
                let title = super::helpers::session_title_from_metadata(row.metadata.as_deref());
                SessionSearchHit {
                    kind: row.kind,
                    score: row.score,
                    session_id: super::helpers::format_uuid_bytes_simple(&row.public_id),
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

    pub async fn list_live_sessions(
        &self,
    ) -> Vec<crate::kernel::agent_manager::LiveSessionSnapshot> {
        self.kernel.agent_manager().list_live_sessions(None).await
    }

    pub async fn subscribe_live_session_events(
        &self,
        session_id: &str,
    ) -> Option<(
        String,
        tokio::sync::broadcast::Receiver<(Option<i64>, KernelEvent)>,
    )> {
        self.kernel
            .agent_manager()
            .subscribe_session_events(session_id)
            .await
    }

    pub async fn open_session(
        &self,
        agent_id: &str,
        slot_id: Option<&str>,
        channel_id: Option<&str>,
    ) -> Result<crate::kernel::agent_manager::LiveSessionSnapshot> {
        self.ensure_enabled_agent(agent_id)?;
        let initial_state_selector = self.resolve_channel_state_selector(channel_id)?;
        let initial_default_store_selector =
            self.resolve_channel_default_store_selector(channel_id)?;
        self.kernel
            .agent_manager()
            .open_session(
                agent_id,
                slot_id,
                initial_state_selector,
                initial_default_store_selector,
            )
            .await
    }

    pub async fn resume_session(
        &self,
        session_id: &str,
        slot_id: Option<&str>,
    ) -> Result<crate::kernel::agent_manager::LiveSessionSnapshot> {
        self.kernel
            .agent_manager()
            .resume_session(session_id, slot_id)
            .await
    }

    pub async fn get_session(&self, session_id: &str) -> Result<Option<SessionDetail>> {
        let Some((store_selector, row)) = self.resolve_persisted_session(session_id).await? else {
            return Ok(None);
        };
        let store = self.kernel.store_manager().open(&store_selector).await?;

        let events = store
            .get_events(row.id)
            .await?
            .into_iter()
            .map(|event| SessionEventDetail {
                id: event.id,
                event_type: event.event_type,
                payload: super::helpers::parse_json_or_string(&event.payload),
                created_at: event.created_at,
            })
            .collect();

        let messages = store
            .get_messages(row.id)
            .await?
            .into_iter()
            .map(|message| SessionMessageDetail {
                id: message.id,
                turn_index: message.turn_index,
                role: message.role,
                content: super::helpers::parse_json_or_string(&message.content),
                token_count: message.token_count,
                created_at: message.created_at,
            })
            .collect();

        let tool_executions = store
            .get_tool_executions(row.id)
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
        }))
    }

    pub async fn set_session_title(
        &self,
        session_id: &str,
        title: Option<&str>,
    ) -> Result<Option<SessionSummary>> {
        let session_ref = parse_session_reference(session_id)?;
        let public_id = Uuid::parse_str(&session_ref.public_id)
            .map_err(|_| anyhow!("Invalid session id '{}'", session_ref.public_id))?;
        // A bare persisted session reference is interpreted against the primary `state` store.
        // Cross-state access must qualify the session id explicitly, e.g. `<session>@telegram`.
        let store_selector = session_ref
            .store_selector
            .unwrap_or_else(|| StoreSelector::Alias("state".to_string()));
        let store = self.kernel.store_manager().open(&store_selector).await?;
        let updated = store.update_session_title(public_id, title).await?;
        Ok(updated
            .as_ref()
            .map(|row| session_summary_from_row_and_selector(row, &store_selector)))
    }

    pub async fn list_session_branches(
        &self,
        session_id: &str,
    ) -> Result<Option<Vec<SessionBranchDetail>>> {
        let Some((store_selector, row)) = self.resolve_persisted_session(session_id).await? else {
            return Ok(None);
        };
        let store = self.kernel.store_manager().open(&store_selector).await?;
        let branches = store
            .list_branch_heads(row.id)
            .await?
            .into_iter()
            .map(branch_detail_from_row)
            .collect();
        Ok(Some(branches))
    }

    pub async fn create_session_branch(
        &self,
        session_id: &str,
        name: &str,
        from_turn_index: Option<u32>,
        activate: bool,
    ) -> Result<Option<SessionBranchDetail>> {
        let Some((store_selector, row)) = self.resolve_persisted_session(session_id).await? else {
            return Ok(None);
        };
        let store = self.kernel.store_manager().open(&store_selector).await?;
        let branch = store
            .create_branch_head_from_turn_index(row.id, name, from_turn_index, activate)
            .await?;
        Ok(Some(branch_detail_from_row(branch)))
    }

    pub async fn checkout_session_branch(
        &self,
        session_id: &str,
        branch: &str,
    ) -> Result<Option<SessionBranchDetail>> {
        let Some((store_selector, row)) = self.resolve_persisted_session(session_id).await? else {
            return Ok(None);
        };
        let live = self.live_session_snapshot(&row.public_id).await;
        if let Some(snapshot) = &live
            && (snapshot.active_tasks > 0 || snapshot.queued_tasks > 0)
        {
            anyhow::bail!(
                "Cannot check out branch for busy live session '{}'",
                session_id
            );
        }
        let store = self.kernel.store_manager().open(&store_selector).await?;
        let branch = if let Ok(branch_id) = Uuid::parse_str(branch) {
            store
                .checkout_branch_head_by_public_id(row.id, branch_id)
                .await?
        } else {
            store.checkout_branch_head_by_name(row.id, branch).await?
        };
        if branch.is_some() && live.is_some() {
            self.kernel
                .agent_manager()
                .reload_session(session_id)
                .await?;
        }
        Ok(branch.map(branch_detail_from_row))
    }

    pub async fn cancel_session(&self, session_id: &str) -> Result<serde_json::Value> {
        let (agent_id, session_id) = self
            .kernel
            .agent_manager()
            .cancel_session(session_id)
            .await?;
        Ok(serde_json::json!({
            "agent_id": agent_id,
            "session_id": session_id,
            "action": "cancel_requested",
        }))
    }

    pub async fn kill_session(&self, session_id: &str) -> Result<serde_json::Value> {
        let (agent_id, session_id) = self.kernel.agent_manager().kill_session(session_id).await?;
        Ok(serde_json::json!({
            "agent_id": agent_id,
            "session_id": session_id,
            "action": "killed",
        }))
    }

    pub(super) fn ensure_enabled_agent(&self, agent_id: &str) -> Result<()> {
        if agent_id == self.bootstrap_config.agent.id {
            return Ok(());
        }

        let agent = self
            .registry_load
            .agents
            .iter()
            .find(|agent| agent.id == agent_id)
            .ok_or_else(|| anyhow!("Agent '{}' not found", agent_id))?;
        if !agent.enabled {
            anyhow::bail!("Agent '{}' is disabled", agent_id);
        }
        Ok(())
    }

    pub(super) async fn list_agent_runtime_statuses(&self) -> Vec<AgentStatusSnapshot> {
        let mut live: HashMap<_, _> = self
            .kernel
            .agent_manager()
            .list_statuses()
            .await
            .into_iter()
            .map(|status| (status.agent_id.clone(), status))
            .collect();

        let mut ids = vec![self.bootstrap_config.agent.id.clone()];
        ids.extend(
            self.registry_load
                .agents
                .iter()
                .map(|agent| agent.id.clone()),
        );
        ids.sort();
        ids.dedup();

        ids.into_iter()
            .map(|agent_id| {
                live.remove(&agent_id).unwrap_or(AgentStatusSnapshot {
                    agent_id,
                    running: false,
                    active_tasks: 0,
                    queued_tasks: 0,
                    awaiting_results: 0,
                    current_session_id: None,
                    current_request_id: None,
                })
            })
            .collect()
    }

    fn resolve_channel_state_selector(
        &self,
        channel_id: Option<&str>,
    ) -> Result<Option<StoreSelector>> {
        let Some(channel_id) = channel_id else {
            return Ok(None);
        };
        let Some(channel) = self
            .registry_load
            .channels
            .iter()
            .find(|channel| channel.id == channel_id)
        else {
            return Ok(None);
        };
        channel
            .persistence
            .state
            .as_ref()
            .map(|_| {
                self.bootstrap_config
                    .persistence
                    .resolve_context_state_selector(Some(&channel.persistence))
            })
            .transpose()
    }

    fn resolve_channel_default_store_selector(
        &self,
        channel_id: Option<&str>,
    ) -> Result<Option<StoreSelector>> {
        let Some(channel_id) = channel_id else {
            return Ok(None);
        };
        let Some(channel) = self
            .registry_load
            .channels
            .iter()
            .find(|channel| channel.id == channel_id)
        else {
            return Ok(None);
        };
        if channel.persistence.store.is_none() && channel.persistence.state.is_none() {
            return Ok(None);
        }
        self.bootstrap_config
            .persistence
            .resolve_context_store_selector(Some(&channel.persistence))
            .map(Some)
    }

    async fn resolve_persisted_session(
        &self,
        session_id: &str,
    ) -> Result<Option<(StoreSelector, SessionRow)>> {
        let session_ref = parse_session_reference(session_id)?;
        let public_id = Uuid::parse_str(&session_ref.public_id)
            .map_err(|_| anyhow!("Invalid session id '{}'", session_ref.public_id))?;
        let store_selector = session_ref
            .store_selector
            .unwrap_or_else(|| StoreSelector::Alias("state".to_string()));
        let store = self.kernel.store_manager().open(&store_selector).await?;
        let row = store.get_session_row_by_public_id(public_id).await?;
        Ok(row.map(|row| (store_selector, row)))
    }

    async fn live_session_snapshot(
        &self,
        public_id: &[u8],
    ) -> Option<crate::kernel::agent_manager::LiveSessionSnapshot> {
        let wanted = super::helpers::format_uuid_bytes_simple(public_id);
        self.kernel
            .agent_manager()
            .list_live_sessions(None)
            .await
            .into_iter()
            .find(|snapshot| {
                parse_session_reference(&snapshot.session_id)
                    .map(|session_ref| session_ref.public_id == wanted)
                    .unwrap_or_else(|_| snapshot.session_id == wanted)
            })
    }
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
