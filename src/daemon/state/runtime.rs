use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Result, anyhow};

use super::{
    DaemonState, SessionDetail, SessionEventDetail, SessionMessageDetail, SessionSummary,
    SessionToolExecutionDetail,
};
use crate::kernel::agent_manager::{AgentStatusSnapshot, TaskStatusSnapshot};
use crate::kernel::session::QueuedTask;

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
    ) -> Result<TaskStatusSnapshot> {
        let request_id = if let Some(session_id) = session_id {
            self.kernel
                .agent_manager()
                .submit_to_session(session_id, QueuedTask::ad_hoc(prompt), None)
                .await?
        } else {
            let agent_id =
                agent_id.ok_or_else(|| anyhow!("task.submit requires agent_id or session_id"))?;
            self.ensure_enabled_agent(agent_id)?;
            self.kernel
                .agent_manager()
                .submit(agent_id, QueuedTask::ad_hoc(prompt), None)
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
        let store = self.kernel.store_manager().get_default().await?;
        let rows = store.list_session_rows(limit, offset).await?;
        Ok(rows
            .iter()
            .map(super::helpers::session_summary_from_row)
            .collect())
    }

    pub async fn list_live_sessions(&self) -> Vec<crate::kernel::agent_manager::LiveSessionSnapshot> {
        self.kernel.agent_manager().list_live_sessions(None).await
    }

    pub async fn open_session(
        &self,
        agent_id: &str,
        slot_id: Option<&str>,
    ) -> Result<crate::kernel::agent_manager::LiveSessionSnapshot> {
        self.ensure_enabled_agent(agent_id)?;
        self.kernel.agent_manager().open_session(agent_id, slot_id).await
    }

    pub async fn get_session(&self, session_id: &str) -> Result<Option<SessionDetail>> {
        let public_id = uuid::Uuid::parse_str(session_id)
            .map_err(|_| anyhow!("Invalid session id '{}'", session_id))?;
        let store = self.kernel.store_manager().get_default().await?;
        let Some(row) = store.get_session_row_by_public_id(public_id).await? else {
            return Ok(None);
        };

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

        Ok(Some(SessionDetail {
            session: super::helpers::session_summary_from_row(&row),
            events,
            messages,
            tool_executions,
        }))
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
}
