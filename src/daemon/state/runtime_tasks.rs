use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Result, anyhow};
use uuid::Uuid;

use super::DaemonState;
use crate::daemon::protocol::{SidestepContextTargetParams, SidestepTaskParams, SubmitTaskParams};
use crate::kernel::agent_manager::{AgentStatusSnapshot, TaskStatusSnapshot};
use crate::kernel::config::InferenceOverrideConfig;
use crate::kernel::event::KernelEvent;
use crate::kernel::session::{
    ExecutionConflictPolicy, ExecutionContextTarget, ExecutionDurability, ExecutionVisibility,
    ExecutionWritePolicy, QueuedTask, TaskExecutionOverrides,
};
use crate::kernel::session_refs::parse_session_reference;
use crate::persistence::manager::StoreSelector;
use turin_types::{TaskInputContent, ToolsConfig};

struct TaskSubmissionRequest<'a> {
    agent_id: Option<&'a str>,
    session_id: Option<&'a str>,
    slot_id: Option<&'a str>,
    prompt: String,
    content: Option<Vec<TaskInputContent>>,
    tools: Option<ToolsConfig>,
    conflict_policy: Option<&'a str>,
    execution: Option<TaskExecutionOverrides>,
}

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
        slot_id: Option<&str>,
        prompt: String,
        content: Option<Vec<TaskInputContent>>,
        tools: Option<ToolsConfig>,
    ) -> Result<TaskStatusSnapshot> {
        self.submit_task_request(TaskSubmissionRequest {
            agent_id,
            session_id,
            slot_id,
            prompt,
            content,
            tools,
            conflict_policy: None,
            execution: None,
        })
        .await
    }

    pub(crate) async fn submit_task_params(
        &self,
        params: SubmitTaskParams,
    ) -> Result<TaskStatusSnapshot> {
        self.submit_task_request(TaskSubmissionRequest {
            agent_id: params.agent_id.as_deref(),
            session_id: params.session_id.as_deref(),
            slot_id: params.slot_id.as_deref(),
            prompt: params.prompt,
            content: params.content,
            tools: params.tools,
            conflict_policy: params.conflict_policy.as_deref(),
            execution: None,
        })
        .await
    }

    pub(crate) async fn sidestep_task_params(
        &self,
        params: SidestepTaskParams,
    ) -> Result<TaskStatusSnapshot> {
        let session_id = params.session_id;
        let sidestep_slot_id = params
            .slot_id
            .unwrap_or_else(|| format!("sd_{}", Uuid::now_v7().simple()));

        self.resume_session(&session_id, Some(&sidestep_slot_id))
            .await?;
        let context_target = self
            .resolve_sidestep_execution_target(&session_id, params.context_target)
            .await?;

        let execution = TaskExecutionOverrides {
            context_target: Some(context_target),
            visibility: Some(ExecutionVisibility::Hidden),
            durability: Some(ExecutionDurability::Ephemeral),
            write_policy: Some(ExecutionWritePolicy::Detached),
        };

        let submitted = self
            .submit_task_request(TaskSubmissionRequest {
                agent_id: None,
                session_id: Some(&session_id),
                slot_id: Some(&sidestep_slot_id),
                prompt: params.prompt,
                content: params.content,
                tools: params.tools,
                conflict_policy: Some(ExecutionConflictPolicy::Detached.as_str()),
                execution: Some(execution),
            })
            .await;

        let task_result = match submitted {
            Ok(task) => {
                self.wait_for_task(&task.request_id, params.timeout_ms)
                    .await
            }
            Err(err) => Err(err),
        };

        let cleanup_result = self
            .kill_session(&session_id, Some(&sidestep_slot_id))
            .await;
        match (task_result, cleanup_result) {
            (Ok(task), Ok(_)) => Ok(task),
            (Ok(_), Err(cleanup_err)) => Err(cleanup_err),
            (Err(task_err), Ok(_)) => Err(task_err),
            (Err(task_err), Err(cleanup_err)) => Err(task_err.context(format!(
                "sidestep cleanup for slot '{}' also failed: {}",
                sidestep_slot_id, cleanup_err
            ))),
        }
    }

    async fn submit_task_request(
        &self,
        request: TaskSubmissionRequest<'_>,
    ) -> Result<TaskStatusSnapshot> {
        if request.session_id.is_none() && request.slot_id.is_some() {
            anyhow::bail!("task.submit slot_id requires session_id");
        }
        let mut task = QueuedTask::ad_hoc(request.prompt);
        task.content = request.content;
        if let Some(tools) = request.tools
            && !tools.is_empty()
        {
            task.tools = Some(tools);
        }
        task.conflict_policy = match request.conflict_policy {
            Some(conflict_policy) => Some(
                conflict_policy
                    .parse::<ExecutionConflictPolicy>()
                    .map_err(anyhow::Error::msg)?,
            ),
            None => None,
        };
        task.execution = request.execution;
        let request_id = if let Some(session_id) = request.session_id {
            self.kernel
                .agent_manager()
                .submit_to_session(session_id, request.slot_id, task, None)
                .await?
        } else {
            let agent_id = request
                .agent_id
                .ok_or_else(|| anyhow!("task.submit requires agent_id or session_id"))?;
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

    pub async fn list_live_sessions(
        &self,
    ) -> Vec<crate::kernel::agent_manager::LiveSessionSnapshot> {
        self.kernel.agent_manager().list_live_sessions(None).await
    }

    pub async fn subscribe_live_session_events(
        &self,
        session_id: &str,
        slot_id: Option<&str>,
    ) -> Option<(
        String,
        String,
        tokio::sync::broadcast::Receiver<(Option<i64>, KernelEvent)>,
    )> {
        self.kernel
            .agent_manager()
            .subscribe_session_events(session_id, slot_id)
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
                channel_id.map(str::to_string),
                self.resolve_channel_inference_override(channel_id),
            )
            .await
    }

    pub async fn resume_session(
        &self,
        session_id: &str,
        slot_id: Option<&str>,
    ) -> Result<crate::kernel::agent_manager::LiveSessionSnapshot> {
        let channel_id = self.resolve_session_channel_id(session_id).await?;
        self.kernel
            .agent_manager()
            .resume_session(
                session_id,
                slot_id,
                channel_id.clone(),
                self.resolve_channel_inference_override(channel_id.as_deref()),
            )
            .await
    }

    pub async fn cancel_session(
        &self,
        session_id: &str,
        slot_id: Option<&str>,
    ) -> Result<serde_json::Value> {
        let (agent_id, slot_id, session_id) = self
            .kernel
            .agent_manager()
            .cancel_session(session_id, slot_id)
            .await?;
        Ok(serde_json::json!({
            "agent_id": agent_id,
            "slot_id": slot_id,
            "session_id": session_id,
            "action": "cancel_requested",
        }))
    }

    pub async fn kill_session(
        &self,
        session_id: &str,
        slot_id: Option<&str>,
    ) -> Result<serde_json::Value> {
        let (agent_id, slot_id, session_id) = self
            .kernel
            .agent_manager()
            .kill_session(session_id, slot_id)
            .await?;
        Ok(serde_json::json!({
            "agent_id": agent_id,
            "slot_id": slot_id,
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

    fn resolve_channel_inference_override(
        &self,
        channel_id: Option<&str>,
    ) -> InferenceOverrideConfig {
        channel_id
            .and_then(|channel_id| {
                self.registry_load
                    .channels
                    .iter()
                    .find(|channel| channel.id == channel_id)
            })
            .map(|channel| channel.inference.clone())
            .unwrap_or_default()
    }

    async fn resolve_session_channel_id(&self, session_id: &str) -> Result<Option<String>> {
        let session_ref = parse_session_reference(session_id)?;
        let public_id = Uuid::parse_str(&session_ref.public_id)
            .map_err(|_| anyhow!("Invalid session id '{}'", session_ref.public_id))?;
        let store_selector = session_ref
            .store_selector
            .unwrap_or_else(|| StoreSelector::Alias("state".to_string()));
        let store = self.kernel.store_manager().open(&store_selector).await?;
        let Some(row) = store.get_session_row_by_public_id(public_id).await? else {
            return Ok(None);
        };

        Ok(row
            .metadata
            .as_deref()
            .and_then(session_channel_id_from_metadata))
    }

    pub(super) async fn live_session_snapshots(
        &self,
        public_id: &[u8],
    ) -> Vec<crate::kernel::agent_manager::LiveSessionSnapshot> {
        let wanted = super::helpers::format_uuid_bytes_simple(public_id);
        self.kernel
            .agent_manager()
            .list_live_sessions(None)
            .await
            .into_iter()
            .filter(|snapshot| {
                parse_session_reference(&snapshot.session_id)
                    .map(|session_ref| session_ref.public_id == wanted)
                    .unwrap_or_else(|_| snapshot.session_id == wanted)
            })
            .collect()
    }

    async fn resolve_sidestep_execution_target(
        &self,
        session_id: &str,
        requested: Option<SidestepContextTargetParams>,
    ) -> Result<ExecutionContextTarget> {
        let session_ref = parse_session_reference(session_id)?;
        let public_id = uuid::Uuid::parse_str(&session_ref.public_id)
            .map_err(|_| anyhow!("Invalid session id '{}'", session_ref.public_id))?;
        let store_selector = session_ref
            .store_selector
            .unwrap_or_else(|| StoreSelector::Alias("state".to_string()));
        let store = self.kernel.store_manager().open(&store_selector).await?;
        let row = store
            .get_session_row_by_public_id(public_id)
            .await?
            .ok_or_else(|| anyhow!("Session '{}' not found", session_id))?;

        match requested {
            None => {
                let branch = store.get_active_branch_head(row.id).await?;
                Ok(snapshot_target_from_branch_head(branch, None))
            }
            Some(SidestepContextTargetParams::BranchHead { branch_head_id }) => {
                let branch = store.get_branch_head(row.id, branch_head_id).await?;
                if branch.is_none() {
                    anyhow::bail!(
                        "Branch head '{}' not found for session '{}'",
                        branch_head_id,
                        session_id
                    );
                }
                Ok(snapshot_target_from_branch_head(
                    branch,
                    Some(branch_head_id),
                ))
            }
            Some(other) => Ok(execution_context_target_from_params(other)),
        }
    }
}

fn execution_context_target_from_params(
    params: SidestepContextTargetParams,
) -> ExecutionContextTarget {
    match params {
        SidestepContextTargetParams::BranchHead { branch_head_id } => {
            ExecutionContextTarget::BranchHead {
                branch_head_id: Some(branch_head_id),
            }
        }
        SidestepContextTargetParams::TurnId { turn_id } => {
            ExecutionContextTarget::TurnId { turn_id }
        }
        SidestepContextTargetParams::SelectedPath { turn_ids } => {
            ExecutionContextTarget::SelectedPath { turn_ids }
        }
        SidestepContextTargetParams::ExternalReference { reference } => {
            ExecutionContextTarget::ExternalReference { reference }
        }
        SidestepContextTargetParams::SummarySource { source_turn_id } => {
            ExecutionContextTarget::SummarySource { source_turn_id }
        }
    }
}

fn snapshot_target_from_branch_head(
    branch: Option<crate::persistence::schema::BranchHeadRow>,
    explicit_branch_head_id: Option<i64>,
) -> ExecutionContextTarget {
    match branch.and_then(|branch| branch.head_turn_id) {
        Some(turn_id) => ExecutionContextTarget::TurnId { turn_id },
        None => ExecutionContextTarget::BranchHead {
            branch_head_id: explicit_branch_head_id,
        },
    }
}

fn session_channel_id_from_metadata(metadata: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(metadata)
        .ok()?
        .get("_turin")?
        .get("channel_id")?
        .as_str()
        .map(ToOwned::to_owned)
}
