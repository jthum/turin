use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Result, anyhow};
use uuid::Uuid;

use super::DaemonState;
use crate::daemon::protocol::{
    PromoteTaskParams, SidestepContextTargetParams, SidestepModeParams, SidestepTaskParams,
    SubmitTaskParams,
};
use crate::kernel::agent_manager::{
    AgentStatusSnapshot, PromotedTaskBranch, TaskStatusFingerprint, TaskStatusSnapshot,
};
use crate::kernel::event::{KernelEvent, TaskBranchOutcome};
use crate::kernel::prepare_persisted_session_sidestep;
use crate::kernel::session::{
    ExecutionConflictPolicy, ExecutionContextTarget, PreparedSidestepExecution, QueuedTask,
    SidestepMode, TaskExecutionOverrides,
};
use crate::kernel::session_refs::session_reference_matches_public_id;
use turin_types::{TaskInputContent, ToolsConfig};

struct TaskSubmissionRequest<'a> {
    agent_id: Option<&'a str>,
    session_id: Option<&'a str>,
    slot_id: Option<&'a str>,
    prompt: String,
    inference_context: Option<String>,
    content: Option<Vec<TaskInputContent>>,
    tools: Option<ToolsConfig>,
    conflict_policy: Option<&'a str>,
    execution: Option<TaskExecutionOverrides>,
    branch_outcome: Option<TaskBranchOutcome>,
}

const TASK_WAIT_POLL_INTERVAL: Duration = Duration::from_millis(10);

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
            inference_context: None,
            content,
            tools,
            conflict_policy: None,
            execution: None,
            branch_outcome: None,
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
            inference_context: params.inference_context,
            content: params.content,
            tools: params.tools,
            conflict_policy: params.conflict_policy.as_deref(),
            execution: None,
            branch_outcome: None,
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
        let PreparedSidestepExecution {
            execution,
            conflict_policy,
            branch_outcome,
        } = self
            .prepare_sidestep_execution(&session_id, params.mode, params.context_target)
            .await?;

        let submitted = self
            .submit_task_request(TaskSubmissionRequest {
                agent_id: None,
                session_id: Some(&session_id),
                slot_id: Some(&sidestep_slot_id),
                prompt: params.prompt,
                inference_context: None,
                content: params.content,
                tools: params.tools,
                conflict_policy: Some(conflict_policy.as_str()),
                execution: Some(execution),
                branch_outcome,
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
        task.inference_context = request
            .inference_context
            .as_deref()
            .map(str::trim)
            .filter(|context| !context.is_empty())
            .map(str::to_string);
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
        task.branch_outcome = request.branch_outcome;
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

    pub(crate) async fn list_task_fingerprints(&self) -> Vec<TaskStatusFingerprint> {
        self.kernel.agent_manager().list_task_fingerprints().await
    }

    pub async fn get_task(&self, request_id: &str) -> Option<TaskStatusSnapshot> {
        self.kernel.agent_manager().get_task(request_id).await
    }

    pub async fn cancel_task(&self, request_id: &str) -> Result<TaskStatusSnapshot> {
        self.kernel.agent_manager().cancel_task(request_id).await
    }

    pub(crate) async fn promote_task_params(
        &self,
        params: PromoteTaskParams,
    ) -> Result<PromotedTaskBranch> {
        self.kernel
            .agent_manager()
            .promote_completed_task(
                &params.request_id,
                params.branch_name.as_deref(),
                params.source_turn_id,
            )
            .await
    }

    pub async fn wait_for_task(
        &self,
        request_id: &str,
        timeout_ms: Option<u64>,
    ) -> Result<TaskStatusSnapshot> {
        let Some(initial) = self.get_task(request_id).await else {
            anyhow::bail!("Task '{}' not found", request_id);
        };
        if !initial.state.is_active() {
            return Ok(initial);
        }

        let deadline = timeout_ms.map(|ms| tokio::time::Instant::now() + Duration::from_millis(ms));
        loop {
            if let Some(snapshot) = self.get_task(request_id).await {
                if !snapshot.state.is_active() {
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

            tokio::time::sleep(TASK_WAIT_POLL_INTERVAL).await;
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
        origin_id: Option<&str>,
    ) -> Result<crate::kernel::agent_manager::LiveSessionSnapshot> {
        self.ensure_enabled_agent(agent_id)?;
        self.kernel
            .agent_manager()
            .open_session(
                agent_id,
                slot_id,
                None,
                None,
                origin_id.map(str::to_string),
                Default::default(),
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
            .resume_session(session_id, slot_id, None, Default::default())
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

    pub async fn cancel_session_family(&self, session_id: &str) -> Result<serde_json::Value> {
        let (agent_id, session_id, affected_tasks) = self
            .kernel
            .agent_manager()
            .cancel_session_family(session_id)
            .await?;
        Ok(serde_json::json!({
            "agent_id": agent_id,
            "slot_id": null,
            "session_id": session_id,
            "action": "family_cancel_requested",
            "recursive": true,
            "affected_tasks": affected_tasks,
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
                    provider: String::new(),
                    model: String::new(),
                    harness_id: String::new(),
                    inference_contexts: Vec::new(),
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
            .filter(|snapshot| session_reference_matches_public_id(&snapshot.session_id, &wanted))
            .collect()
    }

    async fn prepare_sidestep_execution(
        &self,
        session_id: &str,
        mode: SidestepModeParams,
        requested: Option<SidestepContextTargetParams>,
    ) -> Result<PreparedSidestepExecution> {
        prepare_persisted_session_sidestep(
            self.kernel.store_manager(),
            session_id,
            &ExecutionContextTarget::BranchHead {
                branch_head_id: None,
            },
            match mode {
                SidestepModeParams::Ephemeral => SidestepMode::Ephemeral,
                SidestepModeParams::ForkSibling => SidestepMode::ForkSibling,
            },
            requested.map(execution_context_target_from_params),
        )
        .await
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
