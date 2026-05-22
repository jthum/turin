use anyhow::Result;
use turin_daemon_protocol::{
    DaemonRequest, PromoteTaskParams, SidestepContextTargetParams, SidestepModeParams,
    SidestepTaskParams, SubmitTaskParams, TaskIdParams, WaitTaskParams,
};

use crate::client::ControlClient;
use crate::models::{SessionBranchDetail, TaskList, TaskStatus};

impl ControlClient {
    pub async fn list_tasks(&self) -> Result<Vec<TaskStatus>> {
        let response: TaskList = self
            .request_ok(None, DaemonRequest::TaskList(Default::default()))
            .await?;
        Ok(response.tasks)
    }

    pub async fn get_task(&self, request_id: &str) -> Result<TaskStatus> {
        self.request_ok(None, DaemonRequest::TaskGet(task_id(request_id)))
            .await
    }

    pub async fn submit_task_in_slot(
        &self,
        agent_id: Option<String>,
        session_id: Option<String>,
        slot_id: Option<String>,
        prompt: String,
    ) -> Result<TaskStatus> {
        self.submit_task_in_slot_with_conflict_policy(agent_id, session_id, slot_id, prompt, None)
            .await
    }

    pub async fn submit_task_in_slot_with_conflict_policy(
        &self,
        agent_id: Option<String>,
        session_id: Option<String>,
        slot_id: Option<String>,
        prompt: String,
        conflict_policy: Option<String>,
    ) -> Result<TaskStatus> {
        self.request_ok(
            None,
            DaemonRequest::TaskSubmit(SubmitTaskParams {
                agent_id,
                session_id,
                slot_id,
                prompt,
                content: None,
                tools: None,
                conflict_policy,
            }),
        )
        .await
    }

    pub async fn submit_task(
        &self,
        agent_id: Option<String>,
        session_id: Option<String>,
        prompt: String,
    ) -> Result<TaskStatus> {
        self.submit_task_in_slot(agent_id, session_id, None, prompt)
            .await
    }

    pub async fn sidestep_task(
        &self,
        session_id: String,
        slot_id: Option<String>,
        prompt: String,
        mode: SidestepModeParams,
        context_target: Option<SidestepContextTargetParams>,
        timeout_ms: Option<u64>,
    ) -> Result<TaskStatus> {
        self.request_ok(
            None,
            DaemonRequest::TaskSidestep(SidestepTaskParams {
                session_id,
                slot_id,
                prompt,
                content: None,
                tools: None,
                mode,
                context_target,
                timeout_ms,
            }),
        )
        .await
    }

    pub async fn wait_task(&self, request_id: &str, timeout_ms: Option<u64>) -> Result<TaskStatus> {
        self.request_ok(
            None,
            DaemonRequest::TaskWait(WaitTaskParams {
                request_id: request_id.to_string(),
                timeout_ms,
            }),
        )
        .await
    }

    pub async fn cancel_task(&self, request_id: &str) -> Result<TaskStatus> {
        self.request_ok(None, DaemonRequest::TaskCancel(task_id(request_id)))
            .await
    }

    pub async fn promote_task(
        &self,
        request_id: &str,
        branch_name: Option<String>,
    ) -> Result<SessionBranchDetail> {
        self.request_ok(
            None,
            DaemonRequest::TaskPromote(PromoteTaskParams {
                request_id: request_id.to_string(),
                branch_name,
            }),
        )
        .await
    }
}

fn task_id(request_id: &str) -> TaskIdParams {
    TaskIdParams {
        request_id: request_id.to_string(),
    }
}
