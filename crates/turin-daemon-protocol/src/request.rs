use serde::{Deserialize, Serialize};

use crate::{
    BindHarnessParams, ChannelAccessParams, ChannelAccessRoomParams, ChannelRunnerHeartbeatParams,
    ChannelRunnerHelloParams, CreateAgentParams, CreateChannelParams, EntityIdParams,
    HarnessActionRunParams, LiveSessionTargetParams, NoParams, OpenSessionParams,
    PromoteTaskParams, ResumeSessionParams, ScheduleCreateParams, ScheduleRunsParams,
    ScheduleUpdateParams, SessionBranchCheckoutParams, SessionBranchCreateParams,
    SessionBranchSiblingsParams, SessionGetParams, SessionIdParams, SessionListParams,
    SessionSearchParams, SessionTitleParams, SidestepTaskParams, SubmitTaskParams, TaskIdParams,
    UpdateAgentParams, UpdateChannelParams, WaitTaskParams, WorkItemTargetParams,
    WorklistItemsParams, WorklistListParams, WorklistTargetParams,
};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "op", content = "params")]
pub enum DaemonRequest {
    #[serde(rename = "daemon.ping")]
    DaemonPing(NoParams),
    #[serde(rename = "daemon.status")]
    DaemonStatus(NoParams),
    #[serde(rename = "daemon.stop")]
    DaemonStop(NoParams),
    #[serde(rename = "runtime.rescan")]
    RuntimeRescan(NoParams),
    #[serde(rename = "runtime.reload")]
    RuntimeReload(NoParams),
    #[serde(rename = "runtime.errors")]
    RuntimeErrors(NoParams),
    #[serde(rename = "runtime.events.subscribe")]
    RuntimeEventsSubscribe(RuntimeEventsSubscribeParams),
    #[serde(rename = "agent.list")]
    AgentList(NoParams),
    #[serde(rename = "agent.get")]
    AgentGet(EntityIdParams),
    #[serde(rename = "agent.status")]
    AgentStatus(EntityIdParams),
    #[serde(rename = "agent.issues")]
    AgentIssues(EntityIdParams),
    #[serde(rename = "agent.create")]
    AgentCreate(CreateAgentParams),
    #[serde(rename = "agent.enable")]
    AgentEnable(EntityIdParams),
    #[serde(rename = "agent.disable")]
    AgentDisable(EntityIdParams),
    #[serde(rename = "agent.update")]
    AgentUpdate(UpdateAgentParams),
    #[serde(rename = "agent.reload")]
    AgentReload(EntityIdParams),
    #[serde(rename = "agent.bind_harness")]
    AgentBindHarness(BindHarnessParams),
    #[serde(rename = "agent.use_local_harness")]
    AgentUseLocalHarness(EntityIdParams),
    #[serde(rename = "agent.delete")]
    AgentDelete(EntityIdParams),
    #[serde(rename = "task.submit")]
    TaskSubmit(SubmitTaskParams),
    #[serde(rename = "task.sidestep")]
    TaskSidestep(SidestepTaskParams),
    #[serde(rename = "task.get")]
    TaskGet(TaskIdParams),
    #[serde(rename = "task.wait")]
    TaskWait(WaitTaskParams),
    #[serde(rename = "task.promote")]
    TaskPromote(PromoteTaskParams),
    #[serde(rename = "task.cancel")]
    TaskCancel(TaskIdParams),
    #[serde(rename = "task.list")]
    TaskList(NoParams),
    #[serde(rename = "schedule.create")]
    ScheduleCreate(ScheduleCreateParams),
    #[serde(rename = "schedule.update")]
    ScheduleUpdate(ScheduleUpdateParams),
    #[serde(rename = "schedule.get")]
    ScheduleGet(EntityIdParams),
    #[serde(rename = "schedule.list")]
    ScheduleList(NoParams),
    #[serde(rename = "schedule.runs")]
    ScheduleRuns(ScheduleRunsParams),
    #[serde(rename = "schedule.enable")]
    ScheduleEnable(EntityIdParams),
    #[serde(rename = "schedule.disable")]
    ScheduleDisable(EntityIdParams),
    #[serde(rename = "schedule.delete")]
    ScheduleDelete(EntityIdParams),
    #[serde(rename = "worklist.list")]
    WorklistList(WorklistListParams),
    #[serde(rename = "worklist.get")]
    WorklistGet(WorklistTargetParams),
    #[serde(rename = "worklist.items")]
    WorklistItems(WorklistItemsParams),
    #[serde(rename = "workitem.get")]
    WorkItemGet(WorkItemTargetParams),
    #[serde(rename = "session.list")]
    SessionList(SessionListParams),
    #[serde(rename = "session.list_live")]
    SessionListLive(NoParams),
    #[serde(rename = "session.search")]
    SessionSearch(SessionSearchParams),
    #[serde(rename = "session.open")]
    SessionOpen(OpenSessionParams),
    #[serde(rename = "session.resume")]
    SessionResume(ResumeSessionParams),
    #[serde(rename = "session.get")]
    SessionGet(SessionGetParams),
    #[serde(rename = "session.set_title")]
    SessionSetTitle(SessionTitleParams),
    #[serde(rename = "session.branch_list")]
    SessionBranchList(SessionIdParams),
    #[serde(rename = "session.branch_create")]
    SessionBranchCreate(SessionBranchCreateParams),
    #[serde(rename = "session.branch_checkout")]
    SessionBranchCheckout(SessionBranchCheckoutParams),
    #[serde(rename = "session.branch_siblings")]
    SessionBranchSiblings(SessionBranchSiblingsParams),
    #[serde(rename = "session.cancel")]
    SessionCancel(LiveSessionTargetParams),
    #[serde(rename = "session.kill")]
    SessionKill(LiveSessionTargetParams),
    #[serde(rename = "harness.list")]
    HarnessList(NoParams),
    #[serde(rename = "harness.create")]
    HarnessCreate(EntityIdParams),
    #[serde(rename = "harness.get")]
    HarnessGet(EntityIdParams),
    #[serde(rename = "harness.issues")]
    HarnessIssues(EntityIdParams),
    #[serde(rename = "harness.reload")]
    HarnessReload(EntityIdParams),
    #[serde(rename = "harness.validate")]
    HarnessValidate(EntityIdParams),
    #[serde(rename = "harness.action_run")]
    HarnessActionRun(HarnessActionRunParams),
    #[serde(rename = "harness.delete")]
    HarnessDelete(EntityIdParams),
    #[serde(rename = "channel.list")]
    ChannelList(NoParams),
    #[serde(rename = "channel.create")]
    ChannelCreate(CreateChannelParams),
    #[serde(rename = "channel.get")]
    ChannelGet(EntityIdParams),
    #[serde(rename = "channel.status")]
    ChannelStatus(EntityIdParams),
    #[serde(rename = "channel.issues")]
    ChannelIssues(EntityIdParams),
    #[serde(rename = "channel.enable")]
    ChannelEnable(EntityIdParams),
    #[serde(rename = "channel.disable")]
    ChannelDisable(EntityIdParams),
    #[serde(rename = "channel.update")]
    ChannelUpdate(UpdateChannelParams),
    #[serde(rename = "channel.access.get")]
    ChannelAccessGet(ChannelAccessParams),
    #[serde(rename = "channel.access.approve")]
    ChannelAccessApprove(ChannelAccessRoomParams),
    #[serde(rename = "channel.access.reject")]
    ChannelAccessReject(ChannelAccessRoomParams),
    #[serde(rename = "channel.access.revoke")]
    ChannelAccessRevoke(ChannelAccessRoomParams),
    #[serde(rename = "channel.runner.hello")]
    ChannelRunnerHello(ChannelRunnerHelloParams),
    #[serde(rename = "channel.runner.heartbeat")]
    ChannelRunnerHeartbeat(ChannelRunnerHeartbeatParams),
    #[serde(rename = "channel.delete")]
    ChannelDelete(EntityIdParams),
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct RuntimeEventsSubscribeParams {
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub slot_id: Option<String>,
}
