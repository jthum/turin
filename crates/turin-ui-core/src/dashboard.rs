use std::collections::BTreeMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use turin_control_client::{
    AgentSummary, ChannelSummary, ConnectionKind, ControlClient, ControlHealth, DaemonStatus,
    LiveSession, SessionDetail, SessionSummary, TaskStatus,
};
use turin_daemon_protocol::EventEnvelope;

use crate::controller::UiUpdate;

const MAX_RECENT_EVENTS: usize = 64;
const DEFAULT_SESSION_LIMIT: usize = 25;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardState {
    pub connection_kind: ConnectionKind,
    pub connection_target: String,
    pub health: Option<DashboardHealth>,
    pub status: Option<DaemonStatus>,
    pub live_sessions: Vec<LiveSession>,
    pub sessions: Vec<SessionSummary>,
    pub tasks: Vec<TaskStatus>,
    #[serde(default)]
    pub session_details: BTreeMap<String, SessionDetail>,
    pub recent_events: Vec<EventEnvelope>,
    pub last_error: Option<String>,
    pub last_info: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSnapshot {
    pub connection_kind: ConnectionKind,
    pub connection_target: String,
    pub health: DashboardHealth,
    pub status: DaemonStatus,
    pub live_sessions: Vec<LiveSession>,
    pub sessions: Vec<SessionSummary>,
    pub tasks: Vec<TaskStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardHealth {
    pub ready: bool,
    pub version: String,
    pub protocol_version: u32,
    pub transport: String,
    pub wire_format: String,
    pub issue_count: usize,
    pub agent_count: usize,
    pub harness_count: usize,
    pub channel_count: usize,
    pub running_agent_count: usize,
    pub active_task_count: usize,
    pub queued_task_count: usize,
    pub awaiting_result_count: usize,
    pub channel_runtime_count: usize,
    pub failed_channel_count: usize,
}

impl DashboardState {
    pub async fn load(client: &ControlClient) -> Result<Self> {
        let snapshot = Self::snapshot(client).await?;
        Ok(Self {
            connection_kind: snapshot.connection_kind,
            connection_target: snapshot.connection_target,
            health: Some(snapshot.health),
            status: Some(snapshot.status),
            live_sessions: snapshot.live_sessions,
            sessions: snapshot.sessions,
            tasks: snapshot.tasks,
            session_details: BTreeMap::new(),
            recent_events: Vec::new(),
            last_error: None,
            last_info: None,
        })
    }

    pub async fn snapshot(client: &ControlClient) -> Result<DashboardSnapshot> {
        let (health, status) = client.health_and_status().await?;
        let live_sessions = client.list_live_sessions().await?;
        let sessions = client.list_sessions(DEFAULT_SESSION_LIMIT, 0).await?;
        let tasks = client.list_tasks().await?;
        Ok(DashboardSnapshot {
            connection_kind: client.kind(),
            connection_target: client.target(),
            health: health.into(),
            status,
            live_sessions,
            sessions,
            tasks,
        })
    }

    pub fn apply_snapshot(&mut self, snapshot: DashboardSnapshot) {
        let mut retained_details = BTreeMap::new();
        for session_id in snapshot
            .live_sessions
            .iter()
            .map(|session| session.session_id.as_str())
            .chain(
                snapshot
                    .sessions
                    .iter()
                    .map(|session| session.session_id.as_str()),
            )
        {
            if let Some(detail) = self.session_details.remove(session_id) {
                retained_details.insert(session_id.to_string(), detail);
            }
        }
        self.connection_kind = snapshot.connection_kind;
        self.connection_target = snapshot.connection_target;
        self.health = Some(snapshot.health);
        self.status = Some(snapshot.status);
        self.live_sessions = snapshot.live_sessions;
        self.sessions = snapshot.sessions;
        self.tasks = snapshot.tasks;
        self.session_details = retained_details;
        self.last_error = None;
    }

    pub fn apply_update(&mut self, update: UiUpdate) {
        match update {
            UiUpdate::Snapshot(snapshot) => self.apply_snapshot(*snapshot),
            UiUpdate::SessionDetail(detail) => self.record_session_detail(*detail),
            UiUpdate::Event(event) => self.record_event(event),
            UiUpdate::Error(message) => self.record_error(message),
            UiUpdate::Info(message) => self.record_info(message),
        }
    }

    pub fn record_session_detail(&mut self, detail: SessionDetail) {
        self.session_details
            .insert(detail.session.session_id.clone(), detail);
    }

    pub fn record_event(&mut self, event: EventEnvelope) {
        self.recent_events.push(event);
        if self.recent_events.len() > MAX_RECENT_EVENTS {
            let drop_count = self.recent_events.len() - MAX_RECENT_EVENTS;
            self.recent_events.drain(0..drop_count);
        }
    }

    pub fn record_error(&mut self, message: impl Into<String>) {
        self.last_error = Some(message.into());
    }

    pub fn record_info(&mut self, message: impl Into<String>) {
        self.last_info = Some(message.into());
    }

    pub fn status_pretty_json(&self) -> String {
        self.status
            .as_ref()
            .and_then(|value| serde_json::to_string_pretty(value).ok())
            .unwrap_or_else(|| "{}".to_string())
    }

    pub fn agents(&self) -> &[AgentSummary] {
        self.status
            .as_ref()
            .map(|status| status.registry.agents.as_slice())
            .unwrap_or(&[])
    }

    pub fn channels(&self) -> &[ChannelSummary] {
        self.status
            .as_ref()
            .map(|status| status.registry.channels.as_slice())
            .unwrap_or(&[])
    }

    pub fn session_detail(&self, session_id: &str) -> Option<&SessionDetail> {
        self.session_details.get(session_id)
    }
}

impl From<ControlHealth> for DashboardHealth {
    fn from(value: ControlHealth) -> Self {
        Self {
            ready: value.ready,
            version: value.version,
            protocol_version: value.protocol_version,
            transport: value.transport,
            wire_format: value.wire_format,
            issue_count: value.issue_count,
            agent_count: value.agent_count,
            harness_count: value.harness_count,
            channel_count: value.channel_count,
            running_agent_count: value.running_agent_count,
            active_task_count: value.active_task_count,
            queued_task_count: value.queued_task_count,
            awaiting_result_count: value.awaiting_result_count,
            channel_runtime_count: value.channel_runtime_count,
            failed_channel_count: value.failed_channel_count,
        }
    }
}
