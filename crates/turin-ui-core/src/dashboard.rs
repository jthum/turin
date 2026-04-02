use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use turin_control_client::{
    AgentSummary, ChannelSummary, ConnectionKind, ControlClient, ControlHealth, DaemonStatus,
    LiveSession, SessionDetail, SessionSummary, TaskStatus,
};
use turin_daemon_protocol::EventEnvelope;

use crate::controller::UiUpdate;

const MAX_RECENT_EVENTS: usize = 64;
const MAX_RECENT_NOTICES: usize = 16;
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
    #[serde(default)]
    pub recent_notices: Vec<DashboardNotice>,
    #[serde(default = "now_unix_ms")]
    pub last_snapshot_unix_ms: u64,
    #[serde(default)]
    pub last_event_unix_ms: Option<u64>,
    #[serde(default)]
    pub last_notice_unix_ms: Option<u64>,
    #[serde(default)]
    pub total_event_count: u64,
    #[serde(default)]
    pub refresh_success_count: u64,
    #[serde(default)]
    pub refresh_failure_count: u64,
    #[serde(default)]
    pub last_refresh_duration_ms: Option<u64>,
    #[serde(default)]
    pub last_refresh_ok: Option<bool>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardNotice {
    pub level: DashboardNoticeLevel,
    pub message: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DashboardNoticeLevel {
    Error,
    Info,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DashboardFreshness {
    Fresh,
    Quiet,
    Stale,
}

impl DashboardState {
    pub async fn load(client: &ControlClient) -> Result<Self> {
        let snapshot = Self::snapshot(client).await?;
        let now = now_unix_ms();
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
            recent_notices: Vec::new(),
            last_snapshot_unix_ms: now,
            last_event_unix_ms: None,
            last_notice_unix_ms: None,
            total_event_count: 0,
            refresh_success_count: 0,
            refresh_failure_count: 0,
            last_refresh_duration_ms: None,
            last_refresh_ok: None,
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
        let now = now_unix_ms();
        let mut retained_details = BTreeMap::new();
        for session in &snapshot.sessions {
            if let Some(mut detail) = self.session_details.remove(&session.session_id) {
                detail.session.metadata = session.metadata.clone();
                retained_details.insert(session.session_id.clone(), detail);
            }
        }
        for session_id in snapshot
            .live_sessions
            .iter()
            .map(|session| session.session_id.as_str())
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
        self.last_snapshot_unix_ms = now;
        self.last_error = None;
    }

    pub fn apply_update(&mut self, update: UiUpdate) {
        match update {
            UiUpdate::Snapshot(snapshot) => self.apply_snapshot(*snapshot),
            UiUpdate::SessionDetail(detail) => self.record_session_detail(*detail),
            UiUpdate::ChannelDetail { .. } => {}
            UiUpdate::ChannelAccess { .. } => {}
            UiUpdate::SearchResults { .. } => {}
            UiUpdate::Event(event) => self.record_event(event),
            UiUpdate::SessionEvent(_) => {}
            UiUpdate::RefreshTelemetry {
                duration_ms,
                success,
            } => self.record_refresh_telemetry(duration_ms, success),
            UiUpdate::Error(message) => self.record_error(message),
            UiUpdate::Info(message) => self.record_info(message),
        }
    }

    pub fn record_session_detail(&mut self, detail: SessionDetail) {
        self.session_details
            .insert(detail.session.session_id.clone(), detail);
    }

    pub fn record_event(&mut self, event: EventEnvelope) {
        self.last_event_unix_ms = Some(now_unix_ms());
        self.total_event_count += 1;
        self.recent_events.push(event);
        if self.recent_events.len() > MAX_RECENT_EVENTS {
            let drop_count = self.recent_events.len() - MAX_RECENT_EVENTS;
            self.recent_events.drain(0..drop_count);
        }
    }

    pub fn record_error(&mut self, message: impl Into<String>) {
        let message = message.into();
        self.last_error = Some(message.clone());
        self.last_notice_unix_ms = Some(now_unix_ms());
        self.push_notice(DashboardNoticeLevel::Error, message);
    }

    pub fn record_info(&mut self, message: impl Into<String>) {
        let message = message.into();
        self.last_info = Some(message.clone());
        self.last_notice_unix_ms = Some(now_unix_ms());
        self.push_notice(DashboardNoticeLevel::Info, message);
    }

    pub fn record_refresh_telemetry(&mut self, duration_ms: u64, success: bool) {
        self.last_refresh_duration_ms = Some(duration_ms);
        self.last_refresh_ok = Some(success);
        if success {
            self.refresh_success_count += 1;
        } else {
            self.refresh_failure_count += 1;
        }
    }

    pub fn snapshot_freshness(&self) -> DashboardFreshness {
        freshness_at(now_unix_ms(), self.last_snapshot_unix_ms)
    }

    pub fn snapshot_age_label(&self) -> String {
        format_relative_age(age_seconds(now_unix_ms(), Some(self.last_snapshot_unix_ms)))
    }

    pub fn event_age_label(&self) -> String {
        format_relative_age(age_seconds(now_unix_ms(), self.last_event_unix_ms))
    }

    pub fn notice_age_label(&self) -> String {
        format_relative_age(age_seconds(now_unix_ms(), self.last_notice_unix_ms))
    }

    pub fn last_refresh_latency_label(&self) -> String {
        self.last_refresh_duration_ms
            .map(|duration_ms| format!("{duration_ms}ms"))
            .unwrap_or_else(|| "none yet".to_string())
    }

    pub fn last_refresh_status_label(&self) -> &'static str {
        match self.last_refresh_ok {
            Some(true) => "ok",
            Some(false) => "failed",
            None => "none yet",
        }
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

    fn push_notice(&mut self, level: DashboardNoticeLevel, message: String) {
        self.recent_notices.push(DashboardNotice { level, message });
        if self.recent_notices.len() > MAX_RECENT_NOTICES {
            let drop_count = self.recent_notices.len() - MAX_RECENT_NOTICES;
            self.recent_notices.drain(0..drop_count);
        }
    }
}

pub fn format_relative_age(age_seconds: Option<u64>) -> String {
    match age_seconds {
        None => "none yet".to_string(),
        Some(0) => "just now".to_string(),
        Some(seconds) if seconds < 60 => format!("{seconds}s ago"),
        Some(seconds) if seconds < 3600 => format!("{}m ago", seconds / 60),
        Some(seconds) if seconds < 86_400 => format!("{}h ago", seconds / 3600),
        Some(seconds) => format!("{}d ago", seconds / 86_400),
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

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

fn age_seconds(now_ms: u64, timestamp_ms: Option<u64>) -> Option<u64> {
    timestamp_ms.map(|timestamp_ms| now_ms.saturating_sub(timestamp_ms) / 1000)
}

fn freshness_at(now_ms: u64, last_snapshot_ms: u64) -> DashboardFreshness {
    match now_ms.saturating_sub(last_snapshot_ms) / 1000 {
        0..=10 => DashboardFreshness::Fresh,
        11..=30 => DashboardFreshness::Quiet,
        _ => DashboardFreshness::Stale,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DashboardFreshness, DashboardNoticeLevel, DashboardState, MAX_RECENT_NOTICES,
        format_relative_age, freshness_at,
    };
    use turin_control_client::ConnectionKind;

    fn empty_dashboard() -> DashboardState {
        DashboardState {
            connection_kind: ConnectionKind::Local,
            connection_target: ".turin/config.toml".to_string(),
            health: None,
            status: None,
            live_sessions: Vec::new(),
            sessions: Vec::new(),
            tasks: Vec::new(),
            session_details: Default::default(),
            recent_events: Vec::new(),
            recent_notices: Vec::new(),
            last_snapshot_unix_ms: 0,
            last_event_unix_ms: None,
            last_notice_unix_ms: None,
            total_event_count: 0,
            refresh_success_count: 0,
            refresh_failure_count: 0,
            last_refresh_duration_ms: None,
            last_refresh_ok: None,
            last_error: None,
            last_info: None,
        }
    }

    #[test]
    fn recent_notices_are_bounded_and_keep_latest_entries() {
        let mut dashboard = empty_dashboard();
        for idx in 0..(MAX_RECENT_NOTICES + 4) {
            dashboard.record_info(format!("info-{idx}"));
        }
        dashboard.record_error("boom");

        assert_eq!(dashboard.recent_notices.len(), MAX_RECENT_NOTICES);
        assert_eq!(
            dashboard
                .recent_notices
                .first()
                .map(|notice| notice.message.as_str()),
            Some("info-5")
        );
        assert_eq!(
            dashboard
                .recent_notices
                .last()
                .map(|notice| notice.message.as_str()),
            Some("boom")
        );
        assert_eq!(
            dashboard.recent_notices.last().map(|notice| notice.level),
            Some(DashboardNoticeLevel::Error)
        );
        assert_eq!(dashboard.last_error.as_deref(), Some("boom"));
    }

    #[test]
    fn snapshot_freshness_thresholds_are_stable() {
        assert_eq!(freshness_at(10_000, 10_000), DashboardFreshness::Fresh);
        assert_eq!(freshness_at(22_000, 10_000), DashboardFreshness::Quiet);
        assert_eq!(freshness_at(45_000, 10_000), DashboardFreshness::Stale);
    }

    #[test]
    fn relative_age_labels_cover_common_ranges() {
        assert_eq!(format_relative_age(None), "none yet");
        assert_eq!(format_relative_age(Some(0)), "just now");
        assert_eq!(format_relative_age(Some(9)), "9s ago");
        assert_eq!(format_relative_age(Some(90)), "1m ago");
        assert_eq!(format_relative_age(Some(7200)), "2h ago");
    }

    #[test]
    fn refresh_telemetry_updates_status_and_counts() {
        let mut dashboard = empty_dashboard();
        dashboard.record_refresh_telemetry(84, true);
        dashboard.record_refresh_telemetry(120, false);

        assert_eq!(dashboard.refresh_success_count, 1);
        assert_eq!(dashboard.refresh_failure_count, 1);
        assert_eq!(dashboard.last_refresh_duration_ms, Some(120));
        assert_eq!(dashboard.last_refresh_ok, Some(false));
        assert_eq!(dashboard.last_refresh_latency_label(), "120ms");
        assert_eq!(dashboard.last_refresh_status_label(), "failed");
    }
}
