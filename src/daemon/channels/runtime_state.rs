use serde::Serialize;

use super::DesiredChannel;

pub(super) const STATE_STARTING: &str = "starting";
pub(super) const STATE_RUNNING: &str = "running";
pub(super) const STATE_STOPPED: &str = "stopped";
pub(super) const STATE_FAILED: &str = "failed";
pub(super) const STATE_UNSUPPORTED: &str = "unsupported";

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChannelRunnerHandshakeSnapshot {
    pub display_name: String,
    pub protocol_version: u32,
    pub runner_binary: Option<String>,
    pub runner_version: Option<String>,
    pub pid: Option<u32>,
    pub last_handshake_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChannelRuntimeSnapshot {
    pub id: String,
    pub kind: String,
    pub agent_id: String,
    pub directory: String,
    pub state: String,
    pub last_error: Option<String>,
    pub last_error_code: Option<String>,
    pub start_count: u64,
    pub restart_count: u64,
    pub failure_count: u64,
    pub last_transition_unix_ms: u64,
    pub last_started_unix_ms: Option<u64>,
    pub last_stopped_unix_ms: Option<u64>,
    pub handshake: Option<ChannelRunnerHandshakeSnapshot>,
}

impl ChannelRuntimeSnapshot {
    pub(super) fn new_for_channel(channel: &DesiredChannel, state: &'static str, now: u64) -> Self {
        Self {
            id: channel.id.clone(),
            kind: channel.kind.clone(),
            agent_id: channel.agent_id.clone(),
            directory: channel.directory.display().to_string(),
            state: state.to_string(),
            last_error: None,
            last_error_code: None,
            start_count: 0,
            restart_count: 0,
            failure_count: 0,
            last_transition_unix_ms: now,
            last_started_unix_ms: None,
            last_stopped_unix_ms: Some(now),
            handshake: None,
        }
    }

    pub(super) fn refresh_channel_identity(&mut self, channel: &DesiredChannel) {
        self.kind = channel.kind.clone();
        self.agent_id = channel.agent_id.clone();
        self.directory = channel.directory.display().to_string();
    }

    pub(super) fn mark_starting(
        &mut self,
        channel: &DesiredChannel,
        now: u64,
        count_restart: bool,
    ) {
        self.refresh_channel_identity(channel);
        self.state = STATE_STARTING.to_string();
        self.last_error = None;
        self.last_error_code = None;
        self.start_count = self.start_count.saturating_add(1);
        if count_restart {
            self.restart_count = self.restart_count.saturating_add(1);
        }
        self.last_transition_unix_ms = now;
    }

    pub(super) fn mark_running(&mut self, now: u64) {
        self.state = STATE_RUNNING.to_string();
        self.last_error = None;
        self.last_error_code = None;
        self.last_transition_unix_ms = now;
        self.last_started_unix_ms = Some(now);
    }

    pub(super) fn mark_stopped(&mut self, now: u64) {
        self.state = STATE_STOPPED.to_string();
        self.last_transition_unix_ms = now;
        self.last_stopped_unix_ms = Some(now);
    }

    pub(super) fn mark_clean_stopped(&mut self, now: u64) {
        self.mark_stopped(now);
        self.last_error = None;
        self.last_error_code = None;
    }

    pub(super) fn mark_failed(&mut self, error: String, error_code: String, now: u64) {
        self.state = STATE_FAILED.to_string();
        self.last_error = Some(error);
        self.last_error_code = Some(error_code);
        self.failure_count = self.failure_count.saturating_add(1);
        self.last_transition_unix_ms = now;
        self.last_stopped_unix_ms = Some(now);
    }

    pub(super) fn mark_unsupported(&mut self, channel: &DesiredChannel, now: u64) {
        self.refresh_channel_identity(channel);
        self.state = STATE_UNSUPPORTED.to_string();
        self.last_error = Some(format!(
            "No built-in or external runner is available for channel kind '{}'",
            channel.kind,
        ));
        self.last_error_code = Some("channel_kind_unsupported".to_string());
        self.last_transition_unix_ms = now;
        self.last_stopped_unix_ms = Some(now);
    }
}
