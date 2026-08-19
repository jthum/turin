use anyhow::{Context, Result};
use async_trait::async_trait;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::Mutex;
use turin_channel_core::{
    ChannelCapabilities, ChannelConversationKey, ChannelKind, ChannelUser, ConversationBinding,
    InboundEvent, OutboundMessage, RoutingDecision, decide_routing,
};
use turin_daemon_client::DaemonClient;
use turin_daemon_protocol::{
    OpenSessionParams, ResumeSessionParams, SubmitTaskParams, WaitTaskParams,
};
use turin_types::ToolsConfig;

mod access;
mod bindings;
mod config;
mod driver_loop;
mod presence;
mod sidecar;
mod stream;
mod task_payloads;

pub use access::{
    ApprovedRoomView, ChannelAccessPolicy, ChannelAccessSnapshot, ChannelRoomRef,
    FileAccessStateStore, PairingMode, PendingRoomView,
};
pub use bindings::FileBindingStore;
pub use config::{RunnerConfig, task_timeout_ms_from_settings, tools_config_from_settings};
pub use presence::{RunnerPresence, announce_runner_presence, spawn_runner_heartbeat};
pub use sidecar::{
    ChannelSidecarRun, ChannelSidecarRunArgs, init_channel_tracing, parse_auth_flow_poll_request,
    parse_auth_flow_start_request, parse_channel_settings_json, prepare_channel_sidecar_run,
};
pub use stream::{ChannelProgressUpdate, ChannelStreamMode};
pub use task_payloads::TaskSnapshot;

#[cfg(test)]
pub(crate) use access::AccessStateFile;
pub(crate) use access::{
    ApprovedRoom, ChannelRoomKey, PendingRoom, serialize_room_key, unix_seconds,
};
pub(crate) use bindings::serialize_binding_key;
pub(crate) use task_payloads::{
    task_input_content_from_event, task_prompt_for_submission, task_to_outbound,
};

enum EventAccessDecision {
    Allow,
    Pending { notify: bool },
    Ignore,
}

#[async_trait]
pub trait ChannelDriver {
    fn kind(&self) -> ChannelKind;

    fn user_matches_selector(&self, selector: &str, user: &ChannelUser) -> bool;

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities::default()
    }

    async fn next_event(&mut self) -> Result<Option<InboundEvent>>;

    async fn send(
        &mut self,
        conversation: &ChannelConversationKey,
        message: OutboundMessage,
    ) -> Result<()>;

    fn enrich_outbound_for_event(
        &self,
        _event: &InboundEvent,
        outbound: OutboundMessage,
    ) -> OutboundMessage {
        outbound
    }

    fn stream_mode(&self) -> ChannelStreamMode {
        ChannelStreamMode::Off
    }

    fn stream_thinking(&self) -> bool {
        false
    }

    fn persist_thinking(&self) -> bool {
        false
    }

    async fn send_progress(
        &mut self,
        _event: &InboundEvent,
        _update: ChannelProgressUpdate,
    ) -> Result<()> {
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()>;
}

#[derive(Clone)]
pub struct ChannelRunner {
    daemon: DaemonClient,
    channel_id: String,
    bindings: FileBindingStore,
    bindings_lock: Arc<Mutex<()>>,
    access_state: FileAccessStateStore,
    idle_ttl: Option<Duration>,
    access_policy: ChannelAccessPolicy,
    tools: ToolsConfig,
}

impl ChannelRunner {
    pub fn new(daemon: DaemonClient, config: RunnerConfig) -> Self {
        Self {
            daemon,
            channel_id: config.channel_id,
            bindings: FileBindingStore::new(config.state_path),
            bindings_lock: Arc::new(Mutex::new(())),
            access_state: FileAccessStateStore::new(config.access_state_path),
            idle_ttl: config.idle_ttl,
            access_policy: config.access_policy,
            tools: config.tools,
        }
    }

    pub async fn ensure_session(
        &self,
        agent_id: &str,
        key: &ChannelConversationKey,
        reset_requested: bool,
    ) -> Result<ConversationBinding> {
        let binding_key = serialize_binding_key(key)?;
        let (current, decision) = {
            let _guard = self.bindings_lock.lock().await;
            let bindings = self.bindings.load().await?;
            let current = bindings.get(&binding_key).cloned();
            let decision = decide_routing(
                key,
                current.as_ref(),
                SystemTime::now(),
                self.idle_ttl,
                reset_requested,
            );
            (current, decision)
        };

        let session: serde_json::Value = match decision {
            RoutingDecision::Reuse {
                slot_id,
                session_id,
            } => {
                self.daemon
                    .request_ok(
                        None,
                        turin_daemon_protocol::DaemonRequest::SessionResume(ResumeSessionParams {
                            session_id,
                            slot_id: Some(slot_id),
                        }),
                    )
                    .await?
            }
            RoutingDecision::StartFresh { slot_id } => {
                if let Some(binding) = current {
                    let _ = self
                        .daemon
                        .request_ok::<serde_json::Value>(
                            None,
                            turin_daemon_protocol::DaemonRequest::SessionKill(
                                turin_daemon_protocol::LiveSessionTargetParams {
                                    session_id: binding.session_id.clone(),
                                    slot_id: Some(binding.slot_id.clone()),
                                    recursive: false,
                                },
                            ),
                        )
                        .await;
                }
                self.daemon
                    .request_ok(
                        None,
                        turin_daemon_protocol::DaemonRequest::SessionOpen(OpenSessionParams {
                            agent_id: agent_id.to_string(),
                            slot_id: Some(slot_id),
                            channel_id: Some(self.channel_id.clone()),
                        }),
                    )
                    .await?
            }
        };

        let session_id = session
            .get("session_id")
            .and_then(|v| v.as_str())
            .context("daemon session response missing session_id")?;
        let mut binding = ConversationBinding::new(agent_id, session_id, key, SystemTime::now());
        binding.touch(SystemTime::now());
        {
            let _guard = self.bindings_lock.lock().await;
            let mut bindings = self.bindings.load().await?;
            bindings.insert(binding_key, binding.clone());
            self.bindings.save(&bindings).await?;
        }
        Ok(binding)
    }

    pub async fn submit(
        &self,
        agent_id: &str,
        event: &InboundEvent,
        reset_requested: bool,
    ) -> Result<TaskSnapshot> {
        let binding = self
            .ensure_session(agent_id, &event.conversation, reset_requested)
            .await?;
        self.submit_with_binding(&binding, event).await
    }

    pub async fn submit_with_binding(
        &self,
        binding: &ConversationBinding,
        event: &InboundEvent,
    ) -> Result<TaskSnapshot> {
        let content = task_input_content_from_event(event);
        self.daemon
            .request_ok(
                None,
                turin_daemon_protocol::DaemonRequest::TaskSubmit(SubmitTaskParams {
                    agent_id: None,
                    session_id: Some(binding.session_id.clone()),
                    slot_id: Some(binding.slot_id.clone()),
                    prompt: task_prompt_for_submission(event),
                    inference_context: None,
                    content: (!content.is_empty()).then_some(content),
                    tools: (!self.tools.is_empty()).then_some(self.tools.clone()),
                    conflict_policy: None,
                }),
            )
            .await
    }

    pub async fn submit_and_wait(
        &self,
        agent_id: &str,
        event: &InboundEvent,
        reset_requested: bool,
        timeout_ms: Option<u64>,
    ) -> Result<TaskSnapshot> {
        let submitted = self.submit(agent_id, event, reset_requested).await?;
        self.daemon
            .request_ok(
                None,
                turin_daemon_protocol::DaemonRequest::TaskWait(WaitTaskParams {
                    request_id: submitted.request_id,
                    timeout_ms,
                }),
            )
            .await
    }

    pub async fn handle_event(
        &self,
        agent_id: &str,
        event: &InboundEvent,
        reset_requested: bool,
        timeout_ms: Option<u64>,
    ) -> Result<OutboundMessage> {
        let task = self
            .submit_and_wait(agent_id, event, reset_requested, timeout_ms)
            .await?;
        Ok(task_to_outbound(&task))
    }

    async fn authorize_event<D: ChannelDriver + Send>(
        &self,
        driver: &D,
        event: &InboundEvent,
    ) -> Result<EventAccessDecision> {
        if matches!(self.access_policy.pairing_mode, PairingMode::Off) {
            return Ok(
                if self
                    .access_policy
                    .allows_interaction(&event.user, |selector, user| {
                        driver.user_matches_selector(selector, user)
                    })
                {
                    EventAccessDecision::Allow
                } else {
                    EventAccessDecision::Ignore
                },
            );
        }

        let room = ChannelRoomKey::from(&event.conversation);
        let room_key = serialize_room_key(&room)?;
        let mut state = self.access_state.load().await?;
        if state.approved_rooms.contains_key(&room_key) {
            return Ok(
                if self
                    .access_policy
                    .allows_interaction(&event.user, |selector, user| {
                        driver.user_matches_selector(selector, user)
                    })
                {
                    EventAccessDecision::Allow
                } else {
                    EventAccessDecision::Ignore
                },
            );
        }

        if self.access_policy.is_banned(&event.user, |selector, user| {
            driver.user_matches_selector(selector, user)
        }) {
            return Ok(EventAccessDecision::Ignore);
        }

        if !self
            .access_policy
            .allows_pairing(&event.user, |selector, user| {
                driver.user_matches_selector(selector, user)
            })
        {
            return Ok(EventAccessDecision::Ignore);
        }

        match self.access_policy.pairing_mode {
            PairingMode::Off => Ok(EventAccessDecision::Allow),
            PairingMode::Auto => {
                let now = SystemTime::now();
                state.approved_rooms.insert(
                    room_key.clone(),
                    ApprovedRoom {
                        room,
                        approved_at_unix_seconds: unix_seconds(now),
                        approved_by_user_id: Some(event.user.id.clone()),
                        approved_by_username: event.user.username.clone(),
                    },
                );
                state.pending_rooms.remove(&room_key);
                self.access_state.save(&state).await?;
                Ok(EventAccessDecision::Allow)
            }
            PairingMode::Pending => {
                let now_seconds = unix_seconds(SystemTime::now());
                let notify = match state.pending_rooms.get_mut(&room_key) {
                    Some(existing) => {
                        existing.last_seen_unix_seconds = now_seconds;
                        false
                    }
                    None => {
                        state.pending_rooms.insert(
                            room_key,
                            PendingRoom {
                                room,
                                first_seen_unix_seconds: now_seconds,
                                last_seen_unix_seconds: now_seconds,
                                sample_user_id: Some(event.user.id.clone()),
                                sample_username: event.user.username.clone(),
                            },
                        );
                        true
                    }
                };
                self.access_state.save(&state).await?;
                Ok(EventAccessDecision::Pending { notify })
            }
        }
    }

    pub async fn clear_binding(&self, key: &ChannelConversationKey) -> Result<()> {
        let _guard = self.bindings_lock.lock().await;
        let mut bindings = self.bindings.load().await?;
        bindings.remove(&serialize_binding_key(key)?);
        self.bindings.save(&bindings).await
    }

    pub async fn prune_expired(&self) -> Result<()> {
        let Some(ttl) = self.idle_ttl else {
            return Ok(());
        };
        let _guard = self.bindings_lock.lock().await;
        let mut bindings = self.bindings.load().await?;
        let now = SystemTime::now();
        bindings.retain(|_, binding| !binding.is_expired(now, ttl));
        self.bindings.save(&bindings).await
    }
}

#[cfg(test)]
mod tests;
