use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};
use tokio::time::{Instant, MissedTickBehavior};
use turin_channel_core::{
    ChannelCapabilities, ChannelConversationKey, ChannelKind, ConversationBinding, InboundEvent,
    OutboundMessage, RoutingDecision, decide_routing,
};
use turin_daemon_client::DaemonClient;
use turin_daemon_protocol::{
    OpenSessionParams, ResumeSessionParams, RuntimeEventsSubscribeParams, SubmitTaskParams,
    WaitTaskParams,
};

#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub state_path: PathBuf,
    pub idle_ttl: Option<Duration>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct BindingFile {
    bindings: HashMap<String, ConversationBinding>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TaskSnapshot {
    pub request_id: String,
    pub agent_id: String,
    pub slot_id: String,
    pub trace_id: String,
    pub state: String,
    pub runtime_task_id: Option<String>,
    pub status: Option<String>,
    pub task_turn_count: Option<u32>,
    pub output: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelStreamMode {
    Off,
    Typing,
    Draft,
    Block,
}

impl ChannelStreamMode {
    pub fn sends_typing(self) -> bool {
        matches!(self, Self::Typing | Self::Draft | Self::Block)
    }

    pub fn streams_text(self) -> bool {
        matches!(self, Self::Draft | Self::Block)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelProgressUpdate {
    Typing,
    StreamingPreview {
        text: String,
        thinking: Option<String>,
    },
}

pub struct FileBindingStore {
    path: PathBuf,
}

impl FileBindingStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub async fn load(&self) -> Result<HashMap<String, ConversationBinding>> {
        if !self.path.exists() {
            return Ok(HashMap::new());
        }
        let raw = tokio::fs::read_to_string(&self.path)
            .await
            .with_context(|| format!("Failed to read '{}'", self.path.display()))?;
        let file: BindingFile = serde_json::from_str(&raw)
            .with_context(|| format!("Failed to parse '{}'", self.path.display()))?;
        Ok(file.bindings)
    }

    pub async fn save(&self, bindings: &HashMap<String, ConversationBinding>) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let tmp = self.path.with_extension("json.tmp");
        let body = serde_json::to_string_pretty(&BindingFile {
            bindings: bindings.clone(),
        })?;
        tokio::fs::write(&tmp, body).await?;
        tokio::fs::rename(&tmp, &self.path).await?;
        Ok(())
    }
}

#[async_trait]
pub trait ChannelDriver {
    fn kind(&self) -> ChannelKind;

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities::default()
    }

    async fn next_event(&mut self) -> Result<Option<InboundEvent>>;

    async fn send(
        &mut self,
        conversation: &ChannelConversationKey,
        message: OutboundMessage,
    ) -> Result<()>;

    fn stream_mode(&self) -> ChannelStreamMode {
        ChannelStreamMode::Off
    }

    fn stream_include_thinking(&self) -> bool {
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

pub struct ChannelRunner {
    daemon: DaemonClient,
    bindings: FileBindingStore,
    idle_ttl: Option<Duration>,
}

impl ChannelRunner {
    pub fn new(daemon: DaemonClient, config: RunnerConfig) -> Self {
        Self {
            daemon,
            bindings: FileBindingStore::new(config.state_path),
            idle_ttl: config.idle_ttl,
        }
    }

    pub async fn ensure_session(
        &self,
        agent_id: &str,
        key: &ChannelConversationKey,
        reset_requested: bool,
    ) -> Result<ConversationBinding> {
        let mut bindings = self.bindings.load().await?;
        let binding_key = serialize_binding_key(key)?;
        let current = bindings.get(&binding_key);
        let decision = decide_routing(
            key,
            current,
            SystemTime::now(),
            self.idle_ttl,
            reset_requested,
        );

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
                                turin_daemon_protocol::SessionIdParams {
                                    session_id: binding.session_id.clone(),
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
        bindings.insert(binding_key, binding.clone());
        self.bindings.save(&bindings).await?;
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
        self.daemon
            .request_ok(
                None,
                turin_daemon_protocol::DaemonRequest::TaskSubmit(SubmitTaskParams {
                    agent_id: None,
                    session_id: Some(binding.session_id.clone()),
                    prompt: event.text.clone(),
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
        Ok(enrich_outbound_for_event(task_to_outbound(&task), event))
    }

    pub async fn run_driver<D: ChannelDriver + Send>(
        &self,
        agent_id: &str,
        driver: &mut D,
        timeout_ms: Option<u64>,
    ) -> Result<()> {
        let run_result = async {
            while let Some(event) = driver.next_event().await? {
                let reset_requested = event
                    .metadata
                    .get("reset_session")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false);
                let outbound = match self
                    .handle_event_with_driver(agent_id, driver, &event, reset_requested, timeout_ms)
                    .await
                {
                    Ok(message) => message,
                    Err(err) => OutboundMessage::text(format!("Turin error: {}", err)),
                };
                driver.send(&event.conversation, outbound).await?;
            }
            Result::<()>::Ok(())
        }
        .await;

        let shutdown_result = driver.shutdown().await;
        run_result?;
        shutdown_result
    }

    async fn handle_event_with_driver<D: ChannelDriver + Send>(
        &self,
        agent_id: &str,
        driver: &mut D,
        event: &InboundEvent,
        reset_requested: bool,
        timeout_ms: Option<u64>,
    ) -> Result<OutboundMessage> {
        let binding = self
            .ensure_session(agent_id, &event.conversation, reset_requested)
            .await?;
        let stream_mode = driver.stream_mode();
        let session_events = if stream_mode.streams_text() || driver.stream_include_thinking() {
            self.daemon
                .subscribe_managed(RuntimeEventsSubscribeParams {
                    agent_id: None,
                    session_id: Some(binding.session_id.clone()),
                })
                .await
                .ok()
        } else {
            None
        };
        let submitted = self.submit_with_binding(&binding, event).await?;
        let task = self
            .wait_for_task_with_progress(
                driver,
                event,
                &binding,
                &submitted,
                session_events,
                timeout_ms,
            )
            .await?;
        Ok(enrich_outbound_for_event(task_to_outbound(&task), event))
    }

    async fn wait_for_task_with_progress<D: ChannelDriver + Send>(
        &self,
        driver: &mut D,
        event: &InboundEvent,
        binding: &ConversationBinding,
        submitted: &TaskSnapshot,
        mut session_events: Option<turin_daemon_client::ManagedEventStream>,
        timeout_ms: Option<u64>,
    ) -> Result<TaskSnapshot> {
        let stream_mode = driver.stream_mode();
        let include_thinking = driver.stream_include_thinking();
        if stream_mode == ChannelStreamMode::Off {
            return self
                .daemon
                .request_ok(
                    None,
                    turin_daemon_protocol::DaemonRequest::TaskWait(WaitTaskParams {
                        request_id: submitted.request_id.clone(),
                        timeout_ms,
                    }),
                )
                .await;
        }

        if stream_mode.sends_typing() {
            self.try_send_progress(driver, event, ChannelProgressUpdate::Typing)
                .await;
        }

        let wait_task = self.daemon.request_ok(
            None,
            turin_daemon_protocol::DaemonRequest::TaskWait(WaitTaskParams {
                request_id: submitted.request_id.clone(),
                timeout_ms,
            }),
        );
        tokio::pin!(wait_task);

        let mut typing_tick = tokio::time::interval_at(
            Instant::now() + Duration::from_secs(4),
            Duration::from_secs(4),
        );
        typing_tick.set_missed_tick_behavior(MissedTickBehavior::Delay);

        let mut task_started = false;
        let mut text_preview = String::new();
        let mut thinking_preview = String::new();
        let mut last_flushed_chars = 0usize;
        let mut last_flush_at = Instant::now();

        loop {
            tokio::select! {
                result = &mut wait_task => {
                    if stream_mode.streams_text()
                        && preview_char_count(&text_preview, include_thinking.then_some(thinking_preview.as_str())) > last_flushed_chars
                    {
                        self.try_send_progress(
                            driver,
                            event,
                            ChannelProgressUpdate::StreamingPreview {
                                text: text_preview.clone(),
                                thinking: preview_thinking(include_thinking, &thinking_preview),
                            },
                        )
                        .await;
                    }
                    return result;
                }
                _ = typing_tick.tick(), if stream_mode.sends_typing() => {
                    self.try_send_progress(driver, event, ChannelProgressUpdate::Typing).await;
                }
                event_result = next_managed_event(session_events.as_mut()), if session_events.is_some() => {
                    let Ok(kernel_event) = event_result else {
                        session_events = None;
                        continue;
                    };
                    if kernel_event.data.get("session_id").and_then(|value| value.as_str()) != Some(binding.session_id.as_str()) {
                        continue;
                    }

                    match kernel_event.event.as_str() {
                        "task_start" if kernel_event.data.get("trace_id").and_then(|value| value.as_str()) == Some(submitted.trace_id.as_str()) => {
                            task_started = true;
                        }
                        "message_delta" if task_started => {
                            if let Some(delta) = kernel_event.data.get("content_delta").and_then(|value| value.as_str()) {
                                text_preview.push_str(delta);
                            }
                            if should_flush_preview(
                                stream_mode,
                                &text_preview,
                                include_thinking.then_some(thinking_preview.as_str()),
                                last_flushed_chars,
                                last_flush_at,
                            ) {
                                self.try_send_progress(
                                    driver,
                                    event,
                                    ChannelProgressUpdate::StreamingPreview {
                                        text: text_preview.clone(),
                                        thinking: preview_thinking(include_thinking, &thinking_preview),
                                    },
                                )
                                .await;
                                last_flushed_chars = preview_char_count(
                                    &text_preview,
                                    include_thinking.then_some(thinking_preview.as_str()),
                                );
                                last_flush_at = Instant::now();
                            }
                        }
                        "thinking_delta" if task_started && include_thinking => {
                            if let Some(delta) = kernel_event.data.get("thinking").and_then(|value| value.as_str()) {
                                thinking_preview.push_str(delta);
                            }
                            if should_flush_preview(
                                stream_mode,
                                &text_preview,
                                Some(thinking_preview.as_str()),
                                last_flushed_chars,
                                last_flush_at,
                            ) {
                                self.try_send_progress(
                                    driver,
                                    event,
                                    ChannelProgressUpdate::StreamingPreview {
                                        text: text_preview.clone(),
                                        thinking: preview_thinking(include_thinking, &thinking_preview),
                                    },
                                )
                                .await;
                                last_flushed_chars = preview_char_count(
                                    &text_preview,
                                    Some(thinking_preview.as_str()),
                                );
                                last_flush_at = Instant::now();
                            }
                        }
                        "message_end" if task_started => {
                            if preview_char_count(
                                &text_preview,
                                include_thinking.then_some(thinking_preview.as_str()),
                            ) > last_flushed_chars {
                                self.try_send_progress(
                                    driver,
                                    event,
                                    ChannelProgressUpdate::StreamingPreview {
                                        text: text_preview.clone(),
                                        thinking: preview_thinking(include_thinking, &thinking_preview),
                                    },
                                )
                                .await;
                                last_flushed_chars = preview_char_count(
                                    &text_preview,
                                    include_thinking.then_some(thinking_preview.as_str()),
                                );
                                last_flush_at = Instant::now();
                            }
                        }
                        "task_complete" if kernel_event.data.get("trace_id").and_then(|value| value.as_str()) == Some(submitted.trace_id.as_str()) => {
                            task_started = false;
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    async fn try_send_progress<D: ChannelDriver + Send>(
        &self,
        driver: &mut D,
        event: &InboundEvent,
        update: ChannelProgressUpdate,
    ) {
        let _ = driver.send_progress(event, update).await;
    }

    pub async fn clear_binding(&self, key: &ChannelConversationKey) -> Result<()> {
        let mut bindings = self.bindings.load().await?;
        bindings.remove(&serialize_binding_key(key)?);
        self.bindings.save(&bindings).await
    }

    pub async fn prune_expired(&self) -> Result<()> {
        let Some(ttl) = self.idle_ttl else {
            return Ok(());
        };
        let mut bindings = self.bindings.load().await?;
        let now = SystemTime::now();
        bindings.retain(|_, binding| !binding.is_expired(now, ttl));
        self.bindings.save(&bindings).await
    }
}

fn serialize_binding_key(key: &ChannelConversationKey) -> Result<String> {
    Ok(serde_json::to_string(key)?)
}

fn task_to_outbound(task: &TaskSnapshot) -> OutboundMessage {
    if let Some(output) = task.output.as_ref() {
        if let Some(structured) = try_parse_structured_outbound(output) {
            structured
        } else {
            OutboundMessage::text(output.clone())
        }
    } else if let Some(error) = task.error.as_ref() {
        OutboundMessage::text(format!("Turin error: {}", error))
    } else {
        OutboundMessage::text(format!("Task {} finished without output", task.request_id))
    }
}

async fn next_managed_event(
    stream: Option<&mut turin_daemon_client::ManagedEventStream>,
) -> Result<turin_daemon_protocol::EventEnvelope> {
    let stream = stream.context("managed event stream missing")?;
    stream.next_event().await
}

fn should_flush_preview(
    stream_mode: ChannelStreamMode,
    text_preview: &str,
    thinking_preview: Option<&str>,
    last_flushed_chars: usize,
    last_flush_at: Instant,
) -> bool {
    let current_chars = preview_char_count(text_preview, thinking_preview);
    if current_chars <= last_flushed_chars {
        return false;
    }

    let new_chars = current_chars.saturating_sub(last_flushed_chars);
    match stream_mode {
        ChannelStreamMode::Draft => {
            new_chars >= 32 || last_flush_at.elapsed() >= Duration::from_millis(800)
        }
        ChannelStreamMode::Block => {
            new_chars >= 160
                || (new_chars >= 64
                    && last_flush_at.elapsed() >= Duration::from_millis(1500)
                    && (text_preview.ends_with('\n') || text_preview.ends_with(". ")))
        }
        _ => false,
    }
}

fn preview_char_count(text_preview: &str, thinking_preview: Option<&str>) -> usize {
    text_preview.chars().count()
        + thinking_preview
            .map(|thinking| thinking.chars().count())
            .unwrap_or_default()
}

fn preview_thinking(include_thinking: bool, thinking_preview: &str) -> Option<String> {
    if !include_thinking || thinking_preview.trim().is_empty() {
        return None;
    }
    Some(thinking_preview.to_string())
}

fn enrich_outbound_for_event(
    mut outbound: OutboundMessage,
    event: &InboundEvent,
) -> OutboundMessage {
    if event.conversation.channel == ChannelKind::Telegram
        && !outbound
            .metadata
            .contains_key("telegram_reply_to_message_id")
        && let Some(message_id) = event.metadata.get("telegram_message_id")
    {
        outbound.metadata.insert(
            "telegram_reply_to_message_id".to_string(),
            message_id.clone(),
        );
    }
    outbound
}

#[derive(Debug, Clone, Deserialize)]
struct StructuredOutbound {
    #[serde(default)]
    _turin_channel_outbound: bool,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    blocks: Vec<turin_channel_core::MessageBlock>,
    #[serde(default)]
    attachments: Vec<turin_channel_core::ChannelAttachment>,
    #[serde(default)]
    embeds: Vec<Value>,
    #[serde(default)]
    components: Vec<Value>,
    #[serde(default)]
    metadata: Map<String, Value>,
}

fn try_parse_structured_outbound(raw: &str) -> Option<OutboundMessage> {
    let trimmed = raw.trim();
    if !trimmed.starts_with('{') {
        return None;
    }

    let parsed: StructuredOutbound = serde_json::from_str(trimmed).ok()?;
    if !parsed._turin_channel_outbound {
        return None;
    }

    let mut blocks = parsed.blocks;
    if blocks.is_empty()
        && let Some(content) = parsed.content
        && !content.trim().is_empty()
    {
        blocks.push(turin_channel_core::MessageBlock::Text { text: content });
    }

    Some(OutboundMessage {
        blocks,
        attachments: parsed.attachments,
        embeds: parsed.embeds,
        components: parsed.components,
        metadata: parsed.metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use turin_channel_core::{ChannelKind, ChannelMessageRef, ChannelUser, MessageBlock};

    fn sample_key() -> ChannelConversationKey {
        ChannelConversationKey {
            channel: ChannelKind::Discord,
            workspace_id: "guild".into(),
            room_id: Some("room".into()),
            thread_id: "thread".into(),
            user_id: Some("user".into()),
        }
    }

    #[tokio::test]
    async fn file_binding_store_round_trips() {
        let dir = tempdir().unwrap();
        let store = FileBindingStore::new(dir.path().join("bindings.json"));
        let key = serialize_binding_key(&sample_key()).unwrap();
        let mut map = HashMap::new();
        map.insert(
            key,
            ConversationBinding::new("writer", "session-1", &sample_key(), SystemTime::UNIX_EPOCH),
        );
        store.save(&map).await.unwrap();
        let loaded = store.load().await.unwrap();
        assert_eq!(loaded.len(), 1);
    }

    #[test]
    fn inbound_event_shape_is_runner_compatible() {
        let key = sample_key();
        let event = InboundEvent {
            conversation: key.clone(),
            message: ChannelMessageRef {
                conversation: key,
                message_id: "m1".into(),
            },
            user: ChannelUser {
                id: "u1".into(),
                display_name: Some("User".into()),
                username: Some("user".into()),
            },
            text: "hello".into(),
            attachments: vec![],
            metadata: Default::default(),
        };
        assert_eq!(event.text, "hello");
    }

    #[test]
    fn task_to_outbound_prefers_output() {
        let outbound = task_to_outbound(&TaskSnapshot {
            request_id: "req-1".into(),
            agent_id: "writer".into(),
            slot_id: "slot-1".into(),
            trace_id: "trace-1".into(),
            state: "completed".into(),
            runtime_task_id: None,
            status: Some("completed".into()),
            task_turn_count: Some(1),
            output: Some("hello".into()),
            error: Some("bad".into()),
        });
        assert_eq!(
            outbound.blocks,
            vec![MessageBlock::Text {
                text: "hello".into(),
            }]
        );
    }

    #[test]
    fn task_to_outbound_parses_structured_payload() {
        let outbound = task_to_outbound(&TaskSnapshot {
            request_id: "req-1".into(),
            agent_id: "writer".into(),
            slot_id: "slot-1".into(),
            trace_id: "trace-1".into(),
            state: "completed".into(),
            runtime_task_id: None,
            status: Some("completed".into()),
            task_turn_count: Some(1),
            output: Some(
                serde_json::json!({
                    "_turin_channel_outbound": true,
                    "content": "overview",
                    "embeds": [{ "title": "Build result" }],
                    "components": [{ "type": 1, "components": [] }],
                    "metadata": { "priority": "high" }
                })
                .to_string(),
            ),
            error: None,
        });
        assert_eq!(outbound.blocks.len(), 1);
        assert_eq!(outbound.embeds.len(), 1);
        assert_eq!(outbound.components.len(), 1);
        assert_eq!(outbound.metadata["priority"], "high");
    }

    #[test]
    fn telegram_outbound_defaults_to_replying_to_source_message() {
        let key = ChannelConversationKey {
            channel: ChannelKind::Telegram,
            workspace_id: "telegram".into(),
            room_id: Some("-1001".into()),
            thread_id: "-1001".into(),
            user_id: Some("user-1".into()),
        };
        let mut metadata = Map::new();
        metadata.insert("telegram_message_id".to_string(), Value::from(42));
        let event = InboundEvent {
            conversation: key.clone(),
            message: ChannelMessageRef {
                conversation: key,
                message_id: "m-42".into(),
            },
            user: ChannelUser {
                id: "user-1".into(),
                display_name: Some("User One".into()),
                username: Some("user1".into()),
            },
            text: "hello".into(),
            attachments: vec![],
            metadata,
        };

        let enriched = enrich_outbound_for_event(OutboundMessage::text("reply"), &event);
        assert_eq!(enriched.metadata["telegram_reply_to_message_id"], 42);
    }

    #[test]
    fn telegram_outbound_keeps_explicit_reply_override() {
        let key = ChannelConversationKey {
            channel: ChannelKind::Telegram,
            workspace_id: "telegram".into(),
            room_id: Some("-1001".into()),
            thread_id: "-1001".into(),
            user_id: Some("user-1".into()),
        };
        let mut event_metadata = Map::new();
        event_metadata.insert("telegram_message_id".to_string(), Value::from(42));
        let event = InboundEvent {
            conversation: key.clone(),
            message: ChannelMessageRef {
                conversation: key,
                message_id: "m-42".into(),
            },
            user: ChannelUser {
                id: "user-1".into(),
                display_name: Some("User One".into()),
                username: Some("user1".into()),
            },
            text: "hello".into(),
            attachments: vec![],
            metadata: event_metadata,
        };

        let mut outbound = OutboundMessage::text("reply");
        outbound
            .metadata
            .insert("telegram_reply_to_message_id".to_string(), Value::from(7));

        let enriched = enrich_outbound_for_event(outbound, &event);
        assert_eq!(enriched.metadata["telegram_reply_to_message_id"], 7);
    }

    #[test]
    fn telegram_outbound_allows_clearing_default_reply_target() {
        let key = ChannelConversationKey {
            channel: ChannelKind::Telegram,
            workspace_id: "telegram".into(),
            room_id: Some("-1001".into()),
            thread_id: "-1001".into(),
            user_id: Some("user-1".into()),
        };
        let mut event_metadata = Map::new();
        event_metadata.insert("telegram_message_id".to_string(), Value::from(42));
        let event = InboundEvent {
            conversation: key.clone(),
            message: ChannelMessageRef {
                conversation: key,
                message_id: "m-42".into(),
            },
            user: ChannelUser {
                id: "user-1".into(),
                display_name: Some("User One".into()),
                username: Some("user1".into()),
            },
            text: "hello".into(),
            attachments: vec![],
            metadata: event_metadata,
        };

        let mut outbound = OutboundMessage::text("reply");
        outbound
            .metadata
            .insert("telegram_reply_to_message_id".to_string(), Value::Null);

        let enriched = enrich_outbound_for_event(outbound, &event);
        assert_eq!(
            enriched.metadata.get("telegram_reply_to_message_id"),
            Some(&Value::Null)
        );
    }
}
