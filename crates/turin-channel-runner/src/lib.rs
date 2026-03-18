use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};
use turin_channel_core::{
    ChannelCapabilities, ChannelConversationKey, ChannelKind, ConversationBinding, InboundEvent,
    OutboundMessage, RoutingDecision, decide_routing,
};
use turin_daemon_client::DaemonClient;
use turin_daemon_protocol::{
    OpenSessionParams, ResumeSessionParams, SubmitTaskParams, WaitTaskParams,
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
        self.daemon
            .request_ok(
                None,
                turin_daemon_protocol::DaemonRequest::TaskSubmit(SubmitTaskParams {
                    agent_id: None,
                    session_id: Some(binding.session_id),
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
                    .handle_event(agent_id, &event, reset_requested, timeout_ms)
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
