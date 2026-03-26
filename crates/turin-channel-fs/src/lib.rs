use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::watch;
use tokio::time::sleep;
use turin_channel_core::{
    ChannelAttachment, ChannelConversationKey, ChannelKind, ChannelMessageRef, ChannelUser,
    InboundEvent, OutboundMessage,
};
use turin_channel_runner::ChannelDriver;

#[derive(Debug, Clone)]
pub struct FsChannelDriverConfig {
    pub inbox_dir: PathBuf,
    pub outbox_dir: PathBuf,
    pub processed_dir: PathBuf,
    pub failed_dir: PathBuf,
    pub poll_interval: Duration,
}

pub fn validate_settings(channel_dir: &Path, settings: &serde_json::Value) -> Result<()> {
    FsChannelDriverConfig::from_settings(channel_dir, settings).map(|_| ())
}

impl FsChannelDriverConfig {
    pub fn from_settings(channel_dir: &Path, settings: &serde_json::Value) -> Result<Self> {
        let settings = settings
            .as_object()
            .ok_or_else(|| anyhow!("FS channel settings must be a JSON object"))?;

        let inbox_dir = resolve_dir(channel_dir, settings.get("inbox_dir"), "inbox")?;
        let outbox_dir = resolve_dir(channel_dir, settings.get("outbox_dir"), "outbox")?;
        let processed_dir = resolve_dir(channel_dir, settings.get("processed_dir"), "processed")?;
        let failed_dir = resolve_dir(channel_dir, settings.get("failed_dir"), "failed")?;

        let poll_interval_ms = match settings.get("poll_interval_ms") {
            None => 250,
            Some(value) => {
                let interval = value.as_u64().ok_or_else(|| {
                    anyhow!("FS channel setting 'poll_interval_ms' must be a positive integer")
                })?;
                if interval < 10 {
                    anyhow::bail!("FS channel setting 'poll_interval_ms' must be >= 10");
                }
                interval
            }
        };

        Ok(Self {
            inbox_dir,
            outbox_dir,
            processed_dir,
            failed_dir,
            poll_interval: Duration::from_millis(poll_interval_ms),
        })
    }
}

pub struct FsChannelDriver {
    channel_id: String,
    config: FsChannelDriverConfig,
    shutdown_rx: watch::Receiver<bool>,
    backlog: VecDeque<PathBuf>,
}

impl FsChannelDriver {
    pub async fn from_settings(
        channel_id: impl Into<String>,
        channel_dir: impl AsRef<Path>,
        settings: &serde_json::Value,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Self> {
        let channel_dir = channel_dir.as_ref();
        let config = FsChannelDriverConfig::from_settings(channel_dir, settings)?;
        ensure_dir(&config.inbox_dir).await?;
        ensure_dir(&config.outbox_dir).await?;
        ensure_dir(&config.processed_dir).await?;
        ensure_dir(&config.failed_dir).await?;

        Ok(Self {
            channel_id: channel_id.into(),
            config,
            shutdown_rx,
            backlog: VecDeque::new(),
        })
    }

    async fn next_inbound_file(&mut self) -> Result<Option<PathBuf>> {
        if let Some(path) = self.backlog.pop_front() {
            return Ok(Some(path));
        }

        let mut entries = tokio::fs::read_dir(&self.config.inbox_dir).await?;
        let mut paths = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("json") {
                paths.push(path);
            }
        }

        paths.sort();
        self.backlog.extend(paths);
        Ok(self.backlog.pop_front())
    }

    async fn load_event(&self, path: &Path) -> Result<InboundEvent> {
        let raw = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("Failed to read '{}'", path.display()))?;
        let parsed: FsInboundMessage = serde_json::from_str(&raw)
            .with_context(|| format!("Failed to parse '{}'", path.display()))?;

        let message_id = parsed
            .message_id
            .unwrap_or_else(|| default_message_id(path));

        Ok(InboundEvent {
            message: ChannelMessageRef {
                conversation: parsed.conversation.clone(),
                message_id,
            },
            conversation: parsed.conversation,
            user: parsed.user,
            text: parsed.text,
            attachments: parsed.attachments,
            metadata: parsed.metadata,
        })
    }

    async fn mark_processed(&self, path: &Path) -> Result<()> {
        move_file(path, &self.config.processed_dir).await
    }

    async fn mark_failed(&self, path: &Path) -> Result<()> {
        move_file(path, &self.config.failed_dir).await
    }
}

#[async_trait]
impl ChannelDriver for FsChannelDriver {
    fn kind(&self) -> ChannelKind {
        ChannelKind::Other("fs".to_string())
    }

    fn user_matches_selector(&self, selector: &str, user: &ChannelUser) -> bool {
        let selector = selector.trim();
        if selector.is_empty() {
            return false;
        }
        user.id == selector
            || user
                .username
                .as_ref()
                .is_some_and(|username| username.eq_ignore_ascii_case(selector))
    }

    async fn next_event(&mut self) -> Result<Option<InboundEvent>> {
        loop {
            if *self.shutdown_rx.borrow() {
                return Ok(None);
            }

            if let Some(path) = self.next_inbound_file().await? {
                match self.load_event(&path).await {
                    Ok(event) => {
                        self.mark_processed(&path).await?;
                        return Ok(Some(event));
                    }
                    Err(err) => {
                        self.mark_failed(&path).await?;
                        tracing::warn!(
                            channel_id = %self.channel_id,
                            path = %path.display(),
                            error = %err,
                            "Failed to parse filesystem channel message"
                        );
                    }
                }
            }

            tokio::select! {
                changed = self.shutdown_rx.changed() => {
                    if changed.is_ok() && *self.shutdown_rx.borrow() {
                        return Ok(None);
                    }
                }
                _ = sleep(self.config.poll_interval) => {}
            }
        }
    }

    async fn send(
        &mut self,
        conversation: &ChannelConversationKey,
        message: OutboundMessage,
    ) -> Result<()> {
        let out = FsOutboundMessage {
            channel_id: self.channel_id.clone(),
            conversation: conversation.clone(),
            message,
            timestamp_ms: unix_millis(),
        };
        let filename = format!("{}-{}.json", unix_millis(), uuid::Uuid::now_v7().simple());
        let path = self.config.outbox_dir.join(filename);
        let body = serde_json::to_string_pretty(&out)?;
        tokio::fs::write(&path, body)
            .await
            .with_context(|| format!("Failed to write '{}'", path.display()))?;
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
struct FsInboundMessage {
    conversation: ChannelConversationKey,
    #[serde(default)]
    message_id: Option<String>,
    user: ChannelUser,
    text: String,
    #[serde(default)]
    attachments: Vec<ChannelAttachment>,
    #[serde(default)]
    metadata: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
struct FsOutboundMessage {
    channel_id: String,
    conversation: ChannelConversationKey,
    message: OutboundMessage,
    timestamp_ms: u64,
}

fn resolve_dir(
    base: &Path,
    value: Option<&serde_json::Value>,
    default_name: &str,
) -> Result<PathBuf> {
    match value {
        None => Ok(base.join(default_name)),
        Some(value) => {
            let raw = value
                .as_str()
                .ok_or_else(|| anyhow!("Path setting must be a string"))?;
            if raw.trim().is_empty() {
                anyhow::bail!("Path setting must not be empty");
            }
            let path = Path::new(raw);
            if path.is_absolute() {
                Ok(path.to_path_buf())
            } else {
                Ok(base.join(path))
            }
        }
    }
}

fn default_message_id(path: &Path) -> String {
    path.file_stem()
        .and_then(|v| v.to_str())
        .map(std::string::ToString::to_string)
        .unwrap_or_else(|| uuid::Uuid::now_v7().simple().to_string())
}

fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

async fn ensure_dir(path: &Path) -> Result<()> {
    tokio::fs::create_dir_all(path)
        .await
        .with_context(|| format!("Failed to create '{}'", path.display()))
}

async fn move_file(path: &Path, target_dir: &Path) -> Result<()> {
    ensure_dir(target_dir).await?;
    let target = target_dir.join(
        path.file_name()
            .ok_or_else(|| anyhow!("Invalid path '{}': missing file name", path.display()))?,
    );
    tokio::fs::rename(path, &target).await.with_context(|| {
        format!(
            "Failed to move '{}' to '{}'",
            path.display(),
            target.display()
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn sample_settings() -> serde_json::Value {
        serde_json::json!({
            "inbox_dir": "inbox",
            "outbox_dir": "outbox",
            "processed_dir": "processed",
            "failed_dir": "failed",
            "poll_interval_ms": 25,
        })
    }

    fn sample_event() -> serde_json::Value {
        serde_json::json!({
            "conversation": {
                "channel": { "other": "fs" },
                "workspace_id": "workspace",
                "room_id": "room",
                "thread_id": "thread",
                "user_id": "user",
            },
            "message_id": "msg-1",
            "user": {
                "id": "user",
                "display_name": "User",
                "username": "user",
            },
            "text": "hello",
            "metadata": {
                "reset_session": true
            }
        })
    }

    #[tokio::test]
    async fn driver_reads_inbound_and_moves_processed_file() {
        let dir = tempdir().unwrap();
        let channel_dir = dir.path().join("channel");
        tokio::fs::create_dir_all(channel_dir.join("inbox"))
            .await
            .unwrap();
        tokio::fs::write(
            channel_dir.join("inbox/in-1.json"),
            serde_json::to_string_pretty(&sample_event()).unwrap(),
        )
        .await
        .unwrap();

        let (_tx, rx) = watch::channel(false);
        let mut driver = FsChannelDriver::from_settings("fs", &channel_dir, &sample_settings(), rx)
            .await
            .unwrap();

        let event = driver.next_event().await.unwrap().unwrap();
        assert_eq!(event.text, "hello");
        assert!(channel_dir.join("processed/in-1.json").exists());
        assert!(!channel_dir.join("inbox/in-1.json").exists());
    }

    #[tokio::test]
    async fn driver_writes_outbound_messages() {
        let dir = tempdir().unwrap();
        let channel_dir = dir.path().join("channel");
        let (_tx, rx) = watch::channel(false);
        let mut driver = FsChannelDriver::from_settings("fs", &channel_dir, &sample_settings(), rx)
            .await
            .unwrap();

        let conversation = ChannelConversationKey {
            channel: ChannelKind::Other("fs".to_string()),
            workspace_id: "workspace".into(),
            room_id: Some("room".into()),
            thread_id: "thread".into(),
            user_id: Some("user".into()),
        };

        driver
            .send(&conversation, OutboundMessage::text("pong"))
            .await
            .unwrap();

        let outbox = channel_dir.join("outbox");
        let mut files = tokio::fs::read_dir(&outbox).await.unwrap();
        let file = files.next_entry().await.unwrap().unwrap();
        let raw = tokio::fs::read_to_string(file.path()).await.unwrap();
        assert!(raw.contains("pong"));
    }

    #[test]
    fn validate_settings_rejects_too_small_poll_interval() {
        let dir = tempdir().unwrap();
        let channel_dir = dir.path().join("channel");
        let error = validate_settings(
            &channel_dir,
            &serde_json::json!({
                "poll_interval_ms": 1,
            }),
        )
        .expect_err("too-small poll interval should fail");
        assert!(error.to_string().contains("poll_interval_ms"));
    }
}
