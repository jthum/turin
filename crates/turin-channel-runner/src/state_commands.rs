use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use serde::Serialize;
use turin_channel_core::{ChannelConversationKey, ChannelKind};

use crate::sidecar::validate_channel_file;
use crate::{ChannelRoomRef, FileAccessStateStore, FileBindingStore};

#[derive(Debug, Args)]
pub struct ChannelStateArgs {
    /// Channel-owned config whose adjacent runtime state should be managed.
    #[arg(long)]
    config: PathBuf,
    #[command(subcommand)]
    command: ChannelStateCommand,
}

#[derive(Debug, Subcommand)]
enum ChannelStateCommand {
    /// Inspect or update room access state.
    Access(ChannelAccessArgs),
    /// Inspect or clear durable conversation bindings.
    Bindings(ChannelBindingsArgs),
}

#[derive(Debug, Args)]
struct ChannelAccessArgs {
    #[command(subcommand)]
    command: ChannelAccessCommand,
}

#[derive(Debug, Subcommand)]
enum ChannelAccessCommand {
    /// Print approved and pending rooms.
    List,
    /// Approve a room returned by `access list`.
    Approve {
        #[arg(long)]
        room_json: String,
        #[arg(long)]
        approved_by_user_id: Option<String>,
        #[arg(long)]
        approved_by_username: Option<String>,
    },
    /// Remove a room from the pending list without approving it.
    Reject {
        #[arg(long)]
        room_json: String,
    },
    /// Revoke a previously approved room.
    Revoke {
        #[arg(long)]
        room_json: String,
    },
}

#[derive(Debug, Args)]
struct ChannelBindingsArgs {
    #[command(subcommand)]
    command: ChannelBindingsCommand,
}

#[derive(Debug, Subcommand)]
enum ChannelBindingsCommand {
    /// Print durable platform-conversation to Turin-session bindings.
    List,
    /// Clear one exact conversation binding returned by `bindings list`.
    Clear {
        #[arg(long)]
        conversation_json: String,
    },
}

impl ChannelStateArgs {
    pub async fn run(self, expected_kind: &str) -> Result<()> {
        validate_channel_file(&self.config, expected_kind)?;
        let runtime_dir = channel_runtime_dir(&self.config);
        match self.command {
            ChannelStateCommand::Access(args) => {
                run_access(args.command, &runtime_dir, expected_kind).await
            }
            ChannelStateCommand::Bindings(args) => {
                run_bindings(args.command, &runtime_dir, expected_kind).await
            }
        }
    }
}

async fn run_access(
    command: ChannelAccessCommand,
    runtime_dir: &Path,
    expected_kind: &str,
) -> Result<()> {
    let store = FileAccessStateStore::new(runtime_dir.join("access.json"));
    let snapshot = match command {
        ChannelAccessCommand::List => store.snapshot().await?,
        ChannelAccessCommand::Approve {
            room_json,
            approved_by_user_id,
            approved_by_username,
        } => {
            let room = parse_room(&room_json, expected_kind)?;
            store
                .approve(&room, approved_by_user_id, approved_by_username)
                .await?
        }
        ChannelAccessCommand::Reject { room_json } => {
            let room = parse_room(&room_json, expected_kind)?;
            store.reject_pending(&room).await?
        }
        ChannelAccessCommand::Revoke { room_json } => {
            let room = parse_room(&room_json, expected_kind)?;
            store.revoke(&room).await?
        }
    };
    print_json(&snapshot)
}

async fn run_bindings(
    command: ChannelBindingsCommand,
    runtime_dir: &Path,
    expected_kind: &str,
) -> Result<()> {
    let store = FileBindingStore::new(runtime_dir.join("bindings.json"));
    match command {
        ChannelBindingsCommand::List => print_json(&store.snapshot().await?),
        ChannelBindingsCommand::Clear { conversation_json } => {
            let conversation = parse_conversation(&conversation_json, expected_kind)?;
            let removed = store.clear(&conversation).await?;
            print_json(&serde_json::json!({
                "removed": removed,
                "bindings": store.snapshot().await?,
            }))
        }
    }
}

fn parse_room(raw: &str, expected_kind: &str) -> Result<ChannelRoomRef> {
    let room: ChannelRoomRef = serde_json::from_str(raw).context("Invalid --room-json")?;
    validate_kind(&room.channel, expected_kind)?;
    validate_key_fields(&room.workspace_id, &room.thread_id)?;
    Ok(room)
}

fn parse_conversation(raw: &str, expected_kind: &str) -> Result<ChannelConversationKey> {
    let conversation: ChannelConversationKey =
        serde_json::from_str(raw).context("Invalid --conversation-json")?;
    validate_kind(&conversation.channel, expected_kind)?;
    validate_key_fields(&conversation.workspace_id, &conversation.thread_id)?;
    Ok(conversation)
}

fn validate_kind(kind: &ChannelKind, expected_kind: &str) -> Result<()> {
    anyhow::ensure!(
        kind.as_str() == expected_kind,
        "State key belongs to channel kind '{}' rather than '{}'",
        kind,
        expected_kind
    );
    Ok(())
}

fn validate_key_fields(workspace_id: &str, thread_id: &str) -> Result<()> {
    anyhow::ensure!(
        !workspace_id.trim().is_empty(),
        "workspace_id must not be empty"
    );
    anyhow::ensure!(!thread_id.trim().is_empty(), "thread_id must not be empty");
    Ok(())
}

fn channel_runtime_dir(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .unwrap_or(Path::new("."))
        .join("runtime")
}

fn print_json(value: &impl Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn state_commands_manage_disabled_channel_without_daemon() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("telegram/config.toml");
        std::fs::create_dir_all(config.parent().unwrap()).unwrap();
        std::fs::write(
            &config,
            "enabled = false\nkind = \"telegram\"\nagent_id = \"default\"\n",
        )
        .unwrap();
        let room_json = serde_json::json!({
            "channel": "telegram",
            "workspace_id": "telegram",
            "room_id": "room-1",
            "thread_id": "room-1"
        })
        .to_string();

        ChannelStateArgs {
            config: config.clone(),
            command: ChannelStateCommand::Access(ChannelAccessArgs {
                command: ChannelAccessCommand::Approve {
                    room_json,
                    approved_by_user_id: Some("owner".into()),
                    approved_by_username: None,
                },
            }),
        }
        .run("telegram")
        .await
        .unwrap();

        let snapshot =
            FileAccessStateStore::new(config.parent().unwrap().join("runtime/access.json"))
                .snapshot()
                .await
                .unwrap();
        assert_eq!(snapshot.approved_rooms.len(), 1);
    }

    #[test]
    fn state_key_must_match_runner_kind() {
        let error = parse_room(
            r#"{"channel":"discord","workspace_id":"g","room_id":"r","thread_id":"t"}"#,
            "telegram",
        )
        .unwrap_err();
        assert!(error.to_string().contains("rather than 'telegram'"));
    }
}
