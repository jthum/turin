use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde_json::json;
use tempfile::TempDir;
use tokio::task::JoinHandle;
use tokio::time::{Instant, sleep, timeout};
use turin_channel_core::{ChannelConversationKey, ChannelKind, OutboundMessage};
use turin_channel_fs::FsChannelDriver;
use turin_channel_runner::{ChannelRunner, RunnerConfig};

struct DaemonHarness {
    _tempdir: std::sync::Arc<TempDir>,
    workspace_root: PathBuf,
    endpoint: PathBuf,
    join: JoinHandle<Result<()>>,
}

impl DaemonHarness {
    async fn start() -> Result<Self> {
        let tempdir = std::sync::Arc::new(tempfile::tempdir()?);
        let workspace_root = tempdir.path().join("workspace");
        let harness_dir = workspace_root.join(".turin/harnesses");

        std::fs::create_dir_all(&harness_dir)?;
        std::fs::write(
            harness_dir.join("main.lua"),
            "-- channel fs integration harness\n",
        )?;

        let config_path = tempdir.path().join("turin.toml");
        let config_toml = format!(
            r#"[agent]
id = "default"
model = "mock-model"
provider = "mock"
system_prompt = "FS channel integration"

[kernel]
workspace_root = "{workspace_root}"
max_turns = 4
heartbeat_interval_secs = 30
initial_spawn_depth = 0

[persistence]
database_path = "{database_path}"

[harness]
directory = "{harness_directory}"
fs_root = "."

[providers.mock]
type = "mock"
base_url = "PONG"
"#,
            workspace_root = workspace_root.display(),
            database_path = workspace_root.join("test.db").display(),
            harness_directory = harness_dir.display(),
        );
        std::fs::write(&config_path, config_toml)?;

        let endpoint = workspace_root.join(".turin/daemon.sock");
        let serve_config_path = config_path.clone();
        let join =
            tokio::spawn(async move { turin::daemon::server::serve(&serve_config_path).await });

        let deadline = Instant::now() + Duration::from_secs(10);
        let client = turin_daemon_client::DaemonClient::new(&endpoint);
        loop {
            if client.handshake().await.is_ok() {
                break;
            }
            if join.is_finished() {
                let result = join
                    .await
                    .context("daemon task join failed before endpoint bind")?;
                return Err(result
                    .err()
                    .unwrap_or_else(|| anyhow!("daemon exited before creating daemon endpoint")));
            }
            if Instant::now() >= deadline {
                join.abort();
                return Err(anyhow!(
                    "Timed out waiting for daemon endpoint '{}'",
                    endpoint.display()
                ));
            }
            sleep(Duration::from_millis(25)).await;
        }

        Ok(Self {
            _tempdir: tempdir,
            workspace_root,
            endpoint,
            join,
        })
    }

    fn runner(&self) -> ChannelRunner {
        ChannelRunner::new(
            turin_daemon_client::DaemonClient::new(&self.endpoint),
            RunnerConfig {
                state_path: self.workspace_root.join(".turin/channel-bindings.json"),
                access_state_path: self.workspace_root.join(".turin/channel-access.json"),
                idle_ttl: Some(Duration::from_secs(600)),
                access_policy: Default::default(),
                tools: Default::default(),
            },
        )
    }

    async fn stop(self) -> Result<()> {
        let client = turin_daemon_client::DaemonClient::new(&self.endpoint);
        let _: serde_json::Value = client
            .request_ok(
                None,
                turin_daemon_protocol::DaemonRequest::DaemonStop(Default::default()),
            )
            .await?;
        let _ = timeout(Duration::from_secs(5), self.join)
            .await
            .context("timed out waiting for daemon to exit")??;
        Ok(())
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn fs_channel_driver_round_trip_with_daemon_runner() -> Result<()> {
    let daemon = DaemonHarness::start().await?;
    let runner = daemon.runner();

    let channel_dir = daemon.workspace_root.join("channels/fs-test");
    tokio::fs::create_dir_all(channel_dir.join("inbox")).await?;

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let settings = json!({
        "inbox_dir": "inbox",
        "outbox_dir": "outbox",
        "processed_dir": "processed",
        "failed_dir": "failed",
        "poll_interval_ms": 25,
    });
    let mut driver =
        FsChannelDriver::from_settings("fs-test", &channel_dir, &settings, shutdown_rx).await?;

    let run =
        tokio::spawn(async move { runner.run_driver("default", &mut driver, Some(5000)).await });

    let inbound = json!({
        "conversation": {
            "channel": { "other": "fs" },
            "workspace_id": "workspace",
            "room_id": "room",
            "thread_id": "thread-1",
            "user_id": "user-1"
        },
        "message_id": "m-1",
        "user": {
            "id": "user-1",
            "display_name": "User One",
            "username": "user1"
        },
        "text": "Say pong"
    });
    tokio::fs::write(
        channel_dir.join("inbox/in-1.json"),
        serde_json::to_string_pretty(&inbound)?,
    )
    .await?;

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut outbound_body = None;
    while Instant::now() < deadline {
        let outbox = channel_dir.join("outbox");
        if outbox.exists() {
            let mut entries = tokio::fs::read_dir(&outbox).await?;
            if let Some(entry) = entries.next_entry().await? {
                outbound_body = Some(tokio::fs::read_to_string(entry.path()).await?);
                break;
            }
        }
        sleep(Duration::from_millis(25)).await;
    }

    let outbound = outbound_body.context("fs channel did not produce outbound response")?;
    assert!(outbound.contains("PONG"), "outbound payload: {}", outbound);
    assert!(channel_dir.join("processed/in-1.json").exists());

    shutdown_tx.send(true)?;
    let _ = timeout(Duration::from_secs(5), run)
        .await
        .context("timed out waiting for fs channel runner shutdown")??;

    daemon.stop().await
}

#[test]
fn outbound_message_type_is_reachable_for_fs_adapter() {
    let _ = OutboundMessage::text("hello");
    let _ = ChannelConversationKey {
        channel: ChannelKind::Other("fs".to_string()),
        workspace_id: "workspace".into(),
        room_id: Some("room".into()),
        thread_id: "thread".into(),
        user_id: Some("user".into()),
    };
}
