use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use tempfile::TempDir;
use tokio::task::JoinHandle;
use tokio::time::{Instant, sleep};
use turin::remote::{RemoteServeOptions, start as start_remote};
use turin_control_client::{ConnectionKind, ConnectionSpec, ControlClient};
use turin_daemon_protocol::{DaemonRequest, NoParams, RuntimeEventsSubscribeParams};

const DEFAULT_BOOTSTRAP_CONFIG_PATH: &str = ".turin/config.toml";
const DEFAULT_LAYOUT_HARNESSES_DIR: &str = "harnesses";
const DEFAULT_LAYOUT_AGENTS_DIR: &str = "runtime/agents";
const DEFAULT_LAYOUT_ROOT: &str = ".turin";

struct DaemonHarness {
    _tempdir: Arc<TempDir>,
    endpoint: PathBuf,
    config_path: PathBuf,
    join: JoinHandle<Result<()>>,
}

impl DaemonHarness {
    async fn start() -> Result<Self> {
        let tempdir = Arc::new(tempfile::tempdir()?);
        let workspace_root = tempdir.path().join("workspace");
        let turin_root = workspace_root.join(DEFAULT_LAYOUT_ROOT);
        let harness_dir = turin_root.join(DEFAULT_LAYOUT_HARNESSES_DIR);
        let agents_dir = turin_root.join(DEFAULT_LAYOUT_AGENTS_DIR);

        std::fs::create_dir_all(&harness_dir)?;
        std::fs::create_dir_all(&agents_dir)?;
        std::fs::write(
            harness_dir.join("main.lua"),
            "-- control client integration harness\n",
        )?;

        let config_path = workspace_root.join(DEFAULT_BOOTSTRAP_CONFIG_PATH);
        std::fs::create_dir_all(config_path.parent().expect("config parent"))?;
        let config_toml = format!(
            r#"[agent]
id = "default"
model = "mock-model"
provider = "mock"
system_prompt = "Control client integration"

[kernel]
workspace_root = "{workspace_root}"
max_turns = 4
heartbeat_interval_secs = 30
initial_spawn_depth = 0

[persistence.state]
path = "{database_path}"

[harness]
directory = "{harness_directory}"
fs_root = "."

[providers.mock]
type = "mock"
base_url = "PONG"
"#,
            workspace_root = workspace_root.display(),
            database_path = turin_root.join("data/state.db").display(),
            harness_directory = harness_dir.display(),
        );
        std::fs::write(&config_path, config_toml)?;

        let endpoint = turin_root.join("daemon.sock");
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
                return Err(anyhow!(
                    "Timed out waiting for daemon endpoint '{}'",
                    endpoint.display()
                ));
            }
            sleep(Duration::from_millis(25)).await;
        }

        Ok(Self {
            _tempdir: tempdir,
            endpoint,
            config_path,
            join,
        })
    }

    async fn stop(self) -> Result<()> {
        let client = turin_daemon_client::DaemonClient::new(&self.endpoint);
        let _: serde_json::Value = client
            .request_ok(None, DaemonRequest::DaemonStop(NoParams::default()))
            .await?;
        let _ = tokio::time::timeout(Duration::from_secs(5), self.join)
            .await
            .context("timed out waiting for daemon to exit")??;
        Ok(())
    }
}

struct RemoteHarness {
    base_url: String,
    server: turin::remote::RunningRemoteServer,
}

impl RemoteHarness {
    async fn start(config_path: &Path) -> Result<Self> {
        let server = start_remote(
            config_path,
            RemoteServeOptions {
                bind: Some("127.0.0.1:0".to_string()),
                auth_token: Some("test-token".to_string()),
                auth_token_env: None,
                event_keepalive_secs: Some(1),
                allow_non_loopback: Some(false),
            },
        )
        .await?;
        Ok(Self {
            base_url: format!("http://{}", server.local_addr()),
            server,
        })
    }

    async fn stop(self) -> Result<()> {
        self.server.stop().await
    }
}

async fn assert_session_and_task_workflow(client: &ControlClient) -> Result<()> {
    let opened = client.open_session("default", None).await?;
    assert_eq!(opened.agent_id, "default");
    assert!(!opened.session_id.is_empty());

    let live_sessions = client.list_live_sessions().await?;
    assert!(
        live_sessions
            .iter()
            .any(|session| session.session_id == opened.session_id)
    );

    let submitted = client
        .submit_task(
            None,
            Some(opened.session_id.clone()),
            "hello from control client".to_string(),
        )
        .await?;
    assert_eq!(submitted.agent_id, "default");
    assert_eq!(submitted.state, "queued");

    let waited = client.wait_task(&submitted.request_id, Some(5_000)).await?;
    assert_eq!(waited.status.as_deref(), Some("success"));

    let tasks = client.list_tasks().await?;
    assert!(
        tasks
            .iter()
            .any(|task| task.request_id == submitted.request_id)
    );

    let detail = client.get_session(&opened.session_id).await?;
    assert_eq!(detail.session.session_id, opened.session_id);
    assert!(
        detail
            .messages
            .iter()
            .any(|message| message.role == "assistant")
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn control_client_local_health_and_events_work() -> Result<()> {
    let daemon = DaemonHarness::start().await?;
    let client = ControlClient::connect(&ConnectionSpec::LocalEndpoint {
        endpoint: daemon.endpoint.clone(),
    })
    .await?;

    let health = client.health().await?;
    assert!(health.ready);
    assert_eq!(
        health.connection_kind,
        turin_control_client::ConnectionKind::Local
    );

    let mut stream = client
        .subscribe_managed(RuntimeEventsSubscribeParams::default())
        .await?;
    let event = stream.next_event().await?;
    assert_eq!(event.event, "runtime.snapshot");
    assert_session_and_task_workflow(&client).await?;

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn control_client_remote_health_and_events_work() -> Result<()> {
    let daemon = DaemonHarness::start().await?;
    let remote = RemoteHarness::start(&daemon.config_path).await?;
    let client = ControlClient::connect(&ConnectionSpec::Remote {
        base_url: remote.base_url.clone(),
        auth_token: "test-token".to_string(),
    })
    .await?;

    let health = client.health().await?;
    assert!(health.ready);
    assert_eq!(health.connection_kind, ConnectionKind::Remote);

    let mut stream = client
        .subscribe_managed(RuntimeEventsSubscribeParams::default())
        .await?;
    let event = stream.next_event().await?;
    assert_eq!(event.event, "runtime.snapshot");
    assert_session_and_task_workflow(&client).await?;

    remote.stop().await?;
    daemon.stop().await
}
