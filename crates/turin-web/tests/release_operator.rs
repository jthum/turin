use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use futures::StreamExt;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::task::JoinHandle;
use tokio::time::{Instant, sleep};
use turin::remote::{RemoteServeOptions, start as start_remote};
use turin_control_client::ConnectionSpec;
use turin_daemon_protocol::{DaemonRequest, NoParams};
use turin_web::{WebServeOptions, start as start_web};

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
    async fn start_with_harness(harness_body: &str) -> Result<Self> {
        let tempdir = Arc::new(tempfile::tempdir()?);
        let workspace_root = tempdir.path().join("workspace");
        let turin_root = workspace_root.join(DEFAULT_LAYOUT_ROOT);
        let harness_dir = turin_root.join(DEFAULT_LAYOUT_HARNESSES_DIR);
        let agents_dir = turin_root.join(DEFAULT_LAYOUT_AGENTS_DIR);

        std::fs::create_dir_all(&harness_dir)?;
        std::fs::create_dir_all(&agents_dir)?;
        std::fs::write(harness_dir.join("main.lua"), harness_body)?;

        let config_path = workspace_root.join(DEFAULT_BOOTSTRAP_CONFIG_PATH);
        std::fs::create_dir_all(config_path.parent().expect("config parent"))?;
        let config_toml = format!(
            r#"[agent]
id = "default"
model = "mock-model"
provider = "mock"
system_prompt = "turin-web integration"

[kernel]
workspace_root = "{workspace_root}"
max_turns = 4
heartbeat_interval_seconds = 30
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
        let _: Value = client
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
                event_keepalive_seconds: Some(1),
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

#[tokio::test(flavor = "multi_thread")]
async fn turin_web_release_operator_smoke() -> Result<()> {
    let daemon = DaemonHarness::start_with_harness(include_str!(
        "../../../examples/harnesses/ui_release_operator/main.lua"
    ))
    .await?;
    let server = start_web(WebServeOptions {
        bind: "127.0.0.1:0".to_string(),
        connection: ConnectionSpec::LocalEndpoint {
            endpoint: daemon.endpoint.clone(),
        },
        allow_non_loopback: false,
    })
    .await?;
    let base_url = format!("http://{}", server.local_addr());
    let client = reqwest::Client::new();

    assert_release_operator_web(&base_url, &client).await?;

    server.stop().await?;
    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn turin_web_release_operator_smoke_remote() -> Result<()> {
    let daemon = DaemonHarness::start_with_harness(include_str!(
        "../../../examples/harnesses/ui_release_operator/main.lua"
    ))
    .await?;
    let remote = RemoteHarness::start(&daemon.config_path).await?;
    let server = start_web(WebServeOptions {
        bind: "127.0.0.1:0".to_string(),
        connection: ConnectionSpec::Remote {
            base_url: remote.base_url.clone(),
            auth_token: "test-token".to_string(),
        },
        allow_non_loopback: false,
    })
    .await?;
    let base_url = format!("http://{}", server.local_addr());
    let client = reqwest::Client::new();

    assert_release_operator_web(&base_url, &client).await?;

    server.stop().await?;
    remote.stop().await?;
    daemon.stop().await
}

async fn assert_release_operator_web(base_url: &str, client: &reqwest::Client) -> Result<()> {
    let html = client
        .get(format!("{base_url}/"))
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    assert!(html.contains("Turin Web"));
    assert!(html.contains("/assets/app.js"));

    let css = client
        .get(format!("{base_url}/assets/app.css"))
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    assert!(css.contains("--accent"));

    let js = client
        .get(format!("{base_url}/assets/app.js"))
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    assert!(js.contains("/api/status"));
    assert!(js.contains("EventSource"));
    assert!(js.contains("renderActivity"));
    assert!(js.contains("renderDetail"));
    assert!(js.contains("collectFormParams"));
    assert!(js.contains("formDrafts"));
    assert!(js.contains("runningActions"));

    let health: Value = client
        .get(format!("{base_url}/api/healthz"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(health["ok"], true);

    let event_response = client
        .get(format!("{base_url}/api/events"))
        .send()
        .await?
        .error_for_status()?;
    let event_text = read_sse_until(event_response, "event: runtime.snapshot").await?;
    assert!(event_text.contains("event: runtime.snapshot"));

    let apps: Value = client
        .get(format!("{base_url}/api/apps"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let apps = apps["apps"]
        .as_array()
        .context("apps response should include app array")?;
    assert!(
        apps.iter().any(|app| app["id"] == "release-operator"
            && app["definition"]["title"] == "Release Operator")
    );

    let app: Value = client
        .get(format!("{base_url}/api/apps/release-operator"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert!(app["app"]["screens"].get("approvals").is_some());
    assert!(app["app"]["menus"].as_array().is_some_and(|menus| {
        menus.iter().any(|menu| {
            menu["items"].as_array().is_some_and(|items| {
                items.iter().any(|item| {
                    item["label"] == "Work"
                        && item["items"]
                            .as_array()
                            .is_some_and(|children| !children.is_empty())
                })
            })
        })
    }));

    let seeded: Value = client
        .post(format!("{base_url}/api/actions/run"))
        .json(&json!({
            "action": "release.seed_demo_work",
            "harness_id": "default",
            "params": {
                "release": "2026.06",
                "count": 4
            }
        }))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(seeded["result"]["action"], "release.seed_demo_work");
    assert_eq!(seeded["result"]["result"]["status"], "seeded");
    assert_eq!(seeded["result"]["result"]["count"], 4);

    let list: Value = client
        .post(format!("{base_url}/api/ui/list"))
        .json(&json!({
            "source": "worklists.release",
            "where": {
                "kind": "approval",
                "status": "pending"
            },
            "limit": 10
        }))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let items = list["list"]["items"]
        .as_array()
        .context("list response should include items")?;
    assert_eq!(items.len(), 4);
    assert!(items.iter().all(|item| item["status"] == "pending"));
    assert!(items.iter().all(|item| item["kind"] == "approval"));

    Ok(())
}

async fn read_sse_until(response: reqwest::Response, needle: &str) -> Result<String> {
    let mut stream = response.bytes_stream();
    let mut text = String::new();
    let deadline = Instant::now() + Duration::from_secs(5);

    while Instant::now() < deadline {
        let Some(chunk) = tokio::time::timeout(Duration::from_secs(1), stream.next())
            .await
            .context("timed out waiting for SSE chunk")?
        else {
            break;
        };
        let chunk = chunk.context("SSE stream returned an error")?;
        text.push_str(&String::from_utf8_lossy(&chunk));
        if text.contains(needle) {
            return Ok(text);
        }
    }

    Err(anyhow!("SSE stream did not include '{}': {}", needle, text))
}
