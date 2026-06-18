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
    assert!(js.contains("draftValueFromInput"));
    assert!(js.contains("rememberFormDraft(formKey, field.name, draftValueFromInput"));
    assert!(js.contains("draftValueForField(formKey, field, node)"));
    assert!(js.contains("field.name in node.params"));
    assert!(
        js.contains("if (!field.required && (rawValue === null || rawValue === \"\")) continue")
    );
    assert!(js.contains("runningActions"));
    assert!(js.contains("renderReport"));
    assert!(js.contains("renderReportHighlight"));
    assert!(js.contains("highestPriorityPendingItem"));
    assert!(js.contains("Next highest-priority pending item"));
    assert!(js.contains("renderChart"));
    assert!(js.contains("renderState"));
    assert!(js.contains("appendState"));
    assert!(js.contains("surface-state"));
    assert!(js.contains("renderActionResult"));
    assert!(js.contains("latestActionResult"));
    assert!(js.contains("latestActionResultForApp"));
    assert!(js.contains("appId: app?.id || null"));
    assert!(js.contains("title: \"Action failed\""));
    assert!(js.contains("body: error.message"));
    assert!(js.contains("detail: error.envelope || null"));
    assert!(js.contains("error.envelope = body?.error || null"));
    assert!(js.contains("error.status = response.status"));
    assert!(js.contains("pushNotice(\"error\", \"Action failed\", error.message)"));
    assert!(js.contains("pendingAction"));
    assert!(js.contains("renderActionConfirmation"));
    assert!(js.contains("requestActionConfirmation"));
    assert!(js.contains("Confirm and run"));
    assert!(js.contains("Requires confirmation before running."));
    assert!(!js.contains("window.confirm"));
    assert!(js.contains("renderDefaultConsole"));
    assert!(js.contains("Default Operator Console"));
    assert!(js.contains("Runtime Overview"));
    assert!(js.contains("Simple stays simple"));
    assert!(js.contains("applyUiIntentPayload"));
    assert!(js.contains("applyUiIntentMessages"));
    assert!(js.contains("result.result.ui_intents"));
    assert!(js.contains("ui?.refreshes"));
    assert!(js.contains("case \"refresh\""));
    assert!(js.contains("case \"badge\""));
    assert!(js.contains("applyUiRefresh"));
    assert!(js.contains("applyUiBadge"));
    assert!(js.contains("localBadges"));
    assert!(js.contains("applyLocalBadgesToApps"));
    assert!(js.contains("state.localBadges.set(localBadgeKey(appId, target), badge)"));
    assert!(js.contains("app.badges[target]"));
    assert!(js.contains("appById(appId)"));
    assert!(js.contains("invalidateListBinding"));
    assert!(js.contains("parseListKey"));
    assert!(js.contains("focusScreenIdForTarget"));
    assert!(js.contains("renderWorkItemDetail"));
    assert!(js.contains("item.action.name"));
    assert!(js.contains("Review ${item.action.name}"));
    assert!(js.contains("selectedListItems"));
    assert!(js.contains("selectedListItem"));
    assert!(js.contains("aria-selected"));
    assert!(js.contains("counts.done"));
    assert!(js.contains("field === \"id\" || field === \"public_id\""));
    assert!(js.contains("unsupportedSourceMessage"));
    assert!(js.contains("errorMessageFromEnvelope"));
    assert!(js.contains("details?.guidance"));
    assert!(js.contains("Only worklists.* sources load today"));
    assert!(js.contains("isWorklistSource"));
    assert!(js.contains("appendNodeBadge"));
    assert!(js.contains("nodeBadge"));
    assert!(js.contains("renderPlaceholder(node, app)"));
    assert!(js.contains("activePaneId"));
    assert!(js.contains("renderPane"));
    assert!(js.contains("selectedPane"));
    assert_eq!(
        js.matches("const latestActionResult = latestActionResultForApp(app)")
            .count(),
        2
    );
    assert_eq!(
        js.matches("const latestActionResult = latestActionResultForApp(null)")
            .count(),
        1
    );
    assert!(js.contains("state.pendingAction = null;\n    loadVisibleLists().then(render);"));
    assert_eq!(js.matches("appendNodeBadge(row, node, app)").count(), 1);
    assert_eq!(
        js.matches("return node.action === target || node.title === target")
            .count(),
        1
    );
    assert!(css.contains(".list-selection"));
    assert!(css.contains(".surface-state"));
    assert!(css.contains(".surface-state[data-level=\"error\"]"));
    assert!(css.contains(".confirm-overlay"));
    assert!(css.contains(".confirm-dialog"));
    assert!(css.contains(".default-console"));
    assert!(css.contains(".default-grid"));
    assert!(css.contains(".stat-card"));
    assert!(css.contains(".list-row"));
    assert!(css.contains(".node-badge"));
    assert!(css.contains(".pane-overlay"));
    assert!(css.contains(".pane-sheet"));

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
    assert_node(
        &app,
        "home",
        "activity",
        "release-activity",
        Some("worklists.release"),
    )?;
    assert_node(
        &app,
        "approvals",
        "list",
        "pending-approvals",
        Some("worklists.release"),
    )?;
    assert_node(&app, "intake", "form", "seed-demo-form", None)?;
    assert_form_uses_static_count_param_as_default(&app)?;
    assert_node(
        &app,
        "overview",
        "detail",
        "release-snapshot",
        Some("worklists.release"),
    )?;
    assert_node(
        &app,
        "overview",
        "report",
        "release-readiness",
        Some("worklists.release"),
    )?;
    assert_node(
        &app,
        "overview",
        "chart",
        "approval-flow",
        Some("worklists.release"),
    )?;
    assert_pane_node(
        &app,
        "release-notes",
        "detail",
        "pane-release-snapshot",
        Some("worklists.release"),
    )?;
    assert_eq!(
        app["app"]["panes"]["release-notes"]["presentation"],
        "sheet"
    );
    assert_eq!(app["app"]["badges"]["release-readiness"]["label"], "live");
    assert_eq!(app["app"]["badges"]["release-readiness"]["level"], "info");
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

    let invalid_action = client
        .post(format!("{base_url}/api/actions/run"))
        .json(&json!({
            "action": " ",
            "harness_id": "default",
            "params": {}
        }))
        .send()
        .await?;
    assert_eq!(invalid_action.status().as_u16(), 400);
    let invalid_action: Value = invalid_action.json().await?;
    assert_eq!(invalid_action["error"]["code"], "invalid_action_request");
    assert_eq!(invalid_action["error"]["details"]["field"], "action");
    assert!(
        invalid_action["error"]["details"]["guidance"]
            .as_str()
            .is_some_and(|guidance| guidance.contains("declared harness action name"))
    );

    let failed_action = client
        .post(format!("{base_url}/api/actions/run"))
        .json(&json!({
            "action": "release.fail_diagnostic",
            "harness_id": "default",
            "params": {
                "reason": "Release Operator diagnostic failure"
            }
        }))
        .send()
        .await?;
    assert_eq!(failed_action.status().as_u16(), 503);
    let failed_action: Value = failed_action.json().await?;
    assert_eq!(failed_action["error"]["code"], "control_unavailable");
    assert!(
        failed_action["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("Failed to run harness action"))
    );
    assert!(
        failed_action["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("Release Operator diagnostic failure"))
    );

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
    let seeded_intents = ui_intents(&seeded)?;
    assert_has_ui_intent(seeded_intents, "seeded notify", |intent| {
        intent["type"] == "notify"
            && intent["app_id"] == "release-operator"
            && intent["title"] == "Seeded release work"
            && intent["body"] == "Created 4 approval items for 2026.06."
            && intent["level"] == "success"
    });
    assert_has_ui_intent(seeded_intents, "seeded badge", |intent| {
        intent["type"] == "badge"
            && intent["app_id"] == "release-operator"
            && intent["target"] == "approvals"
            && intent["count"] == 4
            && intent["level"] == "info"
    });
    assert_has_ui_intent(seeded_intents, "seeded refresh", |intent| {
        intent["type"] == "refresh"
            && intent["app_id"] == "release-operator"
            && intent["binding"] == "worklists.release"
    });

    let shown: Value = client
        .post(format!("{base_url}/api/actions/run"))
        .json(&json!({
            "action": "release.show_notes",
            "harness_id": "default",
            "params": {}
        }))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(shown["result"]["action"], "release.show_notes");
    assert_eq!(shown["result"]["result"]["status"], "shown");
    assert_eq!(shown["result"]["result"]["target"], "release-notes");
    assert_has_ui_intent(ui_intents(&shown)?, "show release notes", |intent| {
        intent["type"] == "show"
            && intent["app_id"] == "release-operator"
            && intent["target"] == "release-notes"
            && intent["presentation"] == "sheet"
    });

    let opened: Value = client
        .post(format!("{base_url}/api/actions/run"))
        .json(&json!({
            "action": "release.open_intake",
            "harness_id": "default",
            "params": {}
        }))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(opened["result"]["action"], "release.open_intake");
    assert_eq!(opened["result"]["result"]["status"], "opened");
    assert_eq!(opened["result"]["result"]["target"], "intake");
    assert_eq!(opened["result"]["result"]["focus"], "seed-demo-form");
    let opened_intents = ui_intents(&opened)?;
    assert_has_ui_intent(opened_intents, "open intake", |intent| {
        intent["type"] == "open"
            && intent["app_id"] == "release-operator"
            && intent["target"] == "intake"
    });
    assert_has_ui_intent(opened_intents, "focus intake form", |intent| {
        intent["type"] == "focus"
            && intent["app_id"] == "release-operator"
            && intent["target"] == "seed-demo-form"
    });

    let unsupported_source = client
        .post(format!("{base_url}/api/ui/list"))
        .json(&json!({
            "source": "tables.release",
            "limit": 10
        }))
        .send()
        .await?;
    assert_eq!(unsupported_source.status().as_u16(), 400);
    let unsupported_source: Value = unsupported_source.json().await?;
    assert_eq!(
        unsupported_source["error"]["code"],
        "unsupported_ui_list_source"
    );
    assert_eq!(
        unsupported_source["error"]["details"]["source"],
        "tables.release"
    );
    assert_eq!(
        unsupported_source["error"]["details"]["supported_prefixes"][0],
        "worklists."
    );
    assert!(
        unsupported_source["error"]["details"]["guidance"]
            .as_str()
            .is_some_and(|guidance| guidance.contains("deliberate UI list adapter"))
    );

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
    assert!(
        items
            .iter()
            .all(|item| item["action"]["name"] == "release.approve_next")
    );

    Ok(())
}

fn ui_intents(response: &Value) -> Result<&[Value]> {
    response["result"]["ui_intents"]
        .as_array()
        .map(Vec::as_slice)
        .context("action response should include result.ui_intents array")
}

fn assert_has_ui_intent<F>(intents: &[Value], label: &str, predicate: F)
where
    F: Fn(&Value) -> bool,
{
    assert!(
        intents.iter().any(predicate),
        "missing {label} UI intent in {}",
        serde_json::to_string_pretty(intents).unwrap_or_else(|_| "<invalid json>".to_string())
    );
}

fn assert_node(
    app: &Value,
    screen_id: &str,
    kind: &str,
    id: &str,
    source: Option<&str>,
) -> Result<()> {
    let screen = app["app"]["screens"]
        .get(screen_id)
        .with_context(|| format!("missing screen '{screen_id}'"))?;
    let nodes = screen["nodes"]
        .as_array()
        .with_context(|| format!("screen '{screen_id}' should include nodes"))?;
    let found = flatten_nodes(nodes).into_iter().any(|node| {
        node["kind"] == kind
            && node["id"] == id
            && match source {
                Some(source) => node["source"] == source,
                None => true,
            }
    });
    if found {
        Ok(())
    } else {
        Err(anyhow!(
            "screen '{}' did not include {} node '{}' with source {:?}",
            screen_id,
            kind,
            id,
            source
        ))
    }
}

fn assert_pane_node(
    app: &Value,
    pane_id: &str,
    kind: &str,
    id: &str,
    source: Option<&str>,
) -> Result<()> {
    let pane = app["app"]["panes"]
        .get(pane_id)
        .with_context(|| format!("missing pane '{pane_id}'"))?;
    let nodes = pane["nodes"]
        .as_array()
        .with_context(|| format!("pane '{pane_id}' should include nodes"))?;
    let found = flatten_nodes(nodes).into_iter().any(|node| {
        node["kind"] == kind
            && node["id"] == id
            && match source {
                Some(source) => node["source"] == source,
                None => true,
            }
    });
    if found {
        Ok(())
    } else {
        Err(anyhow!(
            "pane '{}' did not include {} node '{}' with source {:?}",
            pane_id,
            kind,
            id,
            source
        ))
    }
}

fn assert_form_uses_static_count_param_as_default(app: &Value) -> Result<()> {
    let screen = app["app"]["screens"]
        .get("intake")
        .context("missing intake screen")?;
    let nodes = screen["nodes"]
        .as_array()
        .context("intake screen should include nodes")?;
    let form = flatten_nodes(nodes)
        .into_iter()
        .find(|node| node["kind"] == "form" && node["id"] == "seed-demo-form")
        .context("missing seed demo form")?;
    assert_eq!(form["params"]["count"], json!(1));
    let count_field = form["fields"]
        .as_array()
        .context("form should include fields")?
        .iter()
        .find(|field| field["name"] == "count")
        .context("missing count field")?;
    assert!(
        count_field.get("default").is_none(),
        "count field should rely on form params instead of duplicating a default"
    );
    Ok(())
}

fn flatten_nodes(nodes: &[Value]) -> Vec<&Value> {
    let mut out = Vec::new();
    for node in nodes {
        out.push(node);
        if let Some(children) = node["nodes"].as_array() {
            out.extend(flatten_nodes(children));
        }
    }
    out
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
