use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use tempfile::TempDir;
use tokio::task::JoinHandle;
use tokio::time::{Instant, sleep};
use turin::remote::{RemoteServeOptions, start as start_remote};
use turin_control_client::{ConnectionKind, ConnectionSpec, ControlClient};
use turin_daemon_protocol::{
    DaemonRequest, HarnessActionRunParams, NoParams, RuntimeEventsSubscribeParams, UiIntent,
    UiIntentMessage, UiNode, UiNoticeLevel, UiPaneIntent, UiScreenIntent, WorklistItemsParams,
    WorklistListParams,
};

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
        Self::start_with_harness("-- control client integration harness\n").await
    }

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
system_prompt = "Control client integration"

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
    assert!(detail.messages.iter().all(|message| message.turn_id > 0));
    assert_eq!(detail.execution.tasks.len(), 1);
    assert_eq!(detail.execution.tasks[0].status, "success");

    let windowed = client.get_session_window(&opened.session_id, 1).await?;
    assert!(!windowed.messages.is_empty());
    assert!(windowed.events.is_empty());
    let window = windowed.message_window.context("message window metadata")?;
    assert_eq!(window.total, detail.messages.len());
    assert_eq!(window.offset + windowed.messages.len(), window.total);
    let final_turn = windowed
        .messages
        .last()
        .expect("windowed transcript message")
        .turn_index;
    assert!(
        windowed
            .messages
            .iter()
            .all(|message| message.turn_index == final_turn)
    );

    let graph = client.get_session_graph(&opened.session_id).await?;
    let first_turn = graph
        .turns
        .iter()
        .min_by_key(|turn| turn.turn_index)
        .context("session graph should expose a turn")?;
    let inspected = client
        .get_session_turn_window(&opened.session_id, first_turn.turn_id, 24)
        .await?;
    assert!(!inspected.messages.is_empty());
    assert!(
        inspected
            .messages
            .iter()
            .all(|message| message.turn_index <= first_turn.turn_index)
    );
    assert_eq!(
        inspected.messages.last().map(|message| message.turn_id),
        Some(first_turn.turn_id)
    );

    Ok(())
}

async fn assert_release_operator_ui_workflow(client: &ControlClient) -> Result<()> {
    let intents = client.list_harness_ui_intents("default").await?;
    assert_release_operator_static_ui_intents(&intents);

    let seeded = client
        .run_harness_action(HarnessActionRunParams {
            action: "release.seed_demo_work".to_string(),
            agent_id: None,
            harness_id: Some("default".to_string()),
            params: serde_json::json!({
                "release": "2026.06",
                "release_mode": "hotfix",
                "count": 4,
            }),
        })
        .await?;
    assert_eq!(seeded.action, "release.seed_demo_work");
    assert_eq!(seeded.result["status"], "seeded");
    assert_eq!(seeded.result["count"], 4);
    assert_eq!(seeded.result["release_mode"], "hotfix");
    assert_release_operator_seeded_ui_intents(&seeded);

    let release_worklist = client
        .list_worklists(WorklistListParams {
            persistence: None,
            name: Some("release".to_string()),
            scope: None,
        })
        .await?
        .into_iter()
        .next()
        .context("release worklist should exist after seeding demo work")?;
    let pending = client
        .list_worklist_items(WorklistItemsParams {
            id: release_worklist.public_id.clone(),
            persistence: release_worklist.persistence.clone(),
            status: Some("pending".to_string()),
            parent_id: None,
            r#where: None,
            claimed_only: false,
            paused_only: false,
            due_only: false,
            limit: None,
        })
        .await?;
    assert_eq!(pending.items.len(), 4);
    assert!(pending.items.iter().all(|item| {
        item.metadata
            .as_ref()
            .and_then(|metadata| metadata.get("release"))
            == Some(&serde_json::json!("2026.06"))
    }));
    assert!(pending.items.iter().all(|item| {
        item.metadata
            .as_ref()
            .and_then(|metadata| metadata.get("release_mode"))
            == Some(&serde_json::json!("hotfix"))
    }));

    let approved = client
        .run_harness_action(HarnessActionRunParams {
            action: "release.approve_next".to_string(),
            agent_id: None,
            harness_id: Some("default".to_string()),
            params: serde_json::json!({
                "release": "2026.06",
            }),
        })
        .await?;
    assert_eq!(approved.action, "release.approve_next");
    assert_eq!(approved.result["status"], "approved");
    assert_release_operator_action_notice_and_refresh(
        &approved,
        "Approved next item",
        UiNoticeLevel::Success,
    );

    let rejected = client
        .run_harness_action(HarnessActionRunParams {
            action: "release.reject_next".to_string(),
            agent_id: None,
            harness_id: Some("default".to_string()),
            params: serde_json::json!({
                "release": "2026.06",
            }),
        })
        .await?;
    assert_eq!(rejected.action, "release.reject_next");
    assert_eq!(rejected.result["status"], "rejected");
    assert_release_operator_action_notice_and_refresh(
        &rejected,
        "Rejected next item",
        UiNoticeLevel::Warning,
    );

    let shown = client
        .run_harness_action(HarnessActionRunParams {
            action: "release.show_notes".to_string(),
            agent_id: None,
            harness_id: Some("default".to_string()),
            params: serde_json::json!({}),
        })
        .await?;
    assert_eq!(shown.action, "release.show_notes");
    assert_eq!(shown.result["status"], "shown");
    assert_release_operator_show_intent(&shown);

    let opened = client
        .run_harness_action(HarnessActionRunParams {
            action: "release.open_intake".to_string(),
            agent_id: None,
            harness_id: Some("default".to_string()),
            params: serde_json::json!({}),
        })
        .await?;
    assert_eq!(opened.action, "release.open_intake");
    assert_eq!(opened.result["status"], "opened");
    assert_release_operator_open_focus_intents(&opened);

    let remaining = client
        .list_worklist_items(WorklistItemsParams {
            id: release_worklist.public_id,
            persistence: release_worklist.persistence,
            status: Some("pending".to_string()),
            parent_id: None,
            r#where: None,
            claimed_only: false,
            paused_only: false,
            due_only: false,
            limit: None,
        })
        .await?;
    assert_eq!(remaining.items.len(), 2);

    Ok(())
}

fn assert_release_operator_static_ui_intents(intents: &[UiIntentMessage]) {
    assert!(intents.iter().any(|message| {
        matches!(
            &message.intent,
            UiIntent::App(app)
                if app.id == "release-operator"
                    && app.title == "Release Operator"
                    && app.about.as_deref()
                        == Some("Coordinate release approvals, intake, readiness, and notes from one focused operator console.")
        )
    }));
    assert!(intents.iter().any(|message| {
        matches!(
            &message.intent,
            UiIntent::OpensWith(opens_with)
                if opens_with.app_id == "release-operator"
                    && opens_with.screen_id == "home"
        )
    }));
    assert!(intents.iter().any(|message| {
        matches!(
            &message.intent,
            UiIntent::Menu(menu)
                if menu.app_id == "release-operator"
                    && menu.items.iter().any(|item| {
                        item.label == "Work"
                            && item.badge.as_deref() == Some("approvals")
                            && item.items.iter().any(|child| child.opens == "intake")
                    })
        )
    }));
    assert!(intents.iter().any(|message| {
        matches!(
            &message.intent,
            UiIntent::Badge(badge)
                if badge.app_id == "release-operator"
                    && badge.target == "release-readiness"
                    && badge.label.as_deref() == Some("live")
                    && badge.level == Some(UiNoticeLevel::Info)
        )
    }));

    let home = release_screen(intents, "home");
    let home_nodes = flattened_nodes(&home.nodes);
    assert!(home_nodes.iter().any(|node| {
        matches!(
            node,
            UiNode::Text(text)
                if text.text.contains("review readiness without leaving the operator console")
        )
    }));
    assert!(home_nodes.iter().any(|node| {
        matches!(
            node,
            UiNode::Action(action)
                if action.id.as_deref() == Some("approve-next")
                    && action.action == "release.approve_next"
                    && action.confirm
        )
    }));
    assert!(home_nodes.iter().any(|node| {
        matches!(
            node,
            UiNode::List(list)
                if list.id.as_deref() == Some("recent-release-work")
                    && list.source == "worklists.release"
                    && list.limit == Some(8)
                    && list.intent.as_deref() == Some("tasks")
                    && list.render_as.as_deref() == Some("table")
        )
    }));
    assert!(home_nodes.iter().any(|node| {
        matches!(
            node,
            UiNode::Activity(activity)
                if activity.id.as_deref() == Some("release-activity")
                    && activity.source == "worklists.release"
        )
    }));

    let approvals = release_screen(intents, "approvals");
    let approvals_nodes = flattened_nodes(&approvals.nodes);
    assert!(approvals_nodes.iter().any(|node| {
        matches!(
            node,
            UiNode::List(list)
                if list.id.as_deref() == Some("pending-approvals")
                    && list.source == "worklists.release"
                    && list.filter.get("kind") == Some(&serde_json::json!("approval"))
                    && list.filter.get("status") == Some(&serde_json::json!("pending"))
                    && list.fields.iter().any(|field| field == "lane")
                    && list.intent.as_deref() == Some("approval")
                    && list.render_as.as_deref() == Some("table")
        )
    }));

    let intake = release_screen(intents, "intake");
    let intake_nodes = flattened_nodes(&intake.nodes);
    assert!(intake_nodes.iter().any(|node| {
        matches!(
            node,
            UiNode::Form(form)
                if form.id.as_deref() == Some("seed-demo-form")
                    && form.action == "release.seed_demo_work"
                    && form.params.get("count") == Some(&serde_json::json!(1))
                    && form.params.get("release_mode") == Some(&serde_json::json!("standard"))
                    && form.fields.iter().any(|field| {
                        field.name == "count"
                            && field.kind.as_deref() == Some("number")
                            && field.default.is_none()
                    })
                    && form.fields.iter().any(|field| {
                        field.name == "release_mode"
                            && field.kind.as_deref() == Some("select")
                            && field.options == vec![
                                serde_json::json!("standard"),
                                serde_json::json!("hotfix"),
                                serde_json::json!("rollback"),
                            ]
                            && field.default.is_none()
                    })
        )
    }));

    let overview = release_screen(intents, "overview");
    let overview_nodes = flattened_nodes(&overview.nodes);
    assert!(overview_nodes.iter().any(|node| {
        matches!(
            node,
            UiNode::Detail(detail)
                if detail.id.as_deref() == Some("release-snapshot")
                    && detail.source == "worklists.release"
        )
    }));
    assert!(overview_nodes.iter().any(|node| {
        matches!(
            node,
            UiNode::Report(report)
                if report.id.as_deref() == Some("release-readiness")
                    && report.source == "worklists.release"
                    && report.prompt.as_deref()
                        == Some("Summarize current release approval readiness.")
        )
    }));
    assert!(overview_nodes.iter().any(|node| {
        matches!(
            node,
            UiNode::Chart(chart)
                if chart.id.as_deref() == Some("approval-flow")
                    && chart.source == "worklists.release"
                    && chart.intent.as_deref() == Some("status_breakdown")
                    && chart.render_as.as_deref() == Some("bar")
        )
    }));

    let pane = release_pane(intents, "release-notes");
    assert_eq!(pane.presentation.as_deref(), Some("sheet"));
    let pane_nodes = flattened_nodes(&pane.nodes);
    assert!(pane_nodes.iter().any(|node| {
        matches!(
            node,
            UiNode::Detail(detail)
                if detail.id.as_deref() == Some("pane-release-snapshot")
                    && detail.source == "worklists.release"
        )
    }));
}

fn release_screen<'a>(intents: &'a [UiIntentMessage], id: &str) -> &'a UiScreenIntent {
    intents
        .iter()
        .find_map(|message| match &message.intent {
            UiIntent::Screen(screen) if screen.app_id == "release-operator" && screen.id == id => {
                Some(screen)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing release-operator screen '{id}'"))
}

fn release_pane<'a>(intents: &'a [UiIntentMessage], id: &str) -> &'a UiPaneIntent {
    intents
        .iter()
        .find_map(|message| match &message.intent {
            UiIntent::Pane(pane) if pane.app_id == "release-operator" && pane.id == id => {
                Some(pane)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing release-operator pane '{id}'"))
}

fn flattened_nodes(nodes: &[UiNode]) -> Vec<&UiNode> {
    let mut out = Vec::new();
    for node in nodes {
        out.push(node);
        if let UiNode::Section(section) = node {
            out.extend(flattened_nodes(&section.nodes));
        }
    }
    out
}

fn assert_release_operator_seeded_ui_intents(
    result: &turin_daemon_protocol::HarnessActionRunResult,
) {
    assert!(result.ui_intents.iter().any(|message| {
        matches!(
            &message.intent,
            UiIntent::Notify(notice)
                if notice.app_id == "release-operator"
                    && notice.title == "Seeded release work"
                    && notice.body.as_deref()
                        == Some("Created 4 approval items for 2026.06.")
                    && notice.level == Some(UiNoticeLevel::Success)
        )
    }));
    assert!(result.ui_intents.iter().any(|message| {
        matches!(
            &message.intent,
            UiIntent::Badge(badge)
                if badge.app_id == "release-operator"
                    && badge.target == "approvals"
                    && badge.count == Some(4)
                    && badge.level == Some(UiNoticeLevel::Info)
        )
    }));
    assert!(result.ui_intents.iter().any(|message| {
        matches!(
            &message.intent,
            UiIntent::Refresh(refresh)
                if refresh.app_id == "release-operator"
                    && refresh.binding == "worklists.release"
        )
    }));
}

fn assert_release_operator_action_notice_and_refresh(
    result: &turin_daemon_protocol::HarnessActionRunResult,
    title: &str,
    level: UiNoticeLevel,
) {
    assert!(result.ui_intents.iter().any(|message| {
        matches!(
            &message.intent,
            UiIntent::Notify(notice)
                if notice.app_id == "release-operator"
                    && notice.title == title
                    && notice.level == Some(level)
        )
    }));
    assert!(result.ui_intents.iter().any(|message| {
        matches!(
            &message.intent,
            UiIntent::Refresh(refresh)
                if refresh.app_id == "release-operator"
                    && refresh.binding == "worklists.release"
        )
    }));
}

fn assert_release_operator_show_intent(result: &turin_daemon_protocol::HarnessActionRunResult) {
    assert!(result.ui_intents.iter().any(|message| {
        matches!(
            &message.intent,
            UiIntent::Show(show)
                if show.app_id == "release-operator"
                    && show.target == "release-notes"
                    && show.presentation.as_deref() == Some("sheet")
        )
    }));
}

fn assert_release_operator_open_focus_intents(
    result: &turin_daemon_protocol::HarnessActionRunResult,
) {
    assert!(result.ui_intents.iter().any(|message| {
        matches!(
            &message.intent,
            UiIntent::Open(open)
                if open.app_id == "release-operator"
                    && open.target == "intake"
        )
    }));
    assert!(result.ui_intents.iter().any(|message| {
        matches!(
            &message.intent,
            UiIntent::Focus(focus)
                if focus.app_id == "release-operator"
                    && focus.target == "seed-demo-form"
        )
    }));
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

#[tokio::test(flavor = "multi_thread")]
async fn control_client_release_operator_ui_smoke() -> Result<()> {
    let daemon = DaemonHarness::start_with_harness(include_str!(
        "../../../tests/fixtures/harnesses/ui_contract/main.lua"
    ))
    .await?;
    let client = ControlClient::connect(&ConnectionSpec::LocalEndpoint {
        endpoint: daemon.endpoint.clone(),
    })
    .await?;

    assert_release_operator_ui_workflow(&client).await?;

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn control_client_release_operator_ui_smoke_remote() -> Result<()> {
    let daemon = DaemonHarness::start_with_harness(include_str!(
        "../../../tests/fixtures/harnesses/ui_contract/main.lua"
    ))
    .await?;
    let remote = RemoteHarness::start(&daemon.config_path).await?;
    let client = ControlClient::connect(&ConnectionSpec::Remote {
        base_url: remote.base_url.clone(),
        auth_token: "test-token".to_string(),
    })
    .await?;

    let health = client.health().await?;
    assert!(health.ready);
    assert_eq!(health.connection_kind, ConnectionKind::Remote);
    assert_release_operator_ui_workflow(&client).await?;

    remote.stop().await?;
    daemon.stop().await
}
