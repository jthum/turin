use super::*;
use crate::harness::context::{ContextWrapper, RequestOptionsOverride};
use crate::harness::scheduler::HarnessSchedulerAccess;
use crate::inference::provider::{
    InferenceEvent, InferenceProvider, InferenceRequest, InferenceStream, ProviderClient, SdkError,
};
use crate::kernel::config::InferenceOverrideConfig;
use crate::persistence::manager::{StoreManager, StorePathScope, StoreSelector};
use crate::persistence::state::{StateStore, WorkItemInsert, WorkItemUpdate};
use futures::future::BoxFuture;
use futures::stream;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tempfile::TempDir;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

#[derive(Clone)]
struct CountingTextProvider {
    counter: Arc<Mutex<u32>>,
}

impl InferenceProvider for CountingTextProvider {
    fn stream<'a>(
        &'a self,
        _request: InferenceRequest,
        _options: Option<crate::inference::provider::RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        let counter = Arc::clone(&self.counter);
        Box::pin(async move {
            let next = {
                let mut lock = counter.lock().unwrap();
                *lock += 1;
                *lock
            };
            let text = format!("summary-{next}");
            let events = vec![
                Ok(InferenceEvent::MessageStart {
                    role: "assistant".to_string(),
                    model: "mock-model".to_string(),
                    provider_id: "mock".to_string(),
                }),
                Ok(InferenceEvent::MessageDelta { content: text }),
                Ok(InferenceEvent::MessageEnd {
                    input_tokens: 1,
                    output_tokens: 1,
                    stop_reason: None,
                }),
            ];
            Ok(Box::pin(stream::iter(events)) as InferenceStream)
        })
    }
}

fn test_app_data_for_root_and_session(root: PathBuf, session_id: &str) -> HarnessAppData {
    HarnessAppData {
        fs_root: root.clone(),
        workspace_root: root.clone(),
        harness_directory: root.clone(),
        store_manager: Arc::new(StoreManager::new(
            root.clone(),
            turin_types::layout::default_stores_dir_for_workspace(&root),
        )),
        agent_manager: Arc::new(crate::kernel::agent_manager::AgentManager::new(
            std::sync::Arc::new(crate::kernel::config::TurinConfig::default()),
            Arc::new(StoreManager::new(
                root.clone(),
                turin_types::layout::default_stores_dir_for_workspace(&root),
            )),
        )),
        policy_manager: Arc::new(crate::kernel::policy::RuntimePolicyManager::new()),
        governance_manager: Arc::new(crate::kernel::governance::GovernanceManager::new(
            crate::kernel::config::GovernanceConfig::default(),
        )),
        scheduler: None,
        clients: std::collections::HashMap::new(),
        embedding_provider: None,
        execution_ctx: std::sync::Arc::new(std::sync::Mutex::new(
            crate::harness::globals::HarnessExecutionContext {
                agent_id: Some("test-agent".to_string()),
                session_id: Some(session_id.to_string()),
                queue: Some(std::sync::Arc::new(tokio::sync::Mutex::new(
                    std::collections::VecDeque::new(),
                ))),
                cancel_token: Some(CancellationToken::new()),
                ..Default::default()
            },
        )),
        config: std::sync::Arc::new(crate::kernel::config::TurinConfig::default()),
        spawn_depth: 0,
        active_modules: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        watch_roots: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        loading_phase: std::sync::Arc::new(std::sync::Mutex::new(true)),
    }
}

fn test_app_data_for_root(root: PathBuf) -> HarnessAppData {
    test_app_data_for_root_and_session(root, "test-session")
}

fn test_app_data() -> HarnessAppData {
    test_app_data_for_root(PathBuf::from("."))
}

async fn test_app_data_with_scheduler(root: PathBuf) -> HarnessAppData {
    let mut app_data = test_app_data_for_root(root);
    app_data.config = std::sync::Arc::new(crate::kernel::config::TurinConfig {
        agent: crate::kernel::config::AgentConfig {
            id: "test-agent".to_string(),
            ..crate::kernel::config::AgentConfig::default()
        },
        ..crate::kernel::config::TurinConfig::default()
    });
    let runtime_store = Arc::new(StateStore::open_memory().await.expect("open runtime store"));
    app_data.scheduler = Some(Arc::new(HarnessSchedulerAccess::new(
        runtime_store,
        Some(Arc::new(Notify::new())),
    )));
    app_data
}

async fn open_default_state_store(app_data: &HarnessAppData) -> Arc<StateStore> {
    app_data
        .store_manager
        .open_with_path_scope(
            &StoreSelector::Alias("state".to_string()),
            StorePathScope::WorkspaceOnly,
        )
        .await
        .expect("open default state store")
}

#[test]
fn test_engine_no_scripts() {
    let engine = HarnessEngine::new(test_app_data()).unwrap();
    assert!(engine.loaded_scripts().is_empty());

    let verdict = engine
        .evaluate("on_tool_call", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[test]
fn test_engine_load_empty_dir() {
    let dir = TempDir::new().unwrap();
    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();
    assert!(engine.loaded_scripts().is_empty());
}

#[test]
fn test_engine_load_nonexistent_dir() {
    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(Path::new("/nonexistent/path")).unwrap();
    assert!(engine.loaded_scripts().is_empty());
}

#[test]
fn test_ui_load_time_intents_are_collected() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("ui.lua"),
        r#"
            local app = ui.app("Release Operator", {
                id = "release",
                about = "Coordinate release checks",
            })

            app:home("Release Desk", function(screen)
                screen:list("Open Work", {
                    id = "open-work",
                    from = "worklists.release",
                    where = { kind = "approval" },
                    intent = "approval",
                    as = "table",
                })
                screen:section("Controls", function(section)
                    section:action("Run Smoke Tests", "qa.run_smoke", {
                        id = "run-smoke",
                        params = { suite = "smoke" },
                    })
                end)
                screen:activity("Release Activity", {
                    from = "signals.release",
                })
            end)

            app:screen("approvals", "Approvals", function(screen)
                screen:worklist("Pending Reviews", {
                    from = "release",
                    where = { kind = "approval" },
                })
            end)

            app:menu("Main", function(menu)
                menu:item("Dashboard", "home")
                menu:item("Approvals", "approvals", { badge = "approvals" })
            end)
        "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let intents = engine.ui_intents().unwrap();
    assert_eq!(intents.len(), 5);
    let turin_daemon_protocol::UiIntent::App(app) = &intents[0].intent else {
        panic!("expected app intent");
    };
    assert_eq!(app.id, "release");
    assert_eq!(app.about.as_deref(), Some("Coordinate release checks"));

    let turin_daemon_protocol::UiIntent::Screen(home) = &intents[1].intent else {
        panic!("expected screen intent");
    };
    assert_eq!(home.title, "Release Desk");
    assert_eq!(home.id, "home");
    assert_eq!(home.app_id, "release");
    assert_eq!(home.nodes.len(), 3);

    let turin_daemon_protocol::UiNode::List(list) = &home.nodes[0] else {
        panic!("expected list node");
    };
    assert_eq!(list.id.as_deref(), Some("open-work"));
    assert_eq!(list.source, "worklists.release");
    assert_eq!(list.intent.as_deref(), Some("approval"));
    assert_eq!(list.render_as.as_deref(), Some("table"));
    assert_eq!(list.filter["kind"], "approval");

    let turin_daemon_protocol::UiNode::Section(section) = &home.nodes[1] else {
        panic!("expected section node");
    };
    assert_eq!(section.title, "Controls");
    let turin_daemon_protocol::UiNode::Action(action) = &section.nodes[0] else {
        panic!("expected action node");
    };
    assert_eq!(action.label, "Run Smoke Tests");
    assert_eq!(action.action, "qa.run_smoke");
    assert_eq!(action.params["suite"], "smoke");

    assert!(matches!(
        intents[2].intent,
        turin_daemon_protocol::UiIntent::OpensWith(_)
    ));

    let turin_daemon_protocol::UiIntent::Screen(approvals) = &intents[3].intent else {
        panic!("expected approvals screen");
    };
    let turin_daemon_protocol::UiNode::List(list) = &approvals.nodes[0] else {
        panic!("expected worklist sugar to create list node");
    };
    assert_eq!(list.source, "worklists.release");
    assert_eq!(list.intent.as_deref(), Some("tasks"));
    assert_eq!(list.render_as.as_deref(), Some("table"));

    let turin_daemon_protocol::UiIntent::Menu(menu) = &intents[4].intent else {
        panic!("expected menu intent");
    };
    assert_eq!(menu.items.len(), 2);
    assert_eq!(menu.items[1].badge.as_deref(), Some("approvals"));
}

#[test]
fn test_ui_release_operator_example_loads() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/harnesses/ui_release_operator");
    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(&dir).unwrap();

    #[derive(Default)]
    struct ObservedNodes {
        confirmed_action: bool,
        list: bool,
        activity: bool,
        detail: bool,
        report: bool,
        chart: bool,
        form: bool,
    }

    fn observe_nodes(nodes: &[turin_daemon_protocol::UiNode], observed: &mut ObservedNodes) {
        for node in nodes {
            match node {
                turin_daemon_protocol::UiNode::Action(action) => {
                    observed.confirmed_action |= action.confirm;
                }
                turin_daemon_protocol::UiNode::List(_) => {
                    observed.list = true;
                }
                turin_daemon_protocol::UiNode::Activity(_) => {
                    observed.activity = true;
                }
                turin_daemon_protocol::UiNode::Detail(_) => {
                    observed.detail = true;
                }
                turin_daemon_protocol::UiNode::Report(_) => {
                    observed.report = true;
                }
                turin_daemon_protocol::UiNode::Chart(_) => {
                    observed.chart = true;
                }
                turin_daemon_protocol::UiNode::Form(_) => {
                    observed.form = true;
                }
                turin_daemon_protocol::UiNode::Section(section) => {
                    observe_nodes(&section.nodes, observed);
                }
                _ => {}
            }
        }
    }

    let intents = engine.ui_intents().unwrap();
    let mut app_seen = false;
    let mut screen_count = 0;
    let mut observed = ObservedNodes::default();
    let mut pane_seen = false;
    let mut badge_seen = false;
    let mut nested_menu_seen = false;

    for intent in intents {
        match intent.intent {
            turin_daemon_protocol::UiIntent::App(app) => {
                app_seen = app.id == "release-operator";
            }
            turin_daemon_protocol::UiIntent::Screen(screen) => {
                screen_count += 1;
                observe_nodes(&screen.nodes, &mut observed);
            }
            turin_daemon_protocol::UiIntent::Pane(pane) => {
                pane_seen = pane.id == "release-notes";
            }
            turin_daemon_protocol::UiIntent::Menu(menu) => {
                nested_menu_seen = menu.items.iter().any(|item| !item.items.is_empty());
            }
            turin_daemon_protocol::UiIntent::Badge(badge) => {
                badge_seen = badge.target == "release-readiness";
            }
            _ => {}
        }
    }

    assert!(app_seen);
    assert!(screen_count >= 4);
    assert!(observed.confirmed_action);
    assert!(observed.list);
    assert!(observed.activity);
    assert!(observed.detail);
    assert!(observed.report);
    assert!(observed.chart);
    assert!(observed.form);
    assert!(pane_seen);
    assert!(badge_seen);
    assert!(nested_menu_seen);
}

#[test]
fn test_ui_worklist_sugar_normalizes_bare_and_prefixed_sources() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("ui.lua"),
        r#"
            local app = ui.app("Release Operator", { id = "release" })

            app:home("Release Desk", function(screen)
                screen:worklist("Bare From", { from = "release" })
                screen:worklist("Prefixed From", { from = "worklists.release" })
                screen:worklist("Bare Source", { source = "release" })
                screen:worklist("Prefixed Source", { source = "worklists.release" })
            end)
        "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let intents = engine.ui_intents().unwrap();
    let screen = intents
        .iter()
        .find_map(|message| match &message.intent {
            turin_daemon_protocol::UiIntent::Screen(screen) => Some(screen),
            _ => None,
        })
        .expect("screen intent");
    let sources = screen
        .nodes
        .iter()
        .map(|node| match node {
            turin_daemon_protocol::UiNode::List(list) => list.source.as_str(),
            other => panic!("expected list node, got {other:?}"),
        })
        .collect::<Vec<_>>();

    assert_eq!(
        sources,
        vec![
            "worklists.release",
            "worklists.release",
            "worklists.release",
            "worklists.release"
        ]
    );
}

#[test]
fn test_ui_dynamic_intents_are_collected_from_hooks() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("ui.lua"),
        r#"
            local app = ui.app("Release Operator", { id = "release" })

            function on_turn_prepare(_ctx)
                app:notice("Release blocked", {
                    body = "QA failed",
                    level = "warning",
                })
                app:open("approvals")
                app:show("release-notes", {
                    area = "pane",
                    presentation = "sheet",
                })
                app:badge("approvals", { count = 3, level = "warning" })
                app:focus("open-work")
                app:refresh("worklists.release")
                return ALLOW
            end
        "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();
    assert_eq!(engine.ui_intents().unwrap().len(), 1);

    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);

    let intents = engine.ui_intents().unwrap();
    assert_eq!(intents.len(), 7);
    assert!(matches!(
        intents[1].intent,
        turin_daemon_protocol::UiIntent::Notify(_)
    ));
    assert!(matches!(
        intents[2].intent,
        turin_daemon_protocol::UiIntent::Open(_)
    ));
    let turin_daemon_protocol::UiIntent::Show(show) = &intents[3].intent else {
        panic!("expected show intent");
    };
    assert_eq!(show.app_id, "release");
    assert_eq!(show.target, "release-notes");
    assert_eq!(show.area.as_deref(), Some("pane"));
    assert_eq!(show.presentation.as_deref(), Some("sheet"));
    assert!(matches!(
        intents[4].intent,
        turin_daemon_protocol::UiIntent::Badge(_)
    ));
    assert!(matches!(
        intents[5].intent,
        turin_daemon_protocol::UiIntent::Focus(_)
    ));
    assert!(matches!(
        intents[6].intent,
        turin_daemon_protocol::UiIntent::Refresh(_)
    ));
}

#[test]
fn test_top_level_ui_dynamic_intents_use_default_app() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("ui.lua"),
        r#"
            local release = ui.app("Release Operator", { id = "release" })
            local qa = ui.app("QA Console", { id = "qa" })

            function on_turn_prepare(_ctx)
                ui.notice("Release blocked", {
                    body = "QA failed",
                    level = "warning",
                })
                ui.open("approvals")
                ui.show("release-notes", {
                    area = "pane",
                    presentation = "sheet",
                })
                ui.badge("approvals", { count = 3, level = "warning" })
                ui.focus("open-work")
                ui.refresh("worklists.release")
                return ALLOW
            end
        "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();
    assert_eq!(engine.ui_intents().unwrap().len(), 2);

    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);

    let intents = engine.ui_intents().unwrap();
    assert_eq!(intents.len(), 8);

    let turin_daemon_protocol::UiIntent::Notify(notice) = &intents[2].intent else {
        panic!("expected notice intent");
    };
    assert_eq!(notice.app_id, "release");
    assert_eq!(notice.title, "Release blocked");

    let turin_daemon_protocol::UiIntent::Open(open) = &intents[3].intent else {
        panic!("expected open intent");
    };
    assert_eq!(open.app_id, "release");
    assert_eq!(open.target, "approvals");

    let turin_daemon_protocol::UiIntent::Show(show) = &intents[4].intent else {
        panic!("expected show intent");
    };
    assert_eq!(show.app_id, "release");
    assert_eq!(show.target, "release-notes");
    assert_eq!(show.area.as_deref(), Some("pane"));

    let turin_daemon_protocol::UiIntent::Badge(badge) = &intents[5].intent else {
        panic!("expected badge intent");
    };
    assert_eq!(badge.app_id, "release");
    assert_eq!(badge.target, "approvals");

    let turin_daemon_protocol::UiIntent::Focus(focus) = &intents[6].intent else {
        panic!("expected focus intent");
    };
    assert_eq!(focus.app_id, "release");
    assert_eq!(focus.target, "open-work");

    let turin_daemon_protocol::UiIntent::Refresh(refresh) = &intents[7].intent else {
        panic!("expected refresh intent");
    };
    assert_eq!(refresh.app_id, "release");
    assert_eq!(refresh.binding, "worklists.release");
}

#[test]
fn test_top_level_ui_dynamic_intents_require_declared_app() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("ui.lua"),
        r#"
            function on_turn_prepare(_ctx)
                ui.notice("No app yet")
                return ALLOW
            end
        "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let err = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("ui intent requires ui.app(...) to be declared first")
    );
}

#[test]
fn test_ui_dynamic_intents_are_emitted_as_ephemeral_session_events() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("ui.lua"),
        r#"
            local app = ui.app("Release Operator", { id = "release" })

            function on_turn_prepare(_ctx)
                app:notice("Release blocked", {
                    body = "QA failed",
                    level = "warning",
                })
                return ALLOW
            end
        "#,
    )
    .unwrap();

    let (event_tx, mut event_rx) = tokio::sync::broadcast::channel(8);
    let app_data = test_app_data();
    {
        let mut lock = app_data.execution_ctx.lock().unwrap();
        lock.event_context = Some(crate::harness::globals::HarnessEventContext {
            json: false,
            internal_id: Some(42),
            branch_head_id: None,
            execution_id: "exec-1".to_string(),
            event_tx,
            durability_tx: None,
        });
    }

    let mut engine = HarnessEngine::new(app_data).unwrap();
    engine.load_dir(dir.path()).unwrap();

    assert!(
        event_rx.try_recv().is_err(),
        "load-time app intent is static"
    );

    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);

    let (internal_id, event) = event_rx.try_recv().expect("dynamic ui intent event");
    assert_eq!(internal_id, Some(42));
    assert_eq!(event.event_type(), turin_daemon_protocol::UI_INTENT_EVENT);

    let crate::kernel::event::KernelEvent::Ui(crate::kernel::event::UiEvent::Intent { intent }) =
        event
    else {
        panic!("expected ui intent kernel event");
    };
    assert_eq!(intent.source.agent_id.as_deref(), Some("test-agent"));
    let turin_daemon_protocol::UiIntent::Notify(notice) = intent.intent else {
        panic!("expected notice intent");
    };
    assert_eq!(notice.app_id, "release");
    assert_eq!(notice.title, "Release blocked");
    assert_eq!(notice.body.as_deref(), Some("QA failed"));
}

#[test]
fn test_runtime_inference_available_reports_named_contexts() {
    let mut app_data = test_app_data();
    let mut config = crate::kernel::config::TurinConfig::default();
    config.agent.id = "test-agent".to_string();
    config.inference.contexts.insert(
        "fast".to_string(),
        crate::kernel::config::InferenceContextConfig {
            provider: "mock".to_string(),
            model: "mock-fast".to_string(),
            fallback: None,
            temperature: None,
            max_tokens: None,
            thinking_budget: None,
        },
    );
    app_data.config = Arc::new(config);

    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("inference_available.lua"),
        r#"
            function on_turn_prepare(ctx)
                if not runtime.inference.available("fast") then
                    error("expected fast inference context to be available")
                end
                if runtime.inference.available("missing") then
                    error("missing inference context should not be available")
                end
                return ALLOW
            end
        "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(app_data).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());
}

#[test]
fn test_engine_allow_verdict() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("allow.lua"),
        r#"
            function on_tool_call(call)
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate(
            "on_tool_call",
            serde_json::json!({"name": "read_file", "args": {}}),
        )
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[test]
fn test_engine_reject_verdict() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("safety.lua"),
        r#"
            function on_tool_call(call)
                if call.name == "shell_exec" then
                    return REJECT, "Shell commands are not allowed"
                end
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate(
            "on_tool_call",
            serde_json::json!({"name": "shell_exec", "args": {"command": "ls"}}),
        )
        .unwrap();
    assert!(verdict.is_rejected());
    assert_eq!(verdict.reason(), Some("Shell commands are not allowed"));

    let verdict = engine
        .evaluate(
            "on_tool_call",
            serde_json::json!({"name": "read_file", "args": {"path": "foo.txt"}}),
        )
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[test]
fn test_engine_escalate_verdict() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("escalation.lua"),
        r#"
            function on_tool_call(call)
                if call.name == "write_file" then
                    return ESCALATE, "File writes need human approval"
                end
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate(
            "on_tool_call",
            serde_json::json!({"name": "write_file", "args": {}}),
        )
        .unwrap();
    assert!(verdict.is_escalated());
    assert_eq!(verdict.reason(), Some("File writes need human approval"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_runtime_schedule_requires_daemon_managed_runtime() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            function on_turn_prepare(ctx)
                local ok, err = pcall(function()
                    return runtime.schedule.list()
                end)
                if ok then
                    error("expected runtime.schedule.list to fail without daemon scheduler")
                end
                local message = tostring(err)
                if not string.find(message, "daemon%-managed runtime") then
                    error("unexpected error: " .. message)
                end
                return ALLOW
            end
        "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(dir.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_runtime_signals_requires_daemon_managed_runtime() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            function on_turn_prepare(_ctx)
                local ok_list, err_list = pcall(function()
                    return runtime.signals.list()
                end)
                if ok_list then
                    error("expected runtime.signals.list to fail without daemon scheduler")
                end
                if not string.find(tostring(err_list), "daemon runtime coordination") then
                    error("unexpected list error: " .. tostring(err_list))
                end

                local ok_subs, err_subs = pcall(function()
                    return runtime.signals.subscribers("code.ready")
                end)
                if ok_subs then
                    error("expected runtime.signals.subscribers to fail without daemon scheduler")
                end
                if not string.find(tostring(err_subs), "daemon runtime coordination") then
                    error("unexpected subscribers error: " .. tostring(err_subs))
                end

                return ALLOW
            end
        "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(dir.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_schedule_dx_helpers_create_and_list_jobs() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            function on_turn_prepare(ctx)
                local one = schedule.after(30, "Check status", {
                    overlap = "queue"
                })
                if one.agent_id ~= "test-agent" then
                    error("expected default agent binding")
                end

                local recurring = schedule.every(60, "Run tests", {
                    state = "qa",
                    store = { path = "./project.db" },
                    work_key = "project:alpha:qa",
                    max_concurrency = 1
                })
                if recurring.interval_seconds ~= 60 then
                    error("expected recurring interval")
                end
                if recurring.persistence == nil then
                    error("expected state persistence override")
                end
                if recurring.work_key ~= "project:alpha:qa" then
                    error("expected work_key to round-trip")
                end
                if recurring.max_concurrency ~= 1 then
                    error("expected max_concurrency to round-trip")
                end

                local fetched = schedule.get(recurring.public_id)
                if fetched.public_id ~= recurring.public_id then
                    error("expected schedule.get to round-trip recurring job")
                end

                local empty_runs = schedule.runs(recurring.public_id, {
                    active_only = true,
                    limit = 5
                })
                if empty_runs.public_id ~= recurring.public_id then
                    error("expected schedule.runs to target the requested job")
                end
                if #empty_runs.runs ~= 0 then
                    error("expected no runs before scheduler tick")
                end

                local disabled = schedule.disable(recurring.public_id)
                if disabled.enabled ~= false then
                    error("expected disable to clear enabled flag")
                end

                local updated = schedule.update(recurring.public_id, {
                    prompt = "Run QA tests",
                    interval_seconds = 120,
                    overlap = "skip",
                    work_key = "project:alpha:qa:updated",
                    max_concurrency = 2
                })
                if updated.prompt ~= "Run QA tests" then
                    error("expected update to replace prompt")
                end
                if updated.interval_seconds ~= 120 then
                    error("expected update to replace interval")
                end
                if updated.overlap_policy ~= "skip" then
                    error("expected update to normalize overlap policy")
                end
                if updated.work_key ~= "project:alpha:qa:updated" then
                    error("expected update to replace work_key")
                end
                if updated.max_concurrency ~= 2 then
                    error("expected update to replace max_concurrency")
                end

                local structured = runtime.schedule.create({
                    prompt = "Review artifact bundle",
                    content = {
                        { type = "text", text = "Use the attached QA context" }
                    },
                    tools = {
                        allow = { "shell_exec" }
                    },
                    conflict_policy = "detached",
                    after_seconds = 15
                })
                if structured.content == nil or structured.content[1].text ~= "Use the attached QA context" then
                    error("expected structured content to round-trip")
                end
                if structured.tools == nil or structured.tools.allow == nil or structured.tools.allow[1] ~= "shell_exec" then
                    error("expected structured tools to round-trip")
                end
                if structured.conflict_policy ~= "detached" then
                    error("expected conflict_policy to round-trip")
                end

                local anchored = schedule.at("2030-01-02T03:04:05Z", "Nightly checks", {
                    recurring = "weekly"
                })
                if anchored.recurring_pattern ~= "weekly" then
                    error("expected recurring pattern to round-trip for anchored schedule")
                end

                local morning = schedule.at("08:00", "Daily digest", {
                    recurring = "daily"
                })
                if morning.recurring_pattern ~= "daily" then
                    error("expected recurring pattern to round-trip for local-time schedule")
                end

                local action_job = schedule.after(45, {
                    action = "agent.disable",
                    params = { id = "night-qa" }
                })
                if action_job.kind ~= "action" then
                    error("expected action job kind")
                end
                if action_job.action == nil or action_job.action.name ~= "agent.disable" then
                    error("expected action payload to round-trip")
                end

                local reenabled = schedule.enable(recurring.public_id)
                if reenabled.enabled ~= true then
                    error("expected enable to restore enabled flag")
                end

                local jobs = schedule.list()
                if #jobs ~= 6 then
                    error("expected 6 scheduled jobs, got " .. tostring(#jobs))
                end

                local deleted = schedule.delete(one.public_id)
                if deleted.public_id ~= one.public_id then
                    error("expected delete to return removed job")
                end

                jobs = schedule.list()
                if #jobs ~= 5 then
                    error("expected 5 scheduled jobs after delete, got " .. tostring(#jobs))
                end

                return ALLOW
            end
        "#,
    )
    .unwrap();

    let mut engine =
        HarnessEngine::new(test_app_data_with_scheduler(dir.path().to_path_buf()).await).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_runtime_signals_list_and_subscribers_helpers() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            runtime.on("code.ready", function(_data, _meta) end)

            action.define("signals.inspect", function(ctx, params)
                return {
                    subscribers = runtime.signals.subscribers(params.topic),
                    signals = runtime.signals.list({
                        topic = params.topic,
                        target_agent = params.target_agent,
                    }),
                }
            end)
        "#,
    )
    .unwrap();

    let app_data = test_app_data_with_scheduler(dir.path().to_path_buf()).await;
    let runtime_store = app_data
        .scheduler
        .clone()
        .expect("scheduler")
        .runtime_store();
    runtime_store
        .replace_signal_subscriptions_for_agents(
            &["test-agent".to_string()],
            &[("test-agent".to_string(), "code.ready".to_string())],
        )
        .await
        .unwrap();
    runtime_store
        .insert_signal(crate::persistence::state::SignalInsert {
            public_id: uuid::Uuid::now_v7().into_bytes().to_vec(),
            topic: "code.ready".to_string(),
            source_agent_id: "publisher".to_string(),
            target_agent_id: "test-agent".to_string(),
            payload: serde_json::json!({ "branch": "feature-x" }).to_string(),
        })
        .await
        .unwrap();

    let mut engine = HarnessEngine::new(app_data).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let result = engine
        .invoke_declared_action_for_agent(
            "test-agent",
            "signals.inspect",
            serde_json::json!({
                "topic": "code.ready",
                "target_agent": "test-agent"
            }),
        )
        .unwrap()
        .expect("declared action result");

    assert_eq!(
        result.get("subscribers"),
        Some(&serde_json::json!(["test-agent"]))
    );

    let signals = result
        .get("signals")
        .and_then(|value| value.as_array())
        .expect("signals array");
    assert_eq!(signals.len(), 1);
    assert_eq!(
        signals[0].get("topic"),
        Some(&serde_json::json!("code.ready"))
    );
    assert_eq!(
        signals[0].get("source_agent_id"),
        Some(&serde_json::json!("publisher"))
    );
    assert_eq!(
        signals[0].get("target_agent_id"),
        Some(&serde_json::json!("test-agent"))
    );
    assert_eq!(
        signals[0].get("payload"),
        Some(&serde_json::json!({ "branch": "feature-x" }))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_runtime_signals_support_terminal_wildcard_subscriptions() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            runtime.on("deploy.complete", function(data, meta)
                session.set("signal_exact", meta.name .. ":" .. data.env)
            end)

            runtime.on("deploy.*", function(data, meta)
                session.set("signal_family", meta.name .. ":" .. data.env)
            end)

            runtime.on("*", function(data, meta)
                session.set("signal_global", meta.name .. ":" .. data.env)
            end)

            action.define("signals.read", function(_ctx, _params)
                return {
                    exact = session.get("signal_exact"),
                    family = session.get("signal_family"),
                    global = session.get("signal_global"),
                }
            end)
        "#,
    )
    .unwrap();

    let app_data = test_app_data_with_scheduler(dir.path().to_path_buf()).await;
    let runtime_store = app_data
        .scheduler
        .clone()
        .expect("scheduler")
        .runtime_store();
    runtime_store
        .replace_signal_subscriptions_for_agents(
            &["test-agent".to_string(), "observer".to_string()],
            &[
                ("test-agent".to_string(), "deploy.complete".to_string()),
                ("test-agent".to_string(), "deploy.*".to_string()),
                ("observer".to_string(), "deploy.*".to_string()),
                ("observer".to_string(), "*".to_string()),
            ],
        )
        .await
        .unwrap();
    let subscribers = runtime_store
        .list_signal_subscriber_agent_ids("deploy.complete")
        .await
        .unwrap();
    assert_eq!(subscribers, vec!["observer", "test-agent"]);

    runtime_store
        .insert_signal(crate::persistence::state::SignalInsert {
            public_id: uuid::Uuid::now_v7().into_bytes().to_vec(),
            topic: "deploy.complete".to_string(),
            source_agent_id: "publisher".to_string(),
            target_agent_id: "test-agent".to_string(),
            payload: serde_json::json!({ "env": "prod" }).to_string(),
        })
        .await
        .unwrap();
    let signals = runtime_store
        .list_signals_for_agent("test-agent", 10)
        .await
        .unwrap();
    assert_eq!(signals.len(), 1);

    let mut engine = HarnessEngine::new(app_data).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let invoked = engine.dispatch_runtime_signal(&signals[0]).unwrap();
    assert_eq!(invoked, 3);

    let result = engine
        .invoke_declared_action_for_agent("test-agent", "signals.read", serde_json::json!({}))
        .unwrap()
        .expect("declared action result");
    assert_eq!(
        result,
        serde_json::json!({
            "exact": "deploy.complete:prod",
            "family": "deploy.complete:prod",
            "global": "deploy.complete:prod",
        })
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_worklist_dx_helpers_support_prompt_and_action_items() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            function on_turn_prepare(ctx)
                local tasks = worklist("sprint", {
                    scope = "project:alpha"
                })

                local fix = tasks:add({
                    title = "Fix login redirect",
                    prompt = "Fix login redirect",
                    content = {
                        { type = "text", text = "Repro starts on /login" }
                    },
                    tools = {
                        allow = { "shell_exec" }
                    },
                    conflict_policy = "detached"
                }, {
                    metadata = { role = "dev" }
                })
                if fix.kind ~= "prompt" or fix.prompt ~= "Fix login redirect" then
                    error("expected prompt item to round-trip")
                end
                if fix.content == nil or fix.content[1].text ~= "Repro starts on /login" then
                    error("expected structured prompt content to round-trip")
                end
                if fix.tools == nil or fix.tools.allow[1] ~= "shell_exec" then
                    error("expected structured prompt tools to round-trip")
                end
                if fix.payload == nil or fix.payload.kind ~= "prompt" then
                    error("expected normalized prompt payload view")
                end
                fix = fix:update({
                    content = {
                        { type = "text", text = "Updated repro starts on /login" }
                    },
                    tools = {
                        allow = { "shell_exec", "read_file" }
                    },
                    conflict_policy = "queue",
                    metadata = { role = "dev", phase = "triage" }
                })
                if fix.content == nil or fix.content[1].text ~= "Updated repro starts on /login" then
                    error("expected updated structured prompt content to round-trip")
                end
                if fix.tools == nil or fix.tools.allow[2] ~= "read_file" then
                    error("expected updated structured prompt tools to round-trip")
                end
                if fix.conflict_policy ~= "queue" then
                    error("expected updated conflict policy")
                end

                local qa = tasks:add({
                    title = "Run checkout smoke test",
                    action = "qa.run_smoke",
                    params = { suite = "checkout" },
                    priority = 10,
                    metadata = { role = "qa" }
                })
                if qa.kind ~= "action" or qa.action_name ~= "qa.run_smoke" then
                    error("expected action item to round-trip")
                end
                if qa.params == nil or qa.params.suite ~= "checkout" then
                    error("expected action params to round-trip")
                end
                if qa.payload == nil or qa.payload.kind ~= "action" then
                    error("expected normalized action payload view")
                end
                if qa.payload.action == nil or qa.payload.action.name ~= "qa.run_smoke" then
                    error("expected normalized action payload name")
                end
                qa = qa:update({
                    params = { suite = "payments" },
                    metadata = { role = "qa", lane = "smoke" },
                    priority = 20
                })
                if qa.params == nil or qa.params.suite ~= "payments" then
                    error("expected updated action params to round-trip")
                end
                if qa.metadata == nil or qa.metadata.lane ~= "smoke" then
                    error("expected updated action metadata to round-trip")
                end
                if qa.priority ~= 20 then
                    error("expected updated priority")
                end

                local found = tasks:find({
                    where = { lane = "smoke" }
                })
                if found == nil or found.id ~= qa.id then
                    error("expected find(where=lane=smoke) to return action item")
                end

                local claimed = tasks:next({
                    where = { role = "qa" }
                })
                if claimed == nil or claimed.id ~= qa.id then
                    error("expected filtered next() to claim qa item")
                end

                local active = tasks:active()
                if active == nil or active.id ~= qa.id then
                    error("expected active() to return claimed qa item")
                end

                claimed:requeue()
                local reclaimed = tasks:next({
                    where = { role = "qa" }
                })
                if reclaimed == nil or reclaimed.id ~= qa.id then
                    error("expected requeued item to be claimable again")
                end
                reclaimed:done({ result = "ok" })

                local current = tasks:current()
                if current == nil or current.id ~= fix.id then
                    error("expected current() to claim remaining prompt item")
                end
                current:done()

                local progress = tasks:progress()
                if progress.done ~= 2 or progress.total ~= 2 then
                    error("expected progress to show 2/2 completion")
                end

                if tasks:empty() ~= true then
                    error("expected worklist to be empty after all items complete")
                end

                return ALLOW
            end
        "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(dir.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_reference_aware_action_round_trips_workitem_snapshot_and_ref_only() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            action.define("tasks.inspect", function(ctx, params)
                local item = params.item
                if item.title ~= "Overlay title" then
                    error("expected overlay title in action payload")
                end
                if item.metadata == nil or item.metadata.role ~= "dev" then
                    error("expected metadata overlay to round-trip")
                end
                item:done({ seen = item.title })
                return item
            end)

            action.define("tasks.echo", function(ctx, params)
                return params
            end)

            function on_turn_prepare(ctx)
                local tasks = worklist("tasks")
                local item = tasks:add({
                    title = "Original title",
                    prompt = "Do it",
                    metadata = { role = "dev" }
                })

                item.title = "Overlay title"
                local returned = action.run("tasks.inspect", { item = item })
                if returned.title ~= "Overlay title" then
                    error("expected returned work item to preserve overlay")
                end
                if type(returned.done) ~= "function" then
                    error("expected returned work item to remain a proxy")
                end

                local fresh = tasks:find({ where = { id = item.id } })
                if fresh.status ~= "done" then
                    error("expected action handler to operate on hydrated proxy")
                end

                local canonical = action.run("tasks.echo", ref(item))
                if canonical.title ~= "Original title" then
                    error("expected ref(item) to pass a lightweight canonical shell")
                end
                if canonical.status ~= "done" then
                    error("expected ref(item) to hydrate current stored state")
                end

                return ALLOW
            end
        "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(dir.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_contextual_action_runs_object_scoped_action() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            action.define("project.describe", function(ctx, params)
                local project = params.subject
                project:set("description", params.params.text)
                return project
            end)

            function on_turn_prepare(ctx)
                local project = scope("project", "turin")
                local returned = project:action("describe", { text = "Reference aware DX" })
                if returned:get("description") ~= "Reference aware DX" then
                    error("expected contextual action to use project.describe")
                end
                if type(returned.action) ~= "function" then
                    error("expected contextual action result to stay hydrated")
                end
                return ALLOW
            end
        "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(dir.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_action_define_on_attaches_methods_by_target_specificity() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            action.define_on("project", "mark", function(ctx, project, params)
                project:set("mark", params.value)
                return project
            end)

            action.define_on(target.workitem(), "label", function(ctx, item, params)
                return { source = "generic", title = item.title }
            end)

            action.define_on(target.workitem("tickets"), "label", function(ctx, item, params)
                return { source = "tickets", title = item.title }
            end)

            action.define_on(target.worklist(), "stats", function(ctx, list, params)
                return list:progress()
            end)

            function on_turn_prepare(ctx)
                local project = scope("project", "turin")
                project:mark({ value = "ok" })
                if project:get("mark") ~= "ok" then
                    error("expected scope method to mutate scoped state")
                end

                local tickets = worklist("tickets")
                local bugs = worklist("bugs")
                local ticket = tickets:add("Ticket one")
                local bug = bugs:add("Bug one")

                local ticket_label = ticket:label()
                if ticket_label.source ~= "tickets" or ticket_label.title ~= "Ticket one" then
                    error("expected specific work item method to override generic method")
                end

                local bug_label = bug:label()
                if bug_label.source ~= "generic" or bug_label.title ~= "Bug one" then
                    error("expected generic work item method for other lists")
                end

                local stats = tickets:stats()
                if stats.total ~= 1 or stats.done ~= 0 then
                    error("expected generic worklist method to attach")
                end

                return ALLOW
            end
        "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(dir.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_worklist_items_support_hierarchy_and_dependencies() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            function on_turn_prepare(ctx)
                local roadmap = worklist("roadmap")
                local epic = roadmap:add("Ship onboarding revamp")

                local spec = epic:add("Write spec")
                local implement = epic:add("Implement flow", {
                    after = { spec.id }
                })

                local children = epic:children()
                if #children ~= 2 then
                    error("expected epic children to round-trip")
                end

                local first = epic:next()
                if first == nil or first.id ~= spec.id then
                    error("expected dependency-free child to claim first")
                end
                first:done()

                local second = epic:next()
                if second == nil or second.id ~= implement.id then
                    error("expected dependent child after prerequisite completion")
                end
                second:done()

                local progress = epic:progress()
                if progress.done ~= 2 or progress.total ~= 2 then
                    error("expected child progress to show 2/2 completion")
                end

                return ALLOW
            end
        "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(dir.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_worklist_claim_heartbeat_and_stale_release() {
    let dir = TempDir::new().unwrap();
    let app_data = test_app_data_for_root(dir.path().to_path_buf());
    let store = open_default_state_store(&app_data).await;
    let list = store.open_worklist("ops", "", None).await.unwrap();
    let claimed = store
        .create_work_item(WorkItemInsert {
            public_id: uuid::Uuid::now_v7(),
            worklist_id: list.id,
            parent_item_id: None,
            title: "Stale health check",
            item_kind: "prompt",
            prompt: Some("Check stale runtime"),
            content: None,
            tools: None,
            conflict_policy: None,
            action_name: None,
            action_params: None,
            priority: 0,
            after_ids: None,
            metadata: None,
        })
        .await
        .unwrap();
    assert!(
        store
            .try_claim_work_item(
                claimed.id,
                "test-agent",
                Some("seed-session"),
                Some("seed-exec"),
                1,
            )
            .await
            .unwrap()
    );

    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            function on_turn_prepare(ctx)
                local tasks = worklist("ops")

                local orphaned = tasks:orphaned({
                    stale_after_seconds = 1
                })
                if #orphaned ~= 1 then
                    error("expected one stale claimed item")
                end

                local released = tasks:release_stale({
                    stale_after_seconds = 1
                })
                if #released ~= 1 or released[1].status ~= "pending" then
                    error("expected stale claim to be released back to pending")
                end

                local fresh = tasks:next()
                if fresh == nil or fresh.title ~= "Stale health check" then
                    error("expected released item to be claimable again")
                end

                local heartbeated = fresh:heartbeat()
                if heartbeated.claim_execution_id == nil then
                    error("expected heartbeat to preserve active claim identity")
                end

                return ALLOW
            end
        "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(app_data).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_worklist_dispatches_prompt_and_action_payloads() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            action.define("qa.run_smoke", function(ctx, params)
                return {
                    status = "queued " .. tostring(params.suite)
                }
            end)

            function on_turn_prepare(ctx)
                local tasks = worklist("dispatch")

                tasks:add({
                    title = "Review latest failures",
                    prompt = "Review latest failures",
                    metadata = { role = "dev" }
                })
                tasks:add({
                    title = "Run checkout smoke test",
                    action = "qa.run_smoke",
                    params = { suite = "checkout" },
                    priority = 10,
                    metadata = { role = "qa" }
                })

                local action_run = tasks:dispatch_next({
                    where = { role = "qa" }
                })
                if action_run == nil or action_run.result.dispatched ~= "action" then
                    error("expected action dispatch result")
                end
                if action_run.result.result.status ~= "queued checkout" then
                    error("expected action handler result to round-trip")
                end

                local prompt_run = tasks:dispatch_next({
                    where = { role = "dev" }
                })
                if prompt_run == nil or prompt_run.result.dispatched ~= "task" then
                    error("expected prompt dispatch result")
                end
                if not prompt_run.result.task_id then
                    error("expected prompt dispatch task id")
                end

                return ALLOW
            end
        "#,
    )
    .unwrap();

    let app_data = test_app_data_for_root(dir.path().to_path_buf());
    let mut engine = HarnessEngine::new(app_data.clone()).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);

    let queue = app_data
        .execution_ctx
        .lock()
        .unwrap()
        .queue
        .clone()
        .expect("active session queue");
    let queued = queue.lock().await;
    assert_eq!(queued.len(), 1);
    assert_eq!(queued[0].prompt, "Review latest failures");
    assert_eq!(queued[0].title.as_deref(), Some("Review latest failures"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_worklist_action_pause_updates_checkpoint_and_schedules_resume() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            action.define("contacts.sync", function(ctx, params)
                return ctx:pause({
                    because = "rate_limited",
                    checkpoint = {
                        cursor = params.cursor or "page-1",
                        processed = 12,
                    },
                    resume_in_seconds = 45,
                    note = "Paused due to API rate limit",
                })
            end)

            function on_turn_prepare(_ctx)
                local tasks = worklist("dispatch")
                tasks:add({
                    title = "Sync contacts",
                    action = "contacts.sync",
                    params = { cursor = "page-1" },
                    metadata = { role = "sync" }
                })

                local run = tasks:dispatch_next({
                    where = { role = "sync" }
                })
                if run == nil or run.result.dispatched ~= "action" then
                    error("expected action dispatch result")
                end
                if run.result.result.status ~= "paused" then
                    error("expected paused action result")
                end
                if run.result.result.reason ~= "rate_limited" then
                    error("expected pause reason")
                end
                if run.result.result.resume_in_seconds ~= 45 then
                    error("expected resume delay")
                end
                if tasks:dispatch_next({
                    where = { role = "sync" }
                }) ~= nil then
                    error("paused item should not be redispatched before resume is due")
                end
                return ALLOW
            end
        "#,
    )
    .unwrap();

    let app_data = test_app_data_with_scheduler(dir.path().to_path_buf()).await;
    let default_store = open_default_state_store(&app_data).await;
    let mut engine = HarnessEngine::new(app_data.clone()).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);

    let worklist = default_store
        .open_worklist("dispatch", "", None)
        .await
        .unwrap();
    let rows = default_store.list_work_items(worklist.id).await.unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].status, "paused");
    let metadata: serde_json::Value =
        serde_json::from_str(rows[0].metadata.as_deref().expect("metadata")).unwrap();
    assert_eq!(metadata["pause_reason"], "rate_limited");
    assert_eq!(metadata["checkpoint"]["cursor"], "page-1");
    assert_eq!(metadata["checkpoint"]["processed"], 12);

    let jobs = app_data
        .scheduler
        .as_ref()
        .expect("scheduler")
        .list_jobs()
        .await
        .unwrap();
    assert_eq!(jobs.len(), 1);
    assert_eq!(
        jobs[0].action.as_ref().map(|action| action.name.as_str()),
        Some("worklist.dispatch_next")
    );
    assert_eq!(
        jobs[0]
            .action
            .as_ref()
            .and_then(|action| action.params.as_ref())
            .and_then(|params| params.get("where"))
            .and_then(|where_map| where_map.get("id")),
        Some(&serde_json::Value::String(
            uuid::Uuid::from_slice(&rows[0].public_id)
                .unwrap()
                .to_string()
        ))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_worklist_dispatch_can_resume_due_paused_item() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            action.define("contacts.resume", function(ctx, params)
                return ctx:complete({
                    cursor = ctx.checkpoint:get("cursor", "start"),
                })
            end)
        "#,
    )
    .unwrap();

    let app_data = test_app_data_for_root(dir.path().to_path_buf());
    let store = open_default_state_store(&app_data).await;
    let list = store.open_worklist("dispatch", "", None).await.unwrap();
    let paused_metadata = serde_json::json!({
        "paused": true,
        "pause_reason": "rate_limited",
        "pause_until_unix_ms": 1,
        "checkpoint": {
            "cursor": "page-2"
        }
    });
    store
        .create_work_item(WorkItemInsert {
            public_id: uuid::Uuid::now_v7(),
            worklist_id: list.id,
            parent_item_id: None,
            title: "Resume due paused action",
            item_kind: "action",
            prompt: None,
            content: None,
            tools: None,
            conflict_policy: None,
            action_name: Some("contacts.resume"),
            action_params: Some("{}"),
            priority: 0,
            after_ids: None,
            metadata: Some(&paused_metadata.to_string()),
        })
        .await
        .unwrap();
    let rows = store.list_work_items(list.id).await.unwrap();
    let paused_row = rows
        .into_iter()
        .find(|row| row.title == "Resume due paused action")
        .unwrap();
    store
        .update_work_item(WorkItemUpdate {
            id: paused_row.id,
            status: Some("paused"),
            ..Default::default()
        })
        .await
        .unwrap();

    let mut engine = HarnessEngine::new(app_data).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let result = engine
        .invoke_declared_action_for_agent(
            "test-agent",
            "worklist.dispatch_next",
            serde_json::json!({ "name": "dispatch" }),
        )
        .unwrap()
        .unwrap();

    assert_eq!(result["item"]["title"], "Resume due paused action");
    assert_eq!(result["result"]["dispatched"], "action");
    assert_eq!(result["result"]["result"]["result"]["cursor"], "page-2");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_worklist_paused_query_helpers_surface_due_and_not_due_items() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            function on_turn_prepare(_ctx)
                local tasks = worklist("dispatch")

                local later = tasks:add({
                    title = "Paused later",
                    prompt = "later",
                    metadata = {
                        pause_reason = "rate_limited",
                        pause_until_unix_ms = 4102444800000,
                        checkpoint = { cursor = "page-8" }
                    }
                })
                later:update({ status = "paused" })

                local due_item = tasks:add({
                    title = "Paused and due",
                    prompt = "due",
                    metadata = {
                        pause_reason = "awaiting_reauth",
                        pause_until_unix_ms = 1,
                        checkpoint = { cursor = "page-9" }
                    }
                })
                due_item:update({ status = "paused" })

                local paused = tasks:paused()
                if #paused ~= 2 then
                    error("expected paused() to return both paused items")
                end

                if paused[1].paused ~= true then
                    error("expected paused field on item")
                end

                local due = tasks:paused({ due_only = true })
                if #due ~= 1 then
                    error("expected one due paused item")
                end

                if due[1].title ~= "Paused and due" then
                    error("expected due paused item")
                end

                local by_reason = tasks:paused({
                    where = { pause_reason = "awaiting_reauth" }
                })
                if #by_reason ~= 1 or by_reason[1].title ~= "Paused and due" then
                    error("expected pause_reason filter to match due item")
                end

                local by_flag = tasks:find({
                    where = { paused = true, pause_reason = "rate_limited" }
                })
                if by_flag == nil or by_flag.title ~= "Paused later" then
                    error("expected find(where=paused) to match paused item")
                end

                return ALLOW
            end
        "#,
    )
    .unwrap();

    let app_data = test_app_data_for_root(dir.path().to_path_buf());
    let mut engine = HarnessEngine::new(app_data.clone()).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_worklist_action_checkpoint_helpers_expose_saved_state() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            action.define("contacts.resume", function(ctx, params)
                return ctx:complete({
                    cursor = ctx.checkpoint:get("cursor", "start"),
                    processed = ctx.checkpoint:get("processed", 0),
                    raw = ctx.checkpoint:all(),
                })
            end)

            function on_turn_prepare(_ctx)
                local tasks = worklist("dispatch")
                tasks:add({
                    title = "Resume contacts sync",
                    action = "contacts.resume",
                    params = {},
                    metadata = {
                        checkpoint = {
                            cursor = "page-7",
                            processed = 42,
                        }
                    }
                })

                local run = tasks:dispatch_next()
                if run == nil or run.result.dispatched ~= "action" then
                    error("expected action dispatch result")
                end
                if run.result.result.result.cursor ~= "page-7" then
                    error("expected checkpoint cursor")
                end
                if run.result.result.result.processed ~= 42 then
                    error("expected checkpoint processed count")
                end
                return ALLOW
            end
        "#,
    )
    .unwrap();

    let app_data = test_app_data_for_root(dir.path().to_path_buf());
    let mut engine = HarnessEngine::new(app_data.clone()).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_worklist_action_pause_for_sets_resume_delay() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            action.define("contacts.sync", function(ctx, params)
                return ctx:pause_for(90, {
                    because = "rate_limited",
                    checkpoint = {
                        cursor = "page-9",
                    }
                })
            end)

            function on_turn_prepare(_ctx)
                local tasks = worklist("dispatch")
                tasks:add({
                    title = "Pause with shorthand",
                    action = "contacts.sync",
                    params = {}
                })

                local run = tasks:dispatch_next()
                if run == nil or run.result.dispatched ~= "action" then
                    error("expected action dispatch result")
                end
                if run.result.result.resume_in_seconds ~= 90 then
                    error("expected shorthand resume delay")
                end
                return ALLOW
            end
        "#,
    )
    .unwrap();

    let app_data = test_app_data_with_scheduler(dir.path().to_path_buf()).await;
    let mut engine = HarnessEngine::new(app_data.clone()).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_engine_invokes_builtin_worklist_dispatch_action() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("main.lua"), "-- no custom actions needed\n").unwrap();

    let app_data = test_app_data_for_root(dir.path().to_path_buf());
    let store = open_default_state_store(&app_data).await;
    let list = store.open_worklist("dispatch", "", None).await.unwrap();
    store
        .create_work_item(WorkItemInsert {
            public_id: uuid::Uuid::now_v7(),
            worklist_id: list.id,
            parent_item_id: None,
            title: "Review latest failures",
            item_kind: "prompt",
            prompt: Some("Review latest failures"),
            content: None,
            tools: None,
            conflict_policy: None,
            action_name: None,
            action_params: None,
            priority: 0,
            after_ids: None,
            metadata: None,
        })
        .await
        .unwrap();

    let mut engine = HarnessEngine::new(app_data.clone()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let result = engine
        .invoke_declared_action_for_agent(
            "test-agent",
            "worklist.dispatch_next",
            serde_json::json!({ "name": "dispatch" }),
        )
        .unwrap()
        .unwrap();

    assert_eq!(result["item"]["title"], "Review latest failures");
    assert_eq!(result["result"]["dispatched"], "task");
    assert!(result["result"]["task_id"].is_string());

    let queue = app_data
        .execution_ctx
        .lock()
        .unwrap()
        .queue
        .clone()
        .expect("active session queue");
    let queued = queue.lock().await;
    assert_eq!(queued.len(), 1);
    assert_eq!(queued[0].prompt, "Review latest failures");
}

#[test]
fn test_engine_composition_reject_wins() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("a_permissive.lua"),
        r#"
            function on_tool_call(call)
                return ALLOW
            end
            "#,
    )
    .unwrap();

    std::fs::write(
        dir.path().join("b_safety.lua"),
        r#"
            function on_tool_call(call)
                if call.name == "shell_exec" then
                    return REJECT, "Blocked by safety harness"
                end
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    assert_eq!(engine.loaded_scripts(), vec!["a_permissive", "b_safety"]);

    let verdict = engine
        .evaluate(
            "on_tool_call",
            serde_json::json!({"name": "shell_exec", "args": {}}),
        )
        .unwrap();
    assert_eq!(
        verdict,
        Verdict::Reject("Blocked by safety harness".to_string())
    );
}

#[test]
fn test_engine_collects_declared_virtual_tools() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            tool.declare("play_song", {
                description = "Play an audio file",
                params = {
                    filename = { type = "string", required = true }
                },
                handler = function(args)
                    return tool.call("shell_exec", {
                        command = "mpg123 " .. shell.quote(args.filename)
                    })
                end
            })
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let tools = engine.declared_virtual_tools().unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "play_song");
    assert_eq!(tools[0].description, "Play an audio file");
    assert_eq!(
        tools[0].input_schema["properties"]["filename"]["type"],
        "string"
    );
}

#[test]
fn test_engine_invokes_virtual_tool_handler_sequence() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            tool.declare("play_playlist", {
                description = "Play multiple songs",
                params = {
                    first = { type = "string", required = true },
                    second = { type = "string", required = true }
                },
                handler = function(args)
                    return tool.sequence({
                        tool.call("shell_exec", {
                            command = "mpg123 " .. shell.quote(args.first)
                        }),
                        tool.call("shell_exec", {
                            command = "mpg123 " .. shell.quote(args.second)
                        })
                    })
                end
            })
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let plan = engine
        .invoke_virtual_tool(
            "play_playlist",
            serde_json::json!({ "first": "one.mp3", "second": "two.mp3" }),
        )
        .unwrap()
        .unwrap();

    assert_eq!(plan.calls.len(), 2);
    assert_eq!(plan.calls[0].name, "shell_exec");
    assert_eq!(plan.calls[0].args["command"], "mpg123 'one.mp3'");
    assert_eq!(plan.calls[1].args["command"], "mpg123 'two.mp3'");
}

#[test]
fn test_engine_invokes_virtual_tool_result_handler() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            tool.declare("read_note_wrapped", {
                description = "Read and wrap a note",
                params = {
                    path = { type = "string", required = true }
                },
                handler = function(args)
                    return tool.call("read_file", { path = args.path }, function(result)
                        return {
                            content = "wrapped: " .. result.content,
                            is_error = result.is_error
                        }
                    end)
                end
            })
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let plan = engine
        .invoke_virtual_tool(
            "read_note_wrapped",
            serde_json::json!({ "path": "note.txt" }),
        )
        .unwrap()
        .unwrap();

    let handler_key = plan
        .result_handler_key
        .clone()
        .expect("expected result handler key");

    let output = engine
        .invoke_virtual_tool_result_handler(
            &handler_key,
            serde_json::json!({
                "id": "tc_1",
                "name": "read_file",
                "args": { "path": "note.txt" },
                "verdict": "allow",
                "duration_ms": 2,
                "content": "hello",
                "is_error": false
            }),
            false,
        )
        .unwrap();

    assert_eq!(
        output,
        VirtualToolResultResolution::Output(
            crate::harness::virtual_tools::VirtualToolResultOutput {
                content: "wrapped: hello".to_string(),
                is_error: false,
            }
        )
    );
}

#[test]
fn test_engine_invokes_declared_action_handler() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            action.define("qa.run_smoke", function(ctx, params)
                return {
                    status = "queued " .. tostring(params.suite)
                }
            end)
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let result = engine
        .invoke_declared_action_for_agent(
            "test-agent",
            "qa.run_smoke",
            serde_json::json!({ "suite": "checkout" }),
        )
        .unwrap()
        .unwrap();

    assert_eq!(result["status"], "queued checkout");
}

#[test]
fn test_engine_custom_events_dispatch_in_order_and_return_count() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            local seen = {}

            on("qa.failed", function(payload, meta)
                table.insert(seen, meta.name .. ":" .. payload.suite)
            end)

            on("qa.failed", function(payload, _meta)
                table.insert(seen, "second:" .. payload.suite)
            end)

            function on_turn_prepare(_ctx)
                local handled = emit("qa.failed", { suite = "checkout" })
                if handled ~= 2 then
                    error("expected two listeners")
                end

                if #seen ~= 2 then
                    error("expected two recorded listeners")
                end

                if seen[1] ~= "qa.failed:checkout" or seen[2] ~= "second:checkout" then
                    error("expected listeners to run in registration order")
                end

                local missing = emit("qa.passed", { suite = "checkout" })
                if missing ~= 0 then
                    error("expected emit without listeners to return 0")
                end

                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(dir.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[test]
fn test_engine_custom_events_support_terminal_wildcard_listeners() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            local seen = {}

            on("deploy.complete", function(payload, meta)
                table.insert(seen, "exact:" .. meta.name .. ":" .. payload.env)
            end)

            on("deploy.*", function(payload, meta)
                table.insert(seen, "family:" .. meta.name .. ":" .. payload.env)
            end)

            on("*", function(payload, meta)
                table.insert(seen, "all:" .. meta.name .. ":" .. payload.env)
            end)

            function on_turn_prepare(_ctx)
                local handled = emit("deploy.complete", { env = "prod" })
                if handled ~= 3 then
                    error("expected exact, family, and global listeners")
                end
                if seen[1] ~= "exact:deploy.complete:prod" then
                    error("expected exact listener first")
                end
                if seen[2] ~= "family:deploy.complete:prod" then
                    error("expected family wildcard listener second")
                end
                if seen[3] ~= "all:deploy.complete:prod" then
                    error("expected global wildcard listener third")
                end
                return ALLOW
            end
        "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(dir.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[test]
fn test_engine_rejects_non_terminal_event_wildcards() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            on("deploy.*.complete", function() end)
        "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(dir.path().to_path_buf())).unwrap();
    let err = engine
        .load_dir(dir.path())
        .expect_err("non-terminal wildcard should be rejected");
    assert!(err.to_string().contains("terminal prefix patterns"));
}

#[test]
fn test_engine_event_listener_can_run_declared_action() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            local observed = nil

            action.define("bugs.create", function(ctx, params)
                return {
                    id = "bug-" .. tostring(params.code)
                }
            end)

            on("qa.failed", function(payload)
                local result = action.run("bugs.create", payload)
                observed = result.id
            end)

            function on_turn_prepare(_ctx)
                emit("qa.failed", { code = "123" })
                if observed ~= "bug-123" then
                    error("expected event listener to call declared action")
                end
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(dir.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_engine_registered_callbacks_preserve_imported_module_subject_context() {
    let dir = TempDir::new().unwrap();
    std::fs::create_dir_all(dir.path().join("plugins")).unwrap();
    std::fs::write(
        dir.path().join("plugins").join("callbacks.lua"),
        r#"
            return {
                register = function()
                    action.define("subject.report", function(ctx, params)
                        return runtime.governance.check("runtime.db.query")
                    end)

                    on("qa.failed", function(payload)
                        local decision = runtime.governance.check("runtime.db.query")
                        session.set("event_subject_module", decision.subject_module_name)
                        session.set("event_subject_root", decision.subject_root_name)
                    end)
                end
            }
            "#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            local callbacks = import("plugins/callbacks")
            callbacks.register()

            function on_turn_prepare(_ctx)
                local action_subject = action.run("subject.report", {})
                if action_subject.subject_module_name ~= "plugins/callbacks" then
                    error("expected action subject module from imported callback")
                end
                if action_subject.subject_root_name ~= "plugins" then
                    error("expected action subject root from imported callback")
                end

                emit("qa.failed", {})
                local event_subject_module = session.get("event_subject_module")
                local event_subject_root = session.get("event_subject_root")
                if event_subject_module == nil or event_subject_root == nil then
                    error("missing event subject")
                end
                if event_subject_module ~= "plugins/callbacks" then
                    error("expected event subject module from imported callback")
                end
                if event_subject_root ~= "plugins" then
                    error("expected event subject root from imported callback")
                end

                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut app_data = test_app_data_for_root(dir.path().to_path_buf());
    let mut config = crate::kernel::config::TurinConfig::default();
    config.governance.roots.insert(
        "plugins".to_string(),
        crate::kernel::config::GovernanceRootConfig {
            path: "plugins".to_string(),
            writable_hint: false,
            default_profile: None,
            max_capabilities: Default::default(),
        },
    );
    app_data.config = Arc::new(config.clone());
    app_data.governance_manager = Arc::new(crate::kernel::governance::GovernanceManager::new(
        config.governance.clone(),
    ));

    let mut engine = HarnessEngine::new(app_data).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[test]
fn test_engine_emit_propagates_listener_failure_and_stops_dispatch() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            local seen = {}

            on("qa.failed", function(_payload)
                table.insert(seen, "first")
                error("listener sentinel")
            end)

            on("qa.failed", function(_payload)
                table.insert(seen, "second")
            end)

            function on_turn_prepare(_ctx)
                local ok, err = pcall(function()
                    emit("qa.failed", {})
                end)

                if ok then
                    error("emit should surface listener failure")
                end
                if not tostring(err):find("listener sentinel", 1, true) then
                    error("unexpected emit error: " .. tostring(err))
                end
                if #seen ~= 1 or seen[1] ~= "first" then
                    error("emit should stop dispatch after listener failure")
                end

                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(dir.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[test]
fn test_engine_on_registration_is_load_time_only() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            function on_turn_prepare(_ctx)
                on("qa.failed", function() end)
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(dir.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let err = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("load-time")
            || message.contains("only during harness load")
            || message.contains("during harness load"),
        "unexpected error: {message}"
    );
}

#[test]
fn test_engine_action_context_reports_cancellation() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            action.define("ops.cancel_check", function(ctx, params)
                return {
                    cancelled = ctx:is_cancelled()
                }
            end)
            "#,
    )
    .unwrap();

    let app_data = test_app_data();
    let cancel_token = app_data
        .execution_ctx
        .lock()
        .unwrap()
        .cancel_token
        .clone()
        .expect("cancel token");
    cancel_token.cancel();

    let mut engine = HarnessEngine::new(app_data).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let result = engine
        .invoke_declared_action_for_agent("test-agent", "ops.cancel_check", serde_json::json!({}))
        .unwrap()
        .unwrap();

    assert_eq!(result["cancelled"], true);
}

#[test]
fn test_engine_imports_nested_module_from_subdirectory() {
    let dir = TempDir::new().unwrap();
    std::fs::create_dir_all(dir.path().join("modules")).unwrap();
    std::fs::write(
        dir.path().join("modules").join("helper.lua"),
        r#"
            return {
                ping = function()
                    return "pong"
                end
            }
            "#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            function on_turn_prepare(ctx)
                local helper = import("modules/helper")
                if helper.ping() ~= "pong" then
                    error("nested import returned wrong value")
                end
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(dir.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[test]
fn test_engine_use_activates_script_and_table_blocks() {
    let dir = TempDir::new().unwrap();
    std::fs::create_dir_all(dir.path().join("blocks")).unwrap();
    std::fs::write(
        dir.path().join("blocks").join("script_style.lua"),
        r#"
            function on_turn_prepare(ctx)
                return ESCALATE, "script-style block"
            end
            "#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("blocks").join("table_style.lua"),
        r#"
            return {
                on_turn_prepare = function(ctx)
                    return REJECT, "table-style block"
                end
            }
            "#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            use("blocks/script_style")
            use("blocks/table_style", {
                when = function(hook, payload)
                    return hook == "on_turn_prepare"
                end
            })
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(dir.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();

    assert_eq!(
        engine.loaded_scripts(),
        vec![
            "blocks/script_style#use1".to_string(),
            "blocks/table_style#use1".to_string(),
            "main".to_string()
        ]
    );

    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Reject("table-style block".to_string()));
}

#[test]
fn test_engine_use_rejected_outside_load_phase() {
    let dir = TempDir::new().unwrap();
    std::fs::create_dir_all(dir.path().join("blocks")).unwrap();
    std::fs::write(
        dir.path().join("blocks").join("late.lua"),
        r#"
            function on_turn_prepare(ctx)
                return ALLOW
            end
            "#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            function on_turn_prepare(ctx)
                use("blocks/late")
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(dir.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let err = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("use(...) can only be called during harness load")
    );
}

#[test]
fn test_engine_watch_registers_explicit_roots() {
    let dir = TempDir::new().unwrap();
    std::fs::create_dir_all(dir.path().join("blocks")).unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            watch("blocks")
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(dir.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();

    assert_eq!(
        engine.explicit_watch_roots(),
        vec![dir.path().join("blocks")]
    );
}

#[test]
fn test_engine_rm_rf_blocked() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("safety.lua"),
        r#"
            function on_tool_call(call)
                if call.name == "shell_exec" then
                    local cmd = call.args.command
                    if cmd and cmd:find("rm %-rf") then
                        return REJECT, "Destructive command 'rm -rf' is not allowed"
                    end
                end
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate(
            "on_tool_call",
            serde_json::json!({"name": "shell_exec", "args": {"command": "rm -rf /"}}),
        )
        .unwrap();
    assert_eq!(
        verdict,
        Verdict::Reject("Destructive command 'rm -rf' is not allowed".to_string())
    );

    let verdict = engine
        .evaluate(
            "on_tool_call",
            serde_json::json!({"name": "shell_exec", "args": {"command": "ls -la"}}),
        )
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[test]
fn test_engine_undefined_hook_returns_allow() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("partial.lua"),
        r#"
            function on_tool_call(call)
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate("on_token_usage", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[test]
fn test_engine_token_usage_hook() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("budget.lua"),
        r#"
            function on_token_usage(usage)
                if usage.total_cost_usd and usage.total_cost_usd > 1.0 then
                    return REJECT, "Budget exceeded: $" .. tostring(usage.total_cost_usd)
                end
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate(
            "on_token_usage",
            serde_json::json!({"total_cost_usd": 0.5, "input_tokens": 100, "output_tokens": 50}),
        )
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);

    let verdict = engine
        .evaluate(
            "on_token_usage",
            serde_json::json!({"total_cost_usd": 1.5, "input_tokens": 100, "output_tokens": 50}),
        )
        .unwrap();
    assert!(verdict.is_rejected());
}

#[test]
fn test_engine_modify_verdict() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("modify.lua"),
        r#"
            function on_plan_submit(payload)
                return MODIFY, { "Modified Task 1", "Modified Task 2" }
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate(
            "on_plan_submit",
            serde_json::json!({"action": "submit_plan"}),
        )
        .unwrap();

    match verdict {
        Verdict::Modify(val) => {
            let arr = val.as_array().unwrap();
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0].as_str().unwrap(), "Modified Task 1");
            assert_eq!(arr[1].as_str().unwrap(), "Modified Task 2");
        }
        _ => panic!("Expected Modify verdict, got {:?}", verdict),
    }
}

#[derive(Clone)]
struct MockContext;
impl mlua::UserData for MockContext {}

#[test]
fn test_on_turn_prepare_reject() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("reject.lua"),
        r#"
            function on_turn_prepare(ctx)
                return REJECT, "Blocked by harness"
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_rejected());
    assert_eq!(verdict.reason(), Some("Blocked by harness"));
}

#[test]
fn test_verdict_helpers_support_or_chains() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("verdict_dx.lua"),
        r#"
            function on_tool_call(call)
                return verdict.reject_if(call.name == "shell_exec", "blocked by dx")
                    or verdict.allow()
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate(
            "on_tool_call",
            serde_json::json!({"name": "shell_exec", "args": {}}),
        )
        .unwrap();
    assert!(verdict.is_rejected());
    assert_eq!(verdict.reason(), Some("blocked by dx"));
}

#[test]
fn test_verdict_modify_helper() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("verdict_modify_dx.lua"),
        r#"
            function on_plan_submit(payload)
                return verdict.modify({ "A", "B" })
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate(
            "on_plan_submit",
            serde_json::json!({ "action": "submit_plan" }),
        )
        .unwrap();

    match verdict {
        Verdict::Modify(val) => {
            let arr = val.as_array().unwrap();
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0].as_str().unwrap(), "A");
            assert_eq!(arr[1].as_str().unwrap(), "B");
        }
        other => panic!("Expected Modify verdict, got {:?}", other),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_access_helpers() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("access_dx.lua"),
        r#"
            function on_turn_prepare(ctx)
                if not allowed("db.exec") then
                    return REJECT, "expected allowed in default config"
                end
                local decision = access.check("db.exec")
                if type(decision) ~= "table" then
                    return REJECT, "access.check did not return table"
                end
                needs("db.exec")
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_session_user_kv_helpers() {
    let root = TempDir::new().unwrap();
    let mut engine = HarnessEngine::new(test_app_data_for_root(root.path().to_path_buf())).unwrap();

    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("data_dx.lua"),
        r#"
            function on_turn_prepare(ctx)
                session.set("counter", "0")
                local a = session.incr("counter")
                local b = session.incr("counter", 2)
                if a ~= 1 or b ~= 3 then
                    return REJECT, "session.incr mismatch"
                end

                if session.get("counter") ~= "3" then
                    return REJECT, "session.get mismatch"
                end

                user.set("tz", "UTC")
                if user.get("tz") ~= "UTC" then
                    return REJECT, "user.set/user.get mismatch"
                end
                user.del("tz")
                if user.get("tz") ~= nil then
                    return REJECT, "user.del mismatch"
                end
                return ALLOW
            end
            "#,
    )
    .unwrap();

    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_scope_helper_supports_custom_scopes() {
    let root = TempDir::new().unwrap();
    let mut engine = HarnessEngine::new(test_app_data_for_root(root.path().to_path_buf())).unwrap();

    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("scope_dx.lua"),
        r#"
            function on_turn_prepare(ctx)
                local project = scope("project", "my-app", {
                    namespace = "notes",
                    visibility = "shared",
                })

                project.set("counter", "0")
                local a = project.incr("counter")
                local b = project.incr("counter", 4)
                if a ~= 1 or b ~= 5 then
                    return REJECT, "scope.incr mismatch"
                end
                if project.get("counter") ~= "5" then
                    return REJECT, "scope.get mismatch"
                end

                project.remember("My app uses event sourcing.", { topic = "architecture" })
                local hits = project.recall("event sourcing")
                if hits == nil or #hits < 1 then
                    return REJECT, "scope.recall returned no hits"
                end
                if hits[1].metadata == nil or hits[1].metadata.topic ~= "architecture" then
                    return REJECT, "scope.recall metadata mismatch"
                end

                project.del("counter")
                if project.get("counter") ~= nil then
                    return REJECT, "scope.del mismatch"
                end

                local global_scope = scope("global")
                global_scope.set("banner", "shared")
                if global_scope.get("banner") ~= "shared" then
                    return REJECT, "scope(global) mismatch"
                end

                return ALLOW
            end
        "#,
    )
    .unwrap();

    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_runtime_db_proxy_one_and_with_error_precedence() {
    let root = TempDir::new().unwrap();
    let mut engine = HarnessEngine::new(test_app_data_for_root(root.path().to_path_buf())).unwrap();

    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("db_dx.lua"),
        r#"
            function on_turn_prepare(ctx)
                runtime.db.with("state", function(db)
                    db:exec("CREATE TABLE IF NOT EXISTS dx_users(id INTEGER PRIMARY KEY, name TEXT)")
                    db:exec("DELETE FROM dx_users")
                    db:exec("INSERT INTO dx_users(name) VALUES (?)", {"alice"})

                    local missing = db:one("SELECT name FROM dx_users WHERE id = ?", { 999 })
                    if missing ~= nil then
                        error("runtime.db:one should return nil when no rows")
                    end

                    local first = db:one("SELECT name FROM dx_users ORDER BY id LIMIT 1")
                    if first == nil or first.name ~= "alice" then
                        error("runtime.db:one returned wrong row")
                    end
                end)

                local ok, err = pcall(function()
                    runtime.db.with("state", function(db)
                        db:close()
                        error("callback error sentinel")
                    end)
                end)
                if ok then
                    return REJECT, "runtime.db.with should have failed"
                end
                if not tostring(err):find("callback error sentinel", 1, true) then
                    return REJECT, "runtime.db.with should prioritize callback error"
                end

                return ALLOW
            end
            "#,
    )
    .unwrap();

    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_try_helper_captures_runtime_errors() {
    let root = TempDir::new().unwrap();
    let mut engine = HarnessEngine::new(test_app_data_for_root(root.path().to_path_buf())).unwrap();

    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("try_dx.lua"),
        r#"
            function on_turn_prepare(ctx)
                local ok, err = try(function()
                    error("boom")
                end)
                if ok ~= nil then
                    return REJECT, "try should return nil on raised error"
                end
                if err == nil or not tostring(err):find("boom", 1, true) then
                    return REJECT, "try should expose the raised error"
                end

                local sum = try(function(a, b)
                    return a + b
                end, 2, 3)
                if sum ~= 5 then
                    return REJECT, "try should preserve successful return values"
                end

                return ALLOW
            end
        "#,
    )
    .unwrap();

    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_runtime_agent_status_proxy_and_fs_json_helpers() {
    let root = TempDir::new().unwrap();
    let mut engine = HarnessEngine::new(test_app_data_for_root(root.path().to_path_buf())).unwrap();

    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("agent_fs_dx.lua"),
        r#"
            function on_turn_prepare(ctx)
                local status = runtime.agent("default"):status()
                if status == nil or status.agent_id ~= "default" then
                    return REJECT, "runtime.agent(...):status() mismatch"
                end

                fs.write_json("dx-config.json", { enabled = true, count = 3 }, { pretty = true })
                local cfg = fs.read_json("dx-config.json")
                if cfg.enabled ~= true or cfg.count ~= 3 then
                    return REJECT, "fs.read_json/fs.write_json mismatch"
                end

                return ALLOW
            end
            "#,
    )
    .unwrap();

    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_hash_sha256_and_fs_stat_track_changes_per_session() {
    let root = TempDir::new().unwrap();
    std::fs::write(root.path().join("SPEC.md"), "alpha").unwrap();

    let script = r#"
        function on_turn_prepare(ctx)
            local run = session.incr("run")
            local stat = fs.stat("SPEC.md")

            if hash.sha256("hello") ~= "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824" then
                return REJECT, "hash.sha256 mismatch"
            end

            if stat.path ~= "SPEC.md" then
                return REJECT, "fs.stat path mismatch"
            end

            if run == 1 then
                if stat.bytes ~= 5 or stat.hash ~= hash.sha256("alpha") then
                    return REJECT, "run1 stat facts mismatch"
                end
                if stat.seen_before ~= false or stat.changed ~= true or stat.previous_hash ~= nil then
                    return REJECT, "run1 tracking mismatch"
                end
            elseif run == 2 then
                if stat.hash ~= hash.sha256("alpha") then
                    return REJECT, "run2 hash mismatch"
                end
                if stat.seen_before ~= true or stat.changed ~= false or stat.previous_hash ~= hash.sha256("alpha") then
                    return REJECT, "run2 tracking mismatch"
                end
            elseif run == 3 then
                if stat.bytes ~= 4 or stat.hash ~= hash.sha256("beta") then
                    return REJECT, "run3 stat facts mismatch"
                end
                if stat.seen_before ~= true or stat.changed ~= true or stat.previous_hash ~= hash.sha256("alpha") then
                    return REJECT, "run3 tracking mismatch"
                end
            else
                return REJECT, "unexpected run"
            end

            return ALLOW
        end
    "#;

    let harness_dir = TempDir::new().unwrap();
    std::fs::write(harness_dir.path().join("stat.lua"), script).unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(root.path().to_path_buf())).unwrap();
    engine.load_dir(harness_dir.path()).unwrap();

    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());

    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());

    std::fs::write(root.path().join("SPEC.md"), "beta").unwrap();

    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());

    let other_session_script = r#"
        function on_turn_prepare(ctx)
            local run = session.incr("run")
            local stat = fs.stat("SPEC.md")

            if run ~= 1 then
                return REJECT, "new session counter mismatch"
            end
            if stat.hash ~= hash.sha256("beta") or stat.bytes ~= 4 then
                return REJECT, "new session stat facts mismatch"
            end
            if stat.seen_before ~= false or stat.changed ~= true or stat.previous_hash ~= nil then
                return REJECT, "new session tracking mismatch"
            end

            return ALLOW
        end
    "#;
    let other_harness_dir = TempDir::new().unwrap();
    std::fs::write(
        other_harness_dir.path().join("other_session.lua"),
        other_session_script,
    )
    .unwrap();

    let mut other_engine = HarnessEngine::new(test_app_data_for_root_and_session(
        root.path().to_path_buf(),
        "other-session",
    ))
    .unwrap();
    other_engine.load_dir(other_harness_dir.path()).unwrap();

    let verdict = other_engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_fs_read_and_stat_reject_oversized_files_before_loading() {
    let root = TempDir::new().unwrap();
    let oversized = root.path().join("BIG.txt");
    let file = std::fs::File::create(&oversized).unwrap();
    file.set_len((10 * 1024 * 1024 + 1) as u64).unwrap();

    let script = r#"
        function on_turn_prepare(ctx)
            local content, read_err = try(fs.read, "BIG.txt")
            if content ~= nil then
                return REJECT, "fs.read should reject oversized files"
            end
            if read_err == nil or not tostring(read_err):find("File exceeds max size", 1, true) then
                return REJECT, "fs.read oversized error mismatch: " .. tostring(read_err)
            end

            local stat, stat_err = try(fs.stat, "BIG.txt")
            if stat ~= nil then
                return REJECT, "fs.stat should reject oversized files"
            end
            if stat_err == nil or not tostring(stat_err):find("File exceeds max size", 1, true) then
                return REJECT, "fs.stat oversized error mismatch: " .. tostring(stat_err)
            end

            return ALLOW
        end
    "#;

    let harness_dir = TempDir::new().unwrap();
    std::fs::write(harness_dir.path().join("oversized.lua"), script).unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(root.path().to_path_buf())).unwrap();
    engine.load_dir(harness_dir.path()).unwrap();

    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_fs_summary_reuses_cached_summary_until_file_changes() {
    let root = TempDir::new().unwrap();
    std::fs::write(root.path().join("SPEC.md"), "alpha").unwrap();

    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("summary.lua"),
        r#"
            function on_turn_prepare(ctx)
                local run = session.incr("summary_run")
                local default = fs.summary("SPEC.md")
                local default_again = fs.summary("SPEC.md")
                local custom = fs.summary("SPEC.md", {
                    prompt = "Summarize only the constraints and obligations."
                })

                if default ~= default_again then
                    return REJECT, "default summary should be cached within one turn"
                end

                if run == 1 then
                    if default ~= "summary-1" then
                        return REJECT, "run1 default summary mismatch"
                    end
                    if custom ~= "summary-2" then
                        return REJECT, "run1 custom summary mismatch"
                    end
                elseif run == 2 then
                    if default ~= "summary-1" then
                        return REJECT, "run2 default summary should be reused"
                    end
                    if custom ~= "summary-2" then
                        return REJECT, "run2 custom summary should be reused"
                    end
                elseif run == 3 then
                    if default ~= "summary-3" then
                        return REJECT, "run3 default summary mismatch"
                    end
                    if custom ~= "summary-4" then
                        return REJECT, "run3 custom summary mismatch"
                    end
                else
                    return REJECT, "unexpected summary run"
                end

                return ALLOW
            end
        "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(root.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let provider_calls = Arc::new(Mutex::new(0u32));
    let provider = Arc::new(CountingTextProvider {
        counter: Arc::clone(&provider_calls),
    });
    let mut clients = std::collections::HashMap::new();
    clients.insert("mock".to_string(), ProviderClient::new("mock", provider));

    let make_ctx = || {
        ContextWrapper::new(
            None,
            "mock-model".to_string(),
            "mock".to_string(),
            "Summarize files.".to_string(),
            Vec::new(),
            1,
            1,
            true,
            "task-1".to_string(),
            None,
            0,
            100000,
            0,
            RequestOptionsOverride::default(),
            clients.clone(),
            Arc::new(crate::kernel::config::TurinConfig::default()),
            "default".to_string(),
            InferenceOverrideConfig::default(),
        )
    };

    let verdict = engine
        .evaluate_userdata("on_turn_prepare", make_ctx())
        .unwrap();
    assert!(verdict.is_allowed());

    let verdict = engine
        .evaluate_userdata("on_turn_prepare", make_ctx())
        .unwrap();
    assert!(verdict.is_allowed());

    std::fs::write(root.path().join("SPEC.md"), "beta").unwrap();

    let verdict = engine
        .evaluate_userdata("on_turn_prepare", make_ctx())
        .unwrap();
    assert!(verdict.is_allowed());

    assert_eq!(*provider_calls.lock().unwrap(), 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_runtime_governance_grant_wrapper() {
    let mut engine = HarnessEngine::new(test_app_data()).unwrap();

    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("governance_dx.lua"),
        r#"
            function on_turn_prepare(ctx)
                local gid = nil
                local result = runtime.governance.grant({
                    ttl_ms = 5000,
                    capabilities = {
                        ["db.query"] = true,
                        ["governance.grant.get"] = true,
                    }
                }, function()
                    local query_dec = access.check("db.query")
                    if query_dec == nil or not query_dec.allowed then
                        error("runtime.db.query should be allowed inside grant")
                    end

                    local policy_dec = access.check("policy.set")
                    if policy_dec == nil then
                        error("runtime.policy.set decision missing")
                    end
                    if policy_dec.allowed then
                        error("runtime.policy.set should be denied by grant ceiling")
                    end

                    gid = query_dec.subject_grant_id
                    if gid == nil or gid == "" then
                        error("missing subject_grant_id in access decision")
                    end

                    return "grant_wrapper_ok"
                end)

                if result ~= "grant_wrapper_ok" then
                    return REJECT, "runtime.governance.grant result mismatch"
                end

                local grant, ge = runtime.governance.grant_get(gid)
                if grant ~= nil then
                    return REJECT, "grant should be revoked after callback returns"
                end
                if ge == nil then
                    return REJECT, "grant_get should report missing grant after revoke"
                end

                local ok, err = pcall(function()
                    runtime.governance.grant({
                        ttl_ms = 5000,
                        capabilities = {
                            ["db.query"] = true,
                            ["governance.grant.revoke"] = true,
                        }
                    }, function()
                        local dec = access.check("db.query")
                        local inner_gid = dec.subject_grant_id
                        local revoked, re = runtime.governance.grant_revoke(inner_gid)
                        if not revoked then
                            error("inner grant_revoke failed: " .. tostring(re))
                        end
                        error("grant callback sentinel")
                    end)
                end)
                if ok then
                    return REJECT, "runtime.governance.grant should fail when callback errors"
                end
                if not tostring(err):find("grant callback sentinel", 1, true) then
                    return REJECT, "runtime.governance.grant should prioritize callback error"
                end

                return ALLOW
            end
            "#,
    )
    .unwrap();

    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_time_helpers() {
    let mut engine = HarnessEngine::new(test_app_data()).unwrap();

    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("time_dx.lua"),
        r#"
            function on_turn_prepare(ctx)
                local now = tonumber(time.now_utc())
                if now == nil then
                    return REJECT, "time.now_utc should be numeric string"
                end

                local since_num = time.since(now - 2)
                if since_num < 1 then
                    return REJECT, "time.since(number) should be positive"
                end

                local since_str = time.since(tostring(now - 1))
                if since_str < 0 then
                    return REJECT, "time.since(string) should parse numeric string"
                end

                if not time.after(now - 1, 0.5) then
                    return REJECT, "time.after should be true when elapsed >= threshold"
                end

                if time.after(now - 1, 10) then
                    return REJECT, "time.after should be false for large threshold"
                end

                return ALLOW
            end
            "#,
    )
    .unwrap();

    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());
}
