use super::*;
use serde_json::json;
use turin_types::{TaskInputContent, ToolsConfig};

#[test]
fn request_envelope_round_trips_typed_shape() {
    let request = RequestEnvelope::new(
        Some("req_1".to_string()),
        DaemonRequest::TaskSubmit(SubmitTaskParams {
            agent_id: Some("writer".to_string()),
            session_id: None,
            slot_id: None,
            prompt: "review this".to_string(),
            inference_context: Some("fast".to_string()),
            content: None,
            tools: Default::default(),
            conflict_policy: Some("detached".to_string()),
        }),
    );

    let value = serde_json::to_value(&request).expect("serialize request");
    assert_eq!(value["id"], "req_1");
    assert_eq!(value["op"], "task.submit");
    assert_eq!(value["params"]["agent_id"], "writer");
    assert_eq!(value["params"]["prompt"], "review this");
    assert_eq!(value["params"]["inference_context"], "fast");
    assert_eq!(value["params"]["conflict_policy"], "detached");

    let decoded: RequestEnvelope = serde_json::from_value(value).expect("deserialize request");
    match decoded.request {
        DaemonRequest::TaskSubmit(params) => {
            assert_eq!(params.agent_id.as_deref(), Some("writer"));
            assert!(params.session_id.is_none());
            assert_eq!(params.prompt, "review this");
            assert_eq!(params.inference_context.as_deref(), Some("fast"));
            assert_eq!(params.conflict_policy.as_deref(), Some("detached"));
        }
        other => panic!("unexpected request variant: {other:?}"),
    }
}

#[test]
fn tool_authorization_requests_support_reasonless_denial() {
    let request = RequestEnvelope::new(
        Some("req_authorize".to_string()),
        DaemonRequest::ToolAuthorizationResolve(ToolAuthorizationResolveParams {
            request_id: "auth_1".to_string(),
            decision: ToolAuthorizationResolution::Deny,
            reason: None,
        }),
    );

    let value = serde_json::to_value(&request).expect("serialize authorization resolution");
    assert_eq!(value["op"], "tool_authorization.resolve");
    assert_eq!(value["params"]["decision"], "deny");
    assert!(value["params"].get("reason").is_none());

    let decoded: RequestEnvelope =
        serde_json::from_value(value).expect("deserialize authorization resolution");
    match decoded.request {
        DaemonRequest::ToolAuthorizationResolve(params) => {
            assert_eq!(params.request_id, "auth_1");
            assert_eq!(params.decision, ToolAuthorizationResolution::Deny);
            assert!(params.reason.is_none());
        }
        other => panic!("unexpected request variant: {other:?}"),
    }
}

#[test]
fn harness_source_candidate_and_save_requests_round_trip() {
    let validate = RequestEnvelope::new(
        Some("req_validate_source".to_string()),
        DaemonRequest::HarnessSourceValidate(HarnessSourceValidateParams {
            id: "operator".to_string(),
            changes: vec![
                HarnessSourceOverlay {
                    path: "main.lua".to_string(),
                    source: Some("use('libs/security')".to_string()),
                },
                HarnessSourceOverlay {
                    path: "libs/security.lua".to_string(),
                    source: Some("return {}".to_string()),
                },
            ],
        }),
    );
    let value = serde_json::to_value(validate).expect("serialize source validation");
    assert_eq!(value["op"], "harness.source.validate");
    assert_eq!(value["params"]["changes"][1]["path"], "libs/security.lua");

    let save = RequestEnvelope::new(
        Some("req_save_source".to_string()),
        DaemonRequest::HarnessSourceSave(HarnessSourceSaveParams {
            id: "operator".to_string(),
            changes: vec![HarnessSourceSaveChange {
                path: "main.lua".to_string(),
                source: Some("return {}".to_string()),
                expected_hash: Some("abc123".to_string()),
            }],
        }),
    );
    let value = serde_json::to_value(save).expect("serialize source save");
    assert_eq!(value["op"], "harness.source.save");
    assert_eq!(value["params"]["changes"][0]["expected_hash"], "abc123");
}

#[test]
fn session_get_accepts_full_and_windowed_request_shapes() {
    let full: RequestEnvelope = serde_json::from_value(json!({
        "op": "session.get",
        "params": { "session_id": "sess_123" }
    }))
    .expect("decode legacy full-detail request");
    match full.request {
        DaemonRequest::SessionGet(params) => {
            assert_eq!(params.session_id, "sess_123");
            assert!(params.target_turn_id.is_none());
            assert!(params.message_limit.is_none());
            assert!(params.message_offset.is_none());
            assert!(params.include_events.is_none());
            assert!(params.include_efficiency.is_none());
        }
        other => panic!("unexpected request variant: {other:?}"),
    }

    let windowed = RequestEnvelope::new(
        Some("req_session".to_string()),
        DaemonRequest::SessionGet(SessionGetParams {
            session_id: "sess_123".to_string(),
            target_turn_id: Some(42),
            message_limit: Some(48),
            message_offset: Some(96),
            include_events: Some(false),
            include_efficiency: Some(true),
        }),
    );
    let value = serde_json::to_value(windowed).expect("serialize windowed request");
    assert_eq!(value["params"]["message_limit"], 48);
    assert_eq!(value["params"]["target_turn_id"], 42);
    assert_eq!(value["params"]["message_offset"], 96);
    assert_eq!(value["params"]["include_events"], false);
    assert_eq!(value["params"]["include_efficiency"], true);
}

#[test]
fn session_graph_and_exact_turn_branch_requests_round_trip() {
    let graph = RequestEnvelope::new(
        Some("req_graph".to_string()),
        DaemonRequest::SessionGraphGet(SessionIdParams {
            session_id: "sess_123".to_string(),
        }),
    );
    let graph_value = serde_json::to_value(graph).expect("serialize graph request");
    assert_eq!(graph_value["op"], "session.graph_get");
    assert_eq!(graph_value["params"]["session_id"], "sess_123");

    let branch = RequestEnvelope::new(
        Some("req_branch".to_string()),
        DaemonRequest::SessionBranchCreate(SessionBranchCreateParams {
            session_id: "sess_123".to_string(),
            name: "alternate".to_string(),
            slot_id: None,
            from_turn_index: None,
            from_turn_id: Some(42),
            activate: false,
        }),
    );
    let branch_value = serde_json::to_value(branch).expect("serialize branch request");
    assert_eq!(branch_value["op"], "session.branch_create");
    assert_eq!(branch_value["params"]["from_turn_id"], 42);

    let decoded: RequestEnvelope =
        serde_json::from_value(branch_value).expect("deserialize branch request");
    match decoded.request {
        DaemonRequest::SessionBranchCreate(params) => {
            assert_eq!(params.from_turn_id, Some(42));
            assert!(params.from_turn_index.is_none());
        }
        other => panic!("unexpected request variant: {other:?}"),
    }
}

#[test]
fn session_delete_request_round_trips() {
    let request = RequestEnvelope::new(
        Some("req_delete_session".to_string()),
        DaemonRequest::SessionDelete(SessionIdParams {
            session_id: "019f-session".to_string(),
        }),
    );
    let value = serde_json::to_value(request).expect("serialize session deletion");
    assert_eq!(value["op"], "session.delete");
    assert_eq!(value["params"]["session_id"], "019f-session");
}

#[test]
fn session_list_can_filter_by_origin_or_target_direct_linked_children() {
    let origin_request = RequestEnvelope::new(
        Some("req_origin_sessions".to_string()),
        DaemonRequest::SessionList(SessionListParams {
            limit: 20,
            offset: 0,
            store: None,
            path: None,
            origin_id: Some("client:desktop".to_string()),
            parent_session_id: None,
        }),
    );
    let origin_value = serde_json::to_value(origin_request).expect("serialize origin session list");
    assert_eq!(origin_value["params"]["origin_id"], "client:desktop");

    let request = RequestEnvelope::new(
        Some("req_linked_sessions".to_string()),
        DaemonRequest::SessionList(SessionListParams {
            limit: 20,
            offset: 0,
            store: None,
            path: None,
            origin_id: None,
            parent_session_id: Some("019f-parent".to_string()),
        }),
    );
    let value = serde_json::to_value(request).expect("serialize linked session list");
    assert_eq!(value["op"], "session.list");
    assert_eq!(value["params"]["parent_session_id"], "019f-parent");
}

#[test]
fn memory_list_round_trips_filters_and_window() {
    let request = RequestEnvelope::new(
        Some("req_memory".to_string()),
        DaemonRequest::MemoryList(MemoryListParams {
            persistence: None,
            scope_kind: Some("agent".to_string()),
            scope_key: Some("researcher".to_string()),
            include_superseded: true,
            limit: Some(40),
            offset: Some(80),
        }),
    );

    let value = serde_json::to_value(&request).expect("serialize memory list request");
    assert_eq!(value["op"], "memory.list");
    assert_eq!(value["params"]["scope_kind"], "agent");
    assert_eq!(value["params"]["scope_key"], "researcher");
    assert_eq!(value["params"]["include_superseded"], true);
    assert_eq!(value["params"]["limit"], 40);
    assert_eq!(value["params"]["offset"], 80);

    let decoded: RequestEnvelope = serde_json::from_value(value).expect("decode memory list");
    match decoded.request {
        DaemonRequest::MemoryList(params) => {
            assert_eq!(params.scope_kind.as_deref(), Some("agent"));
            assert_eq!(params.scope_key.as_deref(), Some("researcher"));
            assert!(params.include_superseded);
            assert_eq!(params.limit, Some(40));
            assert_eq!(params.offset, Some(80));
        }
        other => panic!("unexpected request variant: {other:?}"),
    }
}

#[test]
fn sidestep_request_round_trips_typed_shape() {
    let request = RequestEnvelope::new(
        Some("req_3".to_string()),
        DaemonRequest::TaskSidestep(SidestepTaskParams {
            session_id: "sess_123".to_string(),
            slot_id: Some("sd_manual".to_string()),
            prompt: "What else should we add?".to_string(),
            content: None,
            tools: Default::default(),
            mode: SidestepModeParams::ForkSibling,
            context_target: Some(SidestepContextTargetParams::TurnId { turn_id: 42 }),
            timeout_ms: Some(2_500),
        }),
    );

    let value = serde_json::to_value(&request).expect("serialize request");
    assert_eq!(value["op"], "task.sidestep");
    assert_eq!(value["params"]["session_id"], "sess_123");
    assert_eq!(value["params"]["slot_id"], "sd_manual");
    assert_eq!(value["params"]["mode"], "fork_sibling");
    assert_eq!(value["params"]["context_target"]["kind"], "turn_id");
    assert_eq!(value["params"]["context_target"]["turn_id"], 42);

    let decoded: RequestEnvelope = serde_json::from_value(value).expect("deserialize request");
    match decoded.request {
        DaemonRequest::TaskSidestep(params) => {
            assert_eq!(params.session_id, "sess_123");
            assert_eq!(params.slot_id.as_deref(), Some("sd_manual"));
            assert_eq!(params.prompt, "What else should we add?");
            assert_eq!(params.mode, SidestepModeParams::ForkSibling);
            assert_eq!(params.timeout_ms, Some(2_500));
            assert!(matches!(
                params.context_target,
                Some(SidestepContextTargetParams::TurnId { turn_id: 42 })
            ));
        }
        other => panic!("unexpected request variant: {other:?}"),
    }
}

#[test]
fn promote_task_request_round_trips_typed_shape() {
    let request = RequestEnvelope::new(
        Some("req_4".to_string()),
        DaemonRequest::TaskPromote(PromoteTaskParams {
            request_id: "req_task".to_string(),
            branch_name: Some("kept-idea".to_string()),
            source_turn_id: Some(42),
        }),
    );

    let value = serde_json::to_value(&request).expect("serialize request");
    assert_eq!(value["op"], "task.promote");
    assert_eq!(value["params"]["request_id"], "req_task");
    assert_eq!(value["params"]["branch_name"], "kept-idea");
    assert_eq!(value["params"]["source_turn_id"], 42);

    let decoded: RequestEnvelope = serde_json::from_value(value).expect("deserialize request");
    match decoded.request {
        DaemonRequest::TaskPromote(params) => {
            assert_eq!(params.request_id, "req_task");
            assert_eq!(params.branch_name.as_deref(), Some("kept-idea"));
            assert_eq!(params.source_turn_id, Some(42));
        }
        other => panic!("unexpected request variant: {other:?}"),
    }
}

#[test]
fn linked_family_operations_use_session_id_params() {
    for (request, expected_op) in [
        (
            DaemonRequest::SessionFamilyGet(SessionIdParams {
                session_id: "child@state".to_string(),
            }),
            "session.family_get",
        ),
        (
            DaemonRequest::SessionArchive(SessionIdParams {
                session_id: "child@state".to_string(),
            }),
            "session.archive",
        ),
    ] {
        let value = serde_json::to_value(RequestEnvelope::new(None, request)).unwrap();
        assert_eq!(value["op"], expected_op);
        assert_eq!(value["params"]["session_id"], "child@state");
    }
}

#[test]
fn harness_action_run_request_round_trips_typed_shape() {
    let request = RequestEnvelope::new(
        Some("req_action".to_string()),
        DaemonRequest::HarnessActionRun(HarnessActionRunParams {
            action: "release.approve".to_string(),
            agent_id: Some("release-agent".to_string()),
            harness_id: Some("release".to_string()),
            params: json!({ "item": "work_1" }),
        }),
    );

    let value = serde_json::to_value(&request).expect("serialize request");
    assert_eq!(value["op"], "harness.action_run");
    assert_eq!(value["params"]["action"], "release.approve");
    assert_eq!(value["params"]["agent_id"], "release-agent");
    assert_eq!(value["params"]["harness_id"], "release");
    assert_eq!(value["params"]["params"]["item"], "work_1");

    let decoded: RequestEnvelope = serde_json::from_value(value).expect("deserialize request");
    match decoded.request {
        DaemonRequest::HarnessActionRun(params) => {
            assert_eq!(params.action, "release.approve");
            assert_eq!(params.agent_id.as_deref(), Some("release-agent"));
            assert_eq!(params.harness_id.as_deref(), Some("release"));
            assert_eq!(params.params["item"], "work_1");
        }
        other => panic!("unexpected request variant: {other:?}"),
    }
}

#[test]
fn harness_action_run_result_round_trips_ui_intents() {
    let result = HarnessActionRunResult {
        action: "release.seed_demo_work".to_string(),
        agent_id: "default".to_string(),
        harness_id: Some("default".to_string()),
        result: json!({ "status": "seeded" }),
        ui_intents: vec![UiIntentMessage::new(UiIntent::Refresh(UiRefreshIntent {
            app_id: "release-operator".to_string(),
            binding: "worklists.release".to_string(),
        }))],
    };

    let value = serde_json::to_value(&result).expect("serialize action result");
    assert_eq!(value["action"], "release.seed_demo_work");
    assert_eq!(value["ui_intents"][0]["type"], "refresh");
    assert_eq!(value["ui_intents"][0]["binding"], "worklists.release");

    let decoded: HarnessActionRunResult =
        serde_json::from_value(value).expect("deserialize action result");
    assert_eq!(decoded.ui_intents.len(), 1);
    assert!(matches!(decoded.ui_intents[0].intent, UiIntent::Refresh(_)));
}

#[test]
fn harness_action_run_result_defaults_missing_ui_intents() {
    let decoded: HarnessActionRunResult = serde_json::from_value(json!({
        "action": "release.approve",
        "agent_id": "default",
        "harness_id": "default",
        "result": { "status": "approved" }
    }))
    .expect("deserialize old action result");

    assert!(decoded.ui_intents.is_empty());
}

#[test]
fn raw_daemon_wire_shape_deserializes_into_typed_request() {
    let decoded: RequestEnvelope = serde_json::from_value(json!({
        "id": "req_2",
        "op": "agent.disable",
        "params": { "id": "docs-reviewer" }
    }))
    .expect("deserialize request");

    match decoded.request {
        DaemonRequest::AgentDisable(EntityIdParams { id }) => {
            assert_eq!(id, "docs-reviewer");
        }
        other => panic!("unexpected request variant: {other:?}"),
    }
}

#[test]
fn error_code_serializes_as_snake_case() {
    let response = ResponseEnvelope::err(
        Some("req_2".to_string()),
        ErrorCode::AgentNotFound,
        "missing",
        None,
    );

    let value = serde_json::to_value(&response).expect("serialize response");
    assert_eq!(value["error"]["code"], "agent_not_found");
}

#[test]
fn handshake_round_trips_typed_shape() {
    let handshake = DaemonHandshake {
        pong: true,
        version: env!("CARGO_PKG_VERSION").to_string(),
        protocol_version: DAEMON_PROTOCOL_VERSION,
        transport: DAEMON_TRANSPORT_UNIX.to_string(),
        wire_format: DAEMON_WIRE_FORMAT_NDJSON.to_string(),
        capabilities: DaemonCapabilities {
            runtime_snapshot_v1: true,
            scoped_event_snapshots: true,
            lag_resnapshot: true,
            watcher_rescan_failed_events: true,
        },
    };

    let value = serde_json::to_value(&handshake).expect("serialize handshake");
    assert_eq!(value["protocol_version"], DAEMON_PROTOCOL_VERSION);
    assert_eq!(value["transport"], DAEMON_TRANSPORT_UNIX);

    let decoded: DaemonHandshake = serde_json::from_value(value).expect("deserialize handshake");
    assert!(decoded.capabilities.runtime_snapshot_v1);
}

#[test]
fn ui_screen_intent_round_trips_as_event_payload() {
    let message = UiIntentMessage::new(UiIntent::Screen(UiScreenIntent {
        app_id: "release".to_string(),
        id: "dashboard".to_string(),
        title: "Release Desk".to_string(),
        presentation: None,
        nodes: vec![
            UiNode::List(UiListNode {
                id: Some("open-work".to_string()),
                title: "Open Work".to_string(),
                source: "worklists.release".to_string(),
                filter: serde_json::Map::from_iter([("kind".to_string(), json!("approval"))]),
                fields: Vec::new(),
                sort: Vec::new(),
                limit: None,
                intent: Some("approval".to_string()),
                render_as: Some("table".to_string()),
            }),
            UiNode::Section(UiSectionNode {
                id: Some("controls".to_string()),
                title: "Controls".to_string(),
                nodes: vec![UiNode::Action(UiActionNode {
                    id: Some("run-smoke".to_string()),
                    label: "Run Smoke Tests".to_string(),
                    action: "qa.run_smoke".to_string(),
                    params: json!({ "suite": "smoke" }),
                    confirm: false,
                })],
            }),
            UiNode::Form(UiFormNode {
                id: Some("seed-demo-form".to_string()),
                title: "Create Demo Work".to_string(),
                action: "release.seed_demo_work".to_string(),
                fields: vec![UiFormField {
                    name: "count".to_string(),
                    label: "Count".to_string(),
                    kind: Some("number".to_string()),
                    default: Some(json!(4)),
                    required: Some(true),
                    options: Vec::new(),
                }],
                params: json!({ "release": "2026.06" }),
            }),
            UiNode::Activity(UiActivityNode {
                id: Some("activity".to_string()),
                title: "Release Activity".to_string(),
                source: "signals.release".to_string(),
            }),
        ],
    }))
    .from_harness("release")
    .for_app("release")
    .for_session("ui_123");

    let event = message.into_event().expect("serialize ui intent event");

    assert_eq!(event.event, UI_INTENT_EVENT);
    assert_eq!(event.data["version"], UI_INTENT_VERSION);
    assert_eq!(event.data["type"], "screen");
    assert_eq!(event.data["source"]["harness_id"], "release");
    assert_eq!(event.data["source"]["app_id"], "release");
    assert_eq!(event.data["recipient"]["ui_session_id"], "ui_123");
    assert_eq!(event.data["app_id"], "release");
    assert_eq!(event.data["id"], "dashboard");
    assert_eq!(event.data["title"], "Release Desk");
    assert_eq!(event.data["nodes"][0]["kind"], "list");
    assert_eq!(event.data["nodes"][0]["intent"], "approval");
    assert_eq!(event.data["nodes"][0]["as"], "table");
    assert_eq!(event.data["nodes"][0]["where"]["kind"], "approval");
    assert_eq!(event.data["nodes"][1]["nodes"][0]["kind"], "action");
    assert_eq!(
        event.data["nodes"][1]["nodes"][0]["params"]["suite"],
        "smoke"
    );
    assert_eq!(event.data["nodes"][2]["kind"], "form");
    assert_eq!(event.data["nodes"][2]["fields"][0]["kind"], "number");
    assert_eq!(event.data["nodes"][2]["fields"][0]["default"], 4);

    let decoded = UiIntentMessage::from_event(&event)
        .expect("deserialize ui intent")
        .expect("ui intent event");
    match decoded.intent {
        UiIntent::Screen(screen) => {
            assert_eq!(screen.app_id, "release");
            assert_eq!(screen.id, "dashboard");
            assert_eq!(screen.title, "Release Desk");
            assert_eq!(screen.nodes.len(), 4);
            assert!(matches!(screen.nodes[0], UiNode::List(_)));
            assert!(matches!(screen.nodes[1], UiNode::Section(_)));
            assert!(matches!(screen.nodes[2], UiNode::Form(_)));
            assert!(matches!(screen.nodes[3], UiNode::Activity(_)));
        }
        other => panic!("unexpected ui intent: {other:?}"),
    }
}

#[test]
fn ui_form_field_accepts_type_alias_and_default() {
    let decoded: UiFormField = serde_json::from_value(json!({
        "name": "release",
        "label": "Release",
        "type": "text",
        "default": "2026.06",
        "required": true
    }))
    .expect("deserialize form field");

    assert_eq!(decoded.kind.as_deref(), Some("text"));
    assert_eq!(decoded.default.as_ref(), Some(&json!("2026.06")));
    assert_eq!(decoded.required, Some(true));
}

#[test]
fn ui_menu_intent_supports_nested_items() {
    let message = UiIntentMessage::new(UiIntent::Menu(UiMenuIntent {
        app_id: "ops".to_string(),
        title: "Main".to_string(),
        items: vec![UiMenuItem {
            label: "Release".to_string(),
            opens: "release.dashboard".to_string(),
            id: None,
            icon: Some("rocket".to_string()),
            badge: Some("release".to_string()),
            items: vec![UiMenuItem {
                label: "Approvals".to_string(),
                opens: "release.approvals".to_string(),
                id: None,
                icon: None,
                badge: None,
                items: Vec::new(),
            }],
        }],
    }));

    let value = serde_json::to_value(&message).expect("serialize menu");
    assert_eq!(value["type"], "menu");
    assert_eq!(value["items"][0]["label"], "Release");
    assert_eq!(value["items"][0]["items"][0]["opens"], "release.approvals");

    let decoded: UiIntentMessage = serde_json::from_value(value).expect("deserialize menu");
    let UiIntent::Menu(menu) = decoded.intent else {
        panic!("expected menu intent");
    };
    assert_eq!(menu.items[0].items[0].label, "Approvals");
}

#[test]
fn ui_intent_parser_ignores_unrelated_events() {
    let event = EventEnvelope::new("runtime.ready", json!({ "ok": true }));
    let decoded = UiIntentMessage::from_event(&event).expect("parse event");
    assert!(decoded.is_none());
}

#[test]
fn ui_intent_parser_ignores_event_routing_metadata() {
    let event = EventEnvelope::new(
        UI_INTENT_EVENT,
        json!({
            "version": UI_INTENT_VERSION,
            "type": "notify",
            "app_id": "release",
            "title": "Release blocked",
            "agent_id": "agent-1",
            "session_id": "session-1",
            "slot_id": "slot-1"
        }),
    );

    let decoded = UiIntentMessage::from_event(&event)
        .expect("parse event")
        .expect("ui intent event");
    let UiIntent::Notify(notice) = decoded.intent else {
        panic!("expected notice intent");
    };
    assert_eq!(notice.app_id, "release");
    assert_eq!(notice.title, "Release blocked");
}

#[test]
fn ui_dynamic_intents_have_small_wire_shapes() {
    let notice = UiIntentMessage::new(UiIntent::Notify(UiNoticeIntent {
        app_id: "release".to_string(),
        title: "Release blocked".to_string(),
        body: Some("QA failed".to_string()),
        level: Some(UiNoticeLevel::Warning),
    }));
    let notice_value = serde_json::to_value(&notice).expect("serialize notice");
    assert_eq!(notice_value["type"], "notify");
    assert_eq!(notice_value["level"], "warning");
    assert_eq!(notice_value["title"], "Release blocked");

    let focus = UiIntentMessage::new(UiIntent::Focus(UiFocusIntent {
        app_id: "release".to_string(),
        target: "open-work".to_string(),
    }));
    let focus_value = serde_json::to_value(&focus).expect("serialize focus");
    assert_eq!(focus_value["type"], "focus");
    assert_eq!(focus_value["target"], "open-work");

    let refresh = UiIntentMessage::new(UiIntent::Refresh(UiRefreshIntent {
        app_id: "release".to_string(),
        binding: "worklists.release".to_string(),
    }));
    let refresh_value = serde_json::to_value(&refresh).expect("serialize refresh");
    assert_eq!(refresh_value["type"], "refresh");
    assert_eq!(refresh_value["binding"], "worklists.release");
    assert!(refresh_value.get("source").is_none());
    assert!(refresh_value.get("target").is_none());
}

#[test]
fn schedule_create_request_round_trips_typed_shape() {
    let request = RequestEnvelope::new(
        Some("req_sched".to_string()),
        DaemonRequest::ScheduleCreate(ScheduleCreateParams {
            agent_id: "default".to_string(),
            prompt: Some("Heartbeat".to_string()),
            content: Some(vec![TaskInputContent::Text {
                text: "Follow up with context".to_string(),
            }]),
            tools: Some(ToolsConfig {
                selection: turin_types::ToolSelectionConfig {
                    allow: Some(vec!["shell_exec".to_string()]),
                    exclude: Vec::new(),
                },
                ..ToolsConfig::default()
            }),
            conflict_policy: Some("detached".to_string()),
            action: None,
            next_run_unix_ms: 1_700_000_000_000,
            interval_seconds: Some(300),
            recurring_pattern: None,
            overlap_policy: Some("skip".to_string()),
            work_key: Some("project:alpha:qa".to_string()),
            max_concurrency: Some(1),
            persistence: Some(ContextPersistenceParams {
                state: Some(StoreTargetParams {
                    path: None,
                    alias: Some("project-alpha".to_string()),
                }),
                store: None,
            }),
            enabled: true,
        }),
    );

    let value = serde_json::to_value(&request).expect("serialize request");
    assert_eq!(value["op"], "schedule.create");
    assert_eq!(value["params"]["agent_id"], "default");
    assert_eq!(value["params"]["prompt"], "Heartbeat");
    assert_eq!(value["params"]["content"][0]["type"], "text");
    assert_eq!(value["params"]["tools"]["allow"][0], "shell_exec");
    assert_eq!(value["params"]["conflict_policy"], "detached");
    assert_eq!(value["params"]["next_run_unix_ms"], 1_700_000_000_000i64);
    assert_eq!(value["params"]["interval_seconds"], 300);
    assert_eq!(value["params"]["overlap_policy"], "skip");
    assert_eq!(value["params"]["work_key"], "project:alpha:qa");
    assert_eq!(value["params"]["max_concurrency"], 1);
    assert_eq!(
        value["params"]["persistence"]["state"]["alias"],
        "project-alpha"
    );

    let decoded: RequestEnvelope = serde_json::from_value(value).expect("deserialize request");
    match decoded.request {
        DaemonRequest::ScheduleCreate(params) => {
            assert_eq!(params.agent_id, "default");
            assert_eq!(params.prompt.as_deref(), Some("Heartbeat"));
            assert!(matches!(
                params.content.as_deref(),
                Some([TaskInputContent::Text { text }]) if text == "Follow up with context"
            ));
            assert_eq!(
                params
                    .tools
                    .as_ref()
                    .and_then(|tools| tools.selection.allow.as_ref())
                    .cloned(),
                Some(vec!["shell_exec".to_string()])
            );
            assert_eq!(params.conflict_policy.as_deref(), Some("detached"));
            assert!(params.action.is_none());
            assert_eq!(params.next_run_unix_ms, 1_700_000_000_000i64);
            assert_eq!(params.interval_seconds, Some(300));
            assert_eq!(params.recurring_pattern, None);
            assert_eq!(params.overlap_policy.as_deref(), Some("skip"));
            assert_eq!(params.work_key.as_deref(), Some("project:alpha:qa"));
            assert_eq!(params.max_concurrency, Some(1));
            assert_eq!(
                params
                    .persistence
                    .as_ref()
                    .and_then(|p| p.state.as_ref())
                    .and_then(|state| state.alias.as_deref()),
                Some("project-alpha")
            );
            assert!(params.enabled);
        }
        other => panic!("unexpected request variant: {other:?}"),
    }
}

#[test]
fn schedule_runs_request_round_trips_typed_shape() {
    let request = RequestEnvelope::new(
        Some("req_sched_runs".to_string()),
        DaemonRequest::ScheduleRuns(ScheduleRunsParams {
            id: "0196f8fe-6e6a-7e1a-8da5-3f774f1a8d47".to_string(),
            active_only: true,
            limit: Some(5),
        }),
    );

    let value = serde_json::to_value(&request).expect("serialize request");
    assert_eq!(value["op"], "schedule.runs");
    assert_eq!(
        value["params"]["id"],
        "0196f8fe-6e6a-7e1a-8da5-3f774f1a8d47"
    );
    assert_eq!(value["params"]["active_only"], true);
    assert_eq!(value["params"]["limit"], 5);

    let decoded: RequestEnvelope = serde_json::from_value(value).expect("deserialize request");
    match decoded.request {
        DaemonRequest::ScheduleRuns(params) => {
            assert_eq!(params.id, "0196f8fe-6e6a-7e1a-8da5-3f774f1a8d47");
            assert!(params.active_only);
            assert_eq!(params.limit, Some(5));
        }
        other => panic!("unexpected request variant: {other:?}"),
    }
}

#[test]
fn worklist_items_request_round_trips_typed_shape() {
    let request = RequestEnvelope::new(
        Some("req_worklist_items".to_string()),
        DaemonRequest::WorklistItems(WorklistItemsParams {
            id: "0196f8fe-6e6a-7e1a-8da5-3f774f1a8d47".to_string(),
            persistence: Some(ContextPersistenceParams {
                state: Some(StoreTargetParams {
                    path: None,
                    alias: Some("project_alpha".to_string()),
                }),
                store: None,
            }),
            status: Some("pending".to_string()),
            parent_id: Some("0196f8fe-6e6a-7e1a-8da5-3f774f1a8d48".to_string()),
            r#where: Some(serde_json::Map::from_iter([(
                "role".to_string(),
                json!("browser"),
            )])),
            claimed_only: true,
            paused_only: false,
            due_only: false,
            limit: Some(10),
        }),
    );

    let value = serde_json::to_value(&request).expect("serialize request");
    assert_eq!(value["op"], "worklist.items");
    assert_eq!(
        value["params"]["id"],
        "0196f8fe-6e6a-7e1a-8da5-3f774f1a8d47"
    );
    assert_eq!(value["params"]["status"], "pending");
    assert_eq!(
        value["params"]["parent_id"],
        "0196f8fe-6e6a-7e1a-8da5-3f774f1a8d48"
    );
    assert_eq!(value["params"]["where"]["role"], "browser");
    assert_eq!(value["params"]["claimed_only"], true);
    assert_eq!(value["params"]["limit"], 10);
    assert_eq!(
        value["params"]["persistence"]["state"]["alias"],
        "project_alpha"
    );

    let decoded: RequestEnvelope = serde_json::from_value(value).expect("deserialize request");
    match decoded.request {
        DaemonRequest::WorklistItems(params) => {
            assert_eq!(params.id, "0196f8fe-6e6a-7e1a-8da5-3f774f1a8d47");
            assert_eq!(params.status.as_deref(), Some("pending"));
            assert_eq!(
                params.parent_id.as_deref(),
                Some("0196f8fe-6e6a-7e1a-8da5-3f774f1a8d48")
            );
            assert_eq!(
                params.r#where.as_ref().and_then(|value| value.get("role")),
                Some(&json!("browser"))
            );
            assert!(params.claimed_only);
            assert_eq!(params.limit, Some(10));
            assert_eq!(
                params
                    .persistence
                    .as_ref()
                    .and_then(|p| p.state.as_ref())
                    .and_then(|state| state.alias.as_deref()),
                Some("project_alpha")
            );
        }
        other => panic!("unexpected request variant: {other:?}"),
    }
}

#[test]
fn workitem_get_request_round_trips_typed_shape() {
    let request = RequestEnvelope::new(
        Some("req_workitem_get".to_string()),
        DaemonRequest::WorkItemGet(WorkItemTargetParams {
            id: "0196f8fe-6e6a-7e1a-8da5-3f774f1a8d49".to_string(),
            persistence: Some(ContextPersistenceParams {
                state: Some(StoreTargetParams {
                    path: None,
                    alias: Some("project_alpha".to_string()),
                }),
                store: None,
            }),
        }),
    );

    let value = serde_json::to_value(&request).expect("serialize request");
    assert_eq!(value["op"], "workitem.get");
    assert_eq!(
        value["params"]["id"],
        "0196f8fe-6e6a-7e1a-8da5-3f774f1a8d49"
    );
    assert_eq!(
        value["params"]["persistence"]["state"]["alias"],
        "project_alpha"
    );

    let decoded: RequestEnvelope = serde_json::from_value(value).expect("deserialize request");
    match decoded.request {
        DaemonRequest::WorkItemGet(params) => {
            assert_eq!(params.id, "0196f8fe-6e6a-7e1a-8da5-3f774f1a8d49");
            assert_eq!(
                params
                    .persistence
                    .as_ref()
                    .and_then(|p| p.state.as_ref())
                    .and_then(|state| state.alias.as_deref()),
                Some("project_alpha")
            );
        }
        other => panic!("unexpected request variant: {other:?}"),
    }
}
