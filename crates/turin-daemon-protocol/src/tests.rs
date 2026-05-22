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
    assert_eq!(value["params"]["conflict_policy"], "detached");

    let decoded: RequestEnvelope = serde_json::from_value(value).expect("deserialize request");
    match decoded.request {
        DaemonRequest::TaskSubmit(params) => {
            assert_eq!(params.agent_id.as_deref(), Some("writer"));
            assert!(params.session_id.is_none());
            assert_eq!(params.prompt, "review this");
            assert_eq!(params.conflict_policy.as_deref(), Some("detached"));
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
        }),
    );

    let value = serde_json::to_value(&request).expect("serialize request");
    assert_eq!(value["op"], "task.promote");
    assert_eq!(value["params"]["request_id"], "req_task");
    assert_eq!(value["params"]["branch_name"], "kept-idea");

    let decoded: RequestEnvelope = serde_json::from_value(value).expect("deserialize request");
    match decoded.request {
        DaemonRequest::TaskPromote(params) => {
            assert_eq!(params.request_id, "req_task");
            assert_eq!(params.branch_name.as_deref(), Some("kept-idea"));
        }
        other => panic!("unexpected request variant: {other:?}"),
    }
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
            channels: true,
        },
    };

    let value = serde_json::to_value(&handshake).expect("serialize handshake");
    assert_eq!(value["protocol_version"], DAEMON_PROTOCOL_VERSION);
    assert_eq!(value["transport"], DAEMON_TRANSPORT_UNIX);

    let decoded: DaemonHandshake = serde_json::from_value(value).expect("deserialize handshake");
    assert!(decoded.capabilities.runtime_snapshot_v1);
    assert!(decoded.capabilities.channels);
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
