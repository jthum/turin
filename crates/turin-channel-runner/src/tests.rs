use super::*;
use tempfile::tempdir;
use turin_channel_core::{
    ChannelKind, ChannelMessageRef, ChannelSessionScope, ChannelUser, MessageBlock,
};
use turin_types::TaskInputContent;

#[test]
fn stream_mode_parse_normalizes_known_values() {
    assert_eq!(
        ChannelStreamMode::parse("off"),
        Some(ChannelStreamMode::Off)
    );
    assert_eq!(
        ChannelStreamMode::parse(" Typing "),
        Some(ChannelStreamMode::Typing)
    );
    assert_eq!(
        ChannelStreamMode::parse("DRAFT"),
        Some(ChannelStreamMode::Draft)
    );
    assert_eq!(
        ChannelStreamMode::parse("block"),
        Some(ChannelStreamMode::Block)
    );
    assert_eq!(ChannelStreamMode::parse("partial"), None);
}

struct TestDriver;

#[async_trait::async_trait]
impl ChannelDriver for TestDriver {
    fn kind(&self) -> ChannelKind {
        ChannelKind::new("test")
    }

    fn user_matches_selector(&self, selector: &str, user: &ChannelUser) -> bool {
        let selector = selector.trim();
        if selector.is_empty() {
            return false;
        }
        let selector = selector.strip_prefix('@').unwrap_or(selector);
        user.id == selector
            || user
                .username
                .as_ref()
                .is_some_and(|username| username.eq_ignore_ascii_case(selector))
    }

    async fn next_event(&mut self) -> Result<Option<InboundEvent>> {
        Ok(None)
    }

    async fn send(
        &mut self,
        _conversation: &ChannelConversationKey,
        _message: OutboundMessage,
    ) -> Result<()> {
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

fn sample_key() -> ChannelConversationKey {
    ChannelConversationKey {
        channel: ChannelKind::new("discord"),
        workspace_id: "guild".into(),
        room_id: Some("room".into()),
        thread_id: "thread".into(),
        user_id: Some("user".into()),
    }
}

#[tokio::test]
async fn file_binding_store_round_trips() {
    let dir = tempdir().unwrap();
    let store = FileBindingStore::new(dir.path().join("bindings.json"));
    let key = serialize_binding_key(&sample_key()).unwrap();
    let mut map = HashMap::new();
    map.insert(
        key,
        ConversationBinding::new("writer", "session-1", &sample_key(), SystemTime::UNIX_EPOCH),
    );
    store.save(&map).await.unwrap();
    let loaded = store.load().await.unwrap();
    assert_eq!(loaded.len(), 1);
}

#[tokio::test]
async fn file_access_state_store_round_trips() {
    let dir = tempdir().unwrap();
    let store = FileAccessStateStore::new(dir.path().join("access.json"));
    let room = ChannelRoomKey::from(&sample_key());
    let key = serialize_room_key(&room).unwrap();
    let mut state = AccessStateFile::default();
    state.approved_rooms.insert(
        key,
        ApprovedRoom {
            room,
            approved_at_unix_seconds: 1,
            approved_by_user_id: Some("user".into()),
            approved_by_username: Some("owner".into()),
        },
    );
    store.save(&state).await.unwrap();
    let loaded = store.load().await.unwrap();
    assert_eq!(loaded.approved_rooms.len(), 1);
}

#[tokio::test]
async fn file_access_state_store_manages_public_snapshot() {
    let dir = tempdir().unwrap();
    let store = FileAccessStateStore::new(dir.path().join("access.json"));
    let room = ChannelRoomRef {
        channel: ChannelKind::new("telegram"),
        workspace_id: "telegram".into(),
        room_id: Some("-100123".into()),
        thread_id: "-100123".into(),
    };

    let snapshot = store
        .approve(&room, Some("owner".into()), Some("jay".into()))
        .await
        .unwrap();
    assert_eq!(snapshot.approved_rooms.len(), 1);
    assert!(snapshot.pending_rooms.is_empty());

    let snapshot = store.reject_pending(&room).await.unwrap();
    assert_eq!(snapshot.approved_rooms.len(), 1);
    assert!(snapshot.pending_rooms.is_empty());

    let snapshot = store.revoke(&room).await.unwrap();
    assert!(snapshot.approved_rooms.is_empty());
}

#[test]
fn access_policy_parses_pairing_allowed_and_banned_users() {
    let policy = ChannelAccessPolicy::from_settings(&serde_json::json!({
        "pairing_mode": "auto",
        "pairing_users": ["123", "@owner"],
        "allowed_users": "friend1,friend2",
        "banned_users": ["intruder"]
    }))
    .expect("policy should parse");
    assert_eq!(policy.pairing_mode, PairingMode::Auto);
    assert!(policy.pairing_users.contains("123"));
    assert!(policy.pairing_users.contains("@owner"));
    assert!(policy.allowed_users.contains("friend1"));
    assert!(policy.allowed_users.contains("friend2"));
    assert!(policy.banned_users.contains("intruder"));
}

#[test]
fn task_timeout_ms_defaults_to_none_and_accepts_zero_as_unbounded() {
    assert_eq!(
        task_timeout_ms_from_settings(&serde_json::json!({})).unwrap(),
        None
    );
    assert_eq!(
        task_timeout_ms_from_settings(&serde_json::json!({ "task_timeout_ms": 0 })).unwrap(),
        None
    );
    assert_eq!(
        task_timeout_ms_from_settings(&serde_json::json!({ "task_timeout_ms": 45000 })).unwrap(),
        Some(45_000)
    );
}

#[test]
fn tools_settings_parse_string_lists() {
    let tools = tools_config_from_settings(&serde_json::json!({
        "tools": {
            "allow": ["group:web", "read_file"],
            "exclude": ["web_search"]
        }
    }))
    .unwrap();
    assert_eq!(
        tools.selection.allow,
        Some(vec!["group:web".to_string(), "read_file".to_string()])
    );
    assert_eq!(tools.selection.exclude, vec!["web_search".to_string()]);
}

#[test]
fn persist_thinking_subscribes_to_session_events_even_without_streamed_text() {
    let stream = WorkerStreamConfig {
        mode: ChannelStreamMode::Typing,
        stream_thinking: false,
        persist_thinking: true,
    };
    assert!(should_subscribe_to_session_events(&stream));
}

fn test_runner(dir: &tempfile::TempDir, policy: ChannelAccessPolicy) -> ChannelRunner {
    ChannelRunner::new(
        turin_daemon_client::DaemonClient::new(dir.path().join("dummy.sock")),
        RunnerConfig {
            channel_id: "test-channel".to_string(),
            state_path: dir.path().join("bindings.json"),
            access_state_path: dir.path().join("access.json"),
            idle_ttl: Some(Duration::from_secs(600)),
            access_policy: policy,
            tools: Default::default(),
        },
    )
}

#[tokio::test]
async fn authorize_event_records_pending_rooms_once() {
    let dir = tempdir().unwrap();
    let runner = test_runner(
        &dir,
        ChannelAccessPolicy {
            pairing_mode: PairingMode::Pending,
            ..Default::default()
        },
    );
    let event = InboundEvent {
        conversation: sample_key(),
        message: ChannelMessageRef {
            conversation: sample_key(),
            message_id: "m1".into(),
        },
        user: ChannelUser {
            id: "u1".into(),
            display_name: Some("User".into()),
            username: Some("user".into()),
        },
        session_scope: ChannelSessionScope::User,
        text: "hello".into(),
        attachments: vec![],
        metadata: Default::default(),
    };
    let driver = TestDriver;

    assert!(matches!(
        runner.authorize_event(&driver, &event).await.unwrap(),
        EventAccessDecision::Pending { notify: true }
    ));
    assert!(matches!(
        runner.authorize_event(&driver, &event).await.unwrap(),
        EventAccessDecision::Pending { notify: false }
    ));
}

#[tokio::test]
async fn authorize_event_auto_approves_pairing_users() {
    let dir = tempdir().unwrap();
    let runner = test_runner(
        &dir,
        ChannelAccessPolicy {
            pairing_mode: PairingMode::Auto,
            pairing_users: HashSet::from(["u1".to_string()]),
            ..Default::default()
        },
    );
    let event = InboundEvent {
        conversation: sample_key(),
        message: ChannelMessageRef {
            conversation: sample_key(),
            message_id: "m1".into(),
        },
        user: ChannelUser {
            id: "u1".into(),
            display_name: Some("User".into()),
            username: Some("user".into()),
        },
        session_scope: ChannelSessionScope::User,
        text: "hello".into(),
        attachments: vec![],
        metadata: Default::default(),
    };
    let driver = TestDriver;

    assert!(matches!(
        runner.authorize_event(&driver, &event).await.unwrap(),
        EventAccessDecision::Allow
    ));
    assert!(matches!(
        runner.authorize_event(&driver, &event).await.unwrap(),
        EventAccessDecision::Allow
    ));
}

#[tokio::test]
async fn authorize_event_ignores_senders_not_allowed_to_pair() {
    let dir = tempdir().unwrap();
    let runner = test_runner(
        &dir,
        ChannelAccessPolicy {
            pairing_mode: PairingMode::Auto,
            pairing_users: HashSet::from(["owner".to_string()]),
            ..Default::default()
        },
    );
    let event = InboundEvent {
        conversation: sample_key(),
        message: ChannelMessageRef {
            conversation: sample_key(),
            message_id: "m1".into(),
        },
        user: ChannelUser {
            id: "intruder".into(),
            display_name: Some("Intruder".into()),
            username: Some("intruder".into()),
        },
        session_scope: ChannelSessionScope::User,
        text: "hello".into(),
        attachments: vec![],
        metadata: Default::default(),
    };
    let driver = TestDriver;

    assert!(matches!(
        runner.authorize_event(&driver, &event).await.unwrap(),
        EventAccessDecision::Ignore
    ));
}

#[tokio::test]
async fn authorize_event_allows_open_interaction_after_pairing() {
    let dir = tempdir().unwrap();
    let runner = test_runner(
        &dir,
        ChannelAccessPolicy {
            pairing_mode: PairingMode::Auto,
            pairing_users: HashSet::from(["owner".to_string()]),
            ..Default::default()
        },
    );
    let driver = TestDriver;

    let owner_event = InboundEvent {
        conversation: sample_key(),
        message: ChannelMessageRef {
            conversation: sample_key(),
            message_id: "m1".into(),
        },
        user: ChannelUser {
            id: "owner".into(),
            display_name: Some("Owner".into()),
            username: Some("jay".into()),
        },
        session_scope: ChannelSessionScope::User,
        text: "pair room".into(),
        attachments: vec![],
        metadata: Default::default(),
    };

    let friend_event = InboundEvent {
        conversation: sample_key(),
        message: ChannelMessageRef {
            conversation: sample_key(),
            message_id: "m2".into(),
        },
        user: ChannelUser {
            id: "friend".into(),
            display_name: Some("Friend".into()),
            username: Some("friend".into()),
        },
        session_scope: ChannelSessionScope::User,
        text: "hello".into(),
        attachments: vec![],
        metadata: Default::default(),
    };

    assert!(matches!(
        runner.authorize_event(&driver, &owner_event).await.unwrap(),
        EventAccessDecision::Allow
    ));
    assert!(matches!(
        runner
            .authorize_event(&driver, &friend_event)
            .await
            .unwrap(),
        EventAccessDecision::Allow
    ));
}

#[tokio::test]
async fn authorize_event_applies_allowed_users_after_pairing() {
    let dir = tempdir().unwrap();
    let runner = test_runner(
        &dir,
        ChannelAccessPolicy {
            pairing_mode: PairingMode::Auto,
            pairing_users: HashSet::from(["owner".to_string()]),
            allowed_users: HashSet::from(["friend".to_string()]),
            ..Default::default()
        },
    );
    let driver = TestDriver;

    let owner_event = InboundEvent {
        conversation: sample_key(),
        message: ChannelMessageRef {
            conversation: sample_key(),
            message_id: "m1".into(),
        },
        user: ChannelUser {
            id: "owner".into(),
            display_name: Some("Owner".into()),
            username: Some("jay".into()),
        },
        session_scope: ChannelSessionScope::User,
        text: "pair room".into(),
        attachments: vec![],
        metadata: Default::default(),
    };

    let intruder_event = InboundEvent {
        conversation: sample_key(),
        message: ChannelMessageRef {
            conversation: sample_key(),
            message_id: "m2".into(),
        },
        user: ChannelUser {
            id: "intruder".into(),
            display_name: Some("Intruder".into()),
            username: Some("intruder".into()),
        },
        session_scope: ChannelSessionScope::User,
        text: "hello".into(),
        attachments: vec![],
        metadata: Default::default(),
    };

    let friend_event = InboundEvent {
        conversation: sample_key(),
        message: ChannelMessageRef {
            conversation: sample_key(),
            message_id: "m3".into(),
        },
        user: ChannelUser {
            id: "friend".into(),
            display_name: Some("Friend".into()),
            username: Some("friend".into()),
        },
        session_scope: ChannelSessionScope::User,
        text: "hello".into(),
        attachments: vec![],
        metadata: Default::default(),
    };

    assert!(matches!(
        runner.authorize_event(&driver, &owner_event).await.unwrap(),
        EventAccessDecision::Allow
    ));
    assert!(matches!(
        runner
            .authorize_event(&driver, &intruder_event)
            .await
            .unwrap(),
        EventAccessDecision::Ignore
    ));
    assert!(matches!(
        runner
            .authorize_event(&driver, &friend_event)
            .await
            .unwrap(),
        EventAccessDecision::Allow
    ));
}

#[tokio::test]
async fn authorize_event_banned_users_override_approval() {
    let dir = tempdir().unwrap();
    let runner = test_runner(
        &dir,
        ChannelAccessPolicy {
            pairing_mode: PairingMode::Auto,
            pairing_users: HashSet::from(["owner".to_string()]),
            banned_users: HashSet::from(["friend".to_string()]),
            ..Default::default()
        },
    );
    let driver = TestDriver;

    let owner_event = InboundEvent {
        conversation: sample_key(),
        message: ChannelMessageRef {
            conversation: sample_key(),
            message_id: "m1".into(),
        },
        user: ChannelUser {
            id: "owner".into(),
            display_name: Some("Owner".into()),
            username: Some("jay".into()),
        },
        session_scope: ChannelSessionScope::User,
        text: "pair room".into(),
        attachments: vec![],
        metadata: Default::default(),
    };

    let friend_event = InboundEvent {
        conversation: sample_key(),
        message: ChannelMessageRef {
            conversation: sample_key(),
            message_id: "m2".into(),
        },
        user: ChannelUser {
            id: "friend".into(),
            display_name: Some("Friend".into()),
            username: Some("friend".into()),
        },
        session_scope: ChannelSessionScope::User,
        text: "hello".into(),
        attachments: vec![],
        metadata: Default::default(),
    };

    assert!(matches!(
        runner.authorize_event(&driver, &owner_event).await.unwrap(),
        EventAccessDecision::Allow
    ));
    assert!(matches!(
        runner
            .authorize_event(&driver, &friend_event)
            .await
            .unwrap(),
        EventAccessDecision::Ignore
    ));
}

#[test]
fn inbound_event_shape_is_runner_compatible() {
    let key = sample_key();
    let event = InboundEvent {
        conversation: key.clone(),
        message: ChannelMessageRef {
            conversation: key,
            message_id: "m1".into(),
        },
        user: ChannelUser {
            id: "u1".into(),
            display_name: Some("User".into()),
            username: Some("user".into()),
        },
        session_scope: ChannelSessionScope::User,
        text: "hello".into(),
        attachments: vec![],
        metadata: Default::default(),
    };
    assert_eq!(event.text, "hello");
}

#[test]
fn task_to_outbound_prefers_output() {
    let outbound = task_to_outbound(&TaskSnapshot {
        request_id: "req-1".into(),
        agent_id: "writer".into(),
        slot_id: "slot-1".into(),
        trace_id: "trace-1".into(),
        state: "completed".into(),
        runtime_task_id: None,
        status: Some("completed".into()),
        task_turn_count: Some(1),
        output: Some("hello".into()),
        assistant_content: None,
        error: Some("bad".into()),
    });
    assert_eq!(
        outbound.blocks,
        vec![MessageBlock::Text {
            text: "hello".into(),
        }]
    );
}

#[test]
fn task_to_outbound_parses_structured_payload() {
    let outbound = task_to_outbound(&TaskSnapshot {
        request_id: "req-1".into(),
        agent_id: "writer".into(),
        slot_id: "slot-1".into(),
        trace_id: "trace-1".into(),
        state: "completed".into(),
        runtime_task_id: None,
        status: Some("completed".into()),
        task_turn_count: Some(1),
        output: Some(
            serde_json::json!({
                "_turin_channel_outbound": true,
                "content": "overview",
                "embeds": [{ "title": "Build result" }],
                "components": [{ "type": 1, "components": [] }],
                "metadata": { "priority": "high" }
            })
            .to_string(),
        ),
        assistant_content: None,
        error: None,
    });
    assert_eq!(outbound.blocks.len(), 1);
    assert_eq!(outbound.embeds.len(), 1);
    assert_eq!(outbound.components.len(), 1);
    assert_eq!(outbound.metadata["priority"], "high");
}

#[test]
fn task_to_outbound_maps_assistant_content_when_no_structured_payload() {
    let outbound = task_to_outbound(&TaskSnapshot {
        request_id: "req-2".into(),
        agent_id: "writer".into(),
        slot_id: "slot-1".into(),
        trace_id: "trace-2".into(),
        state: "completed".into(),
        runtime_task_id: None,
        status: Some("completed".into()),
        task_turn_count: Some(1),
        output: Some("[Image: chart.png]".into()),
        assistant_content: Some(vec![
            TaskInputContent::Text {
                text: "Here is the chart".into(),
            },
            TaskInputContent::Image {
                name: Some("chart.png".into()),
                content_type: Some("image/png".into()),
                url: None,
                local_path: Some("/tmp/chart.png".into()),
                detail: None,
            },
        ]),
        error: None,
    });
    assert_eq!(
        outbound.blocks,
        vec![MessageBlock::Text {
            text: "Here is the chart".into(),
        }]
    );
    assert_eq!(outbound.attachments.len(), 1);
    assert_eq!(outbound.attachments[0].name, "chart.png");
    assert_eq!(
        outbound.attachments[0].local_path.as_deref(),
        Some("/tmp/chart.png")
    );
}

#[test]
fn task_input_content_from_event_maps_text_and_attachments() {
    let event = InboundEvent {
        conversation: sample_key(),
        message: ChannelMessageRef {
            conversation: sample_key(),
            message_id: "m1".into(),
        },
        user: ChannelUser {
            id: "u1".into(),
            display_name: Some("User".into()),
            username: Some("user".into()),
        },
        session_scope: ChannelSessionScope::Thread,
        text: "review this".into(),
        attachments: vec![
            turin_channel_core::ChannelAttachment {
                name: "diagram.png".into(),
                content_type: Some("image/png".into()),
                url: Some("https://cdn.test/diagram.png".into()),
                local_path: None,
            },
            turin_channel_core::ChannelAttachment {
                name: "spec.pdf".into(),
                content_type: Some("application/pdf".into()),
                url: Some("https://cdn.test/spec.pdf".into()),
                local_path: None,
            },
        ],
        metadata: Default::default(),
    };

    let content = task_input_content_from_event(&event);
    assert_eq!(content.len(), 3);
    assert!(matches!(
        &content[0],
        TaskInputContent::Text { text } if text.contains("review this")
    ));
    assert!(matches!(
        &content[1],
        TaskInputContent::Image {
            name: Some(name),
            content_type: Some(content_type),
            ..
        } if name == "diagram.png" && content_type == "image/png"
    ));
    assert!(matches!(
        &content[2],
        TaskInputContent::File {
            name: Some(name),
            content_type: Some(content_type),
            ..
        } if name == "spec.pdf" && content_type == "application/pdf"
    ));
}

#[test]
fn task_prompt_for_submission_falls_back_to_attachment_summary() {
    let event = InboundEvent {
        conversation: sample_key(),
        message: ChannelMessageRef {
            conversation: sample_key(),
            message_id: "m1".into(),
        },
        user: ChannelUser {
            id: "u1".into(),
            display_name: Some("User".into()),
            username: Some("user".into()),
        },
        session_scope: ChannelSessionScope::User,
        text: String::new(),
        attachments: vec![turin_channel_core::ChannelAttachment {
            name: "diagram.png".into(),
            content_type: Some("image/png".into()),
            url: Some("https://cdn.test/diagram.png".into()),
            local_path: None,
        }],
        metadata: Default::default(),
    };

    assert_eq!(task_prompt_for_submission(&event), "[image attachment]");
}
