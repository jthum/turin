use super::*;

#[test]
fn session_scope_parse_normalizes_known_values() {
    assert_eq!(
        ChannelSessionScope::parse("user"),
        Some(ChannelSessionScope::User)
    );
    assert_eq!(
        ChannelSessionScope::parse(" Thread "),
        Some(ChannelSessionScope::Thread)
    );
    assert_eq!(
        ChannelSessionScope::parse("ROOM"),
        Some(ChannelSessionScope::Room)
    );
    assert_eq!(ChannelSessionScope::parse("guild"), None);
}

fn key() -> ChannelConversationKey {
    ChannelConversationKey {
        channel: ChannelKind::new("discord"),
        workspace_id: "guild-1".into(),
        room_id: Some("room-2".into()),
        thread_id: "thread-3".into(),
        user_id: Some("user-4".into()),
    }
}

#[test]
fn slot_id_is_stable() {
    let key = key();
    assert_eq!(key.deterministic_slot_id(), key.deterministic_slot_id());
}

#[test]
fn reset_forces_fresh_session() {
    let key = key();
    let binding = ConversationBinding::new("writer", "sess-1", &key, SystemTime::UNIX_EPOCH);
    let decision = decide_routing(&key, Some(&binding), SystemTime::UNIX_EPOCH, None, true);
    assert!(matches!(decision, RoutingDecision::StartFresh { .. }));
}

#[test]
fn ttl_expiry_forces_fresh_session() {
    let key = key();
    let binding = ConversationBinding::new("writer", "sess-1", &key, SystemTime::UNIX_EPOCH);
    let decision = decide_routing(
        &key,
        Some(&binding),
        SystemTime::UNIX_EPOCH + Duration::from_secs(120),
        Some(Duration::from_secs(60)),
        false,
    );
    assert!(matches!(decision, RoutingDecision::StartFresh { .. }));
}

#[test]
fn structured_outbound_message_keeps_code_blocks() {
    let message = OutboundMessage {
        blocks: vec![
            MessageBlock::Text {
                text: "Here is code".into(),
            },
            MessageBlock::CodeBlock {
                language: Some("rust".into()),
                code: "fn main() {}".into(),
            },
        ],
        ..OutboundMessage::default()
    };
    let value = serde_json::to_value(&message).expect("serialize outbound message");
    assert_eq!(value["blocks"].as_array().unwrap().len(), 2);
}

#[test]
fn plain_text_renderer_keeps_text_and_code_blocks() {
    let blocks = vec![
        MessageBlock::Text {
            text: "answer".into(),
        },
        MessageBlock::Text { text: "  ".into() },
        MessageBlock::CodeBlock {
            language: Some("rust".into()),
            code: "fn main() {}".into(),
        },
    ];

    assert_eq!(
        render_plain_text_blocks(&blocks),
        "answer\n\n```rust\nfn main() {}\n```"
    );
}

#[test]
fn line_splitter_packs_lines_and_splits_long_lines() {
    assert_eq!(
        split_text_lines_to_char_limit("aa\nbbb\ncccccc", 5),
        vec![
            "aa".to_string(),
            "bbb".to_string(),
            "ccccc".to_string(),
            "c".to_string()
        ]
    );
}

#[test]
fn line_splitter_trims_empty_content() {
    assert!(split_text_lines_to_char_limit(" \n\t ", 4).is_empty());
}

#[test]
fn shared_scope_prompt_includes_sender_label() {
    let key = key();
    let event = InboundEvent {
        conversation: key.clone(),
        message: ChannelMessageRef {
            conversation: key,
            message_id: "m-1".into(),
        },
        user: ChannelUser {
            id: "user-4".into(),
            display_name: Some("Jay".into()),
            username: Some("jthum".into()),
        },
        session_scope: ChannelSessionScope::Thread,
        text: "hello".into(),
        attachments: vec![],
        metadata: serde_json::Map::new(),
    };

    assert_eq!(event.prompt_text(), "[Message from Jay (@jthum)]\nhello");
}

#[test]
fn bound_inbound_text_leaves_short_messages_unchanged() {
    let mut metadata = serde_json::Map::new();
    let text = bound_inbound_text(
        "hello".into(),
        &mut metadata,
        DEFAULT_MAX_INBOUND_TEXT_CHARS,
    );
    assert_eq!(text, "hello");
    assert!(metadata.is_empty());
}

#[test]
fn bound_inbound_text_truncates_and_marks_metadata() {
    let mut metadata = serde_json::Map::new();
    let input = "a".repeat(DEFAULT_MAX_INBOUND_TEXT_CHARS + 5);
    let text = bound_inbound_text(input, &mut metadata, DEFAULT_MAX_INBOUND_TEXT_CHARS);
    assert_eq!(text.chars().count(), DEFAULT_MAX_INBOUND_TEXT_CHARS);
    assert_eq!(
        metadata.get("turin_text_truncated"),
        Some(&serde_json::Value::Bool(true))
    );
    assert_eq!(
        metadata.get("turin_original_text_chars"),
        Some(&serde_json::Value::Number(
            (DEFAULT_MAX_INBOUND_TEXT_CHARS + 5).into()
        ))
    );
}

#[test]
fn bound_inbound_text_uses_custom_limit() {
    let mut metadata = serde_json::Map::new();
    let text = bound_inbound_text("abcdef".into(), &mut metadata, 3);
    assert_eq!(text, "abc");
    assert_eq!(
        metadata.get("turin_text_char_limit"),
        Some(&serde_json::Value::Number(3usize.into()))
    );
}

#[test]
fn channel_kind_normalizes_to_lowercase() {
    let kind = ChannelKind::parse("TeLeGrAm").expect("normalized");
    assert_eq!(kind.as_str(), "telegram");
}

#[test]
fn channel_kind_rejects_invalid_characters() {
    let err = ChannelKind::parse("telegram!").expect_err("invalid");
    assert!(err.contains("channel kind"));
}

#[test]
fn manifest_validation_rejects_wrong_protocol_version() {
    let manifest = ChannelAdapterManifest {
        protocol_version: CHANNEL_ADAPTER_PROTOCOL_VERSION + 1,
        kind: "telegram".to_string(),
        ..ChannelAdapterManifest::default()
    };
    let err = manifest.validate().expect_err("invalid protocol");
    assert!(err.contains("protocol_version"));
}

#[test]
fn manifest_validation_rejects_duplicate_auth_flows() {
    let manifest = ChannelAdapterManifest {
        protocol_version: CHANNEL_ADAPTER_PROTOCOL_VERSION,
        kind: "telegram".to_string(),
        setup: Some(ChannelSetupManifest {
            auth_flows: vec![
                ChannelAuthFlow {
                    id: "pair".to_string(),
                    kind: ChannelAuthFlowKind::QrPairing,
                    label: None,
                    prompt: None,
                    help: None,
                    hint: None,
                    advanced: false,
                    visible_if: None,
                },
                ChannelAuthFlow {
                    id: "pair".to_string(),
                    kind: ChannelAuthFlowKind::QrPairing,
                    label: None,
                    prompt: None,
                    help: None,
                    hint: None,
                    advanced: false,
                    visible_if: None,
                },
            ],
            ..ChannelSetupManifest::default()
        }),
        ..ChannelAdapterManifest::default()
    };
    let err = manifest.validate().expect_err("duplicate auth flow");
    assert!(err.contains("duplicate auth flow"));
}

#[test]
fn manifest_helpers_build_common_channel_setting_shapes() {
    let setting = channel_enum_setting("session_scope", ["user", "thread"]);
    assert_eq!(setting.key, "session_scope");
    assert_eq!(setting.options, vec!["user", "thread"]);

    let target = channel_setting_target("workspace_id");
    assert_eq!(target.kind, ChannelConfigTargetKind::ChannelSetting);
    assert_eq!(target.name, "workspace_id");

    let options = config_field_options([("auto", "Auto approve"), ("pending", "Pending")]);
    assert_eq!(options[0].value, "auto");
    assert_eq!(options[1].label.as_deref(), Some("Pending"));

    let field = max_inbound_text_chars_field("cap inbound text");
    assert_eq!(field.key, "max_inbound_text_chars");
    assert_eq!(field.field_type, "number");
    assert!(field.advanced);
    assert_eq!(
        field.target.as_ref().map(|target| target.name.as_str()),
        Some("max_inbound_text_chars")
    );
}

#[test]
fn setting_helpers_parse_reused_json_shapes() {
    let settings = serde_json::json!({
        "token_env": "TOKEN",
        "poll_interval_ms": 250,
        "enabled": true,
        "limit": 42
    });
    let settings = settings.as_object().expect("object");

    assert_eq!(
        required_non_empty_setting(settings, "token_env", "missing", "invalid").unwrap(),
        "TOKEN"
    );
    assert_eq!(
        optional_non_empty_setting(settings, "missing_optional", "invalid").unwrap(),
        None
    );
    assert_eq!(
        u64_setting_with_min(settings.get("poll_interval_ms"), 1_000, 100, "invalid").unwrap(),
        250
    );
    assert!(
        u64_setting_with_min(settings.get("poll_interval_ms"), 1_000, 500, "too small").is_err()
    );
    assert!(optional_bool_setting(settings.get("enabled"), false, "invalid").unwrap());
    assert_eq!(
        positive_usize_setting(
            settings.get("limit"),
            DEFAULT_MAX_INBOUND_TEXT_CHARS,
            "invalid",
            "large"
        )
        .unwrap(),
        42
    );
    assert_eq!(
        session_scope_setting(
            Some(&serde_json::json!("thread")),
            ChannelSessionScope::User,
            &[ChannelSessionScope::User, ChannelSessionScope::Thread],
            "invalid type",
            "invalid value"
        )
        .unwrap(),
        ChannelSessionScope::Thread
    );
    assert!(
        session_scope_setting(
            Some(&serde_json::json!("room")),
            ChannelSessionScope::User,
            &[ChannelSessionScope::User, ChannelSessionScope::Thread],
            "invalid type",
            "invalid value"
        )
        .is_err()
    );
    assert_eq!(
        optional_session_scope_setting(
            None,
            &[ChannelSessionScope::User, ChannelSessionScope::Room],
            "invalid type",
            "invalid value"
        )
        .unwrap(),
        None
    );
    assert_eq!(
        string_enum_setting(
            Some(&serde_json::json!("blue")),
            "red",
            |raw| match raw {
                "blue" => Some("blue"),
                _ => None,
            },
            "invalid type",
            "invalid value",
        )
        .unwrap(),
        "blue"
    );
    assert_eq!(
        string_enum_setting::<&str>(None, "red", |_| None, "invalid type", "invalid value")
            .unwrap(),
        "red"
    );
    assert!(
        string_enum_setting(
            Some(&serde_json::json!(42)),
            "red",
            |_| Some("blue"),
            "invalid type",
            "invalid value",
        )
        .is_err()
    );
}
