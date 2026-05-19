use super::*;

#[test]
fn parse_transport_mode_defaults_to_gateway() {
    assert_eq!(
        parse_transport_mode(None).expect("default transport should parse"),
        DiscordTransportMode::Gateway
    );
}

#[test]
fn parse_transport_mode_accepts_polling() {
    assert_eq!(
        parse_transport_mode(Some("polling")).expect("polling transport should parse"),
        DiscordTransportMode::Polling
    );
}

#[test]
fn parse_transport_mode_rejects_invalid_value() {
    let error = parse_transport_mode(Some("unknown")).expect_err("transport should fail");
    assert!(error.to_string().contains("Invalid Discord transport"));
}

#[test]
fn validate_settings_rejects_small_poll_interval() {
    let error = validate_settings(&serde_json::json!({
        "token_env": "DISCORD_TOKEN",
        "channel_id": "123",
        "poll_interval_ms": 10
    }))
    .expect_err("too-small poll interval should fail");
    assert!(error.to_string().contains("poll_interval_ms"));
}

#[test]
fn validate_settings_rejects_zero_gateway_intents() {
    let error = validate_settings(&serde_json::json!({
        "token_env": "DISCORD_TOKEN",
        "channel_id": "123",
        "gateway_intents": 0
    }))
    .expect_err("zero gateway intents should fail");
    assert!(error.to_string().contains("gateway_intents"));
}

#[test]
fn validate_settings_rejects_unsupported_room_session_scope() {
    let error = validate_settings(&serde_json::json!({
        "token_env": "DISCORD_TOKEN",
        "channel_id": "123",
        "session_scope": "room"
    }))
    .expect_err("room scope rejected for discord");
    assert!(error.to_string().contains("session_scope"));
}

#[test]
fn render_outbound_preserves_code_blocks() {
    let batch = render_outbound_messages(OutboundMessage {
        blocks: vec![
            MessageBlock::Text {
                text: "summary".to_string(),
            },
            MessageBlock::CodeBlock {
                language: Some("rust".to_string()),
                code: "fn main() {}".to_string(),
            },
        ],
        ..OutboundMessage::default()
    });
    assert_eq!(batch.len(), 1);
    let output = batch[0]
        .content
        .as_ref()
        .expect("first message should contain content");
    assert!(output.contains("summary"));
    assert!(output.contains("```rust"));
}

#[test]
fn render_outbound_includes_embeds_and_components() {
    let batch = render_outbound_messages(OutboundMessage {
        blocks: vec![MessageBlock::Text {
            text: "summary".to_string(),
        }],
        embeds: vec![serde_json::json!({ "title": "Build Summary" })],
        components: vec![serde_json::json!({ "type": 1, "components": [] })],
        ..OutboundMessage::default()
    });
    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0].embeds.len(), 1);
    assert_eq!(batch[0].components.len(), 1);
}

#[test]
fn render_outbound_splits_long_content() {
    let long_text = "a".repeat(DISCORD_CONTENT_MAX_LEN + 200);
    let batch = render_outbound_messages(OutboundMessage {
        blocks: vec![MessageBlock::Text { text: long_text }],
        ..OutboundMessage::default()
    });
    assert!(
        batch.len() >= 2,
        "long payload should be split into multiple outbound messages"
    );
    assert!(batch.iter().all(|entry| {
        entry
            .content
            .as_ref()
            .map(|text| text.chars().count())
            .unwrap_or(0)
            <= DISCORD_CONTENT_MAX_LEN
    }));
}

#[test]
fn normalize_ignores_bot_messages() {
    let config = DiscordChannelDriverConfig {
        base_url: DEFAULT_BASE_URL.to_string(),
        gateway_url: DEFAULT_GATEWAY_URL.to_string(),
        transport_mode: DiscordTransportMode::Gateway,
        gateway_intents: DEFAULT_GATEWAY_INTENTS,
        workspace_id: "discord".to_string(),
        room_id: None,
        channel_id: "123".to_string(),
        token: "token".to_string(),
        poll_interval: Duration::from_millis(250),
        max_messages_per_poll: 10,
        max_inbound_text_chars: DEFAULT_MAX_INBOUND_TEXT_CHARS,
        start_from_latest: true,
        ignore_bot_messages: true,
        session_scope: ChannelSessionScope::User,
    };
    let (_tx, rx) = watch::channel(false);
    let mut driver = DiscordChannelDriver {
        channel_runtime_id: "discord-runtime".to_string(),
        config,
        client: reqwest::Client::new(),
        shutdown_rx: rx,
        backlog: VecDeque::new(),
        last_seen_message_id: None,
        initialized: false,
        gateway: None,
        last_gateway_seq: None,
        gateway_session_id: None,
        resume_gateway_url: None,
        seen_message_ids: VecDeque::new(),
        seen_message_set: HashSet::new(),
        reconnect_attempts: 0,
    };
    let message = DiscordMessage {
        id: "1".to_string(),
        channel_id: "123".to_string(),
        guild_id: Some("guild".to_string()),
        content: "hello".to_string(),
        author: DiscordAuthor {
            id: "bot".to_string(),
            username: "bot".to_string(),
            global_name: None,
            bot: Some(true),
        },
        attachments: Vec::new(),
    };
    assert!(driver.normalize_message(message).is_none());
}

#[test]
fn normalize_dedupes_message_ids() {
    let config = DiscordChannelDriverConfig {
        base_url: DEFAULT_BASE_URL.to_string(),
        gateway_url: DEFAULT_GATEWAY_URL.to_string(),
        transport_mode: DiscordTransportMode::Gateway,
        gateway_intents: DEFAULT_GATEWAY_INTENTS,
        workspace_id: "discord".to_string(),
        room_id: None,
        channel_id: "123".to_string(),
        token: "token".to_string(),
        poll_interval: Duration::from_millis(250),
        max_messages_per_poll: 10,
        max_inbound_text_chars: DEFAULT_MAX_INBOUND_TEXT_CHARS,
        start_from_latest: true,
        ignore_bot_messages: false,
        session_scope: ChannelSessionScope::User,
    };
    let (_tx, rx) = watch::channel(false);
    let mut driver = DiscordChannelDriver {
        channel_runtime_id: "discord-runtime".to_string(),
        config,
        client: reqwest::Client::new(),
        shutdown_rx: rx,
        backlog: VecDeque::new(),
        last_seen_message_id: None,
        initialized: false,
        gateway: None,
        last_gateway_seq: None,
        gateway_session_id: None,
        resume_gateway_url: None,
        seen_message_ids: VecDeque::new(),
        seen_message_set: HashSet::new(),
        reconnect_attempts: 0,
    };
    let message = DiscordMessage {
        id: "dup".to_string(),
        channel_id: "123".to_string(),
        guild_id: Some("guild".to_string()),
        content: "hello".to_string(),
        author: DiscordAuthor {
            id: "user".to_string(),
            username: "user".to_string(),
            global_name: None,
            bot: Some(false),
        },
        attachments: Vec::new(),
    };

    assert!(driver.normalize_message(message.clone()).is_some());
    assert!(driver.normalize_message(message).is_none());
}

#[test]
fn normalize_thread_scope_shares_channel_across_users() {
    let config = DiscordChannelDriverConfig {
        base_url: DEFAULT_BASE_URL.to_string(),
        gateway_url: DEFAULT_GATEWAY_URL.to_string(),
        transport_mode: DiscordTransportMode::Gateway,
        gateway_intents: DEFAULT_GATEWAY_INTENTS,
        workspace_id: "discord".to_string(),
        room_id: None,
        channel_id: "123".to_string(),
        token: "token".to_string(),
        poll_interval: Duration::from_millis(250),
        max_messages_per_poll: 10,
        max_inbound_text_chars: DEFAULT_MAX_INBOUND_TEXT_CHARS,
        start_from_latest: true,
        ignore_bot_messages: false,
        session_scope: ChannelSessionScope::Thread,
    };
    let (_tx, rx) = watch::channel(false);
    let mut driver = DiscordChannelDriver {
        channel_runtime_id: "discord-runtime".to_string(),
        config,
        client: reqwest::Client::new(),
        shutdown_rx: rx,
        backlog: VecDeque::new(),
        last_seen_message_id: None,
        initialized: false,
        gateway: None,
        last_gateway_seq: None,
        gateway_session_id: None,
        resume_gateway_url: None,
        seen_message_ids: VecDeque::new(),
        seen_message_set: HashSet::new(),
        reconnect_attempts: 0,
    };
    let message = DiscordMessage {
        id: "1".to_string(),
        channel_id: "123".to_string(),
        guild_id: Some("guild".to_string()),
        content: "hello".to_string(),
        author: DiscordAuthor {
            id: "user".to_string(),
            username: "user".to_string(),
            global_name: None,
            bot: Some(false),
        },
        attachments: Vec::new(),
    };

    let event = driver.normalize_message(message).expect("normalized event");
    assert_eq!(event.session_scope, ChannelSessionScope::Thread);
    assert_eq!(event.conversation.thread_id, "123");
    assert_eq!(event.conversation.user_id, None);
}

#[test]
fn adapter_manifest_exposes_discord_enum_settings() {
    let manifest = adapter_manifest();
    assert_eq!(manifest.kind, "discord");
    manifest.validate().expect("valid manifest");
    assert_eq!(
        manifest
            .enum_setting("session_scope")
            .expect("session scope setting")
            .options,
        vec!["user", "thread"]
    );
}

#[test]
fn parse_settings_accepts_custom_max_inbound_text_chars() {
    let parsed = parse_settings(&serde_json::json!({
        "token_env": "DISCORD_BOT_TOKEN",
        "channel_id": "123",
        "max_inbound_text_chars": 2048
    }))
    .expect("settings parse");
    assert_eq!(parsed.max_inbound_text_chars, 2048);
}
