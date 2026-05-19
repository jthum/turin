use super::*;

#[test]
fn adapter_manifest_is_valid() {
    let manifest = adapter_manifest();
    assert_eq!(manifest.kind, "rocketchat");
    manifest.validate().expect("valid manifest");
}

#[test]
fn parse_settings_accepts_room_id_and_defaults() {
    let settings = serde_json::json!({
        "token_env": "ROCKETCHAT_AUTH_TOKEN",
        "user_id": "rbAXPnMktTFbNpwtJ",
        "room_id": "GENERAL123"
    });
    let parsed = parse_settings(&settings, false).expect("settings parse");
    assert_eq!(parsed.base_url, DEFAULT_BASE_URL);
    assert_eq!(parsed.websocket_url, "ws://localhost:3000/websocket");
    assert_eq!(parsed.transport_mode, RocketChatTransportMode::Realtime);
    assert_eq!(parsed.workspace_id, "rocketchat");
    assert!(!parsed.accept_all_rooms);
    assert_eq!(parsed.max_messages_per_poll, DEFAULT_MAX_MESSAGES_PER_POLL);
    assert_eq!(
        parsed.max_inbound_text_chars,
        DEFAULT_MAX_INBOUND_TEXT_CHARS
    );
    assert_eq!(parsed.respond_mode, RocketChatRespondMode::Mentions);
    assert_eq!(parsed.session_scope, ChannelSessionScope::Thread);
    assert_eq!(parsed.session_scope_dm, None);
    assert_eq!(parsed.session_scope_group, None);
    assert_eq!(parsed.session_scope_channel, None);
    assert_eq!(parsed.reply_mode, RocketChatReplyMode::Thread);
    assert_eq!(parsed.stream_mode, ChannelStreamMode::Typing);
    assert!(!parsed.persist_thinking);
}

#[test]
fn parse_settings_requires_room_reference() {
    let settings = serde_json::json!({
        "token_env": "ROCKETCHAT_AUTH_TOKEN",
        "user_id": "rbAXPnMktTFbNpwtJ"
    });
    let error = parse_settings(&settings, false).expect_err("missing room should fail");
    assert!(error.to_string().contains("room_id"));
}

#[test]
fn parse_settings_accepts_dynamic_room_discovery_when_pairing_enabled() {
    let settings = serde_json::json!({
        "token_env": "ROCKETCHAT_AUTH_TOKEN",
        "user_id": "rbAXPnMktTFbNpwtJ"
    });
    let parsed = parse_settings(&settings, true).expect("settings parse");
    assert!(parsed.accept_all_rooms);
    assert!(parsed.room_id.is_none());
    assert!(parsed.room_name.is_none());
}

#[test]
fn render_outbound_preserves_code_blocks() {
    let rendered = render_text_blocks(&[
        MessageBlock::Text {
            text: "hello".to_string(),
        },
        MessageBlock::CodeBlock {
            language: Some("rust".to_string()),
            code: "fn main() {}".to_string(),
        },
    ]);
    assert!(rendered.contains("hello"));
    assert!(rendered.contains("```rust"));
}

#[test]
fn user_scope_uses_room_id_for_top_level_messages() {
    let config = RocketChatChannelDriverConfig {
        base_url: DEFAULT_BASE_URL.to_string(),
        websocket_url: default_websocket_url(DEFAULT_BASE_URL),
        transport_mode: RocketChatTransportMode::Realtime,
        workspace_id: "rocketchat".to_string(),
        accept_all_rooms: false,
        room_id: Some("room1".to_string()),
        room_name: None,
        user_id: "bot".to_string(),
        token: "token".to_string(),
        poll_interval: Duration::from_millis(DEFAULT_POLL_INTERVAL_MS),
        max_messages_per_poll: DEFAULT_MAX_MESSAGES_PER_POLL,
        max_inbound_text_chars: DEFAULT_MAX_INBOUND_TEXT_CHARS,
        start_from_latest: true,
        ignore_bot_messages: true,
        respond_mode: RocketChatRespondMode::Mentions,
        session_scope: ChannelSessionScope::User,
        session_scope_dm: None,
        session_scope_group: None,
        session_scope_channel: None,
        reply_mode: RocketChatReplyMode::Thread,
        stream_mode: ChannelStreamMode::Typing,
        persist_thinking: false,
    };
    let driver = RocketChatChannelDriver {
        channel_id: "rocketchat".to_string(),
        client: Client::new(),
        config,
        shutdown_rx: watch::channel(false).1,
        bot_username: None,
        bot_display_name: None,
        rooms: HashMap::from([(
            "room1".to_string(),
            RocketChatRoomState {
                room: RocketChatResolvedRoom {
                    id: "room1".to_string(),
                    room_type: RocketChatRoomType::Channel,
                    name: Some("general".to_string()),
                    friendly_name: Some("General".to_string()),
                    usernames: vec![],
                    latest_message: None,
                    latest_message_id: None,
                    latest_message_ts: None,
                },
                cursor_ts: None,
            },
        )]),
        ws_stream: None,
        realtime_subscribed_room_ids: HashSet::new(),
        active_thread_keys: HashSet::new(),
        backlog: VecDeque::new(),
        seen_message_ids: HashSet::new(),
        seen_message_order: VecDeque::new(),
        recent_sent_message_ids: HashSet::new(),
        recent_sent_message_order: VecDeque::new(),
        rooms_updated_since: None,
        last_room_refresh: None,
        last_typing_at: HashMap::new(),
        last_realtime_activity_at: None,
        last_realtime_keepalive_at: None,
        next_realtime_request_id: 1,
    };
    let room = driver.rooms.get("room1").expect("room state").room.clone();
    let message = RocketChatMessage {
        id: "m1".to_string(),
        text: Some("hi".to_string()),
        ts: "2026-03-29T00:00:00.000Z".to_string(),
        user: Some(RocketChatMessageUser {
            id: "user1".to_string(),
            username: Some("alice".to_string()),
            name: Some("Alice".to_string()),
        }),
        kind: None,
        thread_root_id: None,
        mentions: vec![],
        attachments: vec![],
        file: None,
    };
    assert_eq!(
        driver.thread_id_for_message(&room, &message, ChannelSessionScope::User),
        "room1"
    );
}

#[test]
fn reset_transport_state_clears_realtime_subscriptions() {
    let config = RocketChatChannelDriverConfig {
        base_url: DEFAULT_BASE_URL.to_string(),
        websocket_url: default_websocket_url(DEFAULT_BASE_URL),
        transport_mode: RocketChatTransportMode::Realtime,
        workspace_id: "rocketchat".to_string(),
        accept_all_rooms: false,
        room_id: Some("room1".to_string()),
        room_name: None,
        user_id: "bot".to_string(),
        token: "token".to_string(),
        poll_interval: Duration::from_millis(DEFAULT_POLL_INTERVAL_MS),
        max_messages_per_poll: DEFAULT_MAX_MESSAGES_PER_POLL,
        max_inbound_text_chars: DEFAULT_MAX_INBOUND_TEXT_CHARS,
        start_from_latest: true,
        ignore_bot_messages: true,
        respond_mode: RocketChatRespondMode::Mentions,
        session_scope: ChannelSessionScope::User,
        session_scope_dm: None,
        session_scope_group: None,
        session_scope_channel: None,
        reply_mode: RocketChatReplyMode::Thread,
        stream_mode: ChannelStreamMode::Typing,
        persist_thinking: false,
    };
    let mut driver = RocketChatChannelDriver {
        channel_id: "rocketchat".to_string(),
        client: Client::new(),
        config,
        shutdown_rx: watch::channel(false).1,
        bot_username: None,
        bot_display_name: None,
        rooms: HashMap::new(),
        ws_stream: None,
        realtime_subscribed_room_ids: HashSet::from(["room1".to_string()]),
        active_thread_keys: HashSet::new(),
        backlog: VecDeque::new(),
        seen_message_ids: HashSet::new(),
        seen_message_order: VecDeque::new(),
        recent_sent_message_ids: HashSet::new(),
        recent_sent_message_order: VecDeque::new(),
        rooms_updated_since: Some("2026-03-29T17:12:01Z".to_string()),
        last_room_refresh: None,
        last_typing_at: HashMap::new(),
        last_realtime_activity_at: Some(Instant::now()),
        last_realtime_keepalive_at: Some(Instant::now()),
        next_realtime_request_id: 1,
    };

    driver.reset_transport_state().expect("transport reset");

    assert!(driver.ws_stream.is_none());
    assert!(driver.realtime_subscribed_room_ids.is_empty());
    assert!(driver.last_realtime_activity_at.is_none());
    assert!(driver.last_realtime_keepalive_at.is_none());
}

#[test]
fn mentions_mode_accepts_followups_in_active_turin_threads() {
    let config = RocketChatChannelDriverConfig {
        base_url: DEFAULT_BASE_URL.to_string(),
        websocket_url: default_websocket_url(DEFAULT_BASE_URL),
        transport_mode: RocketChatTransportMode::Realtime,
        workspace_id: "rocketchat".to_string(),
        accept_all_rooms: false,
        room_id: Some("room1".to_string()),
        room_name: None,
        user_id: "bot".to_string(),
        token: "token".to_string(),
        poll_interval: Duration::from_millis(DEFAULT_POLL_INTERVAL_MS),
        max_messages_per_poll: DEFAULT_MAX_MESSAGES_PER_POLL,
        max_inbound_text_chars: DEFAULT_MAX_INBOUND_TEXT_CHARS,
        start_from_latest: true,
        ignore_bot_messages: true,
        respond_mode: RocketChatRespondMode::Mentions,
        session_scope: ChannelSessionScope::Thread,
        session_scope_dm: None,
        session_scope_group: None,
        session_scope_channel: None,
        reply_mode: RocketChatReplyMode::Thread,
        stream_mode: ChannelStreamMode::Typing,
        persist_thinking: false,
    };
    let driver = RocketChatChannelDriver {
        channel_id: "rocketchat".to_string(),
        client: Client::new(),
        config,
        shutdown_rx: watch::channel(false).1,
        bot_username: None,
        bot_display_name: None,
        rooms: HashMap::new(),
        ws_stream: None,
        realtime_subscribed_room_ids: HashSet::new(),
        active_thread_keys: HashSet::from([active_thread_key("room1", "root-message")]),
        backlog: VecDeque::new(),
        seen_message_ids: HashSet::new(),
        seen_message_order: VecDeque::new(),
        recent_sent_message_ids: HashSet::new(),
        recent_sent_message_order: VecDeque::new(),
        rooms_updated_since: None,
        last_room_refresh: None,
        last_typing_at: HashMap::new(),
        last_realtime_activity_at: None,
        last_realtime_keepalive_at: None,
        next_realtime_request_id: 1,
    };
    let room = RocketChatResolvedRoom {
        id: "room1".to_string(),
        room_type: RocketChatRoomType::Channel,
        name: Some("general".to_string()),
        friendly_name: Some("General".to_string()),
        usernames: vec![],
        latest_message: None,
        latest_message_id: None,
        latest_message_ts: None,
    };
    let message = RocketChatMessage {
        id: "m2".to_string(),
        text: Some("follow up".to_string()),
        ts: "2026-03-29T00:00:00.000Z".to_string(),
        user: Some(RocketChatMessageUser {
            id: "user1".to_string(),
            username: Some("alice".to_string()),
            name: Some("Alice".to_string()),
        }),
        kind: None,
        thread_root_id: Some("root-message".to_string()),
        mentions: vec![],
        attachments: vec![],
        file: None,
    };

    assert!(driver.should_accept_message(&room, &message, message.user.as_ref().expect("user"),));
}

#[test]
fn direct_messages_can_override_session_scope() {
    let config = RocketChatChannelDriverConfig {
        base_url: DEFAULT_BASE_URL.to_string(),
        websocket_url: default_websocket_url(DEFAULT_BASE_URL),
        transport_mode: RocketChatTransportMode::Realtime,
        workspace_id: "rocketchat".to_string(),
        accept_all_rooms: false,
        room_id: Some("dm-room".to_string()),
        room_name: None,
        user_id: "bot".to_string(),
        token: "token".to_string(),
        poll_interval: Duration::from_millis(DEFAULT_POLL_INTERVAL_MS),
        max_messages_per_poll: DEFAULT_MAX_MESSAGES_PER_POLL,
        max_inbound_text_chars: DEFAULT_MAX_INBOUND_TEXT_CHARS,
        start_from_latest: true,
        ignore_bot_messages: true,
        respond_mode: RocketChatRespondMode::Mentions,
        session_scope: ChannelSessionScope::Thread,
        session_scope_dm: Some(ChannelSessionScope::Room),
        session_scope_group: None,
        session_scope_channel: None,
        reply_mode: RocketChatReplyMode::Thread,
        stream_mode: ChannelStreamMode::Typing,
        persist_thinking: false,
    };
    let driver = RocketChatChannelDriver {
        channel_id: "rocketchat".to_string(),
        client: Client::new(),
        config,
        shutdown_rx: watch::channel(false).1,
        bot_username: None,
        bot_display_name: None,
        rooms: HashMap::new(),
        ws_stream: None,
        realtime_subscribed_room_ids: HashSet::new(),
        active_thread_keys: HashSet::new(),
        backlog: VecDeque::new(),
        seen_message_ids: HashSet::new(),
        seen_message_order: VecDeque::new(),
        recent_sent_message_ids: HashSet::new(),
        recent_sent_message_order: VecDeque::new(),
        rooms_updated_since: None,
        last_room_refresh: None,
        last_typing_at: HashMap::new(),
        last_realtime_activity_at: None,
        last_realtime_keepalive_at: None,
        next_realtime_request_id: 1,
    };
    let room = RocketChatResolvedRoom {
        id: "dm-room".to_string(),
        room_type: RocketChatRoomType::DirectMessage,
        name: None,
        friendly_name: None,
        usernames: vec!["bot".to_string(), "alice".to_string()],
        latest_message: None,
        latest_message_id: None,
        latest_message_ts: None,
    };
    let message = RocketChatMessage {
        id: "m1".to_string(),
        text: Some("hi".to_string()),
        ts: "2026-03-29T00:00:00.000Z".to_string(),
        user: Some(RocketChatMessageUser {
            id: "user1".to_string(),
            username: Some("alice".to_string()),
            name: Some("Alice".to_string()),
        }),
        kind: None,
        thread_root_id: None,
        mentions: vec![],
        attachments: vec![],
        file: None,
    };

    assert_eq!(
        driver.effective_session_scope(&room),
        ChannelSessionScope::Room
    );
    assert_eq!(
        driver.thread_id_for_message(&room, &message, driver.effective_session_scope(&room)),
        "dm-room"
    );
}

#[test]
fn channel_reply_mode_downgrades_thread_scope_to_room_scope() {
    let config = RocketChatChannelDriverConfig {
        base_url: DEFAULT_BASE_URL.to_string(),
        websocket_url: default_websocket_url(DEFAULT_BASE_URL),
        transport_mode: RocketChatTransportMode::Realtime,
        workspace_id: "rocketchat".to_string(),
        accept_all_rooms: false,
        room_id: Some("room1".to_string()),
        room_name: None,
        user_id: "bot".to_string(),
        token: "token".to_string(),
        poll_interval: Duration::from_millis(DEFAULT_POLL_INTERVAL_MS),
        max_messages_per_poll: DEFAULT_MAX_MESSAGES_PER_POLL,
        max_inbound_text_chars: DEFAULT_MAX_INBOUND_TEXT_CHARS,
        start_from_latest: true,
        ignore_bot_messages: true,
        respond_mode: RocketChatRespondMode::Mentions,
        session_scope: ChannelSessionScope::Thread,
        session_scope_dm: None,
        session_scope_group: None,
        session_scope_channel: None,
        reply_mode: RocketChatReplyMode::Channel,
        stream_mode: ChannelStreamMode::Typing,
        persist_thinking: false,
    };
    let driver = RocketChatChannelDriver {
        channel_id: "rocketchat".to_string(),
        client: Client::new(),
        config,
        shutdown_rx: watch::channel(false).1,
        bot_username: None,
        bot_display_name: None,
        rooms: HashMap::new(),
        ws_stream: None,
        realtime_subscribed_room_ids: HashSet::new(),
        active_thread_keys: HashSet::new(),
        backlog: VecDeque::new(),
        seen_message_ids: HashSet::new(),
        seen_message_order: VecDeque::new(),
        recent_sent_message_ids: HashSet::new(),
        recent_sent_message_order: VecDeque::new(),
        rooms_updated_since: None,
        last_room_refresh: None,
        last_typing_at: HashMap::new(),
        last_realtime_activity_at: None,
        last_realtime_keepalive_at: None,
        next_realtime_request_id: 1,
    };
    let room = RocketChatResolvedRoom {
        id: "room1".to_string(),
        room_type: RocketChatRoomType::PrivateGroup,
        name: Some("turin".to_string()),
        friendly_name: Some("Turin".to_string()),
        usernames: vec![],
        latest_message: None,
        latest_message_id: None,
        latest_message_ts: None,
    };
    let message = RocketChatMessage {
        id: "m1".to_string(),
        text: Some("@nux hello".to_string()),
        ts: "2026-03-31T00:00:00.000Z".to_string(),
        user: Some(RocketChatMessageUser {
            id: "user1".to_string(),
            username: Some("alice".to_string()),
            name: Some("Alice".to_string()),
        }),
        kind: None,
        thread_root_id: None,
        mentions: vec![],
        attachments: vec![],
        file: None,
    };

    assert_eq!(
        driver.effective_session_scope(&room),
        ChannelSessionScope::Room
    );
    assert_eq!(
        driver.thread_id_for_message(&room, &message, driver.effective_session_scope(&room)),
        "room1"
    );
}

#[test]
fn validate_settings_rejects_deprecated_session_scope_aliases() {
    let error = validate_settings(
        &serde_json::json!({
            "token_env": "ROCKETCHAT_AUTH_TOKEN",
            "user_id": "rbAXPnMktTFbNpwtJ",
            "room_id": "GENERAL123",
            "dm_session_scope": "room"
        }),
        false,
    )
    .expect_err("deprecated alias should fail");

    assert!(error.to_string().contains("session_scope_dm"));
}

#[test]
fn resolve_reply_target_starts_thread_from_triggering_message() {
    let conversation = ChannelConversationKey {
        channel: ChannelKind::new("rocketchat"),
        workspace_id: "rocketchat".to_string(),
        room_id: Some("room1".to_string()),
        thread_id: "room1".to_string(),
        user_id: None,
    };
    let mut outbound = OutboundMessage::text("reply");
    outbound.metadata.insert(
        "rocketchat_reply_to_message_id".to_string(),
        serde_json::json!("message-42"),
    );

    let reply_target = resolve_reply_target(
        "room1",
        &conversation,
        &outbound,
        RocketChatReplyMode::Thread,
    );
    assert_eq!(reply_target.thread_id, Some("message-42"));
    assert!(!reply_target.show_in_channel);
}

#[test]
fn build_rocketchat_send_payload_uses_send_message_shape() {
    let payload = build_rocketchat_send_payload(
        "room1",
        "hello",
        RocketChatReplyTarget {
            thread_id: Some("message-42"),
            show_in_channel: true,
        },
        &[],
    );

    assert_eq!(payload["message"]["rid"], "room1");
    assert_eq!(payload["message"]["msg"], "hello");
    assert_eq!(payload["message"]["parseUrls"], false);
    assert_eq!(payload["message"]["tmid"], "message-42");
    assert_eq!(payload["message"]["tshow"], true);
    assert!(payload["message"].get("attachments").is_none());
    assert!(payload.get("roomId").is_none());
    assert!(payload.get("channel").is_none());
}

#[test]
fn render_rocketchat_message_wraps_markdown_tables_and_thinking() {
    let mut outbound =
        OutboundMessage::text("| Name | Score |\n| --- | --- |\n| Alice | 10 |\n| Bob | 9 |");
    outbound.metadata.insert(
        "channel_final_thinking".to_string(),
        serde_json::json!("brief reasoning"),
    );

    let rendered = render_rocketchat_message(&outbound, true);
    assert!(rendered.contains("Thinking:"));
    assert!(rendered.contains("```"));
    assert!(rendered.contains("| Alice | 10 |"));
}

#[test]
fn channel_reply_quote_renders_reply_context() {
    let mut outbound = OutboundMessage::text("reply");
    outbound.metadata.insert(
        "rocketchat_reply_to_label".to_string(),
        serde_json::json!("Jayadeep Thum (@jayadeep)"),
    );
    outbound.metadata.insert(
        "rocketchat_reply_to_message_link".to_string(),
        serde_json::json!("https://chat.example.com/group/turin?msg=m1"),
    );
    outbound.metadata.insert(
        "rocketchat_reply_to_excerpt".to_string(),
        serde_json::json!("Line one\nLine two\nLine three"),
    );

    let quoted = prepend_channel_reply_quote("reply", &outbound);
    assert_eq!(
        quoted,
        "> [Jayadeep Thum (@jayadeep)](https://chat.example.com/group/turin?msg=m1)\n> Line one\n> Line two\n> Line three\n\nreply"
    );
}

#[test]
fn mentions_mode_accepts_quoted_messages_from_recent_bot_replies() {
    let config = RocketChatChannelDriverConfig {
        base_url: DEFAULT_BASE_URL.to_string(),
        websocket_url: default_websocket_url(DEFAULT_BASE_URL),
        transport_mode: RocketChatTransportMode::Realtime,
        workspace_id: "rocketchat".to_string(),
        accept_all_rooms: false,
        room_id: Some("room1".to_string()),
        room_name: None,
        user_id: "bot".to_string(),
        token: "token".to_string(),
        poll_interval: Duration::from_millis(DEFAULT_POLL_INTERVAL_MS),
        max_messages_per_poll: DEFAULT_MAX_MESSAGES_PER_POLL,
        max_inbound_text_chars: DEFAULT_MAX_INBOUND_TEXT_CHARS,
        start_from_latest: true,
        ignore_bot_messages: true,
        respond_mode: RocketChatRespondMode::Mentions,
        session_scope: ChannelSessionScope::Thread,
        session_scope_dm: None,
        session_scope_group: None,
        session_scope_channel: None,
        reply_mode: RocketChatReplyMode::Channel,
        stream_mode: ChannelStreamMode::Typing,
        persist_thinking: false,
    };
    let driver = RocketChatChannelDriver {
        channel_id: "rocketchat".to_string(),
        client: Client::new(),
        config,
        shutdown_rx: watch::channel(false).1,
        bot_username: Some("turinbot".to_string()),
        bot_display_name: Some("Turin".to_string()),
        rooms: HashMap::new(),
        ws_stream: None,
        realtime_subscribed_room_ids: HashSet::new(),
        active_thread_keys: HashSet::new(),
        backlog: VecDeque::new(),
        seen_message_ids: HashSet::new(),
        seen_message_order: VecDeque::new(),
        recent_sent_message_ids: HashSet::from(["bot-message-1".to_string()]),
        recent_sent_message_order: VecDeque::from(["bot-message-1".to_string()]),
        rooms_updated_since: None,
        last_room_refresh: None,
        last_typing_at: HashMap::new(),
        last_realtime_activity_at: None,
        last_realtime_keepalive_at: None,
        next_realtime_request_id: 1,
    };
    let room = RocketChatResolvedRoom {
        id: "room1".to_string(),
        room_type: RocketChatRoomType::Channel,
        name: Some("general".to_string()),
        friendly_name: Some("General".to_string()),
        usernames: vec![],
        latest_message: None,
        latest_message_id: None,
        latest_message_ts: None,
    };
    let message = RocketChatMessage {
        id: "m2".to_string(),
        text: Some("follow up".to_string()),
        ts: "2026-03-30T00:00:00.000Z".to_string(),
        user: Some(RocketChatMessageUser {
            id: "user1".to_string(),
            username: Some("alice".to_string()),
            name: Some("Alice".to_string()),
        }),
        kind: None,
        thread_root_id: None,
        mentions: vec![],
        attachments: vec![RocketChatApiAttachment {
            text: Some("Earlier reply".to_string()),
            title: None,
            title_link: None,
            message_link: Some(
                "https://chat.example.com/channel/general?msg=bot-message-1".to_string(),
            ),
            author_name: Some("Turin".to_string()),
            image_url: None,
            audio_url: None,
            video_url: None,
        }],
        file: None,
    };

    assert!(driver.should_accept_message(&room, &message, message.user.as_ref().expect("user"),));
}

#[test]
fn build_rocketchat_message_link_matches_room_paths() {
    let channel_room = RocketChatResolvedRoom {
        id: "room1".to_string(),
        room_type: RocketChatRoomType::Channel,
        name: Some("general".to_string()),
        friendly_name: Some("General".to_string()),
        usernames: vec![],
        latest_message: None,
        latest_message_id: None,
        latest_message_ts: None,
    };
    assert_eq!(
        build_rocketchat_message_link(
            "https://chat.example.com",
            &channel_room,
            Some("nux"),
            "abc123"
        )
        .as_deref(),
        Some("https://chat.example.com/channel/general?msg=abc123")
    );

    let group_room = RocketChatResolvedRoom {
        id: "room2".to_string(),
        room_type: RocketChatRoomType::PrivateGroup,
        name: Some("turin".to_string()),
        friendly_name: Some("Turin".to_string()),
        usernames: vec![],
        latest_message: None,
        latest_message_id: None,
        latest_message_ts: None,
    };
    assert_eq!(
        build_rocketchat_message_link(
            "https://chat.example.com",
            &group_room,
            Some("nux"),
            "def456"
        )
        .as_deref(),
        Some("https://chat.example.com/group/turin?msg=def456")
    );

    let dm_room = RocketChatResolvedRoom {
        id: "room3".to_string(),
        room_type: RocketChatRoomType::DirectMessage,
        name: None,
        friendly_name: None,
        usernames: vec!["jayadeep".to_string(), "nux".to_string()],
        latest_message: None,
        latest_message_id: None,
        latest_message_ts: None,
    };
    assert_eq!(
        build_rocketchat_message_link("https://chat.example.com", &dm_room, Some("nux"), "ghi789")
            .as_deref(),
        Some("https://chat.example.com/direct/jayadeep?msg=ghi789")
    );
}

#[test]
fn default_websocket_url_tracks_base_url_scheme() {
    assert_eq!(
        default_websocket_url("https://chat.example.com"),
        "wss://chat.example.com/websocket"
    );
    assert_eq!(
        default_websocket_url("http://chat.example.com"),
        "ws://chat.example.com/websocket"
    );
}

#[test]
fn ddp_frame_deserializes_success_result_payloads() {
    let frame: RocketChatDdpFrame = serde_json::from_value(serde_json::json!({
        "msg": "result",
        "id": "turin-1",
        "result": {
            "id": "user-id",
            "token": "resume-token",
            "tokenExpires": { "$date": null },
            "type": "resume"
        }
    }))
    .expect("frame");

    assert_eq!(frame.msg.as_deref(), Some("result"));
    assert_eq!(frame.id.as_deref(), Some("turin-1"));
    assert!(login_result_error(&frame).is_none());
}

#[test]
fn rocketchat_message_accepts_ejson_timestamp() {
    let message: RocketChatMessage = serde_json::from_value(serde_json::json!({
        "_id": "message-id",
        "msg": "hello",
        "ts": { "$date": "2026-03-29T17:12:01.123Z" },
        "u": {
            "_id": "user-id",
            "username": "alice",
            "name": "Alice"
        },
        "mentions": [],
        "attachments": []
    }))
    .expect("message");

    assert_eq!(message.ts, "2026-03-29T17:12:01.123Z");
}

#[test]
fn rocketchat_room_info_accepts_ejson_timestamps() {
    let room: RocketChatRoomInfo = serde_json::from_value(serde_json::json!({
        "_id": "room-id",
        "t": "c",
        "_updatedAt": { "$date": "2026-03-29T17:12:01.123Z" },
        "lm": { "$date": "2026-03-29T17:10:00.000Z" }
    }))
    .expect("room");

    assert_eq!(room.updated_at.as_deref(), Some("2026-03-29T17:12:01.123Z"));
    assert_eq!(
        room.last_message_at.as_deref(),
        Some("2026-03-29T17:10:00Z")
    );
}
