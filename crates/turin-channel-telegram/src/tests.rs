use super::*;

fn config() -> TelegramChannelDriverConfig {
    TelegramChannelDriverConfig {
        base_url: DEFAULT_BASE_URL.to_string(),
        workspace_id: "telegram".to_string(),
        chat_ids: vec!["-10012345".to_string()],
        accept_all_chats: false,
        token: "token".to_string(),
        poll_timeout_seconds: 30,
        poll_interval: Duration::from_millis(250),
        max_updates_per_poll: 25,
        max_inbound_text_chars: DEFAULT_MAX_INBOUND_TEXT_CHARS,
        start_from_latest: false,
        ignore_bot_messages: true,
        respond_mode: TelegramRespondMode::All,
        session_scope: ChannelSessionScope::User,
        session_scope_dm: None,
        session_scope_group: None,
        session_scope_channel: None,
        stream_mode: ChannelStreamMode::Off,
        stream_thinking: false,
        persist_thinking: false,
    }
}

fn driver() -> TelegramChannelDriver {
    let (_tx, rx) = watch::channel(false);
    TelegramChannelDriver::from_config("telegram-runtime", config(), rx).unwrap()
}

fn sample_event_with_message_id(message_id: i64) -> InboundEvent {
    let key = ChannelConversationKey {
        channel: ChannelKind::new("telegram"),
        workspace_id: "telegram".into(),
        room_id: Some("-10012345".into()),
        thread_id: "-10012345".into(),
        user_id: Some("user-1".into()),
    };
    let mut metadata = serde_json::Map::new();
    metadata.insert(
        "telegram_message_id".to_string(),
        serde_json::json!(message_id),
    );
    InboundEvent {
        conversation: key.clone(),
        message: ChannelMessageRef {
            conversation: key,
            message_id: format!("m-{message_id}"),
        },
        user: ChannelUser {
            id: "user-1".into(),
            display_name: Some("User One".into()),
            username: Some("user1".into()),
        },
        session_scope: ChannelSessionScope::User,
        text: "hello".into(),
        attachments: vec![],
        metadata,
    }
}

#[test]
fn normalize_uses_chat_id_as_default_thread() {
    let driver = driver();
    let update = TelegramUpdate {
        update_id: 1,
        message: Some(TelegramMessage {
            message_id: 99,
            chat: TelegramChat {
                id: -10012345,
                chat_type: "supergroup".to_string(),
                title: Some("Ops".to_string()),
                username: None,
                first_name: None,
            },
            from: Some(TelegramUser {
                id: 7,
                is_bot: Some(false),
                first_name: Some("Ava".to_string()),
                last_name: Some("Stone".to_string()),
                username: Some("ava".to_string()),
            }),
            sender_chat: None,
            text: Some("hello".to_string()),
            caption: None,
            entities: Vec::new(),
            caption_entities: Vec::new(),
            message_thread_id: None,
            reply_to_message: None,
            ..Default::default()
        }),
        channel_post: None,
    };

    let event = driver.normalize_update(update).expect("normalized event");
    assert_eq!(event.conversation.channel, ChannelKind::new("telegram"));
    assert_eq!(event.conversation.room_id.as_deref(), Some("-10012345"));
    assert_eq!(event.conversation.thread_id, "-10012345");
    assert_eq!(event.user.display_name.as_deref(), Some("Ava Stone"));
}

#[test]
fn normalize_uses_topic_thread_id_when_present() {
    let driver = driver();
    let update = TelegramUpdate {
        update_id: 2,
        message: Some(TelegramMessage {
            message_id: 100,
            chat: TelegramChat {
                id: -10012345,
                chat_type: "supergroup".to_string(),
                title: Some("Ops".to_string()),
                username: None,
                first_name: None,
            },
            from: Some(TelegramUser {
                id: 8,
                is_bot: Some(false),
                first_name: Some("Mia".to_string()),
                last_name: None,
                username: Some("mia".to_string()),
            }),
            sender_chat: None,
            text: Some("topic ping".to_string()),
            caption: None,
            entities: Vec::new(),
            caption_entities: Vec::new(),
            message_thread_id: Some(444),
            reply_to_message: None,
            ..Default::default()
        }),
        channel_post: None,
    };

    let event = driver.normalize_update(update).expect("normalized event");
    assert_eq!(event.conversation.thread_id, "444");
    assert_eq!(event.metadata["telegram_message_thread_id"], 444);
}

#[test]
fn normalize_thread_scope_shares_topic_across_users() {
    let mut config = config();
    config.session_scope = ChannelSessionScope::Thread;
    let (_tx, rx) = watch::channel(false);
    let driver = TelegramChannelDriver::from_config("telegram-runtime", config, rx).unwrap();
    let update = TelegramUpdate {
        update_id: 2,
        message: Some(TelegramMessage {
            message_id: 100,
            chat: TelegramChat {
                id: -10012345,
                chat_type: "supergroup".to_string(),
                title: Some("Ops".to_string()),
                username: None,
                first_name: None,
            },
            from: Some(TelegramUser {
                id: 8,
                is_bot: Some(false),
                first_name: Some("Mia".to_string()),
                last_name: None,
                username: Some("mia".to_string()),
            }),
            sender_chat: None,
            text: Some("topic ping".to_string()),
            caption: None,
            entities: Vec::new(),
            caption_entities: Vec::new(),
            message_thread_id: Some(444),
            reply_to_message: None,
            ..Default::default()
        }),
        channel_post: None,
    };

    let event = driver.normalize_update(update).expect("normalized event");
    assert_eq!(event.session_scope, ChannelSessionScope::Thread);
    assert_eq!(event.conversation.thread_id, "444");
    assert_eq!(event.conversation.user_id, None);
}

#[test]
fn normalize_room_scope_collapses_topics_and_users() {
    let mut config = config();
    config.session_scope = ChannelSessionScope::Room;
    let (_tx, rx) = watch::channel(false);
    let driver = TelegramChannelDriver::from_config("telegram-runtime", config, rx).unwrap();
    let update = TelegramUpdate {
        update_id: 2,
        message: Some(TelegramMessage {
            message_id: 100,
            chat: TelegramChat {
                id: -10012345,
                chat_type: "supergroup".to_string(),
                title: Some("Ops".to_string()),
                username: None,
                first_name: None,
            },
            from: Some(TelegramUser {
                id: 8,
                is_bot: Some(false),
                first_name: Some("Mia".to_string()),
                last_name: None,
                username: Some("mia".to_string()),
            }),
            sender_chat: None,
            text: Some("topic ping".to_string()),
            caption: None,
            entities: Vec::new(),
            caption_entities: Vec::new(),
            message_thread_id: Some(444),
            reply_to_message: None,
            ..Default::default()
        }),
        channel_post: None,
    };

    let event = driver.normalize_update(update).expect("normalized event");
    assert_eq!(event.session_scope, ChannelSessionScope::Room);
    assert_eq!(event.conversation.thread_id, "-10012345");
    assert_eq!(event.conversation.user_id, None);
}

#[test]
fn normalize_ignores_bot_messages() {
    let driver = driver();
    let update = TelegramUpdate {
        update_id: 3,
        message: Some(TelegramMessage {
            message_id: 101,
            chat: TelegramChat {
                id: -10012345,
                chat_type: "supergroup".to_string(),
                title: Some("Ops".to_string()),
                username: None,
                first_name: None,
            },
            from: Some(TelegramUser {
                id: 9,
                is_bot: Some(true),
                first_name: Some("Bot".to_string()),
                last_name: None,
                username: Some("bot".to_string()),
            }),
            sender_chat: None,
            text: Some("ignore me".to_string()),
            caption: None,
            entities: Vec::new(),
            caption_entities: Vec::new(),
            message_thread_id: None,
            reply_to_message: None,
            ..Default::default()
        }),
        channel_post: None,
    };

    assert!(driver.normalize_update(update).is_none());
}

#[test]
fn normalize_accepts_updates_from_any_configured_chat_id() {
    let mut config = config();
    config.chat_ids.push("-10099999".to_string());
    let (_tx, rx) = watch::channel(false);
    let driver = TelegramChannelDriver::from_config("telegram-runtime", config, rx).unwrap();
    let update = TelegramUpdate {
        update_id: 4,
        message: Some(TelegramMessage {
            message_id: 102,
            chat: TelegramChat {
                id: -10099999,
                chat_type: "supergroup".to_string(),
                title: Some("Second Ops".to_string()),
                username: None,
                first_name: None,
            },
            from: Some(TelegramUser {
                id: 10,
                is_bot: Some(false),
                first_name: Some("Rei".to_string()),
                last_name: None,
                username: Some("rei".to_string()),
            }),
            sender_chat: None,
            text: Some("hello second room".to_string()),
            caption: None,
            entities: Vec::new(),
            caption_entities: Vec::new(),
            message_thread_id: None,
            reply_to_message: None,
            ..Default::default()
        }),
        channel_post: None,
    };

    let event = driver.normalize_update(update).expect("normalized event");
    assert_eq!(event.conversation.room_id.as_deref(), Some("-10099999"));
    assert_eq!(event.conversation.thread_id, "-10099999");
}

#[test]
fn normalize_mentions_only_requires_explicit_bot_mention_in_groups() {
    let mut config = config();
    config.respond_mode = TelegramRespondMode::Mentions;
    let (_tx, rx) = watch::channel(false);
    let mut driver = TelegramChannelDriver::from_config("telegram-runtime", config, rx).unwrap();
    driver.bot_identity = Some(TelegramBotIdentity {
        id: 42,
        username: Some("turin_bot".to_string()),
    });

    let update_without_mention = TelegramUpdate {
        update_id: 5,
        message: Some(TelegramMessage {
            message_id: 103,
            chat: TelegramChat {
                id: -10012345,
                chat_type: "supergroup".to_string(),
                title: Some("Ops".to_string()),
                username: None,
                first_name: None,
            },
            from: Some(TelegramUser {
                id: 11,
                is_bot: Some(false),
                first_name: Some("Nora".to_string()),
                last_name: None,
                username: Some("nora".to_string()),
            }),
            sender_chat: None,
            text: Some("hello there".to_string()),
            caption: None,
            entities: Vec::new(),
            caption_entities: Vec::new(),
            message_thread_id: None,
            reply_to_message: None,
            ..Default::default()
        }),
        channel_post: None,
    };
    assert!(driver.normalize_update(update_without_mention).is_none());

    let update_with_mention = TelegramUpdate {
        update_id: 6,
        message: Some(TelegramMessage {
            message_id: 104,
            chat: TelegramChat {
                id: -10012345,
                chat_type: "supergroup".to_string(),
                title: Some("Ops".to_string()),
                username: None,
                first_name: None,
            },
            from: Some(TelegramUser {
                id: 11,
                is_bot: Some(false),
                first_name: Some("Nora".to_string()),
                last_name: None,
                username: Some("nora".to_string()),
            }),
            sender_chat: None,
            text: Some("@turin_bot hello there".to_string()),
            caption: None,
            entities: vec![TelegramMessageEntity {
                kind: "mention".to_string(),
                offset: 0,
                length: 10,
                user: None,
            }],
            caption_entities: Vec::new(),
            message_thread_id: None,
            reply_to_message: None,
            ..Default::default()
        }),
        channel_post: None,
    };
    assert!(driver.normalize_update(update_with_mention).is_some());
}

#[test]
fn normalize_replies_mode_accepts_replies_to_the_bot() {
    let mut config = config();
    config.respond_mode = TelegramRespondMode::Replies;
    let (_tx, rx) = watch::channel(false);
    let mut driver = TelegramChannelDriver::from_config("telegram-runtime", config, rx).unwrap();
    driver.bot_identity = Some(TelegramBotIdentity {
        id: 42,
        username: Some("turin_bot".to_string()),
    });

    let update = TelegramUpdate {
        update_id: 7,
        message: Some(TelegramMessage {
            message_id: 105,
            chat: TelegramChat {
                id: -10012345,
                chat_type: "supergroup".to_string(),
                title: Some("Ops".to_string()),
                username: None,
                first_name: None,
            },
            from: Some(TelegramUser {
                id: 12,
                is_bot: Some(false),
                first_name: Some("Ira".to_string()),
                last_name: None,
                username: Some("ira".to_string()),
            }),
            sender_chat: None,
            text: Some("following up".to_string()),
            caption: None,
            entities: Vec::new(),
            caption_entities: Vec::new(),
            message_thread_id: None,
            reply_to_message: Some(Box::new(TelegramMessage {
                message_id: 1000,
                chat: TelegramChat {
                    id: -10012345,
                    chat_type: "supergroup".to_string(),
                    title: Some("Ops".to_string()),
                    username: None,
                    first_name: None,
                },
                from: Some(TelegramUser {
                    id: 42,
                    is_bot: Some(true),
                    first_name: Some("Turin".to_string()),
                    last_name: None,
                    username: Some("turin_bot".to_string()),
                }),
                sender_chat: None,
                text: Some("prior answer".to_string()),
                caption: None,
                entities: Vec::new(),
                caption_entities: Vec::new(),
                message_thread_id: None,
                reply_to_message: None,
                ..Default::default()
            })),
            ..Default::default()
        }),
        channel_post: None,
    };

    assert!(driver.normalize_update(update).is_some());
}

#[test]
fn normalize_mentions_or_replies_accepts_addressed_bot_commands_in_groups() {
    let mut config = config();
    config.respond_mode = TelegramRespondMode::MentionsOrReplies;
    let (_tx, rx) = watch::channel(false);
    let mut driver = TelegramChannelDriver::from_config("telegram-runtime", config, rx).unwrap();
    driver.bot_identity = Some(TelegramBotIdentity {
        id: 42,
        username: Some("turin_bot".to_string()),
    });

    let update = TelegramUpdate {
        update_id: 8,
        message: Some(TelegramMessage {
            message_id: 106,
            chat: TelegramChat {
                id: -10012345,
                chat_type: "supergroup".to_string(),
                title: Some("Ops".to_string()),
                username: None,
                first_name: None,
            },
            from: Some(TelegramUser {
                id: 13,
                is_bot: Some(false),
                first_name: Some("Rin".to_string()),
                last_name: None,
                username: Some("rin".to_string()),
            }),
            sender_chat: None,
            text: Some("/start@turin_bot".to_string()),
            caption: None,
            entities: vec![TelegramMessageEntity {
                kind: "bot_command".to_string(),
                offset: 0,
                length: 16,
                user: None,
            }],
            caption_entities: Vec::new(),
            message_thread_id: None,
            reply_to_message: None,
            ..Default::default()
        }),
        channel_post: None,
    };

    assert!(driver.normalize_update(update).is_some());
}

#[test]
fn adapter_manifest_exposes_telegram_enum_settings() {
    let manifest = adapter_manifest();
    assert_eq!(manifest.kind, "telegram");
    manifest.validate().expect("valid manifest");
    assert_eq!(
        manifest
            .enum_setting("session_scope")
            .expect("session scope setting")
            .options,
        vec!["user", "thread", "room"]
    );
    assert_eq!(
        manifest
            .enum_setting("respond_mode")
            .expect("respond mode setting")
            .options,
        vec!["all", "mentions", "replies", "mentions_or_replies"]
    );
}

#[test]
fn enrich_outbound_defaults_to_replying_to_source_message() {
    let driver = driver();
    let event = sample_event_with_message_id(42);
    let enriched = driver.enrich_outbound_for_event(&event, OutboundMessage::text("reply"));
    assert_eq!(enriched.metadata["telegram_reply_to_message_id"], 42);
}

#[test]
fn enrich_outbound_keeps_explicit_reply_override() {
    let driver = driver();
    let event = sample_event_with_message_id(42);
    let mut outbound = OutboundMessage::text("reply");
    outbound.metadata.insert(
        "telegram_reply_to_message_id".to_string(),
        serde_json::json!(7),
    );
    let enriched = driver.enrich_outbound_for_event(&event, outbound);
    assert_eq!(enriched.metadata["telegram_reply_to_message_id"], 7);
}

#[test]
fn enrich_outbound_allows_clearing_default_reply_target() {
    let driver = driver();
    let event = sample_event_with_message_id(42);
    let mut outbound = OutboundMessage::text("reply");
    outbound.metadata.insert(
        "telegram_reply_to_message_id".to_string(),
        serde_json::Value::Null,
    );
    let enriched = driver.enrich_outbound_for_event(&event, outbound);
    assert_eq!(
        enriched.metadata.get("telegram_reply_to_message_id"),
        Some(&serde_json::Value::Null)
    );
}

#[test]
fn outbound_batches_split_long_messages_and_keep_thread() {
    let long_text = "x".repeat(TELEGRAM_MESSAGE_MAX_LEN + 200);
    let payloads = telegram_batches_from_message(
        "-10012345",
        Some(555),
        &OutboundMessage {
            blocks: vec![MessageBlock::Text { text: long_text }],
            ..OutboundMessage::default()
        },
    )
    .expect("render telegram payloads");

    assert!(payloads.len() >= 2);
    assert!(payloads.iter().all(|payload| {
        payload["text"]
            .as_str()
            .map(|text| text.chars().count() <= TELEGRAM_MESSAGE_MAX_LEN)
            .unwrap_or(false)
    }));
    assert!(
        payloads
            .iter()
            .all(|payload| payload["message_thread_id"] == 555)
    );
}

#[test]
fn code_blocks_render_as_html_with_parse_mode() {
    let payloads = telegram_batches_from_message(
        "-10012345",
        None,
        &OutboundMessage {
            blocks: vec![MessageBlock::CodeBlock {
                language: Some("rust".to_string()),
                code: "fn main() { println!(\"hi\"); }".to_string(),
            }],
            ..OutboundMessage::default()
        },
    )
    .expect("render telegram payloads");

    assert_eq!(payloads.len(), 1);
    assert_eq!(payloads[0]["parse_mode"], "HTML");
    assert!(
        payloads[0]["text"]
            .as_str()
            .is_some_and(|text| text.contains("<pre>") && text.contains("fn main()")),
        "payload should render Telegram HTML code block: {}",
        payloads[0]
    );
}

#[test]
fn text_messages_render_markdown_as_html_by_default() {
    let payloads = telegram_batches_from_message(
        "-10012345",
        None,
        &OutboundMessage::text(
            "# Heading\n\n**bold** and `code`\n\n- first\n- second\n\n[site](https://example.com)",
        ),
    )
    .expect("render telegram payloads");

    assert_eq!(payloads.len(), 1);
    assert_eq!(payloads[0]["parse_mode"], "HTML");
    let text = payloads[0]["text"]
        .as_str()
        .expect("telegram text should be a string");
    assert!(text.contains("<b>Heading</b>"), "payload text: {text}");
    assert!(text.contains("<b>bold</b>"), "payload text: {text}");
    assert!(text.contains("<code>code</code>"), "payload text: {text}");
    assert!(text.contains("• first"), "payload text: {text}");
    assert!(
        text.contains("<a href=\"https://example.com\">site</a>"),
        "payload text: {text}"
    );
}

#[test]
fn markdown_tables_render_as_preformatted_blocks() {
    let payloads = telegram_batches_from_message(
        "-10012345",
        None,
        &OutboundMessage::text("| Name | Value |\n| --- | --- |\n| alpha | 1 |\n| beta | 22 |"),
    )
    .expect("render telegram payloads");

    assert_eq!(payloads.len(), 1);
    assert_eq!(payloads[0]["parse_mode"], "HTML");
    let text = payloads[0]["text"]
        .as_str()
        .expect("telegram text should be a string");
    assert!(text.contains("<pre>"), "payload text: {text}");
    assert!(text.contains("| Name  | Value |"), "payload text: {text}");
    assert!(text.contains("| alpha | 1     |"), "payload text: {text}");
    assert!(text.contains("| beta  | 22    |"), "payload text: {text}");
}

#[test]
fn stream_preview_can_include_thinking_sections() {
    let preview = render_stream_preview("Partial answer", Some("Reasoning step"));
    assert!(preview.contains("Thinking…"));
    assert!(preview.contains("Reasoning step"));
    assert!(preview.contains("Reply"));
    assert!(preview.contains("Partial answer"));
}

#[test]
fn stream_preview_returns_text_only_when_no_thinking_is_present() {
    let preview = render_stream_preview("Partial answer", None);
    assert_eq!(preview, "Partial answer");
}

#[test]
fn final_message_can_include_persisted_thinking() {
    let mut message = OutboundMessage::text("Final answer");
    message.metadata.insert(
        "channel_final_thinking".to_string(),
        serde_json::json!("Step 1\nStep 2"),
    );

    let payloads = telegram_batches_from_message("-10012345", None, &message)
        .expect("render telegram payloads");
    let text = payloads[0]["text"].as_str().expect("telegram text payload");
    assert!(text.contains("Thinking"));
    assert!(text.contains("<pre>"));
    assert!(text.contains("Step 1"));
    assert!(text.contains("Reply"));
    assert!(text.contains("Final answer"));
}

#[test]
fn telegram_api_error_recognizes_not_modified_edit_failures() {
    let error = TelegramApiError {
        code: "telegram_edit_message_failed".to_string(),
        message: "Telegram editMessageText request failed with 400: Bad Request: message is not modified: specified new message content and reply markup are exactly the same as a current content and reply markup of the message".to_string(),
        retriable: false,
        retry_after: None,
    };
    assert!(error.is_message_not_modified());
}

#[test]
fn config_supports_chat_lists_and_telegram_stream_settings() {
    unsafe {
        std::env::set_var("TELEGRAM_BOT_TOKEN", "token");
    }
    let config = TelegramChannelDriverConfig::from_settings(
        &serde_json::json!({
            "token_env": "TELEGRAM_BOT_TOKEN",
            "chat_ids": [498502840, -10012345],
            "respond_mode": "mentions_or_replies",
            "stream_mode": "block",
            "stream_thinking": true,
            "persist_thinking": true
        }),
        false,
    )
    .expect("telegram config should parse");

    assert_eq!(config.chat_ids, vec!["498502840", "-10012345"]);
    assert_eq!(config.respond_mode, TelegramRespondMode::MentionsOrReplies);
    assert_eq!(config.session_scope_dm, None);
    assert_eq!(config.session_scope_group, None);
    assert_eq!(config.session_scope_channel, None);
    assert_eq!(
        config.max_inbound_text_chars,
        DEFAULT_MAX_INBOUND_TEXT_CHARS
    );
    assert!(config.stream_thinking);
    assert!(config.persist_thinking);
}

#[test]
fn private_chats_can_override_session_scope() {
    let config = TelegramChannelDriverConfig {
        base_url: DEFAULT_BASE_URL.to_string(),
        workspace_id: "telegram".to_string(),
        chat_ids: vec!["498502840".to_string()],
        accept_all_chats: false,
        token: "token".to_string(),
        poll_timeout_seconds: 30,
        poll_interval: Duration::from_millis(250),
        max_updates_per_poll: 25,
        max_inbound_text_chars: DEFAULT_MAX_INBOUND_TEXT_CHARS,
        start_from_latest: true,
        ignore_bot_messages: true,
        respond_mode: TelegramRespondMode::MentionsOrReplies,
        session_scope: ChannelSessionScope::User,
        session_scope_dm: Some(ChannelSessionScope::Room),
        session_scope_group: None,
        session_scope_channel: None,
        stream_mode: ChannelStreamMode::Off,
        stream_thinking: false,
        persist_thinking: false,
    };
    let chat = TelegramChat {
        id: 498502840,
        title: None,
        username: Some("jthum".to_string()),
        first_name: Some("Jay".to_string()),
        chat_type: "private".to_string(),
    };

    assert_eq!(
        effective_telegram_session_scope(&config, &chat),
        ChannelSessionScope::Room
    );
}

#[test]
fn validate_settings_rejects_deprecated_session_scope_aliases() {
    let error = validate_settings(
        &serde_json::json!({
            "token_env": "TELEGRAM_BOT_TOKEN",
            "chat_id": "498502840",
            "dm_session_scope": "room"
        }),
        false,
    )
    .expect_err("deprecated alias should fail");

    assert!(error.to_string().contains("session_scope_dm"));
}

#[test]
fn validate_settings_does_not_require_live_token_env() {
    validate_settings(
        &serde_json::json!({
            "token_env": "TELEGRAM_TOKEN_NOT_SET_FOR_VALIDATION",
            "chat_ids": [498502840],
            "respond_mode": "mentions_or_replies"
        }),
        false,
    )
    .expect("settings validation should not require the token env var to exist");
}

#[test]
fn validate_settings_allows_missing_chat_ids_when_unconfigured_chats_are_enabled() {
    validate_settings(
        &serde_json::json!({
            "token_env": "TELEGRAM_TOKEN_NOT_SET_FOR_VALIDATION",
            "respond_mode": "mentions_or_replies"
        }),
        true,
    )
    .expect("discovery mode should allow telegram channels without explicit chat ids");
}

#[test]
fn validate_settings_rejects_invalid_session_scope() {
    let error = validate_settings(
        &serde_json::json!({
            "token_env": "TELEGRAM_TOKEN_NOT_SET_FOR_VALIDATION",
            "chat_ids": [498502840],
            "session_scope": "guild"
        }),
        false,
    )
    .expect_err("invalid session scope rejected");
    assert!(error.to_string().contains("session_scope"));
}

#[test]
fn metadata_can_set_reply_target_and_disable_notification() {
    let mut message = OutboundMessage::text("hello");
    message.metadata.insert(
        "telegram_reply_to_message_id".to_string(),
        serde_json::json!(77),
    );
    message.metadata.insert(
        "telegram_disable_notification".to_string(),
        serde_json::json!(true),
    );

    let payloads = telegram_batches_from_message("-10012345", None, &message)
        .expect("render telegram payloads");
    assert_eq!(payloads[0]["reply_to_message_id"], 77);
    assert_eq!(payloads[0]["allow_sending_without_reply"], true);
    assert_eq!(payloads[0]["disable_notification"], true);
}

#[test]
fn telegram_format_plain_disables_html_rendering_for_code_blocks() {
    let mut message = OutboundMessage {
        blocks: vec![MessageBlock::CodeBlock {
            language: Some("rust".to_string()),
            code: "fn main() {}".to_string(),
        }],
        ..OutboundMessage::default()
    };
    message
        .metadata
        .insert("telegram_format".to_string(), serde_json::json!("plain"));

    let payloads = telegram_batches_from_message("-10012345", None, &message)
        .expect("render telegram payloads");

    assert_eq!(payloads.len(), 1);
    assert!(payloads[0].get("parse_mode").is_none());
    assert_eq!(payloads[0]["text"], "```rust\nfn main() {}\n```");
}
