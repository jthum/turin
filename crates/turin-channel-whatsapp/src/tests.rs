use super::*;
use std::collections::HashMap;

#[test]
fn adapter_manifest_is_valid() {
    let manifest = adapter_manifest();
    assert_eq!(manifest.kind, "whatsapp");
    assert!(manifest.runtime.capabilities.attachments);
    manifest.validate().expect("valid manifest");
    let setup = manifest.setup.expect("setup manifest");
    assert_eq!(setup.auth_flows.len(), 1);
    assert_eq!(setup.auth_flows[0].id, DEFAULT_AUTH_FLOW_ID);
}

#[test]
fn validate_pair_code_requires_phone_number() {
    let err = validate_pair_code_fields(None, Some("ABCD1234")).expect_err("invalid");
    assert!(err.to_string().contains("pair_code_phone_number"));
}

#[test]
fn parse_settings_resolves_runtime_default_store() {
    let temp = tempfile::tempdir().expect("tempdir");
    let config = parse_settings(&json!({}), Some(temp.path())).expect("settings");
    assert_eq!(
        config.session_store_path,
        temp.path().join(DEFAULT_RUNTIME_STORE_BASENAME)
    );
    assert_eq!(config.account_mode, WhatsAppAccountMode::Personal);
    assert_eq!(
        config.trigger_prefix.as_deref(),
        Some(DEFAULT_PERSONAL_TRIGGER_PREFIX)
    );
    assert_eq!(
        config.max_inbound_text_chars,
        DEFAULT_MAX_INBOUND_TEXT_CHARS
    );
    assert_eq!(config.media_dir, temp.path().join("media"));
}

#[test]
fn dedicated_mode_does_not_force_trigger_prefix() {
    let temp = tempfile::tempdir().expect("tempdir");
    let config =
        parse_settings(&json!({"account_mode": "dedicated"}), Some(temp.path())).expect("settings");
    assert_eq!(config.account_mode, WhatsAppAccountMode::Dedicated);
    assert_eq!(config.trigger_prefix, None);
}

#[test]
fn inbound_text_requires_prefix_for_personal_mode() {
    assert_eq!(
        inbound_text(
            "/turin status",
            WhatsAppAccountMode::Personal,
            Some(DEFAULT_PERSONAL_TRIGGER_PREFIX)
        ),
        Some("status".to_string())
    );
    assert_eq!(
        inbound_text("status", WhatsAppAccountMode::Personal, Some("/turin")),
        None
    );
    assert_eq!(
        inbound_text("status", WhatsAppAccountMode::Dedicated, None),
        Some("status".to_string())
    );
}

#[test]
fn banned_chats_override_allowed_chats() {
    let allowed = vec![
        "15551234567@s.whatsapp.net".to_string(),
        "120363123456789@g.us".to_string(),
    ];
    let banned = vec!["15551234567".to_string()];
    assert!(!chat_is_allowed(
        "15551234567@s.whatsapp.net",
        &allowed,
        &banned
    ));
    assert!(chat_is_allowed("120363123456789@g.us", &allowed, &banned));
}

#[test]
fn poll_complete_returns_store_path_and_clears_pair_code_fields() {
    let temp = tempfile::tempdir().expect("tempdir");
    let state_path = temp.path().join("state.json");
    let writer = AuthStateWriter::new(state_path.clone());
    writer
        .write(&WhatsAppAuthState {
            phase: WhatsAppAuthPhase::Complete,
            display: ChannelAuthFlowDisplay::default(),
            message: Some("done".to_string()),
        })
        .expect("state written");

    let response = poll_auth_flow(&ChannelAuthFlowPollRequest {
        flow_id: DEFAULT_AUTH_FLOW_ID.to_string(),
        session: serde_json::to_value(WhatsAppAuthSession {
            ticket: "t".to_string(),
            state_path,
            store_path: PathBuf::from("/tmp/whatsapp.db"),
            phone_number: Some("15551234567".to_string()),
            custom_code: Some("ABCD1234".to_string()),
        })
        .expect("session"),
        current_settings: json!({}),
    })
    .expect("poll response");

    match response {
        ChannelAuthFlowPollResponse::Complete { values, .. } => {
            let values_by_name: HashMap<_, _> = values
                .into_iter()
                .map(|value| (value.target.name, value.value))
                .collect();
            assert_eq!(
                values_by_name.get("session_store_path"),
                Some(&Value::String("/tmp/whatsapp.db".to_string()))
            );
            assert_eq!(
                values_by_name.get("pair_code_phone_number"),
                Some(&Value::Null)
            );
            assert_eq!(
                values_by_name.get("pair_code_custom_code"),
                Some(&Value::Null)
            );
        }
        other => panic!("unexpected response: {other:?}"),
    }
}
