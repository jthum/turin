use super::*;
use turin_types::layout::{
    DEFAULT_BOOTSTRAP_CONFIG_PATH, DEFAULT_BOOTSTRAP_DAEMON_ENDPOINT_PATH, DEFAULT_UI_PROFILES_PATH,
};

#[test]
fn connection_options_default_to_local_config() {
    let options = ConnectionOptions {
        config_path: None,
        endpoint: None,
        remote_url: None,
        auth_token: None,
        auth_token_env: None,
        profile: None,
        profiles_file: None,
        suppress_profile_resolution: false,
    };

    match options.to_spec().expect("spec") {
        ConnectionSpec::LocalConfig { config_path } => {
            assert_eq!(config_path, PathBuf::from(DEFAULT_BOOTSTRAP_CONFIG_PATH));
        }
        other => panic!("unexpected spec: {other:?}"),
    }
}

#[test]
fn missing_ui_worklists_start_as_empty_collections() {
    let items = empty_ui_worklist("release");

    assert_eq!(items.worklist_id, "release");
    assert!(items.items.is_empty());
}

#[test]
fn connection_options_require_remote_auth_material() {
    let options = ConnectionOptions {
        config_path: None,
        endpoint: None,
        remote_url: Some("http://example.test".to_string()),
        auth_token: None,
        auth_token_env: None,
        profile: None,
        profiles_file: None,
        suppress_profile_resolution: false,
    };

    let err = options.to_spec().expect_err("missing auth should error");
    assert!(
        err.to_string()
            .contains("--remote-url requires either --auth-token or --auth-token-env")
    );
}

#[test]
fn connection_options_apply_profile_overrides() {
    let temp = tempfile::NamedTempFile::new().expect("temp profile file");
    fs::write(
        temp.path(),
        r#"
[profiles.lab]
remote_url = "http://example.test"
auth_token_env = "TURIN_REMOTE_TOKEN"
"#,
    )
    .expect("write profile file");

    let options = ConnectionOptions {
        config_path: None,
        endpoint: None,
        remote_url: None,
        auth_token: None,
        auth_token_env: None,
        profile: Some("lab".to_string()),
        profiles_file: Some(temp.path().to_path_buf()),
        suppress_profile_resolution: false,
    };

    match options.to_spec().expect("spec") {
        ConnectionSpec::RemoteEnv {
            base_url,
            auth_token_env,
        } => {
            assert_eq!(base_url, "http://example.test");
            assert_eq!(auth_token_env, "TURIN_REMOTE_TOKEN");
        }
        other => panic!("unexpected spec: {other:?}"),
    }
}

#[test]
fn connection_options_can_save_and_delete_profiles() {
    let temp = tempfile::tempdir().expect("temp dir");
    let profiles_path = temp.path().join(DEFAULT_UI_PROFILES_PATH);

    let remote = ConnectionOptions {
        config_path: None,
        endpoint: None,
        remote_url: Some("http://example.test".to_string()),
        auth_token: None,
        auth_token_env: Some("TURIN_REMOTE_TOKEN".to_string()),
        profile: None,
        profiles_file: Some(profiles_path.clone()),
        suppress_profile_resolution: false,
    };

    let catalog = remote
        .save_profile("lab", true)
        .expect("save remote profile");
    assert_eq!(catalog.default_profile(), Some("lab"));
    assert_eq!(catalog.profiles().len(), 1);

    let local = ConnectionOptions {
        config_path: Some(PathBuf::from("turin-dev.toml")),
        endpoint: None,
        remote_url: None,
        auth_token: None,
        auth_token_env: None,
        profile: None,
        profiles_file: Some(profiles_path.clone()),
        suppress_profile_resolution: false,
    };

    let catalog = local
        .save_profile("local", false)
        .expect("save local profile");
    assert_eq!(catalog.default_profile(), Some("lab"));
    assert_eq!(catalog.profiles().len(), 2);

    let deleted = local.delete_profile("lab").expect("delete profile");
    assert_eq!(deleted.default_profile(), Some("local"));
    assert_eq!(deleted.profiles().len(), 1);
    assert_eq!(deleted.profiles()[0].name, "local");

    let raw = fs::read_to_string(&profiles_path).expect("read saved file");
    assert!(raw.contains("default_profile = \"local\""));
    assert!(raw.contains("[profiles.local]"));
}

#[test]
fn connection_options_can_duplicate_and_rename_profiles() {
    let temp = tempfile::tempdir().expect("temp dir");
    let profiles_path = temp.path().join(DEFAULT_UI_PROFILES_PATH);
    fs::create_dir_all(profiles_path.parent().expect("profiles parent")).expect("profiles dir");
    fs::write(
        &profiles_path,
        r#"
default_profile = "lab"

[profiles.lab]
remote_url = "http://example.test"
auth_token_env = "TURIN_REMOTE_TOKEN"
"#,
    )
    .expect("write initial profiles");

    let options = ConnectionOptions {
        config_path: None,
        endpoint: None,
        remote_url: None,
        auth_token: None,
        auth_token_env: None,
        profile: None,
        profiles_file: Some(profiles_path.clone()),
        suppress_profile_resolution: false,
    };

    let duplicated = options
        .duplicate_profile("lab", "lab-copy", false)
        .expect("duplicate profile");
    assert_eq!(duplicated.profiles().len(), 2);
    assert_eq!(duplicated.default_profile(), Some("lab"));

    let renamed = options
        .rename_profile("lab-copy", "lab-stage", true)
        .expect("rename profile");
    assert_eq!(renamed.profiles().len(), 2);
    assert_eq!(renamed.default_profile(), Some("lab-stage"));
    assert!(
        renamed
            .profiles()
            .iter()
            .any(|profile| profile.name == "lab-stage")
    );
}

#[test]
fn connection_profile_drafts_roundtrip_through_profile_storage() {
    let temp = tempfile::tempdir().expect("temp dir");
    let profiles_path = temp.path().join(DEFAULT_UI_PROFILES_PATH);
    let options = ConnectionOptions {
        config_path: None,
        endpoint: None,
        remote_url: None,
        auth_token: None,
        auth_token_env: None,
        profile: None,
        profiles_file: Some(profiles_path),
        suppress_profile_resolution: false,
    };

    let draft = ConnectionProfileDraft {
        kind: ConnectionProfileKind::Remote,
        target: "http://example.test:9324".to_string(),
        auth_mode: ConnectionProfileDraftAuthMode::TokenEnv,
        auth_value: "TURIN_REMOTE_TOKEN".to_string(),
    };

    options
        .save_profile_draft("lab", &draft, true)
        .expect("save draft");
    let loaded = options.load_profile_draft("lab").expect("load draft");

    assert_eq!(loaded, draft);
}

#[test]
fn connection_options_can_materialize_and_resolve_remote_drafts() {
    let options = ConnectionOptions {
        config_path: Some(PathBuf::from(DEFAULT_BOOTSTRAP_CONFIG_PATH)),
        endpoint: None,
        remote_url: None,
        auth_token: None,
        auth_token_env: None,
        profile: Some("ignored".to_string()),
        profiles_file: Some(PathBuf::from(DEFAULT_UI_PROFILES_PATH)),
        suppress_profile_resolution: false,
    };
    let draft = ConnectionProfileDraft {
        kind: ConnectionProfileKind::Remote,
        target: "https://turin.example.com".to_string(),
        auth_mode: ConnectionProfileDraftAuthMode::TokenEnv,
        auth_value: "TURIN_REMOTE_TOKEN".to_string(),
    };

    let materialized = options
        .connection_options_for_draft(&draft)
        .expect("materialize draft");
    assert_eq!(
        materialized.remote_url.as_deref(),
        Some("https://turin.example.com")
    );
    assert_eq!(
        materialized.auth_token_env.as_deref(),
        Some("TURIN_REMOTE_TOKEN")
    );
    assert!(materialized.profile.is_none());
    assert_eq!(
        materialized.profiles_file.as_deref(),
        Some(Path::new(DEFAULT_UI_PROFILES_PATH))
    );

    match options.draft_to_spec(&draft).expect("draft spec") {
        ConnectionSpec::RemoteEnv {
            base_url,
            auth_token_env,
        } => {
            assert_eq!(base_url, "https://turin.example.com");
            assert_eq!(auth_token_env, "TURIN_REMOTE_TOKEN");
        }
        other => panic!("unexpected spec: {other:?}"),
    }
}

#[test]
fn remote_profile_draft_validation_reports_target_and_auth_errors() {
    let validation = ConnectionProfileDraft {
        kind: ConnectionProfileKind::Remote,
        target: "ftp://bad host".to_string(),
        auth_mode: ConnectionProfileDraftAuthMode::TokenEnv,
        auth_value: "bad-token-name".to_string(),
    }
    .validate();

    assert!(!validation.is_valid());
    assert_eq!(
        validation.target_error.as_deref(),
        Some("Remote base URLs cannot contain whitespace")
    );
    assert_eq!(
        validation.auth_error.as_deref(),
        Some("Env var names may only contain letters, numbers, and underscores")
    );
}

#[test]
fn remote_inline_token_draft_validation_reports_plaintext_notice() {
    let validation = ConnectionProfileDraft {
        kind: ConnectionProfileKind::Remote,
        target: "https://turin.example.com".to_string(),
        auth_mode: ConnectionProfileDraftAuthMode::InlineToken,
        auth_value: "secret".to_string(),
    }
    .validate();

    assert!(validation.is_valid());
    assert_eq!(
        validation.auth_notice.as_deref(),
        Some("Inline tokens are stored in plaintext in the profiles file")
    );
}

#[test]
fn local_config_draft_validation_reports_default_path_notice() {
    let validation = ConnectionProfileDraft {
        kind: ConnectionProfileKind::LocalConfig,
        target: String::new(),
        auth_mode: ConnectionProfileDraftAuthMode::None,
        auth_value: String::new(),
    }
    .validate();

    assert!(validation.is_valid());
    let expected = format!(
        "Blank config path will default to {}",
        DEFAULT_BOOTSTRAP_CONFIG_PATH
    );
    assert_eq!(validation.target_notice.as_deref(), Some(expected.as_str()));
}

#[test]
fn recent_connection_drafts_are_deduped_and_bounded() {
    let mut history = ConnectionDraftHistory::default();
    let remote = ConnectionProfileDraft {
        kind: ConnectionProfileKind::Remote,
        target: "https://turin.example.com".to_string(),
        auth_mode: ConnectionProfileDraftAuthMode::TokenEnv,
        auth_value: "TURIN_REMOTE_TOKEN".to_string(),
    };

    history.record_success(&ConnectionProfileDraft {
        kind: ConnectionProfileKind::LocalConfig,
        target: String::new(),
        auth_mode: ConnectionProfileDraftAuthMode::None,
        auth_value: String::new(),
    });
    history.record_success(&ConnectionProfileDraft {
        kind: ConnectionProfileKind::LocalEndpoint,
        target: DEFAULT_BOOTSTRAP_DAEMON_ENDPOINT_PATH.to_string(),
        auth_mode: ConnectionProfileDraftAuthMode::None,
        auth_value: String::new(),
    });
    history.record_success(&remote);
    history.record_success(&ConnectionProfileDraft {
        kind: ConnectionProfileKind::Remote,
        target: "https://turin-backup.example.com".to_string(),
        auth_mode: ConnectionProfileDraftAuthMode::InlineToken,
        auth_value: "secret".to_string(),
    });
    history.record_success(&ConnectionProfileDraft {
        kind: ConnectionProfileKind::LocalConfig,
        target: "turin-stage.toml".to_string(),
        auth_mode: ConnectionProfileDraftAuthMode::None,
        auth_value: String::new(),
    });
    history.record_success(&remote);

    assert_eq!(history.drafts().len(), MAX_RECENT_CONNECTION_DRAFTS);
    assert_eq!(history.latest(), Some(&remote));
    assert_eq!(
        history
            .drafts()
            .iter()
            .filter(|draft| draft.target == "https://turin.example.com")
            .count(),
        1
    );
}

#[test]
fn profile_draft_diff_redacts_inline_tokens() {
    let baseline = ConnectionProfileDraft {
        kind: ConnectionProfileKind::Remote,
        target: "https://turin.example.com".to_string(),
        auth_mode: ConnectionProfileDraftAuthMode::InlineToken,
        auth_value: "secret-a".to_string(),
    };
    let changed = ConnectionProfileDraft {
        auth_value: "secret-b".to_string(),
        ..baseline.clone()
    };

    let diff = changed.diff_against(&baseline);
    assert_eq!(diff.changed_field_names(), vec!["auth_value"]);
    assert_eq!(diff.changed_fields[0].draft_value, "<inline token set>");
    assert_eq!(
        diff.changed_fields[0].comparison_value,
        "<inline token set>"
    );
}

#[test]
fn remote_preflight_reports_missing_env_before_connect() {
    let options = ConnectionOptions {
        config_path: None,
        endpoint: None,
        remote_url: Some("https://turin.example.com".to_string()),
        auth_token: None,
        auth_token_env: Some("TURIN_UI_CORE_MISSING_TOKEN".to_string()),
        profile: None,
        profiles_file: None,
        suppress_profile_resolution: false,
    };

    let report = preflight_connection_blocking(&options);
    assert_eq!(report.outcome, ConnectionPreflightOutcome::Invalid);
    assert!(
        report
            .message
            .contains("Environment variable 'TURIN_UI_CORE_MISSING_TOKEN' is not set")
    );
}

#[test]
fn connection_profile_activity_book_tracks_success_and_failure() {
    let mut book = ConnectionProfileActivityBook::default();
    book.record_preflight_result("lab", true, "ready");
    book.record_connect_result("lab", false, "failed");
    book.record_connect_result("lab", true, "connected");

    let entry = book.entry("lab").expect("activity entry");
    assert_eq!(entry.preflight_count, 1);
    assert_eq!(entry.connect_count, 2);
    assert_eq!(entry.successful_connect_count, 1);
    assert_eq!(entry.failure_count, 1);
    assert_eq!(entry.last_connect_result.as_deref(), Some("connected"));
    assert_eq!(entry.last_preflight_result.as_deref(), Some("ready"));
}
