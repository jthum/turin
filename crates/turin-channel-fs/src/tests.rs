use super::*;
use tempfile::tempdir;

fn sample_settings() -> serde_json::Value {
    serde_json::json!({
        "inbox_dir": "inbox",
        "outbox_dir": "outbox",
        "processed_dir": "processed",
        "failed_dir": "failed",
        "poll_interval_ms": 25,
    })
}

fn sample_event() -> serde_json::Value {
    serde_json::json!({
        "conversation": {
            "channel": "fs",
            "workspace_id": "workspace",
            "room_id": "room",
            "thread_id": "thread",
            "user_id": "user",
        },
        "message_id": "msg-1",
        "user": {
            "id": "user",
            "display_name": "User",
            "username": "user",
        },
        "text": "hello",
        "metadata": {
            "reset_session": true
        }
    })
}

#[tokio::test]
async fn driver_reads_inbound_and_moves_processed_file() {
    let dir = tempdir().unwrap();
    let channel_dir = dir.path().join("channel");
    tokio::fs::create_dir_all(channel_dir.join("inbox"))
        .await
        .unwrap();
    tokio::fs::write(
        channel_dir.join("inbox/in-1.json"),
        serde_json::to_string_pretty(&sample_event()).unwrap(),
    )
    .await
    .unwrap();

    let (_tx, rx) = watch::channel(false);
    let mut driver = FsChannelDriver::from_settings("fs", &channel_dir, &sample_settings(), rx)
        .await
        .unwrap();

    let event = driver.next_event().await.unwrap().unwrap();
    assert_eq!(event.text, "hello");
    assert!(channel_dir.join("processed/in-1.json").exists());
    assert!(!channel_dir.join("inbox/in-1.json").exists());
}

#[tokio::test]
async fn driver_preserves_existing_processed_file_on_name_collision() {
    let dir = tempdir().unwrap();
    let channel_dir = dir.path().join("channel");
    tokio::fs::create_dir_all(channel_dir.join("inbox"))
        .await
        .unwrap();
    tokio::fs::create_dir_all(channel_dir.join("processed"))
        .await
        .unwrap();
    tokio::fs::write(
        channel_dir.join("inbox/in-1.json"),
        serde_json::to_string_pretty(&sample_event()).unwrap(),
    )
    .await
    .unwrap();
    tokio::fs::write(channel_dir.join("processed/in-1.json"), "existing")
        .await
        .unwrap();

    let (_tx, rx) = watch::channel(false);
    let mut driver = FsChannelDriver::from_settings("fs", &channel_dir, &sample_settings(), rx)
        .await
        .unwrap();

    let event = driver.next_event().await.unwrap().unwrap();
    assert_eq!(event.text, "hello");
    assert_eq!(
        tokio::fs::read_to_string(channel_dir.join("processed/in-1.json"))
            .await
            .unwrap(),
        "existing"
    );

    let mut processed = tokio::fs::read_dir(channel_dir.join("processed"))
        .await
        .unwrap();
    let mut count = 0;
    while processed.next_entry().await.unwrap().is_some() {
        count += 1;
    }
    assert_eq!(count, 2);
}

#[tokio::test]
async fn driver_retries_parse_failures_before_marking_failed() {
    let dir = tempdir().unwrap();
    let channel_dir = dir.path().join("channel");
    tokio::fs::create_dir_all(channel_dir.join("inbox"))
        .await
        .unwrap();
    tokio::fs::write(channel_dir.join("inbox/in-1.json"), "{")
        .await
        .unwrap();

    let (tx, rx) = watch::channel(false);
    let mut driver = FsChannelDriver::from_settings("fs", &channel_dir, &sample_settings(), rx)
        .await
        .unwrap();

    let drive = tokio::spawn(async move { driver.next_event().await });
    tokio::time::sleep(Duration::from_millis(400)).await;
    tx.send(true).unwrap();

    let result = drive.await.unwrap().unwrap();
    assert!(result.is_none(), "shutdown should stop the driver");
    assert!(channel_dir.join("failed/in-1.json").exists());
}

#[tokio::test]
async fn recently_modified_reports_metadata_errors() {
    let dir = tempdir().unwrap();
    let missing = dir.path().join("missing.json");
    let err = path_is_recently_modified(&missing, Duration::from_millis(10))
        .await
        .expect_err("missing file metadata should be explicit");
    assert!(err.to_string().contains("Failed to inspect"));
}

#[cfg(unix)]
#[tokio::test]
async fn driver_ignores_symlink_inbox_messages() {
    let dir = tempdir().unwrap();
    let channel_dir = dir.path().join("channel");
    tokio::fs::create_dir_all(channel_dir.join("inbox"))
        .await
        .unwrap();
    let outside = dir.path().join("outside.json");
    tokio::fs::write(
        &outside,
        serde_json::to_string_pretty(&sample_event()).unwrap(),
    )
    .await
    .unwrap();
    std::os::unix::fs::symlink(&outside, channel_dir.join("inbox/link.json")).unwrap();

    let (tx, rx) = watch::channel(false);
    let mut driver = FsChannelDriver::from_settings("fs", &channel_dir, &sample_settings(), rx)
        .await
        .unwrap();

    let drive = tokio::spawn(async move { driver.next_event().await });
    tokio::time::sleep(Duration::from_millis(50)).await;
    tx.send(true).unwrap();

    let result = drive.await.unwrap().unwrap();
    assert!(result.is_none(), "symlink inbox entry should be ignored");
    assert!(channel_dir.join("inbox/link.json").exists());
    assert!(!channel_dir.join("processed/link.json").exists());
}

#[cfg(unix)]
#[tokio::test]
async fn driver_rejects_symlink_channel_directories() {
    let dir = tempdir().unwrap();
    let channel_dir = dir.path().join("channel");
    let real_inbox = dir.path().join("real-inbox");
    tokio::fs::create_dir_all(&real_inbox).await.unwrap();
    tokio::fs::create_dir_all(&channel_dir).await.unwrap();
    std::os::unix::fs::symlink(&real_inbox, channel_dir.join("inbox")).unwrap();

    let (_tx, rx) = watch::channel(false);
    let err = match FsChannelDriver::from_settings("fs", &channel_dir, &sample_settings(), rx).await
    {
        Ok(_) => panic!("symlink channel directories should be rejected"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("must not be a symlink"));
}

#[tokio::test]
async fn driver_recovers_if_message_becomes_valid_before_retry_limit() {
    let dir = tempdir().unwrap();
    let channel_dir = dir.path().join("channel");
    tokio::fs::create_dir_all(channel_dir.join("inbox"))
        .await
        .unwrap();
    tokio::fs::write(channel_dir.join("inbox/in-1.json"), "{")
        .await
        .unwrap();

    let (_tx, rx) = watch::channel(false);
    let mut driver = FsChannelDriver::from_settings("fs", &channel_dir, &sample_settings(), rx)
        .await
        .unwrap();

    let rewrite_path = channel_dir.join("inbox/in-1.json");
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(35)).await;
        let tmp_path = rewrite_path.with_extension("json.tmp");
        tokio::fs::write(
            &tmp_path,
            serde_json::to_string_pretty(&sample_event()).unwrap(),
        )
        .await
        .unwrap();
        tokio::fs::rename(&tmp_path, &rewrite_path).await.unwrap();
    });

    let event = driver.next_event().await.unwrap().unwrap();
    assert_eq!(event.text, "hello");
    assert!(channel_dir.join("processed/in-1.json").exists());
    assert!(!channel_dir.join("failed/in-1.json").exists());
}

#[tokio::test]
async fn driver_writes_outbound_messages() {
    let dir = tempdir().unwrap();
    let channel_dir = dir.path().join("channel");
    let (_tx, rx) = watch::channel(false);
    let mut driver = FsChannelDriver::from_settings("fs", &channel_dir, &sample_settings(), rx)
        .await
        .unwrap();

    let conversation = ChannelConversationKey {
        channel: ChannelKind::new("fs"),
        workspace_id: "workspace".into(),
        room_id: Some("room".into()),
        thread_id: "thread".into(),
        user_id: Some("user".into()),
    };

    driver
        .send(&conversation, OutboundMessage::text("pong"))
        .await
        .unwrap();

    let outbox = channel_dir.join("outbox");
    let mut files = tokio::fs::read_dir(&outbox).await.unwrap();
    let file = files.next_entry().await.unwrap().unwrap();
    let raw = tokio::fs::read_to_string(file.path()).await.unwrap();
    assert!(raw.contains("pong"));
}

#[cfg(unix)]
#[tokio::test]
async fn driver_rejects_symlink_outbox_on_send() {
    let dir = tempdir().unwrap();
    let channel_dir = dir.path().join("channel");
    let (_tx, rx) = watch::channel(false);
    let mut driver = FsChannelDriver::from_settings("fs", &channel_dir, &sample_settings(), rx)
        .await
        .unwrap();

    tokio::fs::remove_dir(channel_dir.join("outbox"))
        .await
        .unwrap();
    let real_outbox = dir.path().join("real-outbox");
    tokio::fs::create_dir_all(&real_outbox).await.unwrap();
    std::os::unix::fs::symlink(&real_outbox, channel_dir.join("outbox")).unwrap();

    let conversation = ChannelConversationKey {
        channel: ChannelKind::new("fs"),
        workspace_id: "workspace".into(),
        room_id: Some("room".into()),
        thread_id: "thread".into(),
        user_id: Some("user".into()),
    };

    let err = driver
        .send(&conversation, OutboundMessage::text("pong"))
        .await
        .expect_err("symlink outbox should be rejected before writing");
    assert!(err.to_string().contains("must not be a symlink"));
}

#[test]
fn validate_settings_rejects_too_small_poll_interval() {
    let dir = tempdir().unwrap();
    let channel_dir = dir.path().join("channel");
    let error = validate_settings(
        &channel_dir,
        &serde_json::json!({
            "poll_interval_ms": 1,
        }),
    )
    .expect_err("too-small poll interval should fail");
    assert!(error.to_string().contains("poll_interval_ms"));
}

#[test]
fn adapter_manifest_is_valid() {
    let manifest = adapter_manifest();
    assert_eq!(manifest.kind, "fs");
    manifest.validate().expect("valid manifest");
}
