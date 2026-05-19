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
