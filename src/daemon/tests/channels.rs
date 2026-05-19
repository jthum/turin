use super::*;
use std::fs;
use std::sync::Arc;
use tempfile::tempdir;
use tokio::time::{Duration, Instant, sleep};

async fn wait_until(timeout: Duration, mut predicate: impl FnMut() -> bool, description: &str) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if predicate() {
            return;
        }
        sleep(Duration::from_millis(25)).await;
    }
    panic!("timed out waiting for {description}");
}

#[cfg(unix)]
#[tokio::test]
async fn external_channel_runner_process_is_supervised() {
    use std::os::unix::fs::PermissionsExt;

    let _env_guard = crate::test_support::env_lock().lock().await;
    let temp = tempdir().unwrap();
    let workspace_root = temp.path().join("workspace");
    fs::create_dir_all(workspace_root.join(".turin/runtime/channels")).unwrap();

    let runner = temp.path().join("fake-telegram-runner.sh");
    fs::write(
        &runner,
        "#!/bin/sh\nif [ \"$1\" = \"describe\" ]; then\n  printf '%s\\n' '{\"protocol_version\":2,\"kind\":\"telegram\"}'\n  exit 0\nfi\nif [ \"$1\" = \"run\" ]; then\n  sleep 30\n  exit 0\nfi\nif [ \"$1\" = \"validate-settings\" ]; then\n  exit 0\nfi\nif [ \"$1\" = \"setup-auth-flow-start\" ]; then\n  exit 1\nfi\nif [ \"$1\" = \"setup-auth-flow-poll\" ]; then\n  exit 1\nfi\nexit 0\n",
    )
    .unwrap();
    let mut perms = fs::metadata(&runner).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&runner, perms).unwrap();

    let event_tx = broadcast::channel(8).0;
    let manager = ChannelRuntimeManager::new(temp.path().join("daemon.sock"), event_tx);
    let previous = std::env::var_os("TURIN_CHANNEL_TELEGRAM_BIN");
    unsafe {
        std::env::set_var("TURIN_CHANNEL_TELEGRAM_BIN", &runner);
    }

    manager
        .sync(
            workspace_root.clone(),
            vec![DiscoveredChannel {
                id: "telegram-ops".to_string(),
                directory: workspace_root.join(".turin/runtime/channels/telegram-ops"),
                enabled: true,
                kind: "telegram".to_string(),
                agent_id: "default".to_string(),
                idle_timeout_seconds: Some(60),
                persistence: Default::default(),
                inference: Default::default(),
                extra: toml::Table::new(),
            }],
        )
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(200)).await;
    let runtime = manager.get("telegram-ops").await.expect("runtime exists");
    assert_eq!(runtime.state, "starting");

    manager.shutdown().await;
    if let Some(value) = previous {
        unsafe {
            std::env::set_var("TURIN_CHANNEL_TELEGRAM_BIN", value);
        }
    } else {
        unsafe {
            std::env::remove_var("TURIN_CHANNEL_TELEGRAM_BIN");
        }
    }
}

#[cfg(unix)]
#[tokio::test]
async fn supervisor_restarts_exited_external_runner() {
    use std::os::unix::fs::PermissionsExt;

    let _env_guard = crate::test_support::env_lock().lock().await;
    let temp = tempdir().unwrap();
    let workspace_root = temp.path().join("workspace");
    let channel_dir = workspace_root.join(".turin/runtime/channels/telegram-ops");
    fs::create_dir_all(&channel_dir).unwrap();

    let counter_path = temp.path().join("runner-count.txt");
    let runner = temp.path().join("fake-telegram-runner.sh");
    fs::write(
        &runner,
        format!(
            "#!/bin/sh\nif [ \"$1\" = \"describe\" ]; then\n  printf '%s\\n' '{{\"protocol_version\":2,\"kind\":\"telegram\"}}'\n  exit 0\nfi\nif [ \"$1\" = \"run\" ]; then\n  count=0\n  if [ -f \"{counter}\" ]; then count=$(cat \"{counter}\"); fi\n  count=$((count + 1))\n  printf '%s' \"$count\" > \"{counter}\"\n  exit 0\nfi\nif [ \"$1\" = \"validate-settings\" ]; then\n  exit 0\nfi\nif [ \"$1\" = \"setup-auth-flow-start\" ]; then\n  exit 1\nfi\nif [ \"$1\" = \"setup-auth-flow-poll\" ]; then\n  exit 1\nfi\nexit 0\n",
            counter = counter_path.display()
        ),
    )
    .unwrap();
    let mut perms = fs::metadata(&runner).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&runner, perms).unwrap();

    let previous = std::env::var_os("TURIN_CHANNEL_TELEGRAM_BIN");
    unsafe {
        std::env::set_var("TURIN_CHANNEL_TELEGRAM_BIN", &runner);
    }

    let event_tx = broadcast::channel(8).0;
    let manager = Arc::new(ChannelRuntimeManager::new(
        temp.path().join("daemon.sock"),
        event_tx,
    ));
    let supervisor = manager.clone().start_supervisor();

    manager
        .sync(
            workspace_root.clone(),
            vec![DiscoveredChannel {
                id: "telegram-ops".to_string(),
                directory: channel_dir,
                enabled: true,
                kind: "telegram".to_string(),
                agent_id: "default".to_string(),
                idle_timeout_seconds: Some(60),
                persistence: Default::default(),
                inference: Default::default(),
                extra: toml::Table::new(),
            }],
        )
        .await
        .unwrap();

    wait_until(
        Duration::from_secs(5),
        || {
            fs::read_to_string(&counter_path)
                .ok()
                .and_then(|raw| raw.trim().parse::<u64>().ok())
                .is_some_and(|count| count >= 2)
        },
        "external channel auto-restart",
    )
    .await;

    let runtime = manager.get("telegram-ops").await.expect("runtime exists");
    assert!(runtime.restart_count >= 1);

    supervisor.abort();
    manager.shutdown().await;
    if let Some(value) = previous {
        unsafe {
            std::env::set_var("TURIN_CHANNEL_TELEGRAM_BIN", value);
        }
    } else {
        unsafe {
            std::env::remove_var("TURIN_CHANNEL_TELEGRAM_BIN");
        }
    }
}

#[cfg(unix)]
#[tokio::test]
async fn supervisor_restarts_runner_when_heartbeat_goes_stale() {
    use std::os::unix::fs::PermissionsExt;

    let _env_guard = crate::test_support::env_lock().lock().await;
    let temp = tempdir().unwrap();
    let workspace_root = temp.path().join("workspace");
    let channel_dir = workspace_root.join(".turin/runtime/channels/telegram-ops");
    fs::create_dir_all(&channel_dir).unwrap();

    let runner = temp.path().join("fake-telegram-runner.sh");
    fs::write(
        &runner,
        "#!/bin/sh\nif [ \"$1\" = \"describe\" ]; then\n  printf '%s\\n' '{\"protocol_version\":2,\"kind\":\"telegram\"}'\n  exit 0\nfi\nif [ \"$1\" = \"run\" ]; then\n  sleep 30\n  exit 0\nfi\nif [ \"$1\" = \"validate-settings\" ]; then\n  exit 0\nfi\nif [ \"$1\" = \"setup-auth-flow-start\" ]; then\n  exit 1\nfi\nif [ \"$1\" = \"setup-auth-flow-poll\" ]; then\n  exit 1\nfi\nexit 0\n",
    )
    .unwrap();
    let mut perms = fs::metadata(&runner).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&runner, perms).unwrap();

    let previous = std::env::var_os("TURIN_CHANNEL_TELEGRAM_BIN");
    unsafe {
        std::env::set_var("TURIN_CHANNEL_TELEGRAM_BIN", &runner);
    }

    let event_tx = broadcast::channel(8).0;
    let manager = Arc::new(ChannelRuntimeManager::new(
        temp.path().join("daemon.sock"),
        event_tx,
    ));
    let supervisor = manager.clone().start_supervisor();

    manager
        .sync(
            workspace_root.clone(),
            vec![DiscoveredChannel {
                id: "telegram-ops".to_string(),
                directory: channel_dir,
                enabled: true,
                kind: "telegram".to_string(),
                agent_id: "default".to_string(),
                idle_timeout_seconds: Some(60),
                persistence: Default::default(),
                inference: Default::default(),
                extra: toml::Table::new(),
            }],
        )
        .await
        .unwrap();

    let start_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        if manager
            .get("telegram-ops")
            .await
            .is_some_and(|runtime| runtime.state == "starting")
        {
            break;
        }
        assert!(
            Instant::now() < start_deadline,
            "timed out waiting for external channel start"
        );
        sleep(Duration::from_millis(25)).await;
    }

    manager
        .record_external_hello(ChannelRunnerHelloParams {
            channel_id: "telegram-ops".to_string(),
            manifest: ChannelAdapterManifest {
                protocol_version: turin_channel_core::CHANNEL_ADAPTER_PROTOCOL_VERSION,
                kind: "telegram".to_string(),
                display_name: "Telegram".to_string(),
                ..ChannelAdapterManifest::default()
            },
            runner_binary: Some("turin-channel-telegram".to_string()),
            runner_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            pid: Some(1234),
        })
        .await
        .expect("hello recorded");

    let restart_deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if manager
            .get("telegram-ops")
            .await
            .is_some_and(|runtime| runtime.restart_count >= 1)
        {
            break;
        }
        assert!(
            Instant::now() < restart_deadline,
            "timed out waiting for stale runner restart: {:?}",
            manager.get("telegram-ops").await
        );
        sleep(Duration::from_millis(25)).await;
    }

    let runtime = manager.get("telegram-ops").await.expect("runtime exists");
    assert!(matches!(runtime.state.as_str(), "starting" | "running"));
    assert!(runtime.restart_count >= 1);

    supervisor.abort();
    manager.shutdown().await;
    if let Some(value) = previous {
        unsafe {
            std::env::set_var("TURIN_CHANNEL_TELEGRAM_BIN", value);
        }
    } else {
        unsafe {
            std::env::remove_var("TURIN_CHANNEL_TELEGRAM_BIN");
        }
    }
}

#[tokio::test]
async fn external_runner_hello_marks_channel_running() {
    let event_tx = broadcast::channel(8).0;
    let manager = ChannelRuntimeManager::new(PathBuf::from("daemon.sock"), event_tx);

    {
        let mut inner = manager.inner.lock().await;
        inner.by_id.insert(
            "telegram-ops".to_string(),
            ChannelRuntimeSnapshot {
                id: "telegram-ops".to_string(),
                kind: "telegram".to_string(),
                agent_id: "default".to_string(),
                directory: "/tmp/workspace/.turin/channels/telegram-ops".to_string(),
                state: "starting".to_string(),
                last_error: None,
                last_error_code: None,
                start_count: 1,
                restart_count: 0,
                failure_count: 0,
                last_transition_unix_ms: 1,
                last_started_unix_ms: None,
                last_stopped_unix_ms: None,
                handshake: None,
            },
        );
    }

    let snapshot = manager
        .record_external_hello(ChannelRunnerHelloParams {
            channel_id: "telegram-ops".to_string(),
            manifest: ChannelAdapterManifest {
                protocol_version: turin_channel_core::CHANNEL_ADAPTER_PROTOCOL_VERSION,
                kind: "telegram".to_string(),
                display_name: "Telegram".to_string(),
                ..ChannelAdapterManifest::default()
            },
            runner_binary: Some("turin-channel-telegram".to_string()),
            runner_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            pid: Some(1234),
        })
        .await
        .expect("hello recorded");

    assert_eq!(snapshot.state, "running");
    assert_eq!(
        snapshot.handshake.as_ref().expect("handshake").display_name,
        "Telegram"
    );

    let first_contact = snapshot
        .handshake
        .as_ref()
        .expect("handshake")
        .last_handshake_unix_ms;
    tokio::time::sleep(Duration::from_millis(5)).await;

    let heartbeat = manager
        .record_external_heartbeat(ChannelRunnerHeartbeatParams {
            channel_id: "telegram-ops".to_string(),
        })
        .await
        .expect("heartbeat recorded");
    assert_eq!(heartbeat.state, "running");
    assert!(
        heartbeat
            .handshake
            .as_ref()
            .expect("handshake")
            .last_handshake_unix_ms
            >= first_contact
    );
}
