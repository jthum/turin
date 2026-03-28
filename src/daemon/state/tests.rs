use super::*;
use serde_json::json;
use tempfile::tempdir;
use tokio::time::{Duration, sleep};

fn write_bootstrap(root: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(root.join("default-harness"))?;
    std::fs::write(
        root.join("default-harness").join("main.lua"),
        "-- bootstrap\n",
    )?;
    let config_path = root.join("turin.toml");
    std::fs::write(
        &config_path,
        r#"[agent]
id = "default"
system_prompt = "bootstrap"
model = "mock-model"
provider = "mock"

[kernel]
workspace_root = "."

[persistence]
database_path = "state.db"

[harness]
directory = "default-harness"
fs_root = "."

[providers.mock]
type = "mock"

[embeddings]
provider = "noop"
"#,
    )?;
    Ok(config_path)
}

#[tokio::test]
async fn create_disable_and_delete_agent_updates_filesystem_state() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    let created = state
        .create_agent(CreateAgentInput {
            id: "docs-reviewer".to_string(),
            provider: "mock".to_string(),
            model: "mock-model".to_string(),
            system_prompt: Some("Review docs".to_string()),
            thinking: None,
            mode: None,
            harness: None,
            idle_grace_secs: None,
            enabled: true,
            tools: Default::default(),
        })
        .await?;

    assert_eq!(created.id, "docs-reviewer");
    assert!(created.has_local_harness);
    assert!(
        temp.path()
            .join("agents")
            .join("docs-reviewer")
            .join("harness")
            .join("main.lua")
            .exists()
    );

    let disabled = state.set_agent_enabled("docs-reviewer", false).await?;
    assert!(!disabled.enabled);

    let updated = state
        .update_agent(
            "docs-reviewer",
            UpdateAgentInput {
                model: Some("mock-model-2".to_string()),
                system_prompt: Some("Review docs carefully".to_string()),
                ..UpdateAgentInput::default()
            },
        )
        .await?;
    assert_eq!(updated.model, "mock-model-2");
    assert_eq!(
        updated.system_prompt.as_deref(),
        Some("Review docs carefully")
    );

    let status = state.delete_agent("docs-reviewer").await?;
    assert!(
        status
            .registry
            .agents
            .iter()
            .all(|agent| agent.id != "docs-reviewer")
    );
    assert!(!temp.path().join("agents").join("docs-reviewer").exists());

    Ok(())
}

#[tokio::test]
async fn submit_task_exposes_completed_result_and_blocks_rescan_while_active() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    let task = state
        .submit_task(
            Some("default"),
            None,
            "Hello daemon".to_string(),
            Default::default(),
        )
        .await?;
    assert_eq!(task.agent_id, "default");
    assert!(matches!(task.state.as_str(), "queued" | "running"));
    assert!(state.rescan().await.is_err());

    let mut saw_completed = false;
    for _ in 0..50 {
        if let Some(snapshot) = state.get_task(&task.request_id).await
            && snapshot.state == "completed"
        {
            saw_completed = true;
            assert!(snapshot.status.is_some());
            break;
        }
        sleep(Duration::from_millis(20)).await;
    }
    assert!(saw_completed, "daemon task did not complete in time");

    let tasks = state.list_tasks().await;
    assert!(
        tasks
            .iter()
            .any(|entry| entry.request_id == task.request_id)
    );
    assert!(state.rescan().await.is_ok());

    Ok(())
}

#[tokio::test]
async fn wait_for_task_returns_terminal_result() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let state = DaemonState::load(&config_path).await?;

    let task = state
        .submit_task(
            Some("default"),
            None,
            "Hello wait".to_string(),
            Default::default(),
        )
        .await?;
    let completed = state.wait_for_task(&task.request_id, Some(2_000)).await?;
    assert_eq!(completed.request_id, task.request_id);
    assert_eq!(completed.state, "completed");
    assert!(completed.status.is_some());

    Ok(())
}

#[tokio::test]
async fn session_list_and_get_expose_persisted_session_details() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let state = DaemonState::load(&config_path).await?;

    let task = state
        .submit_task(
            Some("default"),
            None,
            "Hello session".to_string(),
            Default::default(),
        )
        .await?;

    let mut saw_completed = false;
    for _ in 0..50 {
        if let Some(snapshot) = state.get_task(&task.request_id).await
            && snapshot.state == "completed"
        {
            saw_completed = true;
            break;
        }
        sleep(Duration::from_millis(20)).await;
    }
    assert!(saw_completed, "daemon task did not complete in time");

    let sessions = state.list_sessions(10, 0).await?;
    assert!(!sessions.is_empty());
    let session = &sessions[0];
    assert_eq!(session.agent_id, "default");

    let detail = state
        .get_session(&session.session_id)
        .await?
        .expect("session detail visible");
    assert_eq!(detail.session.session_id, session.session_id);
    assert_eq!(detail.session.agent_id, "default");
    assert!(!detail.events.is_empty());
    assert!(!detail.messages.is_empty());

    Ok(())
}

#[tokio::test]
async fn harness_reload_and_validate_are_targeted() -> Result<()> {
    let temp = tempdir()?;
    let shared_harness = temp.path().join("harnesses").join("shared");
    std::fs::create_dir_all(&shared_harness)?;
    std::fs::write(shared_harness.join("main.lua"), "-- shared\n")?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    let agent = state
        .create_agent(CreateAgentInput {
            id: "shared-agent".to_string(),
            provider: "mock".to_string(),
            model: "mock-model".to_string(),
            system_prompt: None,
            thinking: None,
            mode: None,
            harness: Some("shared".to_string()),
            idle_grace_secs: None,
            enabled: true,
            tools: Default::default(),
        })
        .await?;
    assert_eq!(agent.harness.as_deref(), Some("shared"));

    let detail = state
        .harness_detail("shared")
        .expect("shared harness visible");
    assert_eq!(detail.harness_id, "shared");
    assert!(detail.bound_agents.contains(&"shared-agent".to_string()));

    std::fs::write(shared_harness.join("extra.lua"), "-- extra\n")?;
    let reloaded = state.reload_harness("shared").await?;
    assert!(reloaded.loaded_scripts.iter().any(|s| s == "extra"));

    let validation = state.validate_harness("shared")?;
    assert_eq!(validation["harness_id"], "shared");
    assert_eq!(validation["valid"], true);
    assert!(
        validation["script_count"]
            .as_u64()
            .expect("script_count number")
            >= 2
    );

    std::fs::write(
        shared_harness.join("broken.lua"),
        "function on_turn_prepare(",
    )?;
    assert!(state.validate_harness("shared").is_err());
    let still_loaded = state
        .harness_detail("shared")
        .expect("shared harness still visible");
    assert!(still_loaded.loaded_scripts.iter().all(|s| s != "broken"));

    Ok(())
}

#[tokio::test]
async fn shared_harness_create_and_delete_are_filesystem_backed() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    let created = state.create_shared_harness("reviewer").await?;
    assert_eq!(created.harness_id, "reviewer");
    assert!(
        temp.path()
            .join("harnesses")
            .join("reviewer")
            .join("main.lua")
            .exists()
    );

    let status = state.delete_shared_harness("reviewer").await?;
    assert!(
        status
            .harnesses
            .iter()
            .all(|harness| harness.harness_id != "reviewer")
    );

    Ok(())
}

#[tokio::test]
async fn channel_create_disable_update_and_delete_are_filesystem_backed() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    let created = state
        .create_channel(CreateChannelInput {
            id: "discord".to_string(),
            kind: "discord".to_string(),
            agent_id: "default".to_string(),
            idle_ttl_secs: Some(600),
            enabled: true,
            settings: json!({
                "token_env": "DISCORD_TOKEN",
                "channel_id": "1234567890",
                "allow_dm": true,
            }),
        })
        .await?;
    assert_eq!(created.id, "discord");
    assert_eq!(created.kind, "discord");
    assert_eq!(created.agent_id, "default");
    assert_eq!(created.settings["token_env"], "DISCORD_TOKEN");

    let disabled = state.set_channel_enabled("discord", false).await?;
    assert!(!disabled.enabled);

    let updated = state
        .update_channel(
            "discord",
            UpdateChannelInput {
                idle_ttl_secs: Some(900),
                settings: Some(json!({
                    "token_env": "NEW_TOKEN",
                    "guild_id": "123",
                })),
                ..UpdateChannelInput::default()
            },
        )
        .await?;
    assert_eq!(updated.idle_ttl_secs, Some(900));
    assert_eq!(updated.settings["token_env"], "NEW_TOKEN");
    assert_eq!(updated.settings["guild_id"], "123");
    assert_eq!(updated.settings["channel_id"], "1234567890");

    let status = state.delete_channel("discord").await?;
    assert!(
        status
            .registry
            .channels
            .iter()
            .all(|channel| channel.id != "discord")
    );
    assert!(!temp.path().join("channels").join("discord").exists());

    Ok(())
}

#[tokio::test]
async fn channel_create_rejects_invalid_discord_settings() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    let error = state
        .create_channel(CreateChannelInput {
            id: "discord".to_string(),
            kind: "discord".to_string(),
            agent_id: "default".to_string(),
            idle_ttl_secs: Some(600),
            enabled: true,
            settings: json!({
                "channel_id": "1234567890"
            }),
        })
        .await
        .expect_err("discord settings without token_env should fail");
    assert!(error.to_string().contains("token_env"));

    Ok(())
}

#[tokio::test]
async fn channel_update_rejects_invalid_fs_poll_interval() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    state
        .create_channel(CreateChannelInput {
            id: "fs-local".to_string(),
            kind: "fs".to_string(),
            agent_id: "default".to_string(),
            idle_ttl_secs: Some(600),
            enabled: true,
            settings: json!({
                "inbox_dir": "inbox",
                "outbox_dir": "outbox"
            }),
        })
        .await?;

    let error = state
        .update_channel(
            "fs-local",
            UpdateChannelInput {
                settings: Some(json!({
                    "poll_interval_ms": 0
                })),
                ..UpdateChannelInput::default()
            },
        )
        .await
        .expect_err("invalid fs poll interval should fail");
    assert!(error.to_string().contains("poll_interval_ms"));

    Ok(())
}

#[tokio::test]
async fn channel_create_and_update_accept_valid_telegram_settings() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    let created = state
        .create_channel(CreateChannelInput {
            id: "telegram".to_string(),
            kind: "telegram".to_string(),
            agent_id: "default".to_string(),
            idle_ttl_secs: Some(600),
            enabled: false,
            settings: json!({
                "token_env": "TELEGRAM_BOT_TOKEN",
                "chat_id": -100123456,
                "poll_timeout_secs": 10,
            }),
        })
        .await?;
    assert_eq!(created.id, "telegram");
    assert_eq!(created.kind, "telegram");
    assert_eq!(created.settings["chat_id"], -100123456);

    let updated = state
        .update_channel(
            "telegram",
            UpdateChannelInput {
                idle_ttl_secs: Some(900),
                settings: Some(json!({
                    "workspace_id": "ops",
                    "poll_interval_ms": 250,
                })),
                ..UpdateChannelInput::default()
            },
        )
        .await?;
    assert_eq!(updated.idle_ttl_secs, Some(900));
    assert_eq!(updated.settings["workspace_id"], "ops");
    assert_eq!(updated.settings["chat_id"], -100123456);

    Ok(())
}

#[tokio::test]
async fn channel_create_accepts_multi_chat_telegram_settings() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    let created = state
        .create_channel(CreateChannelInput {
            id: "telegram-multi".to_string(),
            kind: "telegram".to_string(),
            agent_id: "default".to_string(),
            idle_ttl_secs: Some(600),
            enabled: false,
            settings: json!({
                "token_env": "TELEGRAM_BOT_TOKEN",
                "chat_ids": [-100123456, -100654321],
                "respond_mode": "mentions_or_replies",
            }),
        })
        .await?;

    assert_eq!(created.id, "telegram-multi");
    assert_eq!(
        created.settings["chat_ids"],
        json!([-100123456, -100654321])
    );
    assert_eq!(created.settings["respond_mode"], "mentions_or_replies");

    Ok(())
}

#[tokio::test]
async fn channel_access_snapshot_and_approval_are_filesystem_backed() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    state
        .create_channel(CreateChannelInput {
            id: "telegram".to_string(),
            kind: "telegram".to_string(),
            agent_id: "default".to_string(),
            idle_ttl_secs: Some(600),
            enabled: true,
            settings: json!({
                "token_env": "TELEGRAM_BOT_TOKEN",
                "pairing_mode": "auto",
            }),
        })
        .await?;

    let snapshot = state
        .channel_access_snapshot("telegram")
        .await?
        .expect("channel exists");
    assert!(snapshot.pending_rooms.is_empty());
    assert!(snapshot.approved_rooms.is_empty());

    let snapshot = state
        .approve_channel_room(
            "telegram",
            "telegram".to_string(),
            Some("-100123456".to_string()),
            "-100123456".to_string(),
        )
        .await?
        .expect("channel exists");
    assert_eq!(snapshot.approved_rooms.len(), 1);

    let snapshot = state
        .revoke_channel_room(
            "telegram",
            "telegram".to_string(),
            Some("-100123456".to_string()),
            "-100123456".to_string(),
        )
        .await?
        .expect("channel exists");
    assert!(snapshot.approved_rooms.is_empty());

    Ok(())
}

#[tokio::test]
async fn channel_create_rejects_invalid_telegram_settings() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    let error = state
        .create_channel(CreateChannelInput {
            id: "telegram".to_string(),
            kind: "telegram".to_string(),
            agent_id: "default".to_string(),
            idle_ttl_secs: Some(600),
            enabled: true,
            settings: json!({
                "token_env": "TELEGRAM_BOT_TOKEN",
                "chat_id": "@ops"
            }),
        })
        .await
        .expect_err("telegram settings with non-numeric chat_id should fail");
    assert!(error.to_string().contains("chat_id"));

    Ok(())
}

#[tokio::test]
async fn agent_can_bind_shared_harness_and_switch_back_to_local() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    state.create_shared_harness("reviewer").await?;
    state
        .create_agent(CreateAgentInput {
            id: "writer".to_string(),
            provider: "mock".to_string(),
            model: "mock-model".to_string(),
            system_prompt: None,
            thinking: None,
            mode: None,
            harness: None,
            idle_grace_secs: None,
            enabled: true,
            tools: Default::default(),
        })
        .await?;

    let rebound = state
        .bind_agent_shared_harness("writer", "reviewer")
        .await?;
    assert_eq!(rebound.harness.as_deref(), Some("reviewer"));
    assert!(!rebound.has_local_harness);
    assert!(
        !temp
            .path()
            .join("agents")
            .join("writer")
            .join("harness")
            .exists()
    );

    let local = state.use_local_agent_harness("writer").await?;
    assert_eq!(local.harness, None);
    assert!(local.has_local_harness);
    assert!(
        temp.path()
            .join("agents")
            .join("writer")
            .join("harness")
            .exists()
    );

    Ok(())
}

#[tokio::test]
async fn runtime_errors_surface_invalid_agent_configs_without_global_failure() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let bad_agent_dir = temp.path().join("agents").join("broken");
    std::fs::create_dir_all(&bad_agent_dir)?;
    std::fs::write(bad_agent_dir.join("agent.toml"), "provider = [")?;

    let state = DaemonState::load(&config_path).await?;
    let errors = state.runtime_errors();
    assert_eq!(errors.len(), 1);
    assert!(errors[0].path.contains("broken"));

    let agent_issues = state
        .agent_issues("broken")?
        .expect("broken agent should be addressable");
    assert_eq!(agent_issues.len(), 1);
    assert!(agent_issues[0].path.contains("broken"));

    Ok(())
}

#[tokio::test]
async fn harness_issues_surface_broken_shared_harness_without_loaded_runtime() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    state.create_shared_harness("reviewer").await?;
    let harness_dir = temp.path().join("harnesses").join("reviewer");
    std::fs::write(harness_dir.join("broken.lua"), "function on_turn_prepare(")?;

    let status = state.rescan().await?;
    assert!(status.harnesses.iter().all(|h| h.harness_id != "reviewer"));

    let harness_issues = state
        .harness_issues("reviewer")?
        .expect("broken harness should still expose issues");
    assert_eq!(harness_issues.len(), 1);
    assert!(harness_issues[0].path.contains("reviewer"));

    Ok(())
}

#[tokio::test]
async fn bind_shared_harness_rejects_non_scaffold_local_harness() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    state.create_shared_harness("reviewer").await?;
    state
        .create_agent(CreateAgentInput {
            id: "writer".to_string(),
            provider: "mock".to_string(),
            model: "mock-model".to_string(),
            system_prompt: None,
            thinking: None,
            mode: None,
            harness: None,
            idle_grace_secs: None,
            enabled: true,
            tools: Default::default(),
        })
        .await?;

    let local_main = temp
        .path()
        .join("agents")
        .join("writer")
        .join("harness")
        .join("main.lua");
    std::fs::write(
        &local_main,
        "function on_turn_prepare(ctx)\n  return ALLOW\nend\n",
    )?;

    let err = state
        .bind_agent_shared_harness("writer", "reviewer")
        .await
        .expect_err("non-scaffold local harness should block rebinding");
    assert!(err.to_string().contains("non-scaffold local harness"));
    assert!(local_main.exists());

    Ok(())
}

#[tokio::test]
async fn agent_runtime_status_reflects_live_runtime_state() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    let disabled = state
        .create_agent(CreateAgentInput {
            id: "disabled-reviewer".to_string(),
            provider: "mock".to_string(),
            model: "mock-model".to_string(),
            system_prompt: None,
            thinking: None,
            mode: None,
            harness: None,
            idle_grace_secs: None,
            enabled: false,
            tools: Default::default(),
        })
        .await?;
    assert!(!disabled.enabled);

    let disabled_status = state
        .agent_runtime_status("disabled-reviewer")
        .await?
        .expect("disabled agent status exists");
    assert_eq!(disabled_status.agent_id, "disabled-reviewer");
    assert!(!disabled_status.running);

    let daemon_status = state.status().await;
    assert!(
        daemon_status
            .agent_runtimes
            .iter()
            .any(|status| status.agent_id == "disabled-reviewer")
    );

    let initial = state
        .agent_runtime_status("default")
        .await?
        .expect("default agent status exists");
    assert_eq!(initial.agent_id, "default");
    assert!(!initial.running);

    let task = state
        .submit_task(
            Some("default"),
            None,
            "Hello status".to_string(),
            Default::default(),
        )
        .await?;
    assert!(matches!(task.state.as_str(), "queued" | "running"));

    let mut saw_running = false;
    for _ in 0..50 {
        let status = state
            .agent_runtime_status("default")
            .await?
            .expect("default agent status exists");
        if status.running {
            saw_running = true;
            break;
        }
        sleep(Duration::from_millis(20)).await;
    }

    assert!(
        saw_running,
        "agent runtime status never transitioned to running"
    );

    Ok(())
}
