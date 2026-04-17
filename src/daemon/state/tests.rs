use super::*;
use crate::kernel::event::TaskBranchOutcome;
use crate::kernel::session_refs::parse_session_reference;
use crate::persistence::manager::StoreSelector;
use crate::persistence::state::SessionReadTarget;
use serde_json::json;
use tempfile::tempdir;
use tokio::time::{Duration, sleep};
use turin_daemon_protocol::{SessionSearchScope, SidestepModeParams, SidestepTaskParams};

fn write_agent_with_state_path(root: &Path, agent_id: &str, state_path: &str) -> Result<()> {
    let agent_dir = root
        .join(".turin")
        .join("runtime")
        .join("agents")
        .join(agent_id);
    std::fs::create_dir_all(&agent_dir)?;
    std::fs::create_dir_all(agent_dir.join("harness"))?;
    std::fs::write(
        agent_dir.join("harness").join("main.lua"),
        "-- local harness\n",
    )?;
    std::fs::write(
        agent_dir.join("config.toml"),
        format!(
            r#"id = "{agent_id}"
model = "mock-model"
provider = "mock"

[persistence.state]
path = "{state_path}"
"#
        ),
    )?;
    Ok(())
}

fn write_bootstrap(root: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(root.join("default-harness"))?;
    std::fs::write(
        root.join("default-harness").join("main.lua"),
        "-- bootstrap\n",
    )?;
    let config_path = root.join(".turin").join("config.toml");
    std::fs::create_dir_all(config_path.parent().expect("config parent"))?;
    std::fs::write(
        &config_path,
        r#"[agent]
id = "default"
system_prompt = "bootstrap"
model = "mock-model"
provider = "mock"

[kernel]
workspace_root = "."

[persistence.state]
path = "state.db"

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
            .join(".turin")
            .join("runtime")
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
    assert!(
        !temp
            .path()
            .join(".turin")
            .join("runtime")
            .join("agents")
            .join("docs-reviewer")
            .exists()
    );

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
            None,
            "Hello daemon".to_string(),
            None,
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
            None,
            "Hello wait".to_string(),
            None,
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
async fn sidestep_task_uses_ephemeral_slot_and_does_not_persist_transcript() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let state = DaemonState::load(&config_path).await?;

    let live = state.open_session("default", Some("main"), None).await?;
    let task = state
        .submit_task(
            None,
            Some(&live.session_id),
            Some("main"),
            "Hello session".to_string(),
            None,
            Default::default(),
        )
        .await?;
    let completed = state.wait_for_task(&task.request_id, Some(2_000)).await?;
    assert_eq!(completed.state, "completed");

    let before_detail = state
        .get_session(&live.session_id)
        .await?
        .expect("persisted session detail visible");
    let before_message_count = before_detail.messages.len();

    let sidestep = state
        .sidestep_task_params(SidestepTaskParams {
            session_id: live.session_id.clone(),
            slot_id: None,
            prompt: "Explore a side question".to_string(),
            content: None,
            tools: None,
            mode: SidestepModeParams::Ephemeral,
            context_target: None,
            timeout_ms: Some(2_000),
        })
        .await?;
    assert_eq!(sidestep.state, "completed");
    assert!(sidestep.slot_id.starts_with("sd_"));

    let after_detail = state
        .get_session(&live.session_id)
        .await?
        .expect("persisted session detail visible");
    assert_eq!(after_detail.messages.len(), before_message_count);
    assert!(
        state
            .list_live_sessions()
            .await
            .iter()
            .all(|live| live.slot_id != sidestep.slot_id)
    );

    Ok(())
}

#[tokio::test]
async fn sidestep_task_can_run_durably_on_a_sibling_branch() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let state = DaemonState::load(&config_path).await?;

    let live = state.open_session("default", Some("main"), None).await?;
    let seed = state
        .submit_task(
            None,
            Some(&live.session_id),
            Some("main"),
            "Seed session".to_string(),
            None,
            Default::default(),
        )
        .await?;
    let completed = state.wait_for_task(&seed.request_id, Some(2_000)).await?;
    assert_eq!(completed.state, "completed");

    let before_detail = state
        .get_session(&live.session_id)
        .await?
        .expect("persisted session detail visible");
    let before_message_count = before_detail.messages.len();
    let before_branches = state
        .list_session_branches(&live.session_id)
        .await?
        .expect("branch list visible");

    let sidestep = state
        .sidestep_task_params(SidestepTaskParams {
            session_id: live.session_id.clone(),
            slot_id: None,
            prompt: "Explore on sibling branch".to_string(),
            content: None,
            tools: None,
            mode: SidestepModeParams::ForkSibling,
            context_target: None,
            timeout_ms: Some(2_000),
        })
        .await?;
    assert_eq!(sidestep.state, "completed");
    let branch_outcome = sidestep
        .branch_outcome
        .clone()
        .expect("fork_sibling sidestep should surface branch outcome");
    let sidestep_branch_id = match branch_outcome {
        TaskBranchOutcome::SidestepSibling {
            branch_id,
            persisted_active_head_unchanged,
            ..
        } => {
            assert!(persisted_active_head_unchanged);
            branch_id
        }
        other => panic!("unexpected branch outcome: {other:?}"),
    };

    let after_detail = state
        .get_session(&live.session_id)
        .await?
        .expect("persisted session detail visible");
    assert_eq!(after_detail.messages.len(), before_message_count);

    let after_branches = state
        .list_session_branches(&live.session_id)
        .await?
        .expect("branch list visible");
    assert_eq!(after_branches.len(), before_branches.len() + 1);
    assert!(
        after_branches
            .iter()
            .any(|branch| !branch.active && branch.name.starts_with("sidestep-"))
    );

    let session_ref = parse_session_reference(&live.session_id)?;
    let store_selector = session_ref
        .store_selector
        .unwrap_or_else(|| StoreSelector::Alias("state".to_string()));
    let store = state.kernel.store_manager().open(&store_selector).await?;
    let internal_id = before_detail.session.internal_id;
    let sibling_messages = store
        .get_messages(
            internal_id,
            &SessionReadTarget::BranchHead(sidestep_branch_id),
        )
        .await?;
    assert!(
        sibling_messages
            .iter()
            .any(|message| message.content.contains("Explore on sibling branch"))
    );
    assert!(
        state
            .list_live_sessions()
            .await
            .iter()
            .all(|live| live.slot_id != sidestep.slot_id)
    );

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
            None,
            "Hello session".to_string(),
            None,
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

    let sessions = state.list_sessions(10, 0, None).await?;
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
async fn session_list_and_search_can_target_an_explicit_state_store() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    write_agent_with_state_path(temp.path(), "reviewer", "reviewer.db")?;
    let state = DaemonState::load(&config_path).await?;

    let live = state.open_session("reviewer", None, None).await?;
    assert!(live.session_id.contains("@"));

    let task = state
        .submit_task(
            None,
            Some(&live.session_id),
            None,
            "alternate store session body".to_string(),
            None,
            Default::default(),
        )
        .await?;
    assert_eq!(
        state
            .wait_for_task(&task.request_id, Some(2_000))
            .await?
            .state,
        "completed"
    );

    let default_sessions = state.list_sessions(20, 0, None).await?;
    assert!(
        default_sessions
            .iter()
            .all(|session| session.agent_id != "reviewer"),
        "default session list should remain scoped to primary state"
    );

    let reviewer_path = temp.path().join("reviewer.db").display().to_string();
    let reviewer_sessions = state
        .list_sessions(20, 0, Some(StoreSelector::Path(reviewer_path.clone())))
        .await?;
    assert_eq!(reviewer_sessions.len(), 1);
    assert_eq!(reviewer_sessions[0].agent_id, "reviewer");
    assert_eq!(
        parse_session_reference(&reviewer_sessions[0].session_id)?.public_id,
        parse_session_reference(&live.session_id)?.public_id
    );

    let default_hits = state
        .search_sessions(
            "alternate store session body",
            SessionSearchScope::Messages,
            10,
            0,
            None,
        )
        .await?;
    assert!(default_hits.is_empty());

    let reviewer_hits = state
        .search_sessions(
            "alternate store session body",
            SessionSearchScope::Messages,
            10,
            0,
            Some(StoreSelector::Path(reviewer_path.clone())),
        )
        .await?;
    assert_eq!(reviewer_hits.len(), 1);
    assert_eq!(reviewer_hits[0].agent_id, "reviewer");
    assert_eq!(
        parse_session_reference(&reviewer_hits[0].session_id)?.public_id,
        parse_session_reference(&live.session_id)?.public_id
    );

    Ok(())
}

#[tokio::test]
async fn channel_inference_override_applies_on_open_and_resume() -> Result<()> {
    let temp = tempdir()?;
    let harness_dir = temp.path().join("default-harness");
    std::fs::create_dir_all(&harness_dir)?;
    std::fs::write(
        harness_dir.join("main.lua"),
        r#"
function on_turn_prepare(ctx)
  ctx.inference = "fast"
  return ALLOW
end
"#,
    )?;

    let channel_dir = temp
        .path()
        .join(".turin")
        .join("runtime")
        .join("channels")
        .join("telegram-ops");
    std::fs::create_dir_all(&channel_dir)?;
    std::fs::write(
        channel_dir.join("config.toml"),
        r#"
id = "telegram-ops"
kind = "telegram"
agent_id = "default"

[inference.contexts.fast]
provider = "channel_fast"
model = "channel-fast-model"
"#,
    )?;

    let config_path = temp.path().join(".turin").join("config.toml");
    std::fs::create_dir_all(config_path.parent().expect("config parent"))?;
    std::fs::write(
        &config_path,
        r#"[agent]
id = "default"
system_prompt = "bootstrap"
model = "primary-model"
provider = "primary"

[kernel]
workspace_root = "."

[persistence.state]
path = "state.db"

[harness]
directory = "default-harness"
fs_root = "."

[providers.primary]
type = "mock"
base_url = "PRIMARY"

[providers.root_fast]
type = "mock"
base_url = "ROOT_FAST"

[providers.channel_fast]
type = "mock"
base_url = "CHANNEL_FAST"

[inference.contexts.fast]
provider = "root_fast"
model = "root-fast-model"

[embeddings]
provider = "noop"
"#,
    )?;

    let state = DaemonState::load(&config_path).await?;

    let live = state
        .open_session("default", None, Some("telegram-ops"))
        .await?;
    let original_public_id = parse_session_reference(&live.session_id)?.public_id;

    let first = state
        .submit_task(
            None,
            Some(&live.session_id),
            None,
            "route via channel override".to_string(),
            None,
            Default::default(),
        )
        .await?;
    assert_eq!(
        state
            .wait_for_task(&first.request_id, Some(2_000))
            .await?
            .state,
        "completed"
    );

    let first_detail = state
        .get_session(&live.session_id)
        .await?
        .expect("session detail visible");
    let first_assistant_text = first_detail
        .messages
        .iter()
        .filter(|message| message.role == "assistant")
        .flat_map(|message| message.content.as_array().into_iter().flatten())
        .filter_map(|part| part.get("text").and_then(|value| value.as_str()))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        first_assistant_text.contains("CHANNEL_FAST"),
        "expected channel override provider output, got: {first_assistant_text}"
    );
    assert!(!first_assistant_text.contains("ROOT_FAST"));
    assert!(!first_assistant_text.contains("PRIMARY"));

    state.kill_session(&live.session_id, None).await?;

    let resumed = state.resume_session(&live.session_id, None).await?;
    assert_eq!(
        parse_session_reference(&resumed.session_id)?.public_id,
        original_public_id
    );

    let second = state
        .submit_task(
            None,
            Some(&resumed.session_id),
            None,
            "route via resumed channel override".to_string(),
            None,
            Default::default(),
        )
        .await?;
    assert_eq!(
        state
            .wait_for_task(&second.request_id, Some(2_000))
            .await?
            .state,
        "completed"
    );

    let second_detail = state
        .get_session(&resumed.session_id)
        .await?
        .expect("resumed session detail visible");
    let second_assistant_text = second_detail
        .messages
        .iter()
        .filter(|message| message.role == "assistant")
        .flat_map(|message| message.content.as_array().into_iter().flatten())
        .filter_map(|part| part.get("text").and_then(|value| value.as_str()))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        second_assistant_text.contains("CHANNEL_FAST"),
        "expected resumed session to keep channel override, got: {second_assistant_text}"
    );
    assert!(!second_assistant_text.contains("ROOT_FAST"));
    assert!(!second_assistant_text.contains("PRIMARY"));

    Ok(())
}

#[tokio::test]
async fn session_branches_can_be_created_listed_and_checked_out() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let state = DaemonState::load(&config_path).await?;

    let live = state.open_session("default", None, None).await?;
    let session_id = live.session_id.clone();

    let task = state
        .submit_task(
            None,
            Some(&session_id),
            None,
            "first branch turn".to_string(),
            None,
            Default::default(),
        )
        .await?;
    assert_eq!(
        state
            .wait_for_task(&task.request_id, Some(2_000))
            .await?
            .state,
        "completed"
    );
    let task = state
        .submit_task(
            None,
            Some(&session_id),
            None,
            "second branch turn".to_string(),
            None,
            Default::default(),
        )
        .await?;
    assert_eq!(
        state
            .wait_for_task(&task.request_id, Some(2_000))
            .await?
            .state,
        "completed"
    );

    let initial = state
        .list_session_branches(&session_id)
        .await?
        .expect("session exists");
    assert_eq!(initial.len(), 1);
    assert_eq!(initial[0].name, "main");
    assert!(initial[0].active);

    let created = state
        .create_session_branch(&session_id, "alt", None, Some(0), false)
        .await?
        .expect("branch created");
    assert_eq!(created.name, "alt");
    assert!(!created.active);

    let listed = state
        .list_session_branches(&session_id)
        .await?
        .expect("session exists");
    assert_eq!(listed.len(), 2);
    assert!(listed.iter().any(|branch| branch.name == "main"));
    assert!(listed.iter().any(|branch| branch.name == "alt"));

    let checked_out = state
        .checkout_session_branch(&session_id, "alt", None)
        .await?
        .expect("branch checkout succeeds");
    assert_eq!(checked_out.name, "alt");
    assert!(checked_out.active);

    let task = state
        .submit_task(
            None,
            Some(&session_id),
            None,
            "third branch turn".to_string(),
            None,
            Default::default(),
        )
        .await?;
    assert_eq!(
        state
            .wait_for_task(&task.request_id, Some(2_000))
            .await?
            .state,
        "completed"
    );

    let listed = state
        .list_session_branches(&session_id)
        .await?
        .expect("session exists");
    assert!(
        listed
            .iter()
            .any(|branch| branch.name == "alt" && branch.active)
    );
    assert!(
        listed
            .iter()
            .any(|branch| branch.name == "main" && !branch.active)
    );

    let detail = state
        .get_session(&session_id)
        .await?
        .expect("session detail visible");
    let rendered = serde_json::to_string(&detail.messages)?;
    assert!(rendered.contains("first branch turn"));
    assert!(rendered.contains("third branch turn"));
    assert!(!rendered.contains("second branch turn"));

    Ok(())
}

#[tokio::test]
async fn live_session_control_can_target_a_specific_runtime_slot() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let state = DaemonState::load(&config_path).await?;

    let slot_a = state.open_session("default", Some("slot-a"), None).await?;
    let slot_b = state
        .resume_session(&slot_a.session_id, Some("slot-b"))
        .await?;
    assert_eq!(slot_b.slot_id, "slot-b");

    let ambiguous_submit = state
        .submit_task(
            None,
            Some(&slot_a.session_id),
            None,
            "ambiguous".to_string(),
            None,
            Default::default(),
        )
        .await
        .expect_err("slot-agnostic submit should reject ambiguity");
    assert!(
        ambiguous_submit
            .to_string()
            .contains("multiple runtime slots")
    );

    let targeted = state
        .submit_task(
            None,
            Some(&slot_a.session_id),
            Some("slot-b"),
            "targeted slot".to_string(),
            None,
            Default::default(),
        )
        .await?;
    assert_eq!(targeted.slot_id, "slot-b");
    assert_eq!(
        state
            .wait_for_task(&targeted.request_id, Some(2_000))
            .await?
            .state,
        "completed"
    );

    let ambiguous_kill = state
        .kill_session(&slot_a.session_id, None)
        .await
        .expect_err("slot-agnostic kill should reject ambiguity");
    assert!(
        ambiguous_kill
            .to_string()
            .contains("multiple runtime slots")
    );

    let targeted_kill = state
        .kill_session(&slot_a.session_id, Some("slot-b"))
        .await?;
    assert_eq!(targeted_kill["slot_id"], "slot-b");
    assert_eq!(targeted_kill["session_id"], slot_a.session_id);

    let live = state.list_live_sessions().await;
    assert_eq!(live.len(), 1);
    assert_eq!(live[0].slot_id, "slot-a");

    let final_kill = state.kill_session(&slot_a.session_id, None).await?;
    assert_eq!(final_kill["slot_id"], "slot-a");

    Ok(())
}

#[tokio::test]
async fn live_branch_control_can_target_a_specific_runtime_slot() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let state = DaemonState::load(&config_path).await?;

    let slot_a = state.open_session("default", Some("slot-a"), None).await?;
    let seed = state
        .submit_task(
            None,
            Some(&slot_a.session_id),
            None,
            "seed main branch".to_string(),
            None,
            Default::default(),
        )
        .await?;
    assert_eq!(
        state
            .wait_for_task(&seed.request_id, Some(2_000))
            .await?
            .state,
        "completed"
    );

    let slot_b = state
        .resume_session(&slot_a.session_id, Some("slot-b"))
        .await?;
    assert_eq!(slot_b.slot_id, "slot-b");

    let ambiguous_activate = state
        .create_session_branch(&slot_a.session_id, "alt", None, Some(0), true)
        .await
        .expect_err("slot-agnostic live branch activation should reject ambiguity");
    assert!(
        ambiguous_activate
            .to_string()
            .contains("multiple runtime slots")
    );

    let activated = state
        .create_session_branch(&slot_a.session_id, "alt", Some("slot-b"), Some(0), true)
        .await?
        .expect("targeted branch activation succeeds");
    assert_eq!(activated.name, "alt");
    assert!(activated.active);

    let main_followup = state
        .submit_task(
            None,
            Some(&slot_a.session_id),
            Some("slot-a"),
            "main branch followup".to_string(),
            None,
            Default::default(),
        )
        .await?;
    assert_eq!(
        state
            .wait_for_task(&main_followup.request_id, Some(2_000))
            .await?
            .state,
        "completed"
    );

    let alt_followup = state
        .submit_task(
            None,
            Some(&slot_a.session_id),
            Some("slot-b"),
            "alt branch followup".to_string(),
            None,
            Default::default(),
        )
        .await?;
    assert_eq!(
        state
            .wait_for_task(&alt_followup.request_id, Some(2_000))
            .await?
            .state,
        "completed"
    );

    let branches = state
        .list_session_branches(&slot_a.session_id)
        .await?
        .expect("session exists");
    assert!(branches.iter().any(|branch| branch.name == "main"
        && !branch.active
        && branch.head_turn_index == Some(1)));
    assert!(
        branches.iter().any(|branch| branch.name == "alt"
            && branch.active
            && branch.head_turn_index == Some(1))
    );

    let ambiguous_checkout = state
        .checkout_session_branch(&slot_a.session_id, "main", None)
        .await
        .expect_err("slot-agnostic live branch checkout should reject ambiguity");
    assert!(
        ambiguous_checkout
            .to_string()
            .contains("multiple runtime slots")
    );

    let checked_out = state
        .checkout_session_branch(&slot_a.session_id, "main", Some("slot-b"))
        .await?
        .expect("targeted branch checkout succeeds");
    assert_eq!(checked_out.name, "main");
    assert!(checked_out.active);

    let main_again = state
        .submit_task(
            None,
            Some(&slot_a.session_id),
            Some("slot-b"),
            "main branch resumed".to_string(),
            None,
            Default::default(),
        )
        .await?;
    assert_eq!(
        state
            .wait_for_task(&main_again.request_id, Some(2_000))
            .await?
            .state,
        "completed"
    );

    let branches = state
        .list_session_branches(&slot_a.session_id)
        .await?
        .expect("session exists");
    assert!(
        branches.iter().any(|branch| branch.name == "main"
            && branch.active
            && branch.head_turn_index == Some(2))
    );
    assert!(
        branches.iter().any(|branch| branch.name == "alt"
            && !branch.active
            && branch.head_turn_index == Some(1))
    );

    Ok(())
}

#[tokio::test]
async fn harness_reload_and_validate_are_targeted() -> Result<()> {
    let temp = tempdir()?;
    let shared_harness = temp.path().join(".turin").join("harnesses").join("shared");
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
            .join(".turin")
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
    assert!(
        !temp
            .path()
            .join(".turin")
            .join("runtime")
            .join("channels")
            .join("discord")
            .exists()
    );

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
    use std::os::unix::fs::PermissionsExt;

    let _env_guard = crate::test_support::env_lock().lock().await;
    let temp = tempdir()?;
    let runner = temp.path().join("fake-telegram-runner.sh");
    std::fs::write(
        &runner,
        "#!/bin/sh\nif [ \"$1\" = \"describe\" ]; then\n  printf '%s\\n' '{\"protocol_version\":2,\"kind\":\"telegram\"}'\n  exit 0\nfi\nif [ \"$1\" = \"validate-settings\" ]; then\n  exit 0\nfi\nexit 0\n",
    )?;
    let mut perms = std::fs::metadata(&runner)?.permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&runner, perms)?;
    let previous = std::env::var_os("TURIN_CHANNEL_TELEGRAM_BIN");
    unsafe {
        std::env::set_var("TURIN_CHANNEL_TELEGRAM_BIN", &runner);
    }

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

    if let Some(value) = previous {
        unsafe {
            std::env::set_var("TURIN_CHANNEL_TELEGRAM_BIN", value);
        }
    } else {
        unsafe {
            std::env::remove_var("TURIN_CHANNEL_TELEGRAM_BIN");
        }
    }

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
async fn channel_create_accepts_valid_rocketchat_settings() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    let created = state
        .create_channel(CreateChannelInput {
            id: "rocketchat".to_string(),
            kind: "rocketchat".to_string(),
            agent_id: "default".to_string(),
            idle_ttl_secs: Some(600),
            enabled: false,
            settings: json!({
                "token_env": "ROCKETCHAT_AUTH_TOKEN",
                "user_id": "rbAXPnMktTFbNpwtJ",
                "room_id": "GENERAL123",
                "respond_mode": "mentions",
            }),
        })
        .await?;

    assert_eq!(created.id, "rocketchat");
    assert_eq!(created.kind, "rocketchat");
    assert_eq!(created.settings["room_id"], "GENERAL123");
    assert_eq!(created.settings["respond_mode"], "mentions");

    Ok(())
}

#[tokio::test]
async fn channel_create_rejects_invalid_rocketchat_settings() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    let error = state
        .create_channel(CreateChannelInput {
            id: "rocketchat".to_string(),
            kind: "rocketchat".to_string(),
            agent_id: "default".to_string(),
            idle_ttl_secs: Some(600),
            enabled: false,
            settings: json!({
                "token_env": "ROCKETCHAT_AUTH_TOKEN",
                "room_id": "GENERAL123"
            }),
        })
        .await
        .expect_err("rocketchat settings without user_id should fail");
    assert!(error.to_string().contains("user_id"));

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

#[cfg(unix)]
#[tokio::test]
async fn channel_create_rejects_invalid_telegram_settings() -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let _env_guard = crate::test_support::env_lock().lock().await;
    let temp = tempdir()?;
    let runner = temp.path().join("fake-telegram-runner.sh");
    std::fs::write(
        &runner,
        "#!/bin/sh\nif [ \"$1\" = \"describe\" ]; then\n  printf '%s\\n' '{\"protocol_version\":2,\"kind\":\"telegram\"}'\n  exit 0\nfi\nif [ \"$1\" = \"validate-settings\" ]; then\n  case \"$3\" in\n    *'\"chat_id\":\"@ops\"'*)\n      printf '%s\\n' 'chat_id must be numeric' >&2\n      exit 1\n      ;;\n  esac\n  exit 0\nfi\nexit 0\n",
    )?;
    let mut perms = std::fs::metadata(&runner)?.permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&runner, perms)?;
    let previous = std::env::var_os("TURIN_CHANNEL_TELEGRAM_BIN");
    unsafe {
        std::env::set_var("TURIN_CHANNEL_TELEGRAM_BIN", &runner);
    }

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

    if let Some(value) = previous {
        unsafe {
            std::env::set_var("TURIN_CHANNEL_TELEGRAM_BIN", value);
        }
    } else {
        unsafe {
            std::env::remove_var("TURIN_CHANNEL_TELEGRAM_BIN");
        }
    }

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
            .join(".turin")
            .join("runtime")
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
            .join(".turin")
            .join("runtime")
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
    let bad_agent_dir = temp
        .path()
        .join(".turin")
        .join("runtime")
        .join("agents")
        .join("broken");
    std::fs::create_dir_all(&bad_agent_dir)?;
    std::fs::write(bad_agent_dir.join("config.toml"), "provider = [")?;

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
    let harness_dir = temp
        .path()
        .join(".turin")
        .join("harnesses")
        .join("reviewer");
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
        .join(".turin")
        .join("runtime")
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
            None,
            "Hello status".to_string(),
            None,
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
