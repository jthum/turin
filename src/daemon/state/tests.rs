use super::scheduled_jobs::{
    CreateScheduledJobInput, ScheduledJobOverlapPolicy, UpdateScheduledJobInput,
};
use super::worklists::WorklistItemsQuery;
use super::*;
use crate::kernel::event::TaskBranchOutcome;
use crate::kernel::session_refs::parse_session_reference;
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::LinkedSessionCreate;
use crate::persistence::state::SessionReadTarget;
use serde_json::json;
use tempfile::tempdir;
use tokio::time::{Duration, sleep};
use turin_daemon_protocol::{
    ContextPersistenceParams, HarnessActionRunParams, HarnessSourceOverlay,
    HarnessSourceSaveChange, PromoteTaskParams, ScheduleActionParams, SessionSearchScope,
    SidestepContextTargetParams, SidestepModeParams, SidestepTaskParams, StoreTargetParams,
};
use turin_types::{TaskInputContent, ToolSelectionConfig, ToolsConfig};

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

fn write_bootstrap_with_mock_response(root: &Path, response: &str) -> Result<PathBuf> {
    std::fs::create_dir_all(root.join("default-harness"))?;
    std::fs::write(
        root.join("default-harness").join("main.lua"),
        "-- bootstrap\n",
    )?;
    let config_path = root.join(".turin").join("config.toml");
    std::fs::create_dir_all(config_path.parent().expect("config parent"))?;
    std::fs::write(
        &config_path,
        format!(
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
base_url = "{response}"

[embeddings]
provider = "noop"
"#
        ),
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
            harness: None,
            idle_timeout_seconds: None,
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
async fn scheduled_one_shot_job_submits_and_disables_after_completion() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    let now_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis() as i64;
    let job = state
        .create_scheduled_job(CreateScheduledJobInput {
            agent_id: "default".to_string(),
            prompt: Some("Hello from scheduler".to_string()),
            content: None,
            tools: None,
            conflict_policy: None,
            action: None,
            persistence: None,
            next_run_unix_ms: now_unix_ms - 1,
            interval_seconds: None,
            recurring_pattern: None,
            overlap_policy: ScheduledJobOverlapPolicy::Skip,
            work_key: None,
            max_concurrency: None,
            enabled: true,
        })
        .await?;

    let default_state_store = state.kernel.store_manager().get_default().await?;
    assert!(
        default_state_store.list_scheduled_jobs().await?.is_empty(),
        "scheduled jobs should not be stored in the primary session state DB"
    );
    assert_eq!(state.runtime_store.list_scheduled_jobs().await?.len(), 1);

    let next_sleep = state.scheduler_tick().await?;
    assert!(next_sleep.is_none());

    let jobs = state.list_scheduled_jobs().await?;
    let running = jobs
        .iter()
        .find(|entry| entry.id == job.id)
        .expect("scheduled job visible after tick");
    let task_id = running
        .running_task_id
        .clone()
        .expect("scheduled job should have submitted a task");
    assert!(
        !running.enabled,
        "one-shot job should disable itself after submit"
    );

    let completed = state.wait_for_task(&task_id, Some(2_000)).await?;
    assert_eq!(completed.state, "completed");

    state.scheduler_tick().await?;
    let jobs = state.list_scheduled_jobs().await?;
    let finished = jobs
        .iter()
        .find(|entry| entry.id == job.id)
        .expect("scheduled job still visible");
    assert!(finished.running_task_id.is_none());
    assert!(!finished.enabled);
    assert!(finished.last_status.is_some());

    Ok(())
}

#[tokio::test]
async fn scheduled_interval_job_reschedules_after_submit() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    let now_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis() as i64;
    let job = state
        .create_scheduled_job(CreateScheduledJobInput {
            agent_id: "default".to_string(),
            prompt: Some("Heartbeat run".to_string()),
            content: None,
            tools: None,
            conflict_policy: None,
            action: None,
            persistence: None,
            next_run_unix_ms: now_unix_ms - 1,
            interval_seconds: Some(60),
            recurring_pattern: None,
            overlap_policy: ScheduledJobOverlapPolicy::Skip,
            work_key: None,
            max_concurrency: None,
            enabled: true,
        })
        .await?;

    state.scheduler_tick().await?;
    let jobs = state.list_scheduled_jobs().await?;
    let scheduled = jobs
        .iter()
        .find(|entry| entry.id == job.id)
        .expect("scheduled job visible");
    assert!(scheduled.running_task_id.is_some());
    assert!(scheduled.enabled);
    assert!(
        scheduled.next_run_unix_ms > now_unix_ms,
        "recurring job should advance its next due time"
    );

    Ok(())
}

#[tokio::test]
async fn scheduled_daily_job_reschedules_after_submit() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    let now_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis() as i64;
    let anchored = now_unix_ms - 3_600_000;
    let job = state
        .create_scheduled_job(CreateScheduledJobInput {
            agent_id: "default".to_string(),
            prompt: Some("Morning review".to_string()),
            content: None,
            tools: None,
            conflict_policy: None,
            action: None,
            persistence: None,
            next_run_unix_ms: anchored,
            interval_seconds: None,
            recurring_pattern: Some("daily".to_string()),
            overlap_policy: ScheduledJobOverlapPolicy::Skip,
            work_key: None,
            max_concurrency: None,
            enabled: true,
        })
        .await?;

    state.scheduler_tick().await?;
    let jobs = state.list_scheduled_jobs().await?;
    let scheduled = jobs
        .iter()
        .find(|entry| entry.id == job.id)
        .expect("scheduled job visible");
    assert_eq!(scheduled.recurring_pattern.as_deref(), Some("daily"));
    assert!(scheduled.running_task_id.is_some());
    assert!(scheduled.enabled);
    assert!(
        scheduled.next_run_unix_ms >= anchored + 86_400_000,
        "daily recurring job should advance by at least one day"
    );

    Ok(())
}

#[tokio::test]
async fn scheduled_skip_overlap_advances_without_queued_rerun() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap_with_mock_response(temp.path(), "delay_ms=1800;slow")?;
    let mut state = DaemonState::load(&config_path).await?;

    let now_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis() as i64;
    let job = state
        .create_scheduled_job(CreateScheduledJobInput {
            agent_id: "default".to_string(),
            prompt: Some("Skip overlapping slow run".to_string()),
            content: None,
            tools: None,
            conflict_policy: None,
            action: None,
            persistence: None,
            next_run_unix_ms: now_unix_ms - 1,
            interval_seconds: Some(1),
            recurring_pattern: None,
            overlap_policy: ScheduledJobOverlapPolicy::Skip,
            work_key: None,
            max_concurrency: None,
            enabled: true,
        })
        .await?;

    state.scheduler_tick().await?;
    sleep(Duration::from_millis(1100)).await;
    state.scheduler_tick().await?;

    let jobs = state.list_scheduled_jobs().await?;
    let running = jobs
        .iter()
        .find(|entry| entry.id == job.id)
        .expect("skip-overlap job visible");
    assert_eq!(running.active_run_count, 1);
    assert!(
        !running.pending_rerun,
        "skip overlap should not queue a follow-up run"
    );
    assert!(
        running.running_task_id.is_some(),
        "original slow run should remain active"
    );

    let active_runs = state.runtime_store.list_active_scheduled_job_runs().await?;
    assert_eq!(
        active_runs
            .iter()
            .filter(|run| run.scheduled_job_id == job.id)
            .count(),
        1,
        "skip overlap should not start another active run"
    );

    state
        .set_scheduled_job_enabled(&job.public_id, false)
        .await?
        .expect("skip-overlap job should disable for cleanup");
    let task_id = running.running_task_id.clone().unwrap();
    let completed = state.wait_for_task(&task_id, Some(5_000)).await?;
    assert_eq!(completed.state, "completed");

    Ok(())
}

#[tokio::test]
async fn scheduled_queue_overlap_runs_again_after_active_run_finishes() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap_with_mock_response(temp.path(), "delay_ms=1800;slow")?;
    let mut state = DaemonState::load(&config_path).await?;

    let now_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis() as i64;
    let job = state
        .create_scheduled_job(CreateScheduledJobInput {
            agent_id: "default".to_string(),
            prompt: Some("Queue overlapping slow run".to_string()),
            content: None,
            tools: None,
            conflict_policy: None,
            action: None,
            persistence: None,
            next_run_unix_ms: now_unix_ms - 1,
            interval_seconds: Some(1),
            recurring_pattern: None,
            overlap_policy: ScheduledJobOverlapPolicy::Queue,
            work_key: None,
            max_concurrency: None,
            enabled: true,
        })
        .await?;

    state.scheduler_tick().await?;
    let first_task_id = state
        .list_scheduled_jobs()
        .await?
        .iter()
        .find(|entry| entry.id == job.id)
        .and_then(|entry| entry.running_task_id.clone())
        .expect("first queued-overlap run should start");

    sleep(Duration::from_millis(1100)).await;
    state.scheduler_tick().await?;
    let jobs = state.list_scheduled_jobs().await?;
    let queued = jobs
        .iter()
        .find(|entry| entry.id == job.id)
        .expect("queued-overlap job visible");
    assert_eq!(queued.active_run_count, 1);
    assert!(
        queued.pending_rerun,
        "queue overlap should remember one follow-up run"
    );
    assert_eq!(
        queued.running_task_id.as_deref(),
        Some(first_task_id.as_str())
    );

    let completed = state.wait_for_task(&first_task_id, Some(5_000)).await?;
    assert_eq!(completed.state, "completed");

    state.scheduler_tick().await?;
    let jobs = state.list_scheduled_jobs().await?;
    let rerun = jobs
        .iter()
        .find(|entry| entry.id == job.id)
        .expect("queued-overlap job visible after rerun starts");
    assert_eq!(rerun.active_run_count, 1);
    assert!(!rerun.pending_rerun);
    let second_task_id = rerun
        .running_task_id
        .clone()
        .expect("queued rerun should start after first run finishes");
    assert_ne!(second_task_id, first_task_id);

    state
        .set_scheduled_job_enabled(&job.public_id, false)
        .await?
        .expect("queued-overlap job should disable for cleanup");
    let completed = state.wait_for_task(&second_task_id, Some(5_000)).await?;
    assert_eq!(completed.state, "completed");

    Ok(())
}

#[tokio::test]
async fn scheduled_jobs_with_same_work_key_queue_until_capacity_frees() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    let now_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis() as i64;
    let first = state
        .create_scheduled_job(CreateScheduledJobInput {
            agent_id: "default".to_string(),
            prompt: Some("First grouped run".to_string()),
            content: None,
            tools: None,
            conflict_policy: None,
            action: None,
            persistence: None,
            next_run_unix_ms: now_unix_ms - 1,
            interval_seconds: None,
            recurring_pattern: None,
            overlap_policy: ScheduledJobOverlapPolicy::Queue,
            work_key: Some("project:alpha:qa".to_string()),
            max_concurrency: Some(1),
            enabled: true,
        })
        .await?;
    let second = state
        .create_scheduled_job(CreateScheduledJobInput {
            agent_id: "default".to_string(),
            prompt: Some("Second grouped run".to_string()),
            content: None,
            tools: None,
            conflict_policy: None,
            action: None,
            persistence: None,
            next_run_unix_ms: now_unix_ms - 1,
            interval_seconds: None,
            recurring_pattern: None,
            overlap_policy: ScheduledJobOverlapPolicy::Queue,
            work_key: Some("project:alpha:qa".to_string()),
            max_concurrency: Some(1),
            enabled: true,
        })
        .await?;

    state.scheduler_tick().await?;
    let jobs = state.list_scheduled_jobs().await?;
    let first_job = jobs
        .iter()
        .find(|entry| entry.id == first.id)
        .expect("first grouped job visible");
    let second_job = jobs
        .iter()
        .find(|entry| entry.id == second.id)
        .expect("second grouped job visible");
    let first_task_id = first_job
        .running_task_id
        .clone()
        .expect("first grouped job should start");
    assert!(second_job.running_task_id.is_none());
    assert!(second_job.pending_rerun);
    assert_eq!(
        second_job.last_status.as_deref(),
        Some("blocked: concurrency limit reached")
    );

    let completed = state.wait_for_task(&first_task_id, Some(2_000)).await?;
    assert_eq!(completed.state, "completed");

    state.scheduler_tick().await?;
    let jobs = state.list_scheduled_jobs().await?;
    let second_job = jobs
        .iter()
        .find(|entry| entry.id == second.id)
        .expect("second grouped job visible after wake");
    assert!(
        second_job.running_task_id.is_some(),
        "queued grouped job should start after capacity frees"
    );
    assert!(!second_job.pending_rerun);

    Ok(())
}

#[tokio::test]
async fn scheduled_parallel_job_can_start_multiple_active_runs() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap_with_mock_response(temp.path(), "delay_ms=1800;slow")?;
    let mut state = DaemonState::load(&config_path).await?;

    let now_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis() as i64;
    let job = state
        .create_scheduled_job(CreateScheduledJobInput {
            agent_id: "default".to_string(),
            prompt: Some("Run a slow heartbeat".to_string()),
            content: None,
            tools: None,
            conflict_policy: None,
            action: None,
            persistence: None,
            next_run_unix_ms: now_unix_ms - 1,
            interval_seconds: Some(1),
            recurring_pattern: None,
            overlap_policy: ScheduledJobOverlapPolicy::Parallel,
            work_key: None,
            max_concurrency: None,
            enabled: true,
        })
        .await?;

    state.scheduler_tick().await?;
    sleep(Duration::from_millis(1100)).await;
    state.scheduler_tick().await?;

    let jobs = state.list_scheduled_jobs().await?;
    let running = jobs
        .iter()
        .find(|entry| entry.id == job.id)
        .expect("parallel scheduled job visible");
    assert_eq!(running.overlap_policy, "parallel");
    assert_eq!(running.active_run_count, 2);
    assert!(running.running_task_id.is_some());

    let active_runs = state.runtime_store.list_active_scheduled_job_runs().await?;
    let task_ids: Vec<_> = active_runs
        .into_iter()
        .filter(|run| run.scheduled_job_id == job.id)
        .map(|run| run.task_id)
        .collect();
    assert_eq!(task_ids.len(), 2);
    assert_ne!(task_ids[0], task_ids[1]);

    let active_history = state
        .scheduled_job_runs(&job.public_id, true, Some(1))
        .await?
        .expect("parallel job run history visible");
    assert_eq!(active_history.public_id, job.public_id);
    assert_eq!(active_history.runs.len(), 1);
    assert!(active_history.runs[0].active);

    state
        .set_scheduled_job_enabled(&job.public_id, false)
        .await?
        .expect("parallel job should disable for cleanup");

    for task_id in &task_ids {
        let completed = state.wait_for_task(task_id, Some(5_000)).await?;
        assert_eq!(completed.state, "completed");
    }

    state.scheduler_tick().await?;
    let jobs = state.list_scheduled_jobs().await?;
    let settled = jobs
        .iter()
        .find(|entry| entry.id == job.id)
        .expect("parallel scheduled job still visible");
    assert_eq!(settled.active_run_count, 0);

    let history = state
        .scheduled_job_runs(&job.public_id, false, None)
        .await?
        .expect("parallel job run history visible after completion");
    assert_eq!(history.runs.len(), 2);
    assert!(history.runs.iter().all(|run| !run.active));
    assert!(
        history
            .runs
            .iter()
            .all(|run| run.finished_unix_ms.is_some())
    );

    Ok(())
}

#[tokio::test]
async fn scheduled_job_can_target_named_state_store() -> Result<()> {
    let temp = tempdir()?;
    std::fs::create_dir_all(temp.path().join("default-harness"))?;
    std::fs::write(
        temp.path().join("default-harness").join("main.lua"),
        "-- bootstrap\n",
    )?;
    let config_path = temp.path().join(".turin").join("config.toml");
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

[persistence.states.project_alpha]
path = "project-alpha.db"

[harness]
directory = "default-harness"
fs_root = "."

[providers.mock]
type = "mock"

[embeddings]
provider = "noop"
"#,
    )?;

    let mut state = DaemonState::load(&config_path).await?;
    let now_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis() as i64;
    let job = state
        .create_scheduled_job(CreateScheduledJobInput {
            agent_id: "default".to_string(),
            prompt: Some("Run in project alpha store".to_string()),
            content: None,
            tools: None,
            conflict_policy: None,
            action: None,
            persistence: Some(ContextPersistenceParams {
                state: Some(StoreTargetParams {
                    path: None,
                    alias: Some("project_alpha".to_string()),
                }),
                store: None,
            }),
            next_run_unix_ms: now_unix_ms - 1,
            interval_seconds: None,
            recurring_pattern: None,
            overlap_policy: ScheduledJobOverlapPolicy::Skip,
            work_key: None,
            max_concurrency: None,
            enabled: true,
        })
        .await?;

    assert_eq!(
        job.persistence
            .as_ref()
            .and_then(|p| p.state.as_ref())
            .and_then(|state| state.alias.as_deref()),
        Some("project_alpha")
    );

    state.scheduler_tick().await?;
    let jobs = state.list_scheduled_jobs().await?;
    let running = jobs
        .iter()
        .find(|entry| entry.id == job.id)
        .expect("scheduled job visible after tick");
    let task_id = running
        .running_task_id
        .clone()
        .expect("scheduled job should have submitted a task");
    let completed = state.wait_for_task(&task_id, Some(2_000)).await?;
    assert_eq!(completed.state, "completed");

    let sessions = state
        .list_sessions(
            20,
            0,
            Some(StoreSelector::Alias("project_alpha".to_string())),
        )
        .await?;
    assert!(
        !sessions.is_empty(),
        "scheduled job should persist session state in named state store"
    );

    Ok(())
}

#[tokio::test]
async fn worklist_queries_can_target_named_state_store_and_filter_items() -> Result<()> {
    let temp = tempdir()?;
    std::fs::create_dir_all(temp.path().join("default-harness"))?;
    std::fs::write(
        temp.path().join("default-harness").join("main.lua"),
        "-- bootstrap\n",
    )?;
    let config_path = temp.path().join(".turin").join("config.toml");
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

[persistence.states.project_alpha]
path = "project-alpha.db"

[harness]
directory = "default-harness"
fs_root = "."

[providers.mock]
type = "mock"

[embeddings]
provider = "noop"
"#,
    )?;

    let state = DaemonState::load(&config_path).await?;
    let project_store = state
        .kernel
        .store_manager()
        .open(&StoreSelector::Alias("project_alpha".to_string()))
        .await?;
    let worklist = project_store
        .open_worklist("qa", "project:web-app", Some(r#"{"lane":"qa"}"#))
        .await?;

    let root = project_store
        .create_work_item(crate::persistence::state::WorkItemInsert {
            public_id: uuid::Uuid::now_v7(),
            worklist_id: worklist.id,
            parent_item_id: None,
            title: "Run smoke checks",
            item_kind: "prompt",
            prompt: Some("Run smoke checks"),
            content: None,
            tools: None,
            conflict_policy: None,
            action_name: None,
            action_params: None,
            priority: 10,
            after_ids: None,
            metadata: Some(r#"{"role":"qa"}"#),
        })
        .await?;
    let root_public_id = uuid::Uuid::from_slice(&root.public_id)?.to_string();

    let child = project_store
        .create_work_item(crate::persistence::state::WorkItemInsert {
            public_id: uuid::Uuid::now_v7(),
            worklist_id: worklist.id,
            parent_item_id: Some(root.id),
            title: "Capture browser screenshot",
            item_kind: "action",
            prompt: None,
            content: None,
            tools: None,
            conflict_policy: None,
            action_name: Some("qa.capture"),
            action_params: Some(r#"{"browser":"chromium"}"#),
            priority: 5,
            after_ids: Some(&format!("[\"{}\"]", root_public_id)),
            metadata: Some(r#"{"role":"browser"}"#),
        })
        .await?;
    let claimed = project_store
        .try_claim_work_item(
            child.id,
            "default",
            Some("sess_qa"),
            Some("exec_qa"),
            1_700_000_000_000,
        )
        .await?;
    assert!(claimed);

    let persistence = ContextPersistenceParams {
        state: Some(StoreTargetParams {
            path: None,
            alias: Some("project_alpha".to_string()),
        }),
        store: None,
    };

    let worklists = state
        .list_worklists(Some(&persistence), Some("qa"), Some("project:web-app"))
        .await?;
    assert_eq!(worklists.len(), 1);
    assert_eq!(worklists[0].name, "qa");
    assert_eq!(worklists[0].scope_ref, "project:web-app");
    assert_eq!(worklists[0].metadata, Some(json!({ "lane": "qa" })));

    let detail = state
        .worklist_detail(&worklists[0].public_id, Some(&persistence))
        .await?
        .expect("worklist detail present");
    assert_eq!(detail.public_id, worklists[0].public_id);

    let claimed_items = state
        .worklist_items(WorklistItemsQuery {
            public_id: &worklists[0].public_id,
            persistence: Some(&persistence),
            status: Some("pending"),
            parent_public_id: Some("0196f8fe-6e6a-7e1a-8da5-3f774f1a8d48"),
            where_filter: None,
            claimed_only: true,
            paused_only: false,
            due_only: false,
            limit: Some(10),
        })
        .await?
        .expect("claimed items result present");
    assert!(claimed_items.items.is_empty());

    let child_items = state
        .worklist_items(WorklistItemsQuery {
            public_id: &worklists[0].public_id,
            persistence: Some(&persistence),
            status: Some("active"),
            parent_public_id: Some(&root_public_id),
            where_filter: Some(&serde_json::Map::from_iter([(
                "role".to_string(),
                json!("browser"),
            )])),
            claimed_only: true,
            paused_only: false,
            due_only: false,
            limit: Some(10),
        })
        .await?
        .expect("child items result present");
    assert_eq!(child_items.items.len(), 1);
    assert_eq!(child_items.items[0].title, "Capture browser screenshot");
    assert_eq!(child_items.items[0].kind, "action");
    assert_eq!(
        child_items.items[0].parent_id.as_deref(),
        Some(root_public_id.as_str())
    );
    assert_eq!(
        child_items.items[0]
            .action
            .as_ref()
            .map(|action| action.name.as_str()),
        Some("qa.capture")
    );
    assert_eq!(
        child_items.items[0].metadata,
        Some(json!({ "role": "browser" }))
    );
    assert_eq!(
        child_items.items[0].claim_execution_id.as_deref(),
        Some("exec_qa")
    );
    assert!(!child_items.items[0].paused);

    let item_detail = state
        .work_item_detail(
            &uuid::Uuid::from_slice(&child.public_id)?.to_string(),
            Some(&persistence),
        )
        .await?
        .expect("work item detail present");
    assert_eq!(item_detail.title, "Capture browser screenshot");
    assert_eq!(item_detail.worklist_id, worklists[0].public_id);
    assert_eq!(
        item_detail.parent_id.as_deref(),
        Some(root_public_id.as_str())
    );
    assert_eq!(
        item_detail
            .action
            .as_ref()
            .map(|action| action.name.as_str()),
        Some("qa.capture")
    );
    assert!(!item_detail.paused);

    let paused = project_store
        .create_work_item(crate::persistence::state::WorkItemInsert {
            public_id: uuid::Uuid::now_v7(),
            worklist_id: worklist.id,
            parent_item_id: None,
            title: "Resume sync after rate limit",
            item_kind: "action",
            prompt: None,
            content: None,
            tools: None,
            conflict_policy: None,
            action_name: Some("qa.resume"),
            action_params: Some("{}"),
            priority: 1,
            after_ids: None,
            metadata: Some(
                r#"{"paused":true,"pause_reason":"rate_limited","pause_until_unix_ms":4102444800000,"checkpoint":{"cursor":"page-4"}}"#,
            ),
        })
        .await?;
    let paused = project_store
        .update_work_item(crate::persistence::state::WorkItemUpdate {
            id: paused.id,
            status: Some("paused"),
            ..Default::default()
        })
        .await?;

    let paused_detail = state
        .work_item_detail(
            &uuid::Uuid::from_slice(&paused.public_id)?.to_string(),
            Some(&persistence),
        )
        .await?
        .expect("paused work item detail present");
    assert!(paused_detail.paused);
    assert_eq!(paused_detail.pause_reason.as_deref(), Some("rate_limited"));
    assert_eq!(paused_detail.pause_until_unix_ms, Some(4_102_444_800_000));

    let paused_items = state
        .worklist_items(WorklistItemsQuery {
            public_id: &worklists[0].public_id,
            persistence: Some(&persistence),
            status: None,
            parent_public_id: None,
            where_filter: None,
            claimed_only: false,
            paused_only: true,
            due_only: false,
            limit: Some(10),
        })
        .await?
        .expect("paused items result present");
    assert_eq!(paused_items.items.len(), 1);
    assert_eq!(paused_items.items[0].title, "Resume sync after rate limit");

    let due_paused = project_store
        .create_work_item(crate::persistence::state::WorkItemInsert {
            public_id: uuid::Uuid::now_v7(),
            worklist_id: worklist.id,
            parent_item_id: None,
            title: "Resume sync immediately",
            item_kind: "action",
            prompt: None,
            content: None,
            tools: None,
            conflict_policy: None,
            action_name: Some("qa.resume"),
            action_params: Some("{}"),
            priority: 1,
            after_ids: None,
            metadata: Some(
                r#"{"paused":true,"pause_reason":"awaiting_reauth","pause_until_unix_ms":1,"checkpoint":{"cursor":"page-1"}}"#,
            ),
        })
        .await?;
    let due_paused = project_store
        .update_work_item(crate::persistence::state::WorkItemUpdate {
            id: due_paused.id,
            status: Some("paused"),
            ..Default::default()
        })
        .await?;
    project_store
        .create_work_item(crate::persistence::state::WorkItemInsert {
            public_id: uuid::Uuid::now_v7(),
            worklist_id: worklist.id,
            parent_item_id: None,
            title: "Pending item with stale pause metadata",
            item_kind: "action",
            prompt: None,
            content: None,
            tools: None,
            conflict_policy: None,
            action_name: Some("qa.resume"),
            action_params: Some("{}"),
            priority: 1,
            after_ids: None,
            metadata: Some(
                r#"{"paused":true,"pause_reason":"stale_metadata","pause_until_unix_ms":1}"#,
            ),
        })
        .await?;

    let due_items = state
        .worklist_items(WorklistItemsQuery {
            public_id: &worklists[0].public_id,
            persistence: Some(&persistence),
            status: None,
            parent_public_id: None,
            where_filter: None,
            claimed_only: false,
            paused_only: true,
            due_only: true,
            limit: Some(10),
        })
        .await?
        .expect("due paused items result present");
    assert_eq!(due_items.items.len(), 1);
    assert_eq!(
        due_items.items[0].public_id,
        uuid::Uuid::from_slice(&due_paused.public_id)?.to_string()
    );

    let due_items_without_paused_filter = state
        .worklist_items(WorklistItemsQuery {
            public_id: &worklists[0].public_id,
            persistence: Some(&persistence),
            status: None,
            parent_public_id: None,
            where_filter: None,
            claimed_only: false,
            paused_only: false,
            due_only: true,
            limit: Some(10),
        })
        .await?
        .expect("due items result present");
    assert_eq!(due_items_without_paused_filter.items.len(), 1);
    assert_eq!(
        due_items_without_paused_filter.items[0].public_id,
        uuid::Uuid::from_slice(&due_paused.public_id)?.to_string()
    );

    Ok(())
}

#[tokio::test]
async fn scheduled_action_job_can_disable_agent() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let mut state = DaemonState::load(&config_path).await?;

    let created_agent = state
        .create_agent(CreateAgentInput {
            id: "night-qa".to_string(),
            provider: "mock".to_string(),
            model: "mock-model".to_string(),
            system_prompt: Some("QA".to_string()),
            thinking: None,
            harness: None,
            idle_timeout_seconds: None,
            enabled: true,
            tools: Default::default(),
        })
        .await?;
    assert!(created_agent.enabled);

    let now_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis() as i64;
    let job = state
        .create_scheduled_job(CreateScheduledJobInput {
            agent_id: "default".to_string(),
            prompt: None,
            content: None,
            tools: None,
            conflict_policy: None,
            action: Some(ScheduleActionParams {
                name: "agent.disable".to_string(),
                params: Some(json!({ "id": "night-qa" })),
            }),
            persistence: None,
            next_run_unix_ms: now_unix_ms - 1,
            interval_seconds: None,
            recurring_pattern: None,
            overlap_policy: ScheduledJobOverlapPolicy::Skip,
            work_key: None,
            max_concurrency: None,
            enabled: true,
        })
        .await?;
    assert_eq!(job.kind, "action");
    assert!(job.prompt.is_none());
    assert_eq!(
        job.action.as_ref().map(|action| action.name.as_str()),
        Some("agent.disable")
    );

    state.scheduler_tick().await?;
    let updated_agent = state
        .agent_detail("night-qa")?
        .expect("scheduled action should leave agent visible");
    assert!(!updated_agent.enabled);

    let jobs = state.list_scheduled_jobs().await?;
    let finished = jobs
        .iter()
        .find(|entry| entry.id == job.id)
        .expect("scheduled action job visible after tick");
    assert!(!finished.enabled);
    assert_eq!(
        finished.last_status.as_deref(),
        Some("completed: agent disabled")
    );
    assert_eq!(finished.last_error_code, None);
    assert_eq!(finished.failure_count, 0);
    assert!(finished.running_task_id.is_none());

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn scheduled_harness_action_can_enqueue_followup_job() -> Result<()> {
    let temp = tempdir()?;
    std::fs::create_dir_all(temp.path().join("default-harness"))?;
    std::fs::write(
        temp.path().join("default-harness").join("main.lua"),
        r#"
            action.define("ops.enqueue_followup", function(ctx, params)
                runtime.schedule.create({
                    prompt = params.prompt or "Follow up",
                    after_seconds = params.after_seconds or 30
                })
                return {
                    status = "queued followup"
                }
            end)
        "#,
    )?;
    let config_path = temp.path().join(".turin").join("config.toml");
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

    let mut state = DaemonState::load(&config_path).await?;
    let now_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis() as i64;
    let job = state
        .create_scheduled_job(CreateScheduledJobInput {
            agent_id: "default".to_string(),
            prompt: None,
            content: None,
            tools: None,
            conflict_policy: None,
            action: Some(ScheduleActionParams {
                name: "ops.enqueue_followup".to_string(),
                params: Some(json!({
                    "prompt": "Follow up from harness action",
                    "after_seconds": 30
                })),
            }),
            persistence: None,
            next_run_unix_ms: now_unix_ms - 1,
            interval_seconds: None,
            recurring_pattern: None,
            overlap_policy: ScheduledJobOverlapPolicy::Skip,
            work_key: None,
            max_concurrency: None,
            enabled: true,
        })
        .await?;

    state.scheduler_tick().await?;
    let jobs = state.list_scheduled_jobs().await?;
    assert_eq!(
        jobs.len(),
        2,
        "expected original action job plus follow-up job"
    );
    let finished = jobs
        .iter()
        .find(|entry| entry.id == job.id)
        .expect("scheduled harness action should remain visible");
    assert_eq!(
        finished.last_status.as_deref(),
        Some("completed: queued followup")
    );
    assert_eq!(finished.last_error_code, None);
    assert_eq!(finished.failure_count, 0);
    let follow_up = jobs
        .iter()
        .find(|entry| entry.id != job.id)
        .expect("follow-up job should exist");
    assert_eq!(
        follow_up.prompt.as_deref(),
        Some("Follow up from harness action")
    );
    assert!(follow_up.action.is_none());

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn scheduled_harness_action_can_pause_and_reschedule_itself() -> Result<()> {
    let temp = tempdir()?;
    std::fs::create_dir_all(temp.path().join("default-harness"))?;
    std::fs::write(
        temp.path().join("default-harness").join("main.lua"),
        r#"
            action.define("ops.defer", function(ctx, params)
                return ctx:pause({
                    because = params.because or "later",
                    resume_in_seconds = params.after_seconds or 30,
                    checkpoint = {
                        job = params.job or "unknown"
                    },
                    note = "Pausing scheduled operation"
                })
            end)
        "#,
    )?;
    let config_path = temp.path().join(".turin").join("config.toml");
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

    let mut state = DaemonState::load(&config_path).await?;
    let now_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis() as i64;
    let job = state
        .create_scheduled_job(CreateScheduledJobInput {
            agent_id: "default".to_string(),
            prompt: None,
            content: None,
            tools: None,
            conflict_policy: None,
            action: Some(ScheduleActionParams {
                name: "ops.defer".to_string(),
                params: Some(json!({
                    "after_seconds": 30,
                    "because": "rate_limited",
                    "job": "nightly-sync"
                })),
            }),
            persistence: None,
            next_run_unix_ms: now_unix_ms - 1,
            interval_seconds: None,
            recurring_pattern: None,
            overlap_policy: ScheduledJobOverlapPolicy::Skip,
            work_key: None,
            max_concurrency: None,
            enabled: true,
        })
        .await?;

    state.scheduler_tick().await?;
    let jobs = state.list_scheduled_jobs().await?;
    assert_eq!(
        jobs.len(),
        2,
        "expected original action job plus resume job"
    );
    let finished = jobs
        .iter()
        .find(|entry| entry.id == job.id)
        .expect("scheduled harness action should remain visible");
    assert_eq!(finished.last_error_code, None);
    assert_eq!(finished.failure_count, 0);
    assert_eq!(finished.last_status.as_deref(), Some("completed: paused"));

    let follow_up = jobs
        .iter()
        .find(|entry| entry.id != job.id)
        .expect("resume job should exist");
    assert_eq!(
        follow_up.action.as_ref().map(|action| action.name.as_str()),
        Some("ops.defer")
    );
    assert_eq!(
        follow_up
            .action
            .as_ref()
            .and_then(|action| action.params.as_ref())
            .and_then(|params| params.get("job")),
        Some(&json!("nightly-sync"))
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn scheduled_builtin_worklist_dispatch_action_enqueues_prompt_task() -> Result<()> {
    let temp = tempdir()?;
    std::fs::create_dir_all(temp.path().join("default-harness"))?;
    std::fs::write(
        temp.path().join("default-harness").join("main.lua"),
        "-- builtin worklist actions do not require harness-defined handlers\n",
    )?;
    let config_path = temp.path().join(".turin").join("config.toml");
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

    let mut state = DaemonState::load(&config_path).await?;
    let default_state_store = state.kernel.store_manager().get_default().await?;
    let worklist = default_state_store
        .open_worklist("dispatch", "", None)
        .await?;
    default_state_store
        .create_work_item(crate::persistence::state::WorkItemInsert {
            public_id: uuid::Uuid::now_v7(),
            worklist_id: worklist.id,
            parent_item_id: None,
            title: "Review latest failures",
            item_kind: "prompt",
            prompt: Some("Review latest failures"),
            content: None,
            tools: None,
            conflict_policy: None,
            action_name: None,
            action_params: None,
            priority: 0,
            after_ids: None,
            metadata: Some(r#"{"role":"dev"}"#),
        })
        .await?;

    let now_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis() as i64;
    let job = state
        .create_scheduled_job(CreateScheduledJobInput {
            agent_id: "default".to_string(),
            prompt: None,
            content: None,
            tools: None,
            conflict_policy: None,
            action: Some(ScheduleActionParams {
                name: "worklist.dispatch_next".to_string(),
                params: Some(json!({
                    "name": "dispatch",
                    "where": { "role": "dev" }
                })),
            }),
            persistence: None,
            next_run_unix_ms: now_unix_ms - 1,
            interval_seconds: None,
            recurring_pattern: None,
            overlap_policy: ScheduledJobOverlapPolicy::Skip,
            work_key: None,
            max_concurrency: None,
            enabled: true,
        })
        .await?;

    state.scheduler_tick().await?;
    let jobs = state.list_scheduled_jobs().await?;
    let finished = jobs
        .iter()
        .find(|entry| entry.id == job.id)
        .expect("scheduled worklist action should remain visible");
    assert_eq!(finished.last_error_code, None);
    assert_eq!(finished.failure_count, 0);
    assert!(
        finished
            .last_status
            .as_deref()
            .is_some_and(|status| status.contains("completed"))
    );

    let task_statuses = state.list_tasks().await;
    assert_eq!(task_statuses.len(), 1);
    assert_eq!(task_statuses[0].agent_id, "default");
    assert!(!task_statuses[0].request_id.is_empty());

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn scheduled_missing_harness_action_surfaces_error_code_and_failure_count() -> Result<()> {
    let temp = tempdir()?;
    std::fs::create_dir_all(temp.path().join("default-harness"))?;
    std::fs::write(
        temp.path().join("default-harness").join("main.lua"),
        "-- no action handlers defined here\n",
    )?;
    let config_path = temp.path().join(".turin").join("config.toml");
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

    let mut state = DaemonState::load(&config_path).await?;
    let now_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis() as i64;
    let job = state
        .create_scheduled_job(CreateScheduledJobInput {
            agent_id: "default".to_string(),
            prompt: None,
            content: None,
            tools: None,
            conflict_policy: None,
            action: Some(ScheduleActionParams {
                name: "ops.missing".to_string(),
                params: Some(json!({ "message": "never handled" })),
            }),
            persistence: None,
            next_run_unix_ms: now_unix_ms - 1,
            interval_seconds: None,
            recurring_pattern: None,
            overlap_policy: ScheduledJobOverlapPolicy::Skip,
            work_key: None,
            max_concurrency: None,
            enabled: true,
        })
        .await?;

    state.scheduler_tick().await?;
    let jobs = state.list_scheduled_jobs().await?;
    let failed = jobs
        .iter()
        .find(|entry| entry.id == job.id)
        .expect("failed scheduled action job visible");
    assert_eq!(
        failed.last_error_code.as_deref(),
        Some("schedule_action_missing_handler")
    );
    assert_eq!(failed.failure_count, 1);
    assert!(
        failed
            .last_status
            .as_deref()
            .unwrap_or_default()
            .contains("schedule_action_missing_handler")
    );

    Ok(())
}

#[tokio::test]
async fn scheduled_job_lifecycle_ops_round_trip() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let state = DaemonState::load(&config_path).await?;

    let now_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis() as i64;
    let created = state
        .create_scheduled_job(CreateScheduledJobInput {
            agent_id: "default".to_string(),
            prompt: Some("Lifecycle check".to_string()),
            content: Some(vec![TaskInputContent::Text {
                text: "Use persistent QA context".to_string(),
            }]),
            tools: Some(ToolsConfig {
                selection: ToolSelectionConfig {
                    allow: Some(vec!["shell_exec".to_string()]),
                    exclude: Vec::new(),
                },
                ..ToolsConfig::default()
            }),
            conflict_policy: Some("detached".to_string()),
            action: None,
            persistence: None,
            next_run_unix_ms: now_unix_ms + 60_000,
            interval_seconds: Some(300),
            recurring_pattern: None,
            overlap_policy: ScheduledJobOverlapPolicy::Skip,
            work_key: None,
            max_concurrency: None,
            enabled: true,
        })
        .await?;

    let fetched = state
        .scheduled_job_detail(&created.public_id)
        .await?
        .expect("scheduled job should exist");
    assert_eq!(fetched.public_id, created.public_id);
    assert!(fetched.enabled);
    assert_eq!(fetched.kind, "prompt");
    assert_eq!(fetched.prompt.as_deref(), Some("Lifecycle check"));
    assert!(matches!(
        fetched.content.as_deref(),
        Some([TaskInputContent::Text { text }]) if text == "Use persistent QA context"
    ));
    assert_eq!(
        fetched
            .tools
            .as_ref()
            .and_then(|tools| tools.selection.allow.as_ref())
            .cloned(),
        Some(vec!["shell_exec".to_string()])
    );
    assert_eq!(fetched.conflict_policy.as_deref(), Some("detached"));

    let disabled = state
        .set_scheduled_job_enabled(&created.public_id, false)
        .await?
        .expect("scheduled job should disable");
    assert!(!disabled.enabled);

    let enabled = state
        .set_scheduled_job_enabled(&created.public_id, true)
        .await?
        .expect("scheduled job should enable");
    assert!(enabled.enabled);

    let updated = state
        .update_scheduled_job(
            &created.public_id,
            UpdateScheduledJobInput {
                prompt: Some("Updated lifecycle check".to_string()),
                interval_seconds: Some(120),
                recurring_pattern: None,
                work_key: Some("project:alpha:qa".to_string()),
                max_concurrency: Some(1),
                enabled: Some(false),
                ..UpdateScheduledJobInput::default()
            },
        )
        .await?
        .expect("scheduled job should update");
    assert_eq!(updated.prompt.as_deref(), Some("Updated lifecycle check"));
    assert_eq!(updated.interval_seconds, Some(120));
    assert_eq!(updated.work_key.as_deref(), Some("project:alpha:qa"));
    assert_eq!(updated.max_concurrency, Some(1));
    assert!(!updated.enabled);

    let deleted = state
        .delete_scheduled_job(&created.public_id)
        .await?
        .expect("scheduled job should delete");
    assert_eq!(deleted.public_id, created.public_id);
    assert!(
        state
            .scheduled_job_detail(&created.public_id)
            .await?
            .is_none()
    );

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
async fn session_detail_projection_bounds_messages_and_omits_events() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let state = DaemonState::load(&config_path).await?;

    let live = state.open_session("default", Some("main"), None).await?;
    for prompt in ["First turn", "Second turn"] {
        let task = state
            .submit_task(
                None,
                Some(&live.session_id),
                Some("main"),
                prompt.to_string(),
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
    }

    let full = state
        .get_session(&live.session_id)
        .await?
        .expect("full session detail");
    assert!(full.messages.len() > 2);
    assert!(!full.events.is_empty());
    assert!(full.message_window.is_none());

    let graph = state
        .get_session_graph(&live.session_id)
        .await?
        .expect("session graph");
    let first_turn = graph
        .turns
        .iter()
        .min_by_key(|turn| turn.turn_index)
        .expect("first durable turn");
    let active_branch_before = graph
        .branches
        .iter()
        .find(|branch| branch.active)
        .map(|branch| branch.branch_id.clone());
    let inspected = state
        .get_session_projection(
            &live.session_id,
            Some(first_turn.turn_id),
            Some(24),
            None,
            false,
            false,
        )
        .await?
        .expect("exact-turn session projection");
    assert!(!inspected.messages.is_empty());
    assert!(
        inspected
            .messages
            .iter()
            .all(|message| message.turn_index <= first_turn.turn_index)
    );
    assert!(
        inspected
            .message_window
            .as_ref()
            .is_some_and(|window| window.total < full.messages.len())
    );
    assert_eq!(
        inspected
            .branches
            .iter()
            .find(|branch| branch.active)
            .map(|branch| branch.branch_id.clone()),
        active_branch_before
    );

    let windowed = state
        .get_session_projection(&live.session_id, None, Some(2), None, false, true)
        .await?
        .expect("windowed session detail");
    assert_eq!(windowed.messages.len(), 2);
    assert!(windowed.events.is_empty());
    assert_eq!(windowed.execution.tasks.len(), 2);
    assert!(
        windowed
            .execution
            .tasks
            .iter()
            .all(|task| task.status == "success" && !task.turns.is_empty())
    );
    assert!(!windowed.execution.truncated);
    let efficiency = windowed.efficiency.expect("efficiency projection");
    assert_eq!(efficiency.turns.len(), 1);
    assert_eq!(efficiency.total_request_count, 2);
    assert!(efficiency.total_input_tokens > 0);
    assert!(efficiency.total_output_tokens > 0);
    assert_eq!(
        efficiency
            .turns
            .iter()
            .map(|turn| turn.requests.len())
            .sum::<usize>(),
        1
    );
    assert!(
        efficiency
            .turns
            .iter()
            .flat_map(|turn| &turn.requests)
            .all(|request| request.metrics.as_ref().is_some_and(|metrics| {
                metrics.estimated_input_tokens > 0
                    && metrics.sent_message_count <= metrics.available_message_count
                    && request.input_tokens.is_some()
                    && request.output_tokens.is_some()
            }))
    );
    assert!(
        windowed
            .messages
            .iter()
            .all(|message| { message.estimated_token_count.is_some() || message.role == "system" })
    );
    let window = windowed.message_window.expect("message window metadata");
    assert_eq!(window.total, full.messages.len());
    assert_eq!(window.offset, full.messages.len() - 2);

    let first_window = state
        .get_session_projection(&live.session_id, None, Some(2), Some(0), false, false)
        .await?
        .expect("offset session detail");
    let first_window_meta = first_window.message_window.expect("offset metadata");
    assert_eq!(first_window_meta.total, full.messages.len());
    assert_eq!(first_window_meta.offset, 0);
    assert_eq!(first_window.messages[0].id, full.messages[0].id);

    Ok(())
}

#[tokio::test]
async fn session_delete_rejects_a_live_runtime() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let state = DaemonState::load(&config_path).await?;
    let live = state
        .open_session("default", Some("delete-guard"), None)
        .await?;

    let error = state.delete_session(&live.session_id).await.unwrap_err();
    assert!(error.downcast_ref::<SessionDeleteBusy>().is_some());
    assert!(state.get_session(&live.session_id).await?.is_some());
    Ok(())
}

#[tokio::test]
async fn detached_sidestep_task_can_be_promoted_to_sibling_branch() -> Result<()> {
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
    let promotion = sidestep
        .promotion_candidate
        .clone()
        .expect("detached sidestep should expose a promotion candidate");
    assert_eq!(promotion.session_id, live.session_id);

    let promoted = state
        .promote_task_params(PromoteTaskParams {
            request_id: sidestep.request_id.clone(),
            branch_name: Some("kept-side-question".to_string()),
            source_turn_id: None,
        })
        .await?;
    assert_eq!(promoted.name, "kept-side-question");
    assert_eq!(promoted.source_turn_id, Some(promotion.source_turn_id));
    assert_eq!(promoted.origin_kind, "promotion");
    assert_eq!(
        promoted.origin_task_id.as_deref(),
        Some(sidestep.request_id.as_str())
    );
    assert!(!promoted.active);
    let promoted_again = state
        .promote_task_params(PromoteTaskParams {
            request_id: sidestep.request_id.clone(),
            branch_name: Some("should-not-create-new-branch".to_string()),
            source_turn_id: None,
        })
        .await?;
    assert_eq!(promoted_again.branch_id, promoted.branch_id);
    assert_eq!(promoted_again.name, promoted.name);

    let task_after_promote = state
        .get_task(&sidestep.request_id)
        .await
        .expect("promoted task should remain visible");
    assert_eq!(
        task_after_promote
            .promoted_branch
            .as_ref()
            .map(|branch| branch.branch_id.as_str()),
        Some(promoted.branch_id.as_str())
    );
    assert!(state.list_tasks().await.iter().any(|task| {
        task.request_id == sidestep.request_id
            && task
                .promoted_branch
                .as_ref()
                .is_some_and(|branch| branch.branch_id == promoted.branch_id)
    }));

    let session_ref = parse_session_reference(&live.session_id)?;
    let store_selector = session_ref
        .store_selector
        .unwrap_or_else(|| StoreSelector::Alias("state".to_string()));
    let store = state.kernel.store_manager().open(&store_selector).await?;
    let session_detail = state
        .get_session(&live.session_id)
        .await?
        .expect("persisted session detail visible");
    let branch_uuid = uuid::Uuid::parse_str(&promoted.branch_id)?;
    let branch = store
        .get_branch_head_by_name(session_detail.session.internal_id, &promoted.name)
        .await?
        .expect("promoted branch should be queryable");
    assert_eq!(
        uuid::Uuid::from_slice(&branch.public_id)?.to_string(),
        branch_uuid.to_string()
    );
    assert_eq!(branch.origin_kind, "promotion");
    assert_eq!(
        branch.origin_task_id.as_deref(),
        Some(sidestep.request_id.as_str())
    );
    let messages = store
        .get_messages(
            session_detail.session.internal_id,
            &SessionReadTarget::BranchHead(branch.id),
        )
        .await?;
    assert!(messages.iter().any(|message| {
        message.role == "user" && message.content.contains("Explore a side question")
    }));

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
    let sidestep_branch = after_branches
        .iter()
        .find(|branch| !branch.active && branch.name.starts_with("sidestep-"))
        .expect("sidestep sibling branch visible");
    assert_eq!(sidestep_branch.origin_kind, "sidestep");

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
async fn sidestep_task_rejects_missing_branch_head_target() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let state = DaemonState::load(&config_path).await?;

    let live = state.open_session("default", Some("main"), None).await?;
    let err = state
        .sidestep_task_params(SidestepTaskParams {
            session_id: live.session_id.clone(),
            slot_id: None,
            prompt: "Explore invalid target".to_string(),
            content: None,
            tools: None,
            mode: SidestepModeParams::Ephemeral,
            context_target: Some(SidestepContextTargetParams::BranchHead {
                branch_head_id: 9_999_999,
            }),
            timeout_ms: Some(2_000),
        })
        .await
        .expect_err("missing branch head target should fail during sidestep preparation");

    let err_text = err.to_string();
    assert!(
        err_text.contains("Branch head '9999999' not found"),
        "unexpected sidestep target error: {err_text}"
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
async fn session_listing_separates_roots_from_linked_children() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let state = DaemonState::load(&config_path).await?;
    let store = state.kernel.store_manager().get_default().await?;
    let root_public_id = uuid::Uuid::now_v7();
    let root_id = store
        .create_session(root_public_id, "default", None)
        .await?;
    store
        .create_linked_session(
            uuid::Uuid::now_v7(),
            "reviewer",
            None,
            &LinkedSessionCreate {
                parent_session_id: root_id,
                origin_turn_id: None,
                relation_kind: "delegated".to_string(),
                thread_key: "review".to_string(),
                visibility: "contextual".to_string(),
            },
        )
        .await?;

    let roots = state.list_sessions(10, 0, None).await?;
    assert_eq!(roots.len(), 1);
    assert_eq!(roots[0].visibility, "top_level");

    let children = state
        .list_linked_sessions(&root_public_id.simple().to_string(), 10, 0)
        .await?
        .expect("root exists");
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].agent_id, "reviewer");
    assert_eq!(children[0].parent_internal_id, Some(root_id));
    assert_eq!(children[0].thread_key.as_deref(), Some("review"));
    assert_eq!(children[0].visibility, "contextual");
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

    let graph = state
        .get_session_graph(&session_id)
        .await?
        .expect("session graph visible");
    assert_eq!(graph.turns.len(), 3);
    assert_eq!(graph.branches.len(), 2);
    let inactive_main_turn = graph
        .turns
        .iter()
        .find(|turn| {
            turn.preview
                .as_deref()
                .is_some_and(|preview| preview.contains("second branch turn"))
        })
        .expect("inactive main turn remains in graph");
    let exact = state
        .create_session_branch_from_turn_id(
            &session_id,
            "exact-main-fork",
            None,
            inactive_main_turn.turn_id,
            false,
        )
        .await?
        .expect("exact turn branch created");
    assert_eq!(exact.source_turn_id, Some(inactive_main_turn.turn_id));
    assert_eq!(exact.head_turn_id, Some(inactive_main_turn.turn_id));

    Ok(())
}

#[tokio::test]
async fn session_branch_siblings_can_be_queried_by_source_turn() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let state = DaemonState::load(&config_path).await?;

    let live = state.open_session("default", Some("main"), None).await?;
    let session_id = live.session_id.clone();

    let first = state
        .submit_task(
            None,
            Some(&session_id),
            Some("main"),
            "first branch turn".to_string(),
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

    let alt_a = state
        .create_session_branch(&session_id, "alt-a", None, Some(0), false)
        .await?
        .expect("alt-a branch created");
    let alt_b = state
        .create_session_branch(&session_id, "alt-b", None, Some(0), false)
        .await?
        .expect("alt-b branch created");
    let _alt_c = state
        .create_session_branch(&session_id, "alt-c", None, Some(1), false)
        .await?
        .expect("alt-c branch created");

    let source_turn_id = alt_a.source_turn_id.expect("alt-a source turn id");
    assert_eq!(alt_b.source_turn_id, Some(source_turn_id));

    let siblings = state
        .list_session_branch_siblings(&session_id, source_turn_id)
        .await?
        .expect("session exists");
    assert_eq!(siblings.len(), 2);
    assert!(siblings.iter().any(|branch| branch.name == "alt-a"));
    assert!(siblings.iter().any(|branch| branch.name == "alt-b"));
    assert!(!siblings.iter().any(|branch| branch.name == "alt-c"));

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
            harness: Some("shared".to_string()),
            idle_timeout_seconds: None,
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
async fn harness_source_candidates_resolve_unsaved_nested_modules() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let state = DaemonState::load(&config_path).await?;

    let validation = state.validate_harness_sources(
        "default",
        vec![
            HarnessSourceOverlay {
                path: "main.lua".to_string(),
                source: Some("use('libs/security')".to_string()),
            },
            HarnessSourceOverlay {
                path: "libs/security.lua".to_string(),
                source: Some("return { policy = 'strict' }".to_string()),
            },
        ],
    )?;

    assert!(validation.valid);
    assert_eq!(validation.script_count, 2);
    assert!(
        !temp
            .path()
            .join("default-harness/libs/security.lua")
            .exists()
    );
    Ok(())
}

#[tokio::test]
async fn harness_source_save_is_hash_guarded_and_supports_nested_paths() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let state = DaemonState::load(&config_path).await?;

    let original = state.get_harness_source("default", "main.lua")?;
    let saved = state.save_harness_sources(
        "default",
        vec![
            HarnessSourceSaveChange {
                path: "main.lua".to_string(),
                source: Some("use('libs/security')\n".to_string()),
                expected_hash: Some(original.hash.clone()),
            },
            HarnessSourceSaveChange {
                path: "libs/security.lua".to_string(),
                source: Some("return { policy = 'strict' }\n".to_string()),
                expected_hash: None,
            },
        ],
    )?;

    assert_eq!(saved.saved.len(), 2);
    let nested = state.get_harness_source("default", "libs/security.lua")?;
    assert!(nested.source.contains("strict"));
    assert!(
        state
            .list_harness_sources("default")?
            .files
            .iter()
            .any(|file| file.path == "libs/security.lua")
    );

    let stale = state
        .save_harness_sources(
            "default",
            vec![HarnessSourceSaveChange {
                path: "main.lua".to_string(),
                source: Some("-- stale overwrite\n".to_string()),
                expected_hash: Some(original.hash),
            }],
        )
        .expect_err("stale source hash must conflict");
    assert!(stale.downcast_ref::<HarnessSourceConflict>().is_some());
    assert!(
        state
            .get_harness_source("default", "main.lua")?
            .source
            .contains("libs/security")
    );

    let deleted = state.save_harness_sources(
        "default",
        vec![HarnessSourceSaveChange {
            path: "libs/security.lua".to_string(),
            source: None,
            expected_hash: Some(nested.hash),
        }],
    )?;
    assert_eq!(deleted.deleted, vec!["libs/security.lua"]);
    assert!(
        !temp
            .path()
            .join("default-harness/libs/security.lua")
            .exists()
    );
    assert!(
        state
            .get_harness_source("default", "../escape.lua")
            .is_err()
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn unbound_shared_harness_action_runs_in_its_named_runtime() -> Result<()> {
    let temp = tempdir()?;
    let config_path = write_bootstrap(temp.path())?;
    let shared_harness = temp.path().join(".turin").join("harnesses").join("catalog");
    std::fs::create_dir_all(&shared_harness)?;
    std::fs::write(
        shared_harness.join("main.lua"),
        r#"
            action.define("catalog.seed", function(_ctx, params)
                local list = worklist("catalog")
                list:add({
                    title = params.title,
                    kind = "approval",
                    action = "catalog.approve",
                })
                return { status = "seeded" }
            end)
        "#,
    )?;

    let state = DaemonState::load(&config_path).await?;
    let snapshot = state
        .kernel
        .harness_snapshot("catalog")
        .expect("shared harness should be visible");
    assert!(snapshot.bound_agents.is_empty());

    let result = state.run_harness_action(HarnessActionRunParams {
        action: "catalog.seed".to_string(),
        agent_id: None,
        harness_id: Some("catalog".to_string()),
        params: json!({ "title": "Review catalog item" }),
    })?;
    assert_eq!(result.agent_id, "default");
    assert_eq!(result.harness_id.as_deref(), Some("catalog"));
    assert_eq!(result.result["status"], "seeded");

    let worklist = state
        .list_worklists(None, Some("catalog"), None)
        .await?
        .into_iter()
        .next()
        .expect("action should create its worklist");
    let items = state
        .worklist_items(WorklistItemsQuery {
            public_id: &worklist.public_id,
            persistence: None,
            status: None,
            parent_public_id: None,
            where_filter: None,
            claimed_only: false,
            paused_only: false,
            due_only: false,
            limit: None,
        })
        .await?
        .expect("created worklist should remain readable");
    assert_eq!(items.items.len(), 1);
    assert_eq!(items.items[0].title, "Review catalog item");

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
            idle_timeout_seconds: Some(600),
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
                idle_timeout_seconds: Some(900),
                settings: Some(json!({
                    "token_env": "NEW_TOKEN",
                    "guild_id": "123",
                })),
                ..UpdateChannelInput::default()
            },
        )
        .await?;
    assert_eq!(updated.idle_timeout_seconds, Some(900));
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
            idle_timeout_seconds: Some(600),
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
            idle_timeout_seconds: Some(600),
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
            idle_timeout_seconds: Some(600),
            enabled: false,
            settings: json!({
                "token_env": "TELEGRAM_BOT_TOKEN",
                "chat_id": -100123456,
                "poll_timeout_seconds": 10,
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
                idle_timeout_seconds: Some(900),
                settings: Some(json!({
                    "workspace_id": "ops",
                    "poll_interval_ms": 250,
                })),
                ..UpdateChannelInput::default()
            },
        )
        .await?;
    assert_eq!(updated.idle_timeout_seconds, Some(900));
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
            idle_timeout_seconds: Some(600),
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
            idle_timeout_seconds: Some(600),
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
            idle_timeout_seconds: Some(600),
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
            idle_timeout_seconds: Some(600),
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
            idle_timeout_seconds: Some(600),
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
            harness: None,
            idle_timeout_seconds: None,
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
            harness: None,
            idle_timeout_seconds: None,
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
            harness: None,
            idle_timeout_seconds: None,
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
