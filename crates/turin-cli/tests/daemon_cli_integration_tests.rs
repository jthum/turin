mod support;

use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

use anyhow::{Context, Result, anyhow};
use serde_json::Value;
use tempfile::TempDir;

struct DaemonCliHarness {
    _tempdir: TempDir,
    config_path: PathBuf,
}

impl DaemonCliHarness {
    fn new() -> Result<Self> {
        let tempdir = tempfile::tempdir()?;
        let workspace_root = tempdir.path().join("workspace");
        let config_path =
            support::write_mock_runtime_config(&workspace_root, "Daemon CLI integration", "PONG")?;

        Ok(Self {
            _tempdir: tempdir,
            config_path,
        })
    }

    fn config_arg(&self) -> String {
        self.config_path.display().to_string()
    }

    fn command(&self, args: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_turin"));
        command.arg("--log-level").arg("error");
        command.args(args);
        command
    }

    fn output(&self, args: &[&str]) -> Result<Output> {
        let output = self
            .command(args)
            .output()
            .with_context(|| format!("failed to run turin {:?}", args))?;
        Ok(output)
    }

    fn json(&self, args: &[&str]) -> Result<Value> {
        let output = self.output(args)?;
        Self::parse_successful_json(args, output)
    }

    fn parse_successful_json(args: &[&str], output: Output) -> Result<Value> {
        if !output.status.success() {
            return Err(anyhow!(
                "turin {:?} failed\nstdout:\n{}\nstderr:\n{}",
                args,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        serde_json::from_slice(&output.stdout).with_context(|| {
            format!(
                "failed to parse json output for {:?}: {}",
                args,
                String::from_utf8_lossy(&output.stdout)
            )
        })
    }

    fn stop_best_effort(&self) {
        let config = self.config_arg();
        let _ = self.output(&["daemon", "stop", "--config", &config, "--json"]);
    }
}

impl Drop for DaemonCliHarness {
    fn drop(&mut self) {
        self.stop_best_effort();
    }
}

#[test]
fn daemon_ensure_health_and_logs_round_trip() -> Result<()> {
    let harness = DaemonCliHarness::new()?;
    let config = harness.config_arg();

    let health = harness.json(&["daemon", "health", "--config", &config, "--json"])?;
    assert_eq!(health["state"], "offline");
    assert_eq!(health["ready"], false);

    let ensured = harness.json(&[
        "daemon",
        "ensure",
        "--config",
        &config,
        "--json",
        "--timeout-ms",
        "10000",
    ])?;
    assert_eq!(ensured["started"], true);
    assert_eq!(ensured["health"]["state"], "ready");
    assert_eq!(ensured["health"]["ready"], true);

    let ensured_again = harness.json(&[
        "daemon",
        "ensure",
        "--config",
        &config,
        "--json",
        "--timeout-ms",
        "10000",
    ])?;
    assert_eq!(ensured_again["started"], false);
    assert_eq!(ensured_again["health"]["state"], "ready");

    let waited = harness.json(&[
        "daemon",
        "wait",
        "--config",
        &config,
        "--json",
        "--timeout-ms",
        "5000",
    ])?;
    assert_eq!(waited["state"], "ready");
    assert_eq!(waited["ready"], true);

    let logs = harness.json(&[
        "daemon", "logs", "--config", &config, "--json", "--lines", "5",
    ])?;
    assert_eq!(logs["exists"], true);
    assert!(
        logs["path"]
            .as_str()
            .is_some_and(|path| path.ends_with(".turin/daemon.log"))
    );

    let stopped = harness.json(&["daemon", "stop", "--config", &config, "--json"])?;
    assert_eq!(stopped["ok"], true);

    let health = harness.json(&["daemon", "health", "--config", &config, "--json"])?;
    assert_eq!(health["state"], "offline");
    assert_eq!(health["ready"], false);

    Ok(())
}

#[test]
fn daemon_task_survives_stop_restart_and_resume() -> Result<()> {
    let harness = DaemonCliHarness::new()?;
    let config = harness.config_arg();

    harness.json(&[
        "daemon",
        "ensure",
        "--config",
        &config,
        "--json",
        "--timeout-ms",
        "10000",
    ])?;

    let first_task = harness.json(&[
        "daemon",
        "task",
        "submit",
        "--agent",
        "default",
        "before restart",
        "--wait",
        "--timeout-ms",
        "10000",
        "--config",
        &config,
        "--json",
    ])?;
    assert_eq!(first_task["ok"], true);
    assert_eq!(first_task["result"]["status"], "success");
    assert_eq!(first_task["result"]["output"], "PONG");

    let live = harness.json(&["daemon", "session", "live", "--config", &config, "--json"])?;
    let session_id = live["result"]["sessions"][0]["session_id"]
        .as_str()
        .context("live task session should expose its session id")?
        .to_string();
    let public_session_id = session_id
        .split_once('@')
        .map_or(session_id.as_str(), |ids| ids.0);

    harness.json(&["daemon", "stop", "--config", &config, "--json"])?;
    let stopped = harness.json(&["daemon", "health", "--config", &config, "--json"])?;
    assert_eq!(stopped["state"], "offline");

    harness.json(&[
        "daemon",
        "ensure",
        "--config",
        &config,
        "--json",
        "--timeout-ms",
        "10000",
    ])?;
    let persisted = harness.json(&["daemon", "session", "list", "--config", &config, "--json"])?;
    assert!(
        persisted["result"]["sessions"]
            .as_array()
            .is_some_and(|sessions| {
                sessions
                    .iter()
                    .any(|session| session["session_id"] == public_session_id)
            }),
        "expected session {public_session_id}; persisted sessions after restart: {persisted:#}"
    );

    let resumed = harness.json(&[
        "daemon",
        "session",
        "resume",
        &session_id,
        "--config",
        &config,
        "--json",
    ])?;
    assert_eq!(resumed["ok"], true);
    assert_eq!(resumed["result"]["session_id"], session_id);

    let second_task = harness.json(&[
        "daemon",
        "task",
        "submit",
        "--session-id",
        &session_id,
        "after restart",
        "--wait",
        "--timeout-ms",
        "10000",
        "--config",
        &config,
        "--json",
    ])?;
    assert_eq!(second_task["ok"], true);
    assert_eq!(second_task["result"]["status"], "success");
    assert_eq!(second_task["result"]["output"], "PONG");

    Ok(())
}

#[test]
fn concurrent_ensure_calls_converge_on_one_ready_daemon() -> Result<()> {
    let harness = DaemonCliHarness::new()?;
    let config = harness.config_arg();
    let mut children = Vec::new();

    for _ in 0..4 {
        let mut command = harness.command(&[
            "daemon",
            "ensure",
            "--config",
            &config,
            "--json",
            "--timeout-ms",
            "10000",
        ]);
        command.stdout(Stdio::piped()).stderr(Stdio::piped());
        children.push(command.spawn()?);
    }

    for child in children {
        let output = child.wait_with_output()?;
        let report = DaemonCliHarness::parse_successful_json(&["daemon", "ensure"], output)?;
        assert_eq!(report["health"]["ready"], true);
    }

    let health = harness.json(&["daemon", "health", "--config", &config, "--json"])?;
    assert_eq!(health["state"], "ready");
    assert_eq!(health["ready"], true);
    Ok(())
}

#[test]
fn json_protocol_errors_preserve_json_and_fail_the_process() -> Result<()> {
    let harness = DaemonCliHarness::new()?;
    let config = harness.config_arg();
    harness.json(&[
        "daemon",
        "ensure",
        "--config",
        &config,
        "--json",
        "--timeout-ms",
        "10000",
    ])?;

    let output = harness.output(&[
        "daemon",
        "agent",
        "get",
        "missing-agent",
        "--config",
        &config,
        "--json",
    ])?;
    assert!(!output.status.success());
    let response: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(response["ok"], false);
    assert_eq!(response["error"]["code"], "agent_not_found");
    Ok(())
}
