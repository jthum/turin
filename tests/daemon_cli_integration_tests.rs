mod support;

use std::path::PathBuf;
use std::process::{Command, Output};

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
