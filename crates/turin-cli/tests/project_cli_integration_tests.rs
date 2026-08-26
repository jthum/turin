use std::process::{Command, Output};

use anyhow::{Context, Result, anyhow};
use tempfile::TempDir;

struct ProjectCliHarness {
    tempdir: TempDir,
}

impl ProjectCliHarness {
    fn new() -> Result<Self> {
        Ok(Self {
            tempdir: tempfile::tempdir()?,
        })
    }

    fn root(&self) -> &std::path::Path {
        self.tempdir.path()
    }

    fn command(&self, args: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_turin"));
        command.arg("--log-level").arg("error");
        command.args(args);
        command.current_dir(self.root());
        command
    }

    fn output(&self, args: &[&str]) -> Result<Output> {
        self.command(args)
            .output()
            .with_context(|| format!("failed to run turin {:?}", args))
    }

    fn successful_output(&self, args: &[&str]) -> Result<Output> {
        let output = self.output(args)?;
        if !output.status.success() {
            return Err(anyhow!(
                "turin {:?} failed\nstdout:\n{}\nstderr:\n{}",
                args,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok(output)
    }

    fn combined_text(output: &Output) -> String {
        format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    }
}

#[test]
fn init_scaffolds_project_and_gitignore() -> Result<()> {
    let harness = ProjectCliHarness::new()?;
    harness.successful_output(&[
        "init",
        "--yes",
        "--provider",
        "mock",
        "--model",
        "mock-model",
        "--harness-template",
        "starter",
    ])?;

    assert!(harness.root().join(".turin/config.toml").exists());
    assert!(harness.root().join(".turin/harnesses/main.lua").exists());
    assert!(harness.root().join(".turin/data/state.db").exists());
    let config = std::fs::read_to_string(harness.root().join(".turin/config.toml"))?;
    assert!(config.contains("enforcement_enabled = false"));
    assert!(config.contains("unmatched_capability = \"allow\""));
    assert!(!config.contains("profile ="));

    let gitignore = std::fs::read_to_string(harness.root().join(".gitignore"))?;
    assert!(gitignore.contains(".turin/"));

    Ok(())
}

#[test]
fn quickstart_creates_project_and_runs_prompt() -> Result<()> {
    let harness = ProjectCliHarness::new()?;
    let output =
        harness.successful_output(&["quickstart", "--yes", "--prompt", "Say QUICKSTART_OK"])?;

    assert!(harness.root().join(".turin/config.toml").exists());
    assert!(
        harness
            .root()
            .join(".turin/harnesses/00_safety.lua")
            .exists()
    );
    let text = ProjectCliHarness::combined_text(&output);
    assert!(text.contains("Quickstart scaffolded"));
    assert!(text.contains("Turin quickstart is wired correctly"));

    Ok(())
}

#[test]
fn harness_new_and_test_round_trip() -> Result<()> {
    let harness = ProjectCliHarness::new()?;
    harness.successful_output(&[
        "init",
        "--yes",
        "--provider",
        "mock",
        "--model",
        "mock-model",
    ])?;

    harness.successful_output(&[
        "harness",
        "new",
        "reviewer",
        "--dir",
        ".turin/harnesses-reviewer",
    ])?;

    assert!(
        harness
            .root()
            .join(".turin/harnesses-reviewer/main.lua")
            .exists()
    );

    let output = harness.successful_output(&[
        "harness",
        "test",
        "--dir",
        ".turin/harnesses-reviewer",
        "--response",
        "HARNESS_TEST_OK",
        "--prompt",
        "Say HARNESS_TEST_OK",
    ])?;

    let text = ProjectCliHarness::combined_text(&output);
    assert!(text.contains("HARNESS_TEST_OK"));

    Ok(())
}
