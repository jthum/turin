use anyhow::{Result, bail};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use turin::display;
use turin::kernel::config::TurinConfig;
use turin::persistence::manager::StoreSelector;
use turin_daemon_client::{DaemonClient, DaemonHealthState};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CheckStatus {
    Pass,
    Warning,
    Failed,
}

#[derive(Debug, Serialize)]
struct CheckResult {
    name: String,
    status: CheckStatus,
    message: String,
}

#[derive(Debug, Serialize)]
struct ProjectCheckReport {
    config_path: String,
    status: CheckStatus,
    checks: Vec<CheckResult>,
}

impl ProjectCheckReport {
    fn new(config_path: &Path) -> Self {
        Self {
            config_path: config_path.display().to_string(),
            status: CheckStatus::Pass,
            checks: Vec::new(),
        }
    }

    fn push(&mut self, name: impl Into<String>, status: CheckStatus, message: impl Into<String>) {
        self.status = match (self.status, status) {
            (CheckStatus::Failed, _) | (_, CheckStatus::Failed) => CheckStatus::Failed,
            (CheckStatus::Warning, _) | (_, CheckStatus::Warning) => CheckStatus::Warning,
            _ => CheckStatus::Pass,
        };
        self.checks.push(CheckResult {
            name: name.into(),
            status,
            message: message.into(),
        });
    }

    fn failed(&self) -> bool {
        self.status == CheckStatus::Failed
    }
}

struct ProjectInspection {
    report: ProjectCheckReport,
    config: Option<TurinConfig>,
}

pub async fn run_check(config_path: &Path, json_output: bool) -> Result<()> {
    let inspection = inspect_project(config_path).await;
    print_report("Turin project check", &inspection.report, json_output)?;
    if inspection.report.failed() {
        bail!("project validation failed");
    }
    Ok(())
}

pub async fn run_doctor(config_path: &Path, json_output: bool) -> Result<()> {
    let mut inspection = inspect_project(config_path).await;
    if let Some(config) = inspection.config.as_ref() {
        inspect_daemon(config_path, config, &mut inspection.report).await;
    } else {
        inspection.report.push(
            "daemon",
            CheckStatus::Warning,
            "Skipped because the Turin configuration could not be loaded.",
        );
    }

    print_report("Turin doctor", &inspection.report, json_output)?;
    if inspection.report.failed() {
        bail!("Turin doctor found blocking problems");
    }
    Ok(())
}

async fn inspect_project(config_path: &Path) -> ProjectInspection {
    let mut report = ProjectCheckReport::new(config_path);
    let config = match TurinConfig::from_file(config_path) {
        Ok(config) => {
            report.push(
                "configuration",
                CheckStatus::Pass,
                "Configuration is valid TOML and passes Turin validation.",
            );
            config
        }
        Err(err) => {
            report.push("configuration", CheckStatus::Failed, format!("{err:#}"));
            return ProjectInspection {
                report,
                config: None,
            };
        }
    };

    inspect_provider_credentials(&config, &mut report);
    inspect_harness_directories(&config, &mut report);
    inspect_harness_runtime(&config, &mut report).await;
    inspect_state_database(&config, &mut report);

    ProjectInspection {
        report,
        config: Some(config),
    }
}

fn inspect_provider_credentials(config: &TurinConfig, report: &mut ProjectCheckReport) {
    let mut users = BTreeMap::<&str, BTreeSet<&str>>::new();
    users
        .entry(&config.agent.provider)
        .or_default()
        .insert(&config.agent.id);
    for agent in config.agents.values() {
        users.entry(&agent.provider).or_default().insert(&agent.id);
    }

    for (provider_name, agent_ids) in users {
        let check_name = format!("provider.{provider_name}");
        let agents = agent_ids.into_iter().collect::<Vec<_>>().join(", ");
        let Some(provider) = config.providers.get(provider_name) else {
            report.push(
                check_name,
                CheckStatus::Failed,
                format!("Provider is used by {agents} but is not configured."),
            );
            continue;
        };

        let Some(env_name) = provider.api_key_env.as_deref() else {
            report.push(
                check_name,
                CheckStatus::Pass,
                format!("Provider '{}' is configured for {agents}.", provider.kind),
            );
            continue;
        };

        if config
            .environment_value(env_name)
            .is_some_and(|value| !value.trim().is_empty())
        {
            report.push(
                check_name,
                CheckStatus::Pass,
                format!("Credential {env_name} is available for {agents}."),
            );
        } else {
            report.push(
                check_name,
                CheckStatus::Warning,
                format!(
                    "Credential {env_name} is not set for {agents}; set it in the process environment or the configured workspace env file."
                ),
            );
        }
    }
}

fn inspect_harness_directories(config: &TurinConfig, report: &mut ProjectCheckReport) {
    let entries = std::iter::once(("default", &config.harness)).chain(
        config
            .harnesses
            .iter()
            .map(|(id, harness)| (id.as_str(), harness)),
    );

    for (harness_id, harness) in entries {
        let directory = Path::new(&harness.directory);
        let (status, message) = if directory.is_dir() {
            (
                CheckStatus::Pass,
                format!("Harness directory '{}' exists.", directory.display()),
            )
        } else {
            (
                CheckStatus::Warning,
                format!(
                    "Harness directory '{}' does not exist; an empty harness will be used.",
                    directory.display()
                ),
            )
        };
        report.push(format!("harness.{harness_id}.directory"), status, message);
    }
}

async fn inspect_harness_runtime(config: &TurinConfig, report: &mut ProjectCheckReport) {
    let mut kernel = match crate::composition::kernel_builder(config.clone()).build() {
        Ok(kernel) => kernel,
        Err(err) => {
            report.push("harness.runtime", CheckStatus::Failed, format!("{err:#}"));
            return;
        }
    };

    match kernel.init_harness().await {
        Ok(()) => {
            let snapshots = kernel.harness_snapshots();
            let script_count = snapshots
                .iter()
                .map(|snapshot| snapshot.loaded_scripts.len())
                .sum::<usize>();
            report.push(
                "harness.runtime",
                CheckStatus::Pass,
                format!(
                    "Validated {} harness runtime(s) and {script_count} Lua script(s).",
                    snapshots.len()
                ),
            );
        }
        Err(err) => report.push("harness.runtime", CheckStatus::Failed, format!("{err:#}")),
    }
    kernel.shutdown().await;
}

fn inspect_state_database(config: &TurinConfig, report: &mut ProjectCheckReport) {
    let state_path = match resolve_state_path(config) {
        Ok(path) => path,
        Err(err) => {
            report.push("persistence.state", CheckStatus::Failed, format!("{err:#}"));
            return;
        }
    };

    let message = if state_path.exists() {
        format!("State database is available at '{}'.", state_path.display())
    } else {
        format!(
            "State database will be created at '{}' on first runtime start.",
            state_path.display()
        )
    };
    report.push("persistence.state", CheckStatus::Pass, message);
}

fn resolve_state_path(config: &TurinConfig) -> Result<PathBuf> {
    let path = match config.persistence.top_level_state_selector()? {
        StoreSelector::Alias(alias) => config
            .persistence
            .states
            .get(&alias)
            .map(|target| PathBuf::from(&target.path))
            .unwrap_or_else(|| Path::new(&config.layout.data_dir).join("state.db")),
        StoreSelector::Path(path) => PathBuf::from(path),
        StoreSelector::Handle(_) => Path::new(&config.layout.data_dir).join("state.db"),
    };
    Ok(path)
}

async fn inspect_daemon(config_path: &Path, config: &TurinConfig, report: &mut ProjectCheckReport) {
    let config_base = config_path.parent().unwrap_or_else(|| Path::new("."));
    let endpoint = config.resolve_daemon_endpoint(config_base);
    let client = DaemonClient::new(&endpoint);

    match client.health().await {
        Ok(health) => {
            let status = if health.state == DaemonHealthState::Ready {
                CheckStatus::Pass
            } else {
                CheckStatus::Warning
            };
            report.push(
                "daemon",
                status,
                format!(
                    "{} at '{}' with {} agent(s), {} harness(es), and {} issue(s).",
                    match health.state {
                        DaemonHealthState::Ready => "Ready",
                        DaemonHealthState::Degraded => "Degraded",
                    },
                    endpoint.display(),
                    health.agent_count,
                    health.harness_count,
                    health.issue_count
                ),
            );
        }
        Err(err) if daemon_is_offline(&err) => report.push(
            "daemon",
            CheckStatus::Warning,
            format!(
                "Not running at '{}'. Direct CLI runs still work; start the service with `turin daemon ensure --config {}`.",
                endpoint.display(),
                config_path.display()
            ),
        ),
        Err(err) => report.push("daemon", CheckStatus::Failed, format!("{err:#}")),
    }
}

fn daemon_is_offline(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| cause.is::<std::io::Error>())
        || err.to_string().contains("Failed to connect to")
        || err
            .to_string()
            .contains("Daemon closed connection before response")
}

fn print_report(title: &str, report: &ProjectCheckReport, json_output: bool) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string_pretty(report)?);
        return Ok(());
    }

    let ansi = display::stdout_ansi();
    println!("{}", display::header(title, ansi));
    println!("Config: {}\n", report.config_path);
    for check in &report.checks {
        let mark = match check.status {
            CheckStatus::Pass => display::ok_mark(ansi),
            CheckStatus::Warning => display::warn_mark(ansi),
            CheckStatus::Failed => display::err_mark(ansi),
        };
        println!("{mark} {}: {}", check.name, check.message);
    }
    println!("\nResult: {}", status_label(report.status));
    Ok(())
}

fn status_label(status: CheckStatus) -> &'static str {
    match status {
        CheckStatus::Pass => "ready",
        CheckStatus::Warning => "ready with warnings",
        CheckStatus::Failed => "failed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::scaffold::{
        GovernancePreset, HarnessTemplate, InitOptions, InitProvider, scaffold_project,
    };
    use tempfile::TempDir;

    #[tokio::test]
    async fn scaffolded_mock_project_passes_reference_check() -> Result<()> {
        let temp = TempDir::new()?;
        scaffold_project(
            temp.path(),
            &InitOptions {
                provider: InitProvider::Mock,
                model: "mock-model".to_string(),
                harness_template: HarnessTemplate::Starter,
                governance: GovernancePreset::Balanced,
                force: false,
            },
        )?;

        let inspection = inspect_project(&temp.path().join(".turin/config.toml")).await;
        assert_eq!(inspection.report.status, CheckStatus::Pass);
        assert!(inspection.config.is_some());
        Ok(())
    }

    #[tokio::test]
    async fn malformed_config_is_a_blocking_failure() -> Result<()> {
        let temp = TempDir::new()?;
        let config_path = temp.path().join("config.toml");
        std::fs::write(&config_path, "not = [valid")?;

        let inspection = inspect_project(&config_path).await;
        assert!(inspection.report.failed());
        assert!(inspection.config.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn project_check_reads_credential_from_workspace_env_file() -> Result<()> {
        let temp = TempDir::new()?;
        scaffold_project(
            temp.path(),
            &InitOptions {
                provider: InitProvider::Openai,
                model: "test-model".to_string(),
                harness_template: HarnessTemplate::Starter,
                governance: GovernancePreset::Balanced,
                force: false,
            },
        )?;
        let config_path = temp.path().join(".turin/config.toml");
        let config_source = std::fs::read_to_string(&config_path)?
            .replace("OPENAI_API_KEY", "TURIN_REFERENCE_CHECK_TEST_KEY");
        std::fs::write(&config_path, config_source)?;
        std::fs::write(
            temp.path().join(".turin/.env"),
            "TURIN_REFERENCE_CHECK_TEST_KEY=from-workspace\n",
        )?;

        let inspection = inspect_project(&config_path).await;
        assert_eq!(inspection.report.status, CheckStatus::Pass);
        Ok(())
    }

    #[tokio::test]
    async fn invalid_harness_is_a_blocking_failure() -> Result<()> {
        let temp = TempDir::new()?;
        scaffold_project(
            temp.path(),
            &InitOptions {
                provider: InitProvider::Mock,
                model: "mock-model".to_string(),
                harness_template: HarnessTemplate::Starter,
                governance: GovernancePreset::Balanced,
                force: false,
            },
        )?;
        std::fs::write(
            temp.path().join(".turin/harnesses/main.lua"),
            "function broken(",
        )?;

        let inspection = inspect_project(&temp.path().join(".turin/config.toml")).await;
        assert!(inspection.report.failed());
        assert!(inspection.report.checks.iter().any(|check| {
            check.name == "harness.runtime" && check.status == CheckStatus::Failed
        }));
        Ok(())
    }

    #[test]
    fn report_retains_the_most_severe_status() {
        let mut report = ProjectCheckReport::new(Path::new("config.toml"));
        report.push("warning", CheckStatus::Warning, "warning");
        report.push("pass", CheckStatus::Pass, "pass");
        assert_eq!(report.status, CheckStatus::Warning);

        report.push("failure", CheckStatus::Failed, "failure");
        report.push("later warning", CheckStatus::Warning, "warning");
        assert_eq!(report.status, CheckStatus::Failed);
    }
}
