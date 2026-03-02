use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde::Serialize;

use crate::daemon::registry::{
    AgentFileConfig, RegistryIssue, RegistryLoad, RegistrySnapshot, build_effective_config,
    read_agent_file, scan_registry, snapshot, write_agent_file,
};
use crate::kernel::Kernel;
use crate::kernel::agent_manager::{AgentStatusSnapshot, TaskStatusSnapshot};
use crate::kernel::config::{AgentMode, ThinkingConfig, TurinConfig};
use crate::kernel::session::QueuedTask;

#[derive(Debug, Clone)]
pub struct DaemonWatchPaths {
    pub config_path: PathBuf,
    pub agents_dir: PathBuf,
    pub harnesses_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
pub struct DaemonStatus {
    pub config_path: String,
    pub workspace_root: String,
    pub socket_path: String,
    pub registry: RegistrySnapshot,
    pub harnesses: Vec<crate::kernel::HarnessRuntimeSnapshot>,
    pub agent_runtimes: Vec<AgentStatusSnapshot>,
}

pub struct DaemonState {
    config_path: PathBuf,
    config_base: PathBuf,
    bootstrap_config: TurinConfig,
    socket_path: PathBuf,
    registry_load: RegistryLoad,
    kernel: Kernel,
}

#[derive(Debug, Clone)]
pub struct CreateAgentInput {
    pub id: String,
    pub provider: String,
    pub model: String,
    pub system_prompt: Option<String>,
    pub thinking: Option<ThinkingConfig>,
    pub mode: Option<AgentMode>,
    pub harness: Option<String>,
    pub idle_grace_secs: Option<u64>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct UpdateAgentInput {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub thinking: Option<ThinkingConfig>,
    pub mode: Option<AgentMode>,
    pub idle_grace_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentDetail {
    pub id: String,
    pub directory: String,
    pub enabled: bool,
    pub provider: String,
    pub model: String,
    pub system_prompt: Option<String>,
    pub mode: Option<String>,
    pub harness: Option<String>,
    pub idle_grace_secs: Option<u64>,
    pub has_local_harness: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct HarnessDetail {
    pub harness_id: String,
    pub directory: String,
    pub bound_agents: Vec<String>,
    pub watched_roots: Vec<String>,
    pub loaded_scripts: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionSummary {
    pub internal_id: i64,
    pub session_id: String,
    pub agent_id: String,
    pub metadata: Option<serde_json::Value>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionEventDetail {
    pub id: i64,
    pub event_type: String,
    pub payload: serde_json::Value,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionMessageDetail {
    pub id: i64,
    pub turn_index: u32,
    pub role: String,
    pub content: serde_json::Value,
    pub token_count: Option<u64>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionToolExecutionDetail {
    pub id: i64,
    pub turn_index: u32,
    pub tool_call_id: String,
    pub tool_name: String,
    pub args: serde_json::Value,
    pub output: Option<serde_json::Value>,
    pub is_error: bool,
    pub duration_ms: Option<u64>,
    pub verdict: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionDetail {
    pub session: SessionSummary,
    pub events: Vec<SessionEventDetail>,
    pub messages: Vec<SessionMessageDetail>,
    pub tool_executions: Vec<SessionToolExecutionDetail>,
}

impl DaemonState {
    pub async fn load(config_path: &Path) -> Result<Self> {
        let config_path = config_path.to_path_buf();
        let config_base = config_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));

        let mut bootstrap_config = TurinConfig::from_file(&config_path)
            .with_context(|| format!("Failed to load '{}'", config_path.display()))?;
        normalize_bootstrap_paths(&mut bootstrap_config, &config_base);

        let registry_load = scan_registry(&bootstrap_config, &config_base)?;
        let effective_config = build_effective_config(&bootstrap_config, &registry_load)?;
        let socket_path = bootstrap_config.resolve_daemon_socket_path(&config_base);

        let mut kernel = Kernel::builder(effective_config).build()?;
        kernel.init_state().await?;
        kernel.init_clients()?;
        kernel.init_harness().await?;
        kernel.start_watcher()?;

        Ok(Self {
            config_path,
            config_base,
            bootstrap_config,
            socket_path,
            registry_load,
            kernel,
        })
    }

    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    pub async fn status(&self) -> DaemonStatus {
        DaemonStatus {
            config_path: self.config_path.display().to_string(),
            workspace_root: self.bootstrap_config.kernel.workspace_root.clone(),
            socket_path: self.socket_path.display().to_string(),
            registry: snapshot(&self.registry_load),
            harnesses: self.kernel.harness_snapshots(),
            agent_runtimes: self.list_agent_runtime_statuses().await,
        }
    }

    pub fn watch_paths(&self) -> DaemonWatchPaths {
        DaemonWatchPaths {
            config_path: self.config_path.clone(),
            agents_dir: self
                .bootstrap_config
                .resolve_daemon_agents_dir(&self.config_base),
            harnesses_dir: self
                .bootstrap_config
                .resolve_daemon_harnesses_dir(&self.config_base),
        }
    }

    pub async fn rescan(&mut self) -> Result<DaemonStatus> {
        let active = self.kernel.agent_manager().list_statuses().await;
        if active.iter().any(|status| {
            status.active_tasks > 0 || status.queued_tasks > 0 || status.awaiting_results > 0
        }) {
            anyhow::bail!("Cannot rescan while agent runtimes have active or queued tasks");
        }

        let mut bootstrap_config = TurinConfig::from_file(&self.config_path)
            .with_context(|| format!("Failed to load '{}'", self.config_path.display()))?;
        normalize_bootstrap_paths(&mut bootstrap_config, &self.config_base);

        let registry_load = scan_registry(&bootstrap_config, &self.config_base)?;
        let effective_config = build_effective_config(&bootstrap_config, &registry_load)?;

        let mut new_kernel = Kernel::builder(effective_config).build()?;
        new_kernel.init_state().await?;
        new_kernel.init_clients()?;
        new_kernel.init_harness().await?;
        new_kernel.start_watcher()?;

        let old_kernel = std::mem::replace(&mut self.kernel, new_kernel);
        self.bootstrap_config = bootstrap_config;
        self.registry_load = registry_load;

        tokio::spawn(async move {
            let mut kernel = old_kernel;
            kernel.shutdown_mcp_clients().await;
        });

        Ok(self.status().await)
    }

    pub fn registry_snapshot(&self) -> RegistrySnapshot {
        snapshot(&self.registry_load)
    }

    pub fn runtime_errors(&self) -> Vec<crate::daemon::registry::RegistryIssue> {
        self.registry_snapshot().issues
    }

    pub fn agent_issues(&self, agent_id: &str) -> Result<Option<Vec<RegistryIssue>>> {
        if agent_id == self.bootstrap_config.agent.id {
            return Ok(Some(Vec::new()));
        }

        let agent_dir = self.agent_dir(agent_id);
        if !agent_dir.exists() {
            return Ok(None);
        }

        Ok(Some(
            self.runtime_errors()
                .into_iter()
                .filter(|issue| issue_path_is_under(&issue.path, &agent_dir))
                .collect(),
        ))
    }

    pub async fn create_agent(&mut self, input: CreateAgentInput) -> Result<AgentDetail> {
        validate_agent_id(&input.id)?;
        let agent_dir = self.agent_dir(&input.id);
        if agent_dir.exists() {
            anyhow::bail!("Agent '{}' already exists", input.id);
        }

        if let Some(shared_harness) = &input.harness {
            let shared_dir = self.watch_paths().harnesses_dir.join(shared_harness);
            if !shared_dir.is_dir() {
                anyhow::bail!("Shared harness '{}' does not exist", shared_harness);
            }
        } else {
            scaffold_local_harness(&agent_dir)?;
        }

        let file = AgentFileConfig {
            id: None,
            enabled: input.enabled,
            system_prompt: input.system_prompt,
            model: input.model,
            provider: input.provider,
            thinking: input.thinking,
            mode: input.mode,
            harness: input.harness,
            idle_grace_secs: input.idle_grace_secs,
        };
        write_agent_file(&agent_dir, &file)?;
        self.rescan().await?;
        self.agent_detail(&input.id)?
            .ok_or_else(|| anyhow!("Agent '{}' was created but not loaded", input.id))
    }

    pub async fn set_agent_enabled(
        &mut self,
        agent_id: &str,
        enabled: bool,
    ) -> Result<AgentDetail> {
        let agent_dir = self.agent_dir(agent_id);
        let mut file = read_agent_file(&agent_dir)?
            .ok_or_else(|| anyhow!("Agent '{}' does not exist", agent_id))?;
        file.enabled = enabled;
        write_agent_file(&agent_dir, &file)?;
        self.rescan().await?;
        self.agent_detail(agent_id)?
            .ok_or_else(|| anyhow!("Agent '{}' could not be reloaded", agent_id))
    }

    pub async fn update_agent(
        &mut self,
        agent_id: &str,
        input: UpdateAgentInput,
    ) -> Result<AgentDetail> {
        let agent_dir = self.agent_dir(agent_id);
        let mut file = read_agent_file(&agent_dir)?
            .ok_or_else(|| anyhow!("Agent '{}' does not exist", agent_id))?;

        if let Some(provider) = input.provider {
            file.provider = provider;
        }
        if let Some(model) = input.model {
            file.model = model;
        }
        if let Some(system_prompt) = input.system_prompt {
            file.system_prompt = Some(system_prompt);
        }
        if let Some(thinking) = input.thinking {
            file.thinking = Some(thinking);
        }
        if let Some(mode) = input.mode {
            file.mode = Some(mode);
        }
        if let Some(idle_grace_secs) = input.idle_grace_secs {
            file.idle_grace_secs = Some(idle_grace_secs);
        }

        write_agent_file(&agent_dir, &file)?;
        self.rescan().await?;
        self.agent_detail(agent_id)?
            .ok_or_else(|| anyhow!("Agent '{}' could not be reloaded", agent_id))
    }

    pub async fn bind_agent_shared_harness(
        &mut self,
        agent_id: &str,
        harness_id: &str,
    ) -> Result<AgentDetail> {
        validate_harness_id(harness_id)?;

        let shared_dir = self.watch_paths().harnesses_dir.join(harness_id);
        if !shared_dir.is_dir() {
            anyhow::bail!("Shared harness '{}' does not exist", harness_id);
        }

        let agent_dir = self.agent_dir(agent_id);
        let mut file = read_agent_file(&agent_dir)?
            .ok_or_else(|| anyhow!("Agent '{}' does not exist", agent_id))?;

        let local_harness_dir = agent_dir.join("harness");
        if local_harness_dir.is_dir() {
            if local_harness_is_scaffold(&local_harness_dir)? {
                std::fs::remove_dir_all(&local_harness_dir).with_context(|| {
                    format!(
                        "Failed to remove scaffold local harness '{}'",
                        local_harness_dir.display()
                    )
                })?;
            } else {
                anyhow::bail!(
                    "Agent '{}' has a non-scaffold local harness; remove or migrate it before binding a shared harness",
                    agent_id
                );
            }
        }

        file.harness = Some(harness_id.to_string());
        write_agent_file(&agent_dir, &file)?;
        self.rescan().await?;
        self.agent_detail(agent_id)?
            .ok_or_else(|| anyhow!("Agent '{}' could not be rebound", agent_id))
    }

    pub async fn use_local_agent_harness(&mut self, agent_id: &str) -> Result<AgentDetail> {
        let agent_dir = self.agent_dir(agent_id);
        let mut file = read_agent_file(&agent_dir)?
            .ok_or_else(|| anyhow!("Agent '{}' does not exist", agent_id))?;

        file.harness = None;
        write_agent_file(&agent_dir, &file)?;
        scaffold_local_harness(&agent_dir)?;
        self.rescan().await?;
        self.agent_detail(agent_id)?.ok_or_else(|| {
            anyhow!(
                "Agent '{}' could not be switched to local harness",
                agent_id
            )
        })
    }

    pub async fn delete_agent(&mut self, agent_id: &str) -> Result<DaemonStatus> {
        let agent_dir = self.agent_dir(agent_id);
        if !agent_dir.is_dir() {
            anyhow::bail!("Agent '{}' does not exist", agent_id);
        }
        std::fs::remove_dir_all(&agent_dir)
            .with_context(|| format!("Failed to remove '{}'", agent_dir.display()))?;
        self.rescan().await
    }

    pub fn agent_detail(&self, agent_id: &str) -> Result<Option<AgentDetail>> {
        let agent_dir = self.agent_dir(agent_id);
        let Some(file) = read_agent_file(&agent_dir)? else {
            return Ok(None);
        };

        Ok(Some(AgentDetail {
            id: agent_id.to_string(),
            directory: agent_dir.display().to_string(),
            enabled: file.enabled,
            provider: file.provider,
            model: file.model,
            system_prompt: file.system_prompt,
            mode: file.mode.map(|mode| format!("{:?}", mode).to_lowercase()),
            harness: file.harness,
            idle_grace_secs: file.idle_grace_secs,
            has_local_harness: agent_dir.join("harness").is_dir(),
        }))
    }

    pub async fn agent_runtime_status(
        &self,
        agent_id: &str,
    ) -> Result<Option<AgentStatusSnapshot>> {
        Ok(self
            .list_agent_runtime_statuses()
            .await
            .into_iter()
            .find(|status| status.agent_id == agent_id))
    }

    fn agent_dir(&self, agent_id: &str) -> PathBuf {
        self.watch_paths().agents_dir.join(agent_id)
    }

    pub async fn submit_task(&self, agent_id: &str, prompt: String) -> Result<TaskStatusSnapshot> {
        self.ensure_enabled_agent(agent_id)?;
        let request_id = self
            .kernel
            .agent_manager()
            .submit(agent_id, QueuedTask::ad_hoc(prompt), None)
            .await?;
        self.kernel
            .agent_manager()
            .get_task(&request_id)
            .await
            .ok_or_else(|| anyhow!("Task '{}' was submitted but is not visible", request_id))
    }

    pub async fn list_tasks(&self) -> Vec<TaskStatusSnapshot> {
        self.kernel.agent_manager().list_tasks().await
    }

    pub async fn get_task(&self, request_id: &str) -> Option<TaskStatusSnapshot> {
        self.kernel.agent_manager().get_task(request_id).await
    }

    pub async fn cancel_task(&self, request_id: &str) -> Result<TaskStatusSnapshot> {
        self.kernel.agent_manager().cancel_task(request_id).await
    }

    pub async fn wait_for_task(
        &self,
        request_id: &str,
        timeout_ms: Option<u64>,
    ) -> Result<TaskStatusSnapshot> {
        let Some(initial) = self.get_task(request_id).await else {
            anyhow::bail!("Task '{}' not found", request_id);
        };
        if initial.state != "queued" && initial.state != "running" {
            return Ok(initial);
        }

        let deadline = timeout_ms.map(|ms| tokio::time::Instant::now() + Duration::from_millis(ms));
        loop {
            if let Some(snapshot) = self.get_task(request_id).await {
                if snapshot.state != "queued" && snapshot.state != "running" {
                    return Ok(snapshot);
                }
            } else {
                anyhow::bail!("Task '{}' disappeared while waiting", request_id);
            }

            if let Some(deadline) = deadline
                && tokio::time::Instant::now() >= deadline
            {
                anyhow::bail!("Timed out waiting for task '{}'", request_id);
            }

            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    pub async fn list_sessions(&self, limit: usize, offset: usize) -> Result<Vec<SessionSummary>> {
        let store = self.kernel.store_manager().get_default().await?;
        let rows = store.list_session_rows(limit, offset).await?;
        Ok(rows.iter().map(session_summary_from_row).collect())
    }

    pub async fn get_session(&self, session_id: &str) -> Result<Option<SessionDetail>> {
        let public_id = uuid::Uuid::parse_str(session_id)
            .with_context(|| format!("Invalid session id '{}'", session_id))?;
        let store = self.kernel.store_manager().get_default().await?;
        let Some(row) = store.get_session_row_by_public_id(public_id).await? else {
            return Ok(None);
        };

        let events = store
            .get_events(row.id)
            .await?
            .into_iter()
            .map(|event| SessionEventDetail {
                id: event.id,
                event_type: event.event_type,
                payload: parse_json_or_string(&event.payload),
                created_at: event.created_at,
            })
            .collect();

        let messages = store
            .get_messages(row.id)
            .await?
            .into_iter()
            .map(|message| SessionMessageDetail {
                id: message.id,
                turn_index: message.turn_index,
                role: message.role,
                content: parse_json_or_string(&message.content),
                token_count: message.token_count,
                created_at: message.created_at,
            })
            .collect();

        let tool_executions = store
            .get_tool_executions(row.id)
            .await?
            .into_iter()
            .map(|execution| SessionToolExecutionDetail {
                id: execution.id,
                turn_index: execution.turn_index,
                tool_call_id: execution.tool_call_id,
                tool_name: execution.tool_name,
                args: parse_json_or_string(&execution.args),
                output: execution.output.as_deref().map(parse_json_or_string),
                is_error: execution.is_error,
                duration_ms: execution.duration_ms,
                verdict: execution.verdict,
                created_at: execution.created_at,
            })
            .collect();

        Ok(Some(SessionDetail {
            session: session_summary_from_row(&row),
            events,
            messages,
            tool_executions,
        }))
    }

    pub fn harness_detail(&self, harness_id: &str) -> Option<HarnessDetail> {
        self.kernel
            .harness_snapshot(harness_id)
            .map(|snapshot| HarnessDetail {
                harness_id: snapshot.harness_id,
                directory: snapshot.directory,
                bound_agents: snapshot.bound_agents,
                watched_roots: snapshot.watched_roots,
                loaded_scripts: snapshot.loaded_scripts,
            })
    }

    pub fn harness_issues(&self, harness_id: &str) -> Result<Option<Vec<RegistryIssue>>> {
        let Some(harness_dir) = self.resolve_harness_issue_root(harness_id) else {
            return Ok(None);
        };
        Ok(Some(
            self.runtime_errors()
                .into_iter()
                .filter(|issue| issue_path_is_under(&issue.path, &harness_dir))
                .collect(),
        ))
    }

    pub async fn reload_harness(&mut self, harness_id: &str) -> Result<HarnessDetail> {
        self.kernel.reload_named_harness(harness_id).await?;
        self.harness_detail(harness_id)
            .ok_or_else(|| anyhow!("Harness '{}' was reloaded but is not visible", harness_id))
    }

    pub fn validate_harness(&self, harness_id: &str) -> Result<serde_json::Value> {
        let script_count = self.kernel.validate_named_harness(harness_id)?;
        let detail = self
            .harness_detail(harness_id)
            .ok_or_else(|| anyhow!("Harness '{}' not found", harness_id))?;
        Ok(serde_json::json!({
            "harness_id": detail.harness_id,
            "directory": detail.directory,
            "script_count": script_count,
            "valid": true,
        }))
    }

    pub async fn create_shared_harness(&mut self, harness_id: &str) -> Result<HarnessDetail> {
        validate_harness_id(harness_id)?;
        let harness_dir = self.watch_paths().harnesses_dir.join(harness_id);
        if harness_dir.exists() {
            anyhow::bail!("Harness '{}' already exists", harness_id);
        }
        scaffold_shared_harness(&harness_dir)?;
        self.rescan().await?;
        self.harness_detail(harness_id)
            .ok_or_else(|| anyhow!("Harness '{}' was created but not loaded", harness_id))
    }

    pub async fn delete_shared_harness(&mut self, harness_id: &str) -> Result<DaemonStatus> {
        if harness_id == "default" || harness_id.starts_with("agent::") {
            anyhow::bail!("Harness '{}' is not a managed shared harness", harness_id);
        }

        if let Some(detail) = self.harness_detail(harness_id)
            && !detail.bound_agents.is_empty()
        {
            anyhow::bail!(
                "Harness '{}' is still bound to agents: {}",
                harness_id,
                detail.bound_agents.join(", ")
            );
        }

        let harness_dir = self.watch_paths().harnesses_dir.join(harness_id);
        if !harness_dir.is_dir() {
            anyhow::bail!("Harness '{}' does not exist", harness_id);
        }

        std::fs::remove_dir_all(&harness_dir)
            .with_context(|| format!("Failed to remove '{}'", harness_dir.display()))?;
        self.rescan().await
    }

    fn ensure_enabled_agent(&self, agent_id: &str) -> Result<()> {
        if agent_id == self.bootstrap_config.agent.id {
            return Ok(());
        }

        let agent = self
            .registry_load
            .agents
            .iter()
            .find(|agent| agent.id == agent_id)
            .ok_or_else(|| anyhow!("Agent '{}' not found", agent_id))?;
        if !agent.enabled {
            anyhow::bail!("Agent '{}' is disabled", agent_id);
        }
        Ok(())
    }

    async fn list_agent_runtime_statuses(&self) -> Vec<AgentStatusSnapshot> {
        let mut live: std::collections::HashMap<_, _> = self
            .kernel
            .agent_manager()
            .list_statuses()
            .await
            .into_iter()
            .map(|status| (status.agent_id.clone(), status))
            .collect();

        let mut ids = vec![self.bootstrap_config.agent.id.clone()];
        ids.extend(
            self.registry_load
                .agents
                .iter()
                .map(|agent| agent.id.clone()),
        );
        ids.sort();
        ids.dedup();

        ids.into_iter()
            .map(|agent_id| {
                live.remove(&agent_id).unwrap_or(AgentStatusSnapshot {
                    agent_id,
                    running: false,
                    active_tasks: 0,
                    queued_tasks: 0,
                    awaiting_results: 0,
                })
            })
            .collect()
    }

    fn resolve_harness_issue_root(&self, harness_id: &str) -> Option<PathBuf> {
        if harness_id == "default" {
            return Some(PathBuf::from(&self.bootstrap_config.harness.directory));
        }

        if let Some(agent_id) = harness_id.strip_prefix("agent::") {
            let harness_dir = self.agent_dir(agent_id).join("harness");
            if harness_dir.is_dir() {
                return Some(harness_dir);
            }
        }

        let shared_dir = self.watch_paths().harnesses_dir.join(harness_id);
        if shared_dir.exists() {
            return Some(shared_dir);
        }

        self.harness_detail(harness_id)
            .map(|detail| PathBuf::from(detail.directory))
    }
}

fn normalize_bootstrap_paths(config: &mut TurinConfig, config_base: &Path) {
    let workspace_root = config.resolve_workspace_root(config_base);
    config.kernel.workspace_root = workspace_root.display().to_string();

    if Path::new(&config.harness.directory).is_relative() {
        config.harness.directory = workspace_root
            .join(&config.harness.directory)
            .display()
            .to_string();
    }

    if Path::new(&config.harness.fs_root).is_relative() && config.harness.fs_root != "." {
        config.harness.fs_root = workspace_root
            .join(&config.harness.fs_root)
            .display()
            .to_string();
    }
}

fn validate_agent_id(agent_id: &str) -> Result<()> {
    if agent_id.trim().is_empty() {
        anyhow::bail!("Agent ID cannot be empty");
    }
    if agent_id == "default" {
        anyhow::bail!("'default' is reserved for the bootstrap agent");
    }
    if agent_id.contains('/') || agent_id.contains('\\') || agent_id.contains("..") {
        anyhow::bail!("Agent ID '{}' contains invalid path characters", agent_id);
    }
    Ok(())
}

fn validate_harness_id(harness_id: &str) -> Result<()> {
    if harness_id.trim().is_empty() {
        anyhow::bail!("Harness ID cannot be empty");
    }
    if harness_id == "default" {
        anyhow::bail!("'default' is reserved for the bootstrap harness");
    }
    if harness_id.starts_with("agent::") {
        anyhow::bail!("Harness IDs cannot start with 'agent::'");
    }
    if harness_id.contains('/') || harness_id.contains('\\') || harness_id.contains("..") {
        anyhow::bail!(
            "Harness ID '{}' contains invalid path characters",
            harness_id
        );
    }
    Ok(())
}

fn scaffold_local_harness(agent_dir: &Path) -> Result<()> {
    let harness_dir = agent_dir.join("harness");
    std::fs::create_dir_all(&harness_dir)
        .with_context(|| format!("Failed to create '{}'", harness_dir.display()))?;
    scaffold_harness_main(&harness_dir)
}

fn local_harness_is_scaffold(harness_dir: &Path) -> Result<bool> {
    let mut entries = std::fs::read_dir(harness_dir)
        .with_context(|| format!("Failed to read '{}'", harness_dir.display()))?
        .collect::<std::io::Result<Vec<_>>>()
        .with_context(|| format!("Failed to enumerate '{}'", harness_dir.display()))?;
    entries.sort_by_key(|entry| entry.file_name());

    if entries.len() != 1 {
        return Ok(false);
    }

    let entry = &entries[0];
    if entry.file_name() != "main.lua" {
        return Ok(false);
    }

    let body = std::fs::read_to_string(entry.path())
        .with_context(|| format!("Failed to read '{}'", entry.path().display()))?;
    Ok(body == "-- Turin daemon scaffold\n")
}

fn scaffold_shared_harness(harness_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(harness_dir)
        .with_context(|| format!("Failed to create '{}'", harness_dir.display()))?;
    scaffold_harness_main(harness_dir)
}

fn scaffold_harness_main(harness_dir: &Path) -> Result<()> {
    let main_lua = harness_dir.join("main.lua");
    if main_lua.exists() {
        return Ok(());
    }

    let tmp_path = harness_dir.join(format!(".main.lua.{}.tmp", uuid::Uuid::now_v7().simple()));
    std::fs::write(&tmp_path, "-- Turin daemon scaffold\n")
        .with_context(|| format!("Failed to write '{}'", tmp_path.display()))?;
    std::fs::rename(&tmp_path, &main_lua).with_context(|| {
        format!(
            "Failed to atomically replace '{}' from '{}'",
            main_lua.display(),
            tmp_path.display()
        )
    })?;
    Ok(())
}

fn session_summary_from_row(row: &crate::persistence::schema::SessionRow) -> SessionSummary {
    SessionSummary {
        internal_id: row.id,
        session_id: format_uuid_bytes_simple(&row.public_id),
        agent_id: row.agent_id.clone(),
        metadata: row
            .metadata
            .as_deref()
            .and_then(|raw| serde_json::from_str(raw).ok())
            .or_else(|| {
                row.metadata
                    .as_ref()
                    .map(|raw| serde_json::Value::String(raw.clone()))
            }),
        created_at: row.created_at.clone(),
    }
}

fn format_uuid_bytes_simple(bytes: &[u8]) -> String {
    uuid::Uuid::from_slice(bytes)
        .map(|uuid| uuid.simple().to_string())
        .unwrap_or_else(|_| {
            let mut out = String::with_capacity(bytes.len() * 2);
            for byte in bytes {
                use std::fmt::Write as _;
                let _ = write!(&mut out, "{:02x}", byte);
            }
            out
        })
}

fn parse_json_or_string(raw: &str) -> serde_json::Value {
    serde_json::from_str(raw).unwrap_or_else(|_| serde_json::Value::String(raw.to_string()))
}

fn issue_path_is_under(issue_path: &str, root: &Path) -> bool {
    let issue_path = Path::new(issue_path);
    issue_path == root || issue_path.starts_with(root)
}

#[cfg(test)]
mod tests {
    use super::*;
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
type = "no_op"
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
            .submit_task("default", "Hello daemon".to_string())
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
            .submit_task("default", "Hello wait".to_string())
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
            .submit_task("default", "Hello session".to_string())
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
            .submit_task("default", "Hello status".to_string())
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
}
