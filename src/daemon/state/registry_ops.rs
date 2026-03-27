use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use turin_channel_core::ChannelKind;
use turin_channel_runner::{ChannelAccessSnapshot, ChannelRoomRef, FileAccessStateStore};

use super::{
    AgentDetail, ChannelDetail, CreateAgentInput, CreateChannelInput, DaemonState, DaemonStatus,
    HarnessDetail, UpdateAgentInput, UpdateChannelInput,
};
use crate::daemon::registry::{
    AgentFileConfig, ChannelFileConfig, RegistryIssue, read_agent_file, read_channel_file,
    write_agent_file, write_channel_file,
};

impl DaemonState {
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
                .filter(|issue| super::helpers::issue_path_is_under(&issue.path, &agent_dir))
                .collect(),
        ))
    }

    pub async fn create_agent(&mut self, input: CreateAgentInput) -> Result<AgentDetail> {
        super::helpers::validate_agent_id(&input.id)?;
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
            super::helpers::scaffold_local_harness(&agent_dir)?;
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
            tool_selection: input.tool_selection,
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
        if let Some(tool_selection) = input.tool_selection {
            file.tool_selection = tool_selection;
        }

        write_agent_file(&agent_dir, &file)?;
        self.rescan().await?;
        self.agent_detail(agent_id)?
            .ok_or_else(|| anyhow!("Agent '{}' could not be reloaded", agent_id))
    }

    pub async fn reload_agent(&mut self, agent_id: &str) -> Result<AgentDetail> {
        let agent_dir = self.agent_dir(agent_id);
        if !agent_dir.exists() {
            anyhow::bail!("Agent '{}' does not exist", agent_id);
        }
        self.rescan().await?;
        self.agent_detail(agent_id)?
            .ok_or_else(|| anyhow!("Agent '{}' could not be reloaded", agent_id))
    }

    pub async fn bind_agent_shared_harness(
        &mut self,
        agent_id: &str,
        harness_id: &str,
    ) -> Result<AgentDetail> {
        super::helpers::validate_harness_id(harness_id)?;

        let shared_dir = self.watch_paths().harnesses_dir.join(harness_id);
        if !shared_dir.is_dir() {
            anyhow::bail!("Shared harness '{}' does not exist", harness_id);
        }

        let agent_dir = self.agent_dir(agent_id);
        let mut file = read_agent_file(&agent_dir)?
            .ok_or_else(|| anyhow!("Agent '{}' does not exist", agent_id))?;

        let local_harness_dir = agent_dir.join("harness");
        if local_harness_dir.is_dir() {
            if super::helpers::local_harness_is_scaffold(&local_harness_dir)? {
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
        super::helpers::scaffold_local_harness(&agent_dir)?;
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
                .filter(|issue| super::helpers::issue_path_is_under(&issue.path, &harness_dir))
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
        super::helpers::validate_harness_id(harness_id)?;
        let harness_dir = self.watch_paths().harnesses_dir.join(harness_id);
        if harness_dir.exists() {
            anyhow::bail!("Harness '{}' already exists", harness_id);
        }
        super::helpers::scaffold_shared_harness(&harness_dir)?;
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

    pub(super) fn agent_dir(&self, agent_id: &str) -> PathBuf {
        self.watch_paths().agents_dir.join(agent_id)
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

    pub fn channel_detail(&self, channel_id: &str) -> Option<ChannelDetail> {
        self.registry_load
            .channels
            .iter()
            .find(|channel| channel.id == channel_id)
            .map(|channel| ChannelDetail {
                id: channel.id.clone(),
                directory: channel.directory.display().to_string(),
                enabled: channel.enabled,
                kind: channel.kind.clone(),
                agent_id: channel.agent_id.clone(),
                idle_ttl_secs: channel.idle_ttl_secs,
                settings: serde_json::to_value(&channel.extra).unwrap_or_default(),
            })
    }

    pub fn channel_issues(&self, channel_id: &str) -> Result<Option<Vec<RegistryIssue>>> {
        let channel_dir = self.watch_paths().channels_dir.join(channel_id);
        if !channel_dir.exists() {
            return Ok(None);
        }

        Ok(Some(
            self.runtime_errors()
                .into_iter()
                .filter(|issue| super::helpers::issue_path_is_under(&issue.path, &channel_dir))
                .collect(),
        ))
    }

    pub async fn channel_access_snapshot(
        &self,
        channel_id: &str,
    ) -> Result<Option<ChannelAccessSnapshot>> {
        let Some(_) = self.channel_detail(channel_id) else {
            return Ok(None);
        };
        let store = FileAccessStateStore::new(self.channel_access_state_path(channel_id));
        Ok(Some(store.snapshot().await?))
    }

    pub async fn approve_channel_room(
        &self,
        channel_id: &str,
        workspace_id: String,
        room_id: Option<String>,
        thread_id: String,
    ) -> Result<Option<ChannelAccessSnapshot>> {
        let Some(channel) = self.channel_detail(channel_id) else {
            return Ok(None);
        };
        let store = FileAccessStateStore::new(self.channel_access_state_path(channel_id));
        let snapshot = store
            .approve(
                &ChannelRoomRef {
                    channel: parse_channel_kind(&channel.kind),
                    workspace_id,
                    room_id,
                    thread_id,
                },
                None,
                Some("operator".to_string()),
            )
            .await?;
        Ok(Some(snapshot))
    }

    pub async fn reject_channel_room(
        &self,
        channel_id: &str,
        workspace_id: String,
        room_id: Option<String>,
        thread_id: String,
    ) -> Result<Option<ChannelAccessSnapshot>> {
        let Some(channel) = self.channel_detail(channel_id) else {
            return Ok(None);
        };
        let store = FileAccessStateStore::new(self.channel_access_state_path(channel_id));
        let snapshot = store
            .reject_pending(&ChannelRoomRef {
                channel: parse_channel_kind(&channel.kind),
                workspace_id,
                room_id,
                thread_id,
            })
            .await?;
        Ok(Some(snapshot))
    }

    pub async fn revoke_channel_room(
        &self,
        channel_id: &str,
        workspace_id: String,
        room_id: Option<String>,
        thread_id: String,
    ) -> Result<Option<ChannelAccessSnapshot>> {
        let Some(channel) = self.channel_detail(channel_id) else {
            return Ok(None);
        };
        let store = FileAccessStateStore::new(self.channel_access_state_path(channel_id));
        let snapshot = store
            .revoke(&ChannelRoomRef {
                channel: parse_channel_kind(&channel.kind),
                workspace_id,
                room_id,
                thread_id,
            })
            .await?;
        Ok(Some(snapshot))
    }

    pub async fn create_channel(&mut self, input: CreateChannelInput) -> Result<ChannelDetail> {
        super::helpers::validate_channel_id(&input.id)?;
        if input.kind.trim().is_empty() {
            anyhow::bail!("Channel kind cannot be empty");
        }
        self.ensure_channel_agent_exists(&input.agent_id)?;
        let access_policy =
            turin_channel_runner::ChannelAccessPolicy::from_settings(&input.settings)?;
        let tool_selection = turin_channel_runner::tool_selection_from_settings(&input.settings)?;
        turin_channel_runner::task_timeout_ms_from_settings(&input.settings)?;
        crate::tools::policy::resolve_effective_native_tools(
            self.kernel.config(),
            &input.agent_id,
            Some(&tool_selection),
        )?;
        let channel_dir = self.watch_paths().channels_dir.join(&input.id);
        super::channel_validation::validate_channel_settings(
            &input.kind,
            &channel_dir,
            &input.settings,
            &access_policy,
        )?;
        if channel_dir.exists() {
            anyhow::bail!("Channel '{}' already exists", input.id);
        }

        let file = ChannelFileConfig {
            id: None,
            enabled: input.enabled,
            kind: input.kind,
            agent_id: input.agent_id,
            idle_ttl_secs: input.idle_ttl_secs,
            extra: super::helpers::json_object_to_toml_table(input.settings)?,
        };

        write_channel_file(&channel_dir, &file)?;
        self.rescan().await?;
        self.channel_detail(&input.id)
            .ok_or_else(|| anyhow!("Channel '{}' was created but not loaded", input.id))
    }

    pub async fn set_channel_enabled(
        &mut self,
        channel_id: &str,
        enabled: bool,
    ) -> Result<ChannelDetail> {
        let channel_dir = self.watch_paths().channels_dir.join(channel_id);
        let mut file = read_channel_file(&channel_dir)?
            .ok_or_else(|| anyhow!("Channel '{}' does not exist", channel_id))?;
        file.enabled = enabled;
        write_channel_file(&channel_dir, &file)?;
        self.rescan().await?;
        self.channel_detail(channel_id)
            .ok_or_else(|| anyhow!("Channel '{}' could not be reloaded", channel_id))
    }

    pub async fn update_channel(
        &mut self,
        channel_id: &str,
        input: UpdateChannelInput,
    ) -> Result<ChannelDetail> {
        let channel_dir = self.watch_paths().channels_dir.join(channel_id);
        let mut file = read_channel_file(&channel_dir)?
            .ok_or_else(|| anyhow!("Channel '{}' does not exist", channel_id))?;

        if let Some(kind) = input.kind {
            if kind.trim().is_empty() {
                anyhow::bail!("Channel kind cannot be empty");
            }
            file.kind = kind;
        }
        if let Some(agent_id) = input.agent_id {
            self.ensure_channel_agent_exists(&agent_id)?;
            file.agent_id = agent_id;
        }
        if let Some(idle_ttl_secs) = input.idle_ttl_secs {
            file.idle_ttl_secs = Some(idle_ttl_secs);
        }
        if let Some(settings) = input.settings {
            super::helpers::merge_json_object_into_toml_table(&mut file.extra, settings)?;
        }
        let settings_value = serde_json::to_value(file.extra.clone())
            .context("Failed to serialize channel settings for validation")?;
        let access_policy =
            turin_channel_runner::ChannelAccessPolicy::from_settings(&settings_value)?;
        let tool_selection = turin_channel_runner::tool_selection_from_settings(&settings_value)?;
        turin_channel_runner::task_timeout_ms_from_settings(&settings_value)?;
        crate::tools::policy::resolve_effective_native_tools(
            self.kernel.config(),
            &file.agent_id,
            Some(&tool_selection),
        )?;
        super::channel_validation::validate_channel_settings(
            &file.kind,
            &channel_dir,
            &settings_value,
            &access_policy,
        )?;

        write_channel_file(&channel_dir, &file)?;
        self.rescan().await?;
        self.channel_detail(channel_id)
            .ok_or_else(|| anyhow!("Channel '{}' could not be reloaded", channel_id))
    }

    pub async fn delete_channel(&mut self, channel_id: &str) -> Result<DaemonStatus> {
        let channels_dir = self.watch_paths().channels_dir.clone();
        let channel_dir = channels_dir.join(channel_id);
        if !channel_dir.is_dir() {
            anyhow::bail!("Channel '{}' does not exist", channel_id);
        }
        let tombstone = channels_dir.join(format!(
            ".deleted-{}-{}",
            channel_id,
            uuid::Uuid::now_v7().simple()
        ));
        std::fs::rename(&channel_dir, &tombstone).with_context(|| {
            format!(
                "Failed to move '{}' to '{}'",
                channel_dir.display(),
                tombstone.display()
            )
        })?;

        let status = match self.rescan().await {
            Ok(status) => status,
            Err(err) => {
                let _ = std::fs::rename(&tombstone, &channel_dir);
                return Err(err);
            }
        };

        std::fs::remove_dir_all(&tombstone)
            .with_context(|| format!("Failed to remove '{}'", tombstone.display()))?;
        Ok(status)
    }

    fn ensure_channel_agent_exists(&self, agent_id: &str) -> Result<()> {
        if agent_id == self.bootstrap_config.agent.id {
            return Ok(());
        }
        if self
            .registry_load
            .agents
            .iter()
            .any(|agent| agent.id == agent_id)
        {
            return Ok(());
        }
        anyhow::bail!("Channel agent '{}' does not exist", agent_id)
    }

    fn channel_access_state_path(&self, channel_id: &str) -> PathBuf {
        PathBuf::from(&self.bootstrap_config.kernel.workspace_root)
            .join(".turin")
            .join("channels")
            .join(format!("{channel_id}-access.json"))
    }
}

fn parse_channel_kind(kind: &str) -> ChannelKind {
    match kind.trim().to_ascii_lowercase().as_str() {
        "discord" => ChannelKind::Discord,
        "telegram" => ChannelKind::Telegram,
        "slack" => ChannelKind::Slack,
        "matrix" => ChannelKind::Matrix,
        other => ChannelKind::Other(other.to_string()),
    }
}
