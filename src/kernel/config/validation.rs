use anyhow::{Context, Result};

use super::*;

fn require_non_empty(value: &str, label: impl std::fmt::Display) -> Result<()> {
    anyhow::ensure!(!value.trim().is_empty(), "{} must not be empty", label);
    Ok(())
}

fn require_non_empty_with_message(value: &str, message: impl std::fmt::Display) -> Result<()> {
    anyhow::ensure!(!value.trim().is_empty(), "{}", message);
    Ok(())
}

fn require_optional_non_empty(value: Option<&str>, label: impl std::fmt::Display) -> Result<()> {
    if let Some(value) = value {
        anyhow::ensure!(
            !value.trim().is_empty(),
            "{} must not be empty when set",
            label
        );
    }
    Ok(())
}

fn require_positive<T>(value: T, label: impl std::fmt::Display) -> Result<()>
where
    T: Copy + From<u8> + PartialOrd,
{
    anyhow::ensure!(value > T::from(0), "{} must be greater than 0", label);
    Ok(())
}

fn require_optional_positive<T>(value: Option<T>, label: impl std::fmt::Display) -> Result<()>
where
    T: Copy + From<u8> + PartialOrd,
{
    if let Some(value) = value {
        require_positive(value, label)?;
    }
    Ok(())
}

fn validate_bool_rule_map(
    map: &std::collections::BTreeMap<String, serde_json::Value>,
    label: &str,
) -> Result<()> {
    for (rule, value) in map {
        require_non_empty(rule, format!("{label} contains an empty capability rule"))?;
        anyhow::ensure!(value.is_boolean(), "{}.{} must be a boolean", label, rule);
    }
    Ok(())
}

impl TurinConfig {
    /// Validate semantic invariants that serde can't enforce.
    pub fn validate(&self) -> Result<()> {
        require_non_empty(&self.agent.model, "agent.model")?;
        if !self.providers.contains_key(&self.agent.provider) {
            anyhow::bail!(
                "Provider '{}' configured in [agent] but not found in [providers]",
                self.agent.provider
            );
        }
        require_positive(self.kernel.max_turns, "kernel.max_turns")?;
        require_positive(
            self.kernel.heartbeat_interval_seconds,
            "kernel.heartbeat_interval_seconds",
        )?;
        require_positive(
            self.runtime.linked_runtime_lanes,
            "runtime.linked_runtime_lanes",
        )?;
        require_optional_non_empty(self.layout.root.as_deref(), "layout.root")?;
        require_non_empty(&self.layout.data_dir, "layout.data_dir")?;
        require_non_empty(&self.layout.states_dir, "layout.states_dir")?;
        require_non_empty(&self.layout.stores_dir, "layout.stores_dir")?;
        require_non_empty(&self.layout.harnesses_dir, "layout.harnesses_dir")?;
        require_non_empty(&self.layout.agents_dir, "layout.agents_dir")?;
        require_non_empty(&self.layout.scopes_dir, "layout.scopes_dir")?;
        require_non_empty(&self.layout.env_file, "layout.env_file")?;
        require_non_empty(&self.layout.daemon_socket, "layout.daemon_socket")?;

        require_non_empty(&self.harness.directory, "harness.directory")?;
        require_positive(self.harness.memory_limit_mb, "harness.memory_limit_mb")?;
        require_non_empty(&self.daemon.agents_dir, "daemon.agents_dir")?;
        require_non_empty(&self.daemon.harnesses_dir, "daemon.harnesses_dir")?;
        require_non_empty(&self.daemon.runtime_db, "daemon.runtime_db")?;
        require_non_empty(&self.daemon.endpoint, "daemon.endpoint")?;
        require_non_empty(&self.remote.bind, "remote.bind")?;
        require_non_empty(&self.remote.auth_token_env, "remote.auth_token_env")?;
        require_positive(
            self.remote.event_keepalive_seconds,
            "remote.event_keepalive_seconds",
        )?;
        self.persistence.state.validate("persistence.state")?;
        if let Some(store) = &self.persistence.store {
            store.validate("persistence.store")?;
        }
        for (state_name, state) in &self.persistence.states {
            require_non_empty_with_message(
                state_name,
                "persistence.states contains an empty state name",
            )?;
            anyhow::ensure!(
                state_name != "state" && state_name != "store",
                "persistence.states.{} is reserved",
                state_name
            );
            require_non_empty(
                &state.path,
                format!("persistence.states.{}.path", state_name),
            )?;
        }
        for (store_name, store) in &self.persistence.stores {
            require_non_empty_with_message(
                store_name,
                "persistence.stores contains an empty store name",
            )?;
            anyhow::ensure!(
                store_name != "state" && store_name != "store",
                "persistence.stores.{} is reserved",
                store_name
            );
            require_non_empty(
                &store.path,
                format!("persistence.stores.{}.path", store_name),
            )?;
        }
        for state_name in self.persistence.states.keys() {
            anyhow::ensure!(
                !self.persistence.stores.contains_key(state_name),
                "persistence.states.{} collides with persistence.stores.{}",
                state_name,
                state_name
            );
        }
        for (idx, placement) in self.persistence.placements.iter().enumerate() {
            require_non_empty(
                &placement.scope_kind,
                format!("persistence.placements[{}].scope_kind", idx),
            )?;
            require_optional_non_empty(
                placement.scope_key.as_deref(),
                format!("persistence.placements[{}].scope_key", idx),
            )?;
            require_optional_non_empty(
                placement.namespace.as_deref(),
                format!("persistence.placements[{}].namespace", idx),
            )?;
            require_non_empty(
                &placement.store,
                format!("persistence.placements[{}].store", idx),
            )?;
            anyhow::ensure!(
                placement.store == "state"
                    || self.persistence.states.contains_key(&placement.store)
                    || self.persistence.stores.contains_key(&placement.store),
                "persistence.placements[{}].store '{}' not found in persistence.states or persistence.stores",
                idx,
                placement.store
            );
        }

        for (harness_id, harness_cfg) in &self.harnesses {
            require_non_empty_with_message(harness_id, "harnesses contains an empty harness id")?;
            anyhow::ensure!(
                harness_id != "default",
                "harnesses.default is reserved; use [harness] for the default harness"
            );
            require_non_empty(
                &harness_cfg.directory,
                format!("harnesses.{}.directory", harness_id),
            )?;
            require_positive(
                harness_cfg.memory_limit_mb,
                format!("harnesses.{}.memory_limit_mb", harness_id),
            )?;
        }

        for (provider_name, provider) in &self.providers {
            require_optional_positive(
                provider.request_timeout_seconds,
                format!("providers.{}.request_timeout_seconds", provider_name),
            )?;
            require_optional_positive(
                provider.total_timeout_seconds,
                format!("providers.{}.total_timeout_seconds", provider_name),
            )?;
            require_optional_positive(
                provider.context_window_tokens,
                format!("providers.{}.context_window_tokens", provider_name),
            )?;

            if let (Some(request_seconds), Some(total_seconds)) = (
                provider.request_timeout_seconds,
                provider.total_timeout_seconds,
            ) {
                anyhow::ensure!(
                    total_seconds >= request_seconds,
                    "providers.{}.total_timeout_seconds must be >= request_timeout_seconds",
                    provider_name
                );
            }

            for header in provider.headers.keys() {
                require_non_empty_with_message(
                    header,
                    format!(
                        "providers.{}.headers contains an empty header name",
                        provider_name
                    ),
                )?;
            }
        }

        self.inference
            .validate_complete(&self.providers, "inference")?;
        require_optional_positive(
            self.inference.hot_history.max_messages,
            "inference.hot_history.max_messages",
        )?;
        require_optional_positive(
            self.inference.hot_history.max_tool_result_bytes,
            "inference.hot_history.max_tool_result_bytes",
        )?;

        require_optional_positive(
            self.governance.grants.max_ttl_ms,
            "governance.grants.max_ttl_ms",
        )?;
        require_non_empty(&self.governance.profile, "governance.profile")?;
        validate_bool_rule_map(&self.governance.capabilities, "governance.capabilities")?;

        for (root_name, root) in &self.governance.roots {
            require_non_empty_with_message(
                root_name,
                "governance.roots contains an empty root name",
            )?;
            require_non_empty(&root.path, format!("governance.roots.{}.path", root_name))?;
        }

        for profile_name in self.governance.capability_profiles.keys() {
            require_non_empty_with_message(
                profile_name,
                "governance.capability_profiles contains an empty profile name",
            )?;
        }

        for (agent_id, agent_cfg) in &self.governance.agents {
            if let Some(profile_name) = &agent_cfg.capability_profile {
                anyhow::ensure!(
                    self.governance
                        .capability_profiles
                        .contains_key(profile_name),
                    "governance.agents.{}.capability_profile '{}' not found in governance.capability_profiles",
                    agent_id,
                    profile_name
                );
            }
        }

        for (agent_id, agent_cfg) in
            std::iter::once((&self.agent.id, &self.agent)).chain(self.agents.iter())
        {
            require_optional_positive(
                agent_cfg.linked_runtime_lanes,
                format!("agent '{}'.linked_runtime_lanes", agent_id),
            )?;
            agent_cfg
                .inference
                .validate_shallow(&self.providers, &format!("agent '{}'.inference", agent_id))?;
            if !agent_cfg.inference.is_empty() {
                let effective_inference = self.inference.merged_with(&agent_cfg.inference);
                effective_inference.validate_complete(
                    &self.providers,
                    &format!("agent '{}'.inference", agent_id),
                )?;
            }
            if let Some(state) = &agent_cfg.persistence.state {
                state.validate(&format!("agent '{}'.persistence.state", agent_id))?;
                self.persistence
                    .resolve_state_target(state)
                    .with_context(|| {
                        format!("agent '{}': invalid persistence.state target", agent_id)
                    })?;
            }
            if let Some(store) = &agent_cfg.persistence.store {
                store.validate(&format!("agent '{}'.persistence.store", agent_id))?;
                self.persistence
                    .resolve_store_target(store)
                    .with_context(|| {
                        format!("agent '{}': invalid persistence.store target", agent_id)
                    })?;
            }
            if let Some(harness_id) = &agent_cfg.harness {
                anyhow::ensure!(
                    harness_id == "default" || self.harnesses.contains_key(harness_id),
                    "agent '{}': harness '{}' not found in [harnesses.*]",
                    agent_id,
                    harness_id
                );
            }
        }

        for (agent_id, _) in
            std::iter::once((&self.agent.id, &self.agent)).chain(self.agents.iter())
        {
            let _ = crate::tools::policy::resolve_effective_tools_config(self, agent_id, None)
                .with_context(|| {
                    format!("invalid [tools] configuration for agent '{}'", agent_id)
                })?;
        }

        Ok(())
    }
}
