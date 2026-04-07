use anyhow::{Context, Result};
use std::collections::HashSet;

use super::*;

impl TurinConfig {
    /// Validate semantic invariants that serde can't enforce.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            !self.agent.model.trim().is_empty(),
            "agent.model must not be empty"
        );
        if !self.providers.contains_key(&self.agent.provider) {
            anyhow::bail!(
                "Provider '{}' configured in [agent] but not found in [providers]",
                self.agent.provider
            );
        }
        anyhow::ensure!(
            self.kernel.max_turns > 0,
            "kernel.max_turns must be greater than 0"
        );
        anyhow::ensure!(
            self.kernel.heartbeat_interval_secs > 0,
            "kernel.heartbeat_interval_secs must be greater than 0"
        );
        if let Some(root) = self.layout.root.as_ref() {
            anyhow::ensure!(
                !root.trim().is_empty(),
                "layout.root must not be empty when set"
            );
        }
        anyhow::ensure!(
            !self.layout.data_dir.trim().is_empty(),
            "layout.data_dir must not be empty"
        );
        anyhow::ensure!(
            !self.layout.states_dir.trim().is_empty(),
            "layout.states_dir must not be empty"
        );
        anyhow::ensure!(
            !self.layout.stores_dir.trim().is_empty(),
            "layout.stores_dir must not be empty"
        );
        anyhow::ensure!(
            !self.layout.harnesses_dir.trim().is_empty(),
            "layout.harnesses_dir must not be empty"
        );
        anyhow::ensure!(
            !self.layout.agents_dir.trim().is_empty(),
            "layout.agents_dir must not be empty"
        );
        anyhow::ensure!(
            !self.layout.channels_dir.trim().is_empty(),
            "layout.channels_dir must not be empty"
        );
        anyhow::ensure!(
            !self.layout.scopes_dir.trim().is_empty(),
            "layout.scopes_dir must not be empty"
        );
        anyhow::ensure!(
            !self.layout.env_file.trim().is_empty(),
            "layout.env_file must not be empty"
        );
        anyhow::ensure!(
            !self.layout.daemon_socket.trim().is_empty(),
            "layout.daemon_socket must not be empty"
        );

        anyhow::ensure!(
            !self.harness.directory.trim().is_empty(),
            "harness.directory must not be empty"
        );
        anyhow::ensure!(
            self.harness.memory_limit_mb > 0,
            "harness.memory_limit_mb must be greater than 0"
        );
        anyhow::ensure!(
            !self.daemon.agents_dir.trim().is_empty(),
            "daemon.agents_dir must not be empty"
        );
        anyhow::ensure!(
            !self.daemon.harnesses_dir.trim().is_empty(),
            "daemon.harnesses_dir must not be empty"
        );
        anyhow::ensure!(
            !self.daemon.channels_dir.trim().is_empty(),
            "daemon.channels_dir must not be empty"
        );
        anyhow::ensure!(
            !self.daemon.endpoint.trim().is_empty(),
            "daemon.endpoint must not be empty"
        );
        anyhow::ensure!(
            !self.remote.bind.trim().is_empty(),
            "remote.bind must not be empty"
        );
        anyhow::ensure!(
            !self.remote.auth_token_env.trim().is_empty(),
            "remote.auth_token_env must not be empty"
        );
        anyhow::ensure!(
            self.remote.event_keepalive_secs > 0,
            "remote.event_keepalive_secs must be greater than 0"
        );
        self.persistence.state.validate("persistence.state")?;
        if let Some(store) = &self.persistence.store {
            store.validate("persistence.store")?;
        }
        for (state_name, state) in &self.persistence.states {
            anyhow::ensure!(
                !state_name.trim().is_empty(),
                "persistence.states contains an empty state name"
            );
            anyhow::ensure!(
                state_name != "state" && state_name != "store",
                "persistence.states.{} is reserved",
                state_name
            );
            anyhow::ensure!(
                !state.path.trim().is_empty(),
                "persistence.states.{}.path must not be empty",
                state_name
            );
        }
        for (store_name, store) in &self.persistence.stores {
            anyhow::ensure!(
                !store_name.trim().is_empty(),
                "persistence.stores contains an empty store name"
            );
            anyhow::ensure!(
                store_name != "state" && store_name != "store",
                "persistence.stores.{} is reserved",
                store_name
            );
            anyhow::ensure!(
                !store.path.trim().is_empty(),
                "persistence.stores.{}.path must not be empty",
                store_name
            );
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
            anyhow::ensure!(
                !placement.scope_kind.trim().is_empty(),
                "persistence.placements[{}].scope_kind must not be empty",
                idx
            );
            if let Some(scope_key) = &placement.scope_key {
                anyhow::ensure!(
                    !scope_key.trim().is_empty(),
                    "persistence.placements[{}].scope_key must not be empty when set",
                    idx
                );
            }
            if let Some(namespace) = &placement.namespace {
                anyhow::ensure!(
                    !namespace.trim().is_empty(),
                    "persistence.placements[{}].namespace must not be empty when set",
                    idx
                );
            }
            anyhow::ensure!(
                !placement.store.trim().is_empty(),
                "persistence.placements[{}].store must not be empty",
                idx
            );
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
            anyhow::ensure!(
                !harness_id.trim().is_empty(),
                "harnesses contains an empty harness id"
            );
            anyhow::ensure!(
                harness_id != "default",
                "harnesses.default is reserved; use [harness] for the default harness"
            );
            anyhow::ensure!(
                !harness_cfg.directory.trim().is_empty(),
                "harnesses.{}.directory must not be empty",
                harness_id
            );
            anyhow::ensure!(
                harness_cfg.memory_limit_mb > 0,
                "harnesses.{}.memory_limit_mb must be greater than 0",
                harness_id
            );
        }

        for (provider_name, provider) in &self.providers {
            if let Some(timeout_secs) = provider.request_timeout_secs {
                anyhow::ensure!(
                    timeout_secs > 0,
                    "providers.{}.request_timeout_secs must be greater than 0",
                    provider_name
                );
            }

            if let Some(timeout_secs) = provider.total_timeout_secs {
                anyhow::ensure!(
                    timeout_secs > 0,
                    "providers.{}.total_timeout_secs must be greater than 0",
                    provider_name
                );
            }

            if let (Some(request_secs), Some(total_secs)) =
                (provider.request_timeout_secs, provider.total_timeout_secs)
            {
                anyhow::ensure!(
                    total_secs >= request_secs,
                    "providers.{}.total_timeout_secs must be >= request_timeout_secs",
                    provider_name
                );
            }

            for header in provider.headers.keys() {
                anyhow::ensure!(
                    !header.trim().is_empty(),
                    "providers.{}.headers contains an empty header name",
                    provider_name
                );
            }
        }

        if let Some(default_context) = self.inference.default.as_deref() {
            anyhow::ensure!(
                !default_context.trim().is_empty(),
                "inference.default must not be empty when set"
            );
            anyhow::ensure!(
                self.inference.contexts.contains_key(default_context),
                "inference.default '{}' not found in inference.contexts",
                default_context
            );
        }

        for (context_name, context) in &self.inference.contexts {
            anyhow::ensure!(
                !context_name.trim().is_empty(),
                "inference.contexts contains an empty context name"
            );
            anyhow::ensure!(
                !context.provider.trim().is_empty(),
                "inference.contexts.{}.provider must not be empty",
                context_name
            );
            anyhow::ensure!(
                !context.model.trim().is_empty(),
                "inference.contexts.{}.model must not be empty",
                context_name
            );
            anyhow::ensure!(
                self.providers.contains_key(&context.provider),
                "inference.contexts.{}.provider '{}' not found in [providers]",
                context_name,
                context.provider
            );
            if let Some(fallback) = context.fallback.as_deref() {
                anyhow::ensure!(
                    !fallback.trim().is_empty(),
                    "inference.contexts.{}.fallback must not be empty when set",
                    context_name
                );
                anyhow::ensure!(
                    self.inference.contexts.contains_key(fallback),
                    "inference.contexts.{}.fallback '{}' not found in inference.contexts",
                    context_name,
                    fallback
                );
            }
            if let Some(temperature) = context.temperature {
                anyhow::ensure!(
                    temperature.is_finite(),
                    "inference.contexts.{}.temperature must be finite",
                    context_name
                );
            }
            if let Some(max_tokens) = context.max_tokens {
                anyhow::ensure!(
                    max_tokens > 0,
                    "inference.contexts.{}.max_tokens must be greater than 0",
                    context_name
                );
            }
        }

        for context_name in self.inference.contexts.keys() {
            let mut seen = HashSet::new();
            let mut current = context_name.as_str();
            while let Some(next) = self
                .inference
                .contexts
                .get(current)
                .and_then(|context| context.fallback.as_deref())
            {
                anyhow::ensure!(
                    seen.insert(current.to_string()),
                    "inference context fallback cycle detected at '{}'",
                    current
                );
                current = next;
            }
        }

        if let Some(ttl_ms) = self.governance.grants.max_ttl_ms {
            anyhow::ensure!(
                ttl_ms > 0,
                "governance.grants.max_ttl_ms must be greater than 0"
            );
        }

        for (root_name, root) in &self.governance.roots {
            anyhow::ensure!(
                !root_name.trim().is_empty(),
                "governance.roots contains an empty root name"
            );
            anyhow::ensure!(
                !root.path.trim().is_empty(),
                "governance.roots.{}.path must not be empty",
                root_name
            );
        }

        for profile_name in self.governance.capability_profiles.keys() {
            anyhow::ensure!(
                !profile_name.trim().is_empty(),
                "governance.capability_profiles contains an empty profile name"
            );
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
