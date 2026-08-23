//! Kernel initialization — provider clients, state store, harness, and file watcher.
//!
//! This module contains all one-time setup methods that must be called before
//! the agent loop runs. They are separated from the core kernel module to keep
//! each file focused on a single responsibility.

use anyhow::{Context, Result};
use notify::Event;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{error, info, instrument, warn};

use super::Kernel;
use super::execution_host::ExecutionHost;
use super::harness_runtime::{HarnessDefinition, HarnessRuntimeInitContext};
use crate::inference::provider::{self, ProviderClient};

impl ExecutionHost {
    pub(crate) async fn sync_runtime_signal_subscriptions(&self) -> Result<()> {
        if self.scheduler.is_none() {
            return Ok(());
        }

        let harness_ids: Vec<_> = self
            .harness_manager
            .definition_entries()
            .map(|(harness_id, _)| harness_id.clone())
            .collect();
        self.sync_runtime_signal_subscriptions_for_harnesses(&harness_ids)
            .await?;
        Ok(())
    }

    pub(crate) async fn sync_runtime_signal_subscriptions_for_harnesses(
        &self,
        harness_ids: &[String],
    ) -> Result<()> {
        let Some(scheduler) = self.scheduler.as_ref() else {
            return Ok(());
        };

        let agent_ids = self.harness_manager.agent_ids_for_harnesses(harness_ids);
        let subscriptions = self
            .harness_manager
            .signal_subscriptions_for_harnesses(harness_ids);
        scheduler
            .runtime_store()
            .replace_signal_subscriptions_for_agents(&agent_ids, &subscriptions)
            .await
    }

    pub(crate) fn harness_init_context(&self) -> HarnessRuntimeInitContext {
        HarnessRuntimeInitContext {
            config: self.config.clone(),
            store_manager: self.store_manager.clone(),
            agent_manager: self.agent_manager.clone(),
            policy_manager: self.policy_manager.clone(),
            governance_manager: self.governance_manager.clone(),
            scheduler: self.scheduler.clone(),
            embedding_provider: self.embedding_provider.clone(),
        }
    }

    /// Create the appropriate provider client from config.
    pub(crate) fn create_client(
        &self,
        _name: &str,
        config: &crate::kernel::config::ProviderConfig,
    ) -> Result<ProviderClient> {
        let client = provider::create_provider_client(config, &self.config)?;
        Ok(ProviderClient::new(config.kind.clone(), client))
    }
}

impl ExecutionHost {
    /// Initialize all configured provider clients. Call before `init_harness()` and `run()`.
    pub fn init_clients(&mut self) -> Result<()> {
        let providers: Vec<_> = self
            .config
            .providers
            .iter()
            .map(|(name, config)| (name.clone(), config.clone()))
            .collect();
        for (name, config) in providers {
            let client = self.create_client(&name, &config)?;
            self.clients.insert(name, client);
        }

        for agent in std::iter::once(&self.config.agent).chain(self.config.agents.values()) {
            let provider_name = &agent.provider;
            if !self.clients.contains_key(provider_name)
                && !self.config.providers.contains_key(provider_name)
            {
                anyhow::bail!(
                    "Provider '{}' required by agent '{}' not found in [providers] configuration",
                    provider_name,
                    agent.id
                );
            }
        }

        // Initialize embedding provider
        let embedding_provider = if let Some(ref config) = self.config.embeddings {
            Some(crate::inference::embeddings::create_embedding_provider(
                config,
                &self.config.providers,
                &self.config,
            )?)
        } else {
            // No embeddings configured means lexical-only memory behavior.
            None
        };

        self.embedding_provider = embedding_provider;
        self.agent_manager
            .bind_inference_state(self.clients.clone(), self.embedding_provider.clone());

        Ok(())
    }

    /// Initialize the default state store alias. Call before `run()`.
    pub async fn init_state(&mut self) -> Result<()> {
        let state_selector = self.config.persistence.top_level_state_selector()?;
        let db_path = self
            .store_manager
            .resolve_path_for_selector(
                &state_selector,
                crate::persistence::manager::StorePathScope::AllowAny,
            )
            .await?
            .display()
            .to_string();
        self.store_manager.register_alias("state", &db_path).await?;
        for (alias, store) in &self.config.persistence.states {
            self.store_manager
                .register_alias(alias, &store.path)
                .await?;
        }
        for (alias, store) in &self.config.persistence.stores {
            self.store_manager
                .register_alias(alias, &store.path)
                .await?;
        }
        let _ = self.store_manager.get_default().await.with_context(|| {
            format!("Failed to initialize default state store at '{}'", db_path)
        })?;
        info!(db_path = %db_path, "State store initialized");

        Ok(())
    }

    /// Initialize the harness engine. Call after `init_state()` and before `run()`.
    #[instrument(skip(self), fields(directory = %self.config.harness.directory))]
    pub async fn init_harness(&mut self) -> Result<()> {
        info!("Initializing harness");
        for definition in self.harness_manager.definitions() {
            definition.init(self.harness_init_context())?;
        }
        self.sync_runtime_signal_subscriptions().await?;
        Ok(())
    }

    /// Reload the harness from disk (atomic swap).
    #[instrument(skip(self))]
    pub async fn reload_harness(&mut self) -> Result<()> {
        info!("Reloading harness");
        for definition in self.harness_manager.definitions() {
            definition.reload(self.harness_init_context())?;
        }
        self.sync_runtime_signal_subscriptions().await?;
        Ok(())
    }

    pub async fn reload_named_harness(&mut self, harness_id: &str) -> Result<()> {
        let definition = self
            .harness_manager
            .definition_by_id(harness_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown harness '{}'", harness_id))?;
        info!(harness_id = %harness_id, "Reloading named harness");
        definition.reload(self.harness_init_context())?;
        self.sync_runtime_signal_subscriptions_for_harnesses(&[harness_id.to_string()])
            .await?;
        Ok(())
    }

    pub fn validate_named_harness(&self, harness_id: &str) -> Result<usize> {
        let definition = self
            .harness_manager
            .definition_by_id(harness_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown harness '{}'", harness_id))?;
        definition.validate(self.harness_init_context())
    }

    pub(crate) fn validate_named_harness_sources(
        &self,
        harness_id: &str,
        source_overlay: crate::harness::source::HarnessSourceOverlay,
    ) -> Result<usize> {
        let definition = self
            .harness_manager
            .definition_by_id(harness_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown harness '{}'", harness_id))?;
        definition.validate_sources(self.harness_init_context(), source_overlay)
    }
}

impl Kernel {
    pub(crate) async fn reconcile_agent_catalog(
        &mut self,
        config: crate::kernel::config::TurinConfig,
        affected_agents: &HashSet<String>,
    ) -> Result<()> {
        self.agent_manager
            .ensure_agents_reconfigurable(affected_agents)
            .await?;
        let config = Arc::new(config);
        let empty_rust_harness_factories = crate::kernel::harness::RustHarnessFactories::new();
        let rust_harness_factories = self
            .host
            .rust_harness_factories
            .as_deref()
            .unwrap_or(&empty_rust_harness_factories);
        let harness_manager = Arc::new(
            crate::kernel::harness_manager::HarnessManager::from_config_with_harnesses(
                config.as_ref(),
                rust_harness_factories,
                self.host.script_harness_adapter.as_ref(),
            )?,
        );
        let mut init_context = self.harness_init_context();
        init_context.config = Arc::clone(&config);
        for definition in harness_manager.definitions() {
            definition.init(init_context.clone())?;
        }

        self.agent_manager
            .reconcile_runtime_catalog(
                Arc::clone(&config),
                Arc::clone(&harness_manager),
                affected_agents,
            )
            .await?;
        self.host.config = config;
        self.host.harness_manager = harness_manager;
        if let Err(err) = self.sync_runtime_signal_subscriptions().await {
            warn!(error = %err, "Failed to reconcile runtime signal subscriptions after agent catalog update");
        }
        if let Err(err) = self.start_watcher() {
            warn!(error = %err, "Failed to refresh harness watcher after agent catalog update");
        }
        Ok(())
    }

    /// Start watching the harness directory for changes (background thread).
    #[instrument(skip(self))]
    pub fn start_watcher(&mut self) -> Result<()> {
        use std::time::Duration;

        let definitions: Vec<_> = self.harness_manager.definitions().cloned().collect();
        let task_definitions = definitions.clone();
        let init_ctx = self.harness_init_context();
        let harness_manager = Arc::clone(&self.harness_manager);
        let watcher_slot = Arc::clone(&self.check_watcher);

        // We use an async channel to debounce events
        let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<PathBuf>>(32);
        let reload_tx = tx.clone();

        // Spawn background task to handle reloads with debouncing
        tokio::spawn(async move {
            while let Some(mut changed_paths) = rx.recv().await {
                // Debounce: Wait for more events
                tokio::time::sleep(Duration::from_millis(200)).await;
                // Clear any pending events that arrived during sleep
                while let Ok(mut more_paths) = rx.try_recv() {
                    changed_paths.append(&mut more_paths);
                }

                let affected = affected_definitions(&task_definitions, &changed_paths);
                if affected.is_empty() {
                    continue;
                }

                info!(
                    count = affected.len(),
                    "Hot-reload triggered by file change"
                );
                let ctx = init_ctx.clone();
                let harness_ids: Vec<_> = affected
                    .iter()
                    .map(|runtime| runtime.harness_id().to_string())
                    .collect();
                info!(?harness_ids, "Reloading affected harness runtimes");

                for runtime in &affected {
                    if let Err(err) = runtime.reload(ctx.clone()) {
                        error!(
                            harness_id = %runtime.harness_id(),
                            error = %err,
                            "Harness hot-reload failed"
                        );
                    }
                }

                if let Some(scheduler) = ctx.scheduler.as_ref() {
                    let agent_ids = harness_manager.agent_ids_for_harnesses(&harness_ids);
                    let subscriptions =
                        harness_manager.signal_subscriptions_for_harnesses(&harness_ids);
                    if let Err(err) = scheduler
                        .runtime_store()
                        .replace_signal_subscriptions_for_agents(&agent_ids, &subscriptions)
                        .await
                    {
                        error!(error = %err, "Failed to sync durable runtime signal subscriptions");
                    }
                }

                match build_harness_watcher(&task_definitions, reload_tx.clone()) {
                    Ok(watcher) => {
                        let mut slot = watcher_slot
                            .lock()
                            .expect("watcher mutex poisoned during rebuild");
                        *slot = watcher;
                    }
                    Err(err) => {
                        error!(error = %err, "Failed to rebuild harness watcher");
                    }
                }
            }
        });

        let watcher = build_harness_watcher(&definitions, tx)?;
        let mut slot = self
            .check_watcher
            .lock()
            .expect("watcher mutex poisoned during startup");
        *slot = watcher;

        Ok(())
    }
}

fn affected_definitions(
    definitions: &[Arc<HarnessDefinition>],
    changed_paths: &[PathBuf],
) -> Vec<Arc<HarnessDefinition>> {
    let mut seen = HashSet::new();
    let mut affected = Vec::new();

    for definition in definitions {
        if changed_paths.iter().any(|path| definition.owns_path(path))
            && seen.insert(definition.harness_id().to_string())
        {
            affected.push(Arc::clone(definition));
        }
    }

    affected
}

fn build_harness_watcher(
    definitions: &[Arc<HarnessDefinition>],
    tx: tokio::sync::mpsc::Sender<Vec<PathBuf>>,
) -> Result<Option<notify::RecommendedWatcher>> {
    use notify::{RecursiveMode, Watcher};

    let roots = collect_watch_roots(definitions);
    if roots.is_empty() {
        warn!("No harness directories or explicit watch roots available, skipping watcher");
        return Ok(None);
    }

    let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| match res {
        Ok(event) => {
            if event.kind.is_modify() || event.kind.is_create() || event.kind.is_remove() {
                let _ = tx.blocking_send(event.paths.clone());
            }
        }
        Err(e) => error!(error = ?e, "Watcher channel error"),
    })?;

    for root in roots {
        if !root.path.exists() {
            warn!(
                path = %root.path.display(),
                harness_id = %root.harness_id,
                "Watch path does not exist, skipping"
            );
            continue;
        }
        let mode = if root.recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };
        watcher.watch(&root.path, mode)?;
        info!(
            harness_id = %root.harness_id,
            path = %root.path.display(),
            recursive = matches!(mode, RecursiveMode::Recursive),
            "Watching harness path"
        );
    }

    Ok(Some(watcher))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct OwnedWatchRoot {
    harness_id: String,
    path: PathBuf,
    recursive: bool,
}

fn collect_watch_roots(definitions: &[Arc<HarnessDefinition>]) -> Vec<OwnedWatchRoot> {
    let mut roots = Vec::new();
    for definition in definitions {
        for root in definition.watch_roots() {
            let owned = OwnedWatchRoot {
                harness_id: definition.harness_id().to_string(),
                path: root.path,
                recursive: root.recursive,
            };
            if !roots.contains(&owned) {
                roots.push(owned);
            }
        }
    }
    roots
}
