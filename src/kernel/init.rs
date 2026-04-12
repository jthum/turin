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
use super::harness_runtime::{HarnessRuntime, HarnessRuntimeInitContext};
use crate::inference::provider::{self, ProviderClient};

impl ExecutionHost {
    pub(crate) fn harness_init_context(&self) -> HarnessRuntimeInitContext {
        HarnessRuntimeInitContext {
            config: self.config.clone(),
            clients: self.clients.clone(),
            store_manager: self.store_manager.clone(),
            agent_manager: self.agent_manager.clone(),
            policy_manager: self.policy_manager.clone(),
            governance_manager: self.governance_manager.clone(),
            embedding_provider: self.embedding_provider.clone(),
        }
    }

    /// Create the appropriate provider client from config.
    pub(crate) fn create_client(
        &self,
        _name: &str,
        config: &crate::kernel::config::ProviderConfig,
    ) -> Result<ProviderClient> {
        let client = provider::create_provider_client(config)?;
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
        for runtime in self.harness_manager.runtimes() {
            runtime.init(self.harness_init_context())?;
        }
        Ok(())
    }

    /// Reload the harness from disk (atomic swap).
    #[instrument(skip(self))]
    pub async fn reload_harness(&mut self) -> Result<()> {
        info!("Reloading harness");
        for runtime in self.harness_manager.runtimes() {
            runtime.reload(self.harness_init_context())?;
        }
        Ok(())
    }

    pub async fn reload_named_harness(&mut self, harness_id: &str) -> Result<()> {
        let runtime = self
            .harness_manager
            .runtime_by_id(harness_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown harness '{}'", harness_id))?;
        info!(harness_id = %harness_id, "Reloading named harness");
        runtime.reload(self.harness_init_context())?;
        Ok(())
    }

    pub fn validate_named_harness(&self, harness_id: &str) -> Result<usize> {
        let runtime = self
            .harness_manager
            .runtime_by_id(harness_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown harness '{}'", harness_id))?;
        runtime.validate(self.harness_init_context())
    }
}

impl Kernel {
    /// Start watching the harness directory for changes (background thread).
    #[instrument(skip(self))]
    pub fn start_watcher(&mut self) -> Result<()> {
        use std::time::Duration;

        let runtimes: Vec<_> = self.harness_manager.runtimes().cloned().collect();
        let task_runtimes = runtimes.clone();
        let init_ctx = self.harness_init_context();
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

                let affected = affected_runtimes(&task_runtimes, &changed_paths);
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

                match build_harness_watcher(&task_runtimes, reload_tx.clone()) {
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

        let watcher = build_harness_watcher(&runtimes, tx)?;
        let mut slot = self
            .check_watcher
            .lock()
            .expect("watcher mutex poisoned during startup");
        *slot = watcher;

        Ok(())
    }
}

fn affected_runtimes(
    runtimes: &[Arc<HarnessRuntime>],
    changed_paths: &[PathBuf],
) -> Vec<Arc<HarnessRuntime>> {
    let mut seen = HashSet::new();
    let mut affected = Vec::new();

    for runtime in runtimes {
        if changed_paths.iter().any(|path| runtime.owns_path(path))
            && seen.insert(runtime.harness_id().to_string())
        {
            affected.push(Arc::clone(runtime));
        }
    }

    affected
}

fn build_harness_watcher(
    runtimes: &[Arc<HarnessRuntime>],
    tx: tokio::sync::mpsc::Sender<Vec<PathBuf>>,
) -> Result<Option<notify::RecommendedWatcher>> {
    use notify::{RecursiveMode, Watcher};

    let roots = collect_watch_roots(runtimes);
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

fn collect_watch_roots(runtimes: &[Arc<HarnessRuntime>]) -> Vec<OwnedWatchRoot> {
    let mut roots = Vec::new();
    for runtime in runtimes {
        for root in runtime.watch_roots() {
            let owned = OwnedWatchRoot {
                harness_id: runtime.harness_id().to_string(),
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
