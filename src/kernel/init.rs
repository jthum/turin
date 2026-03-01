//! Kernel initialization — provider clients, state store, harness, and file watcher.
//!
//! This module contains all one-time setup methods that must be called before
//! the agent loop runs. They are separated from the core kernel module to keep
//! each file focused on a single responsibility.

use anyhow::{Context, Result};
use notify::Event;
use std::sync::Arc;
use tracing::{error, info, instrument, warn};

use super::Kernel;
use super::harness_runtime::HarnessRuntimeInitContext;
use crate::inference::provider::{self, ProviderClient};

impl Kernel {
    fn harness_init_context(&self) -> HarnessRuntimeInitContext {
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

    /// Initialize all configured provider clients. Call before `init_harness()` and `run()`.
    pub fn init_clients(&mut self) -> Result<()> {
        for (name, config) in &self.config.providers {
            let client = self.create_client(name, config)?;
            self.clients.insert(name.clone(), client);
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
            match config {
                crate::kernel::config::EmbeddingConfig::OpenAI => {
                    // Find a provider with type="openai"
                    let openai_config = self
                        .config
                        .providers
                        .values()
                        .find(|p| p.kind == "openai")
                        .with_context(
                            || "OpenAI embeddings selected but no OpenAI provider configured",
                        )?;

                    let api_key_env = openai_config.api_key_env.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("OpenAI provider missing 'api_key_env' configuration")
                    })?;
                    let api_key = std::env::var(api_key_env).with_context(|| {
                        format!("Environment variable '{}' not set", api_key_env)
                    })?;

                    crate::inference::embeddings::create_embedding_provider(
                        &crate::inference::embeddings::EmbeddingConfig::OpenAI {
                            api_key,
                            model: "text-embedding-3-small".to_string(),
                        },
                    )
                }
                crate::kernel::config::EmbeddingConfig::NoOp => {
                    crate::inference::embeddings::create_embedding_provider(
                        &crate::inference::embeddings::EmbeddingConfig::NoOp,
                    )
                }
            }
        } else {
            // No embeddings configured — use NoOp (no hidden fallback to OpenAI)
            crate::inference::embeddings::create_embedding_provider(
                &crate::inference::embeddings::EmbeddingConfig::NoOp,
            )
        };

        self.embedding_provider = Some(Arc::from(embedding_provider));
        self.agent_manager
            .bind_inference_state(self.clients.clone(), self.embedding_provider.clone());

        Ok(())
    }

    /// Initialize the default state store alias. Call before `run()`.
    pub async fn init_state(&mut self) -> Result<()> {
        let db_path = &self.config.persistence.database_path;
        self.store_manager.register_alias("state", db_path).await?;
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

    /// Start watching the harness directory for changes (background thread).
    #[instrument(skip(self))]
    pub fn start_watcher(&mut self) -> Result<()> {
        use notify::{RecursiveMode, Watcher};
        use std::time::Duration;

        let runtimes: Vec<_> = self.harness_manager.runtimes().cloned().collect();
        let init_ctx = self.harness_init_context();
        let harness_roots: Vec<_> = self
            .harness_manager
            .runtimes()
            .map(|runtime| runtime.directory().to_path_buf())
            .collect();

        let explicit_watch_roots = self.harness_manager.explicit_watch_roots();

        // We use an async channel to debounce events
        let (tx, mut rx) = tokio::sync::mpsc::channel::<()>(10);

        // Spawn background task to handle reloads with debouncing
        tokio::spawn(async move {
            while rx.recv().await.is_some() {
                // Debounce: Wait for more events
                tokio::time::sleep(Duration::from_millis(200)).await;
                // Clear any pending events that arrived during sleep
                while rx.try_recv().is_ok() {}

                info!("Hot-reload triggered by file change");
                let runtimes = runtimes.clone();
                let ctx = init_ctx.clone();

                tokio::spawn(async move {
                    for runtime in runtimes {
                        if let Err(err) = runtime.reload(ctx.clone()) {
                            error!(
                                harness_id = %runtime.directory().display(),
                                error = %err,
                                "Harness hot-reload failed"
                            );
                        }
                    }
                });
            }
        });

        let mut watcher =
            notify::recommended_watcher(move |res: notify::Result<Event>| match res {
                Ok(event) => {
                    if event.kind.is_modify() || event.kind.is_create() || event.kind.is_remove() {
                        let _ = tx.blocking_send(());
                    }
                }
                Err(e) => error!(error = ?e, "Watcher channel error"),
            })?;

        let mut watched_any = false;
        for harness_dir in harness_roots {
            if !harness_dir.exists() {
                warn!(directory = %harness_dir.display(), "Harness directory does not exist, skipping watcher");
                continue;
            }
            watcher.watch(&harness_dir, RecursiveMode::NonRecursive)?;
            watched_any = true;
            info!(directory = %harness_dir.display(), "Watching harness directory");
        }
        for extra_root in explicit_watch_roots {
            if !extra_root.exists() {
                warn!(path = %extra_root.display(), "Explicit watch path does not exist, skipping");
                continue;
            }
            let mode = if extra_root.is_dir() {
                RecursiveMode::Recursive
            } else {
                RecursiveMode::NonRecursive
            };
            watcher.watch(&extra_root, mode)?;
            info!(path = %extra_root.display(), recursive = matches!(mode, RecursiveMode::Recursive), "Watching explicit harness path");
            watched_any = true;
        }
        if !watched_any {
            warn!("No harness directories or explicit watch roots available, skipping watcher");
            return Ok(());
        }
        self.check_watcher = Some(watcher);

        Ok(())
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
