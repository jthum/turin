//! Kernel initialization — provider clients, state store, harness, and file watcher.
//!
//! This module contains all one-time setup methods that must be called before
//! the agent loop runs. They are separated from the core kernel module to keep
//! each file focused on a single responsibility.

use anyhow::{Context, Result};
use notify::Event;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, error, info, instrument, warn};

use super::Kernel;
use super::config::TurinConfig;
use crate::harness::engine::HarnessEngine;
use crate::harness::globals::HarnessAppData;
use crate::inference::embeddings::EmbeddingProvider;
use crate::inference::provider::{self, ProviderClient};
use crate::persistence::state::StateStore;

impl Kernel {
    /// Initialize all configured provider clients. Call before `init_harness()` and `run()`.
    pub fn init_clients(&mut self) -> Result<()> {
        for (name, config) in &self.config.providers {
            let client = self.create_client(name, config)?;
            self.clients.insert(name.clone(), client);
        }

        // Ensure the default provider is available
        let default_provider_name = &self.config.agent.provider;

        if !self.clients.contains_key(default_provider_name)
            && !self.config.providers.contains_key(default_provider_name)
        {
            anyhow::bail!(
                "Default provider '{}' not found in [providers] configuration",
                default_provider_name
            );
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

        Ok(())
    }

    /// Initialize the state store. Call before `run()`.
    pub async fn init_state(&mut self) -> Result<()> {
        let db_path = &self.config.persistence.database_path;
        let store = StateStore::open(db_path)
            .await
            .with_context(|| format!("Failed to initialize state store at '{}'", db_path))?;
        info!(db_path = %db_path, "State store initialized");
        self.state = Some(store.clone());

        // Start background persistence task - MOVED to create_session
        // init_state now only initializes the store.

        Ok(())
    }

    /// Initialize the harness engine. Call after `init_state()` and before `run()`.
    #[instrument(skip(self), fields(directory = %self.config.harness.directory))]
    pub async fn init_harness(&mut self) -> Result<()> {
        info!("Initializing harness");
        let harness_dir = PathBuf::from(&self.config.harness.directory);

        // Resolve fs_root: "." means workspace root, otherwise use as-is
        let fs_root = if self.config.harness.fs_root == "." {
            PathBuf::from(&self.config.kernel.workspace_root)
        } else {
            PathBuf::from(&self.config.harness.fs_root)
        };

        let app_data = HarnessAppData {
            fs_root,
            workspace_root: PathBuf::from(&self.config.kernel.workspace_root),
            state_store: self.state.clone(),
            clients: self.clients.clone(),
            embedding_provider: self.embedding_provider.clone(),
            queue: self.active_queue.clone(),
            active_session_id: Arc::new(std::sync::Mutex::new(None)),
            config: self.config.clone(),
            spawn_depth: self.config.kernel.initial_spawn_depth,
        };

        let mut engine =
            HarnessEngine::new(app_data).with_context(|| "Failed to create harness engine")?;

        engine.load_dir(&harness_dir).with_context(|| {
            format!(
                "Failed to load harness scripts from '{}'",
                harness_dir.display()
            )
        })?;

        let script_count = engine.loaded_scripts().len();
        if script_count > 0 {
            info!(count = script_count, directory = %harness_dir.display(), "Harness scripts loaded");
            for name in engine.loaded_scripts() {
                debug!(script = %name, "Loaded harness script");
            }
        } else {
            warn!(directory = %harness_dir.display(), "No harness scripts found");
        }

        {
            let mut h = self.lock_harness();
            *h = Some(engine);
        }
        Ok(())
    }

    /// Reload the harness from disk (atomic swap).
    #[instrument(skip(self))]
    pub async fn reload_harness(&mut self) -> Result<()> {
        info!("Reloading harness");
        self.init_harness().await
    }

    #[instrument(skip_all)]
    pub fn reload_harness_static(
        harness: Arc<std::sync::Mutex<Option<HarnessEngine>>>,
        config: Arc<TurinConfig>,
        clients: HashMap<String, ProviderClient>,
        state: Option<StateStore>,
        embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
        active_queue: crate::harness::globals::ActiveSessionQueue,
    ) -> Result<()> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let harness_dir = PathBuf::from(&config.harness.directory);
                let fs_root = if config.harness.fs_root == "." {
                    PathBuf::from(&config.kernel.workspace_root)
                } else {
                    PathBuf::from(&config.harness.fs_root)
                };

                let app_data = HarnessAppData {
                    fs_root,
                    workspace_root: PathBuf::from(&config.kernel.workspace_root),
                    state_store: state.clone(),
                    clients,
                    embedding_provider,
                    queue: active_queue,
                    active_session_id: Arc::new(std::sync::Mutex::new(None)),
                    config: config.clone(),
                    spawn_depth: config.kernel.initial_spawn_depth,
                };

                match HarnessEngine::new(app_data) {
                    Ok(mut engine) => match engine.load_dir(&harness_dir) {
                        Ok(_) => {
                            let script_count = engine.loaded_scripts().len();
                            let mut h = harness.lock().expect("harness mutex poisoned");
                            *h = Some(engine);
                            info!(count = script_count, "Harness reloaded successfully");
                        }
                        Err(e) => error!(error = %e, "Failed to load harness scripts"),
                    },
                    Err(e) => {
                        error!(error = %e, "Failed to create harness engine during static reload")
                    }
                }
                Ok(())
            })
        })
    }

    /// Start watching the harness directory for changes (background thread).
    #[instrument(skip(self))]
    pub fn start_watcher(&mut self) -> Result<()> {
        use notify::{RecursiveMode, Watcher};
        use std::time::Duration;

        let harness_clone = self.harness.clone();
        let config_clone = self.config.clone();
        let clients_clone = self.clients.clone();
        let state_clone = self.state.clone();
        let embedding_clone = self.embedding_provider.clone();
        let queue_clone = self.active_queue.clone();
        let harness_dir = PathBuf::from(&config_clone.harness.directory);

        if !harness_dir.exists() {
            warn!(directory = %harness_dir.display(), "Harness directory does not exist, skipping watcher");
            return Ok(());
        }

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
                let h = harness_clone.clone();
                let c = config_clone.clone();
                let cl = clients_clone.clone();
                let s = state_clone.clone();
                let e = embedding_clone.clone();
                let q = queue_clone.clone();

                tokio::spawn(async move {
                    if let Err(err) = Self::reload_harness_static(h, c, cl, s, e, q) {
                        error!(error = %err, "Harness hot-reload failed");
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

        watcher.watch(&harness_dir, RecursiveMode::NonRecursive)?;
        self.check_watcher = Some(watcher);

        info!(directory = %harness_dir.display(), "Watching harness directory");

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
