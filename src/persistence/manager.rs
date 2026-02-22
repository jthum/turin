//! Database connection pooling and dynamic Store routing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::RwLock;
use tracing::info;

use super::state::StateStore;

/// Specifies how to locate or select a StateStore.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StoreSelector {
    /// Use a predefined alias (e.g., "state", "agent", "user")
    Alias(String),
    /// Use an explicit file path (relative to workspace or absolute)
    Path(String),
}

/// Manages multiple independent StateStore instances,
/// enabling dynamic DB routing and connection pooling.
pub struct StoreManager {
    /// Active open stores, keyed by their canonical absolute path
    stores: RwLock<HashMap<PathBuf, Arc<StateStore>>>,

    /// Mapping of alias names to canonical paths
    aliases: RwLock<HashMap<String, PathBuf>>,

    /// The root directory for resolving relative paths safely
    workspace_root: PathBuf,
}

impl StoreManager {
    /// Create a new StoreManager bound to a workspace root.
    pub fn new(workspace_root: impl Into<PathBuf>) -> Self {
        Self {
            stores: RwLock::new(HashMap::new()),
            aliases: RwLock::new(HashMap::new()),
            workspace_root: workspace_root.into(),
        }
    }

    /// Register an alias mapped to a specific database path.
    pub async fn register_alias(
        &self,
        alias: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<()> {
        let alias = alias.into();
        let canonical_path = self.resolve_path(path.as_ref());

        let mut aliases = self.aliases.write().await;
        aliases.insert(alias, canonical_path);
        Ok(())
    }

    /// Open or retrieve a cached database connection using a selector.
    pub async fn open(&self, selector: &StoreSelector) -> Result<Arc<StateStore>> {
        let target_path = match selector {
            StoreSelector::Alias(alias) => {
                let aliases = self.aliases.read().await;
                if let Some(path) = aliases.get(alias) {
                    path.clone()
                } else {
                    anyhow::bail!("Unknown store alias: {}", alias);
                }
            }
            StoreSelector::Path(p) => self.resolve_path(Path::new(p)),
        };

        // Fast path: check if it's already open in the cache
        {
            let stores = self.stores.read().await;
            if let Some(store) = stores.get(&target_path) {
                return Ok(Arc::clone(store));
            }
        }

        // Slow path: open DB, initialize it, and insert it
        let db_path_str = target_path
            .to_str()
            .context("Database path contains invalid UTF-8")?;

        info!("Opening dynamic database store at: {}", db_path_str);

        // StateStore::open handles creating the schema automatically
        let store = match StateStore::open(db_path_str).await {
            Ok(s) => Arc::new(s),
            Err(e) => anyhow::bail!("Failed to open store at {}: {}", db_path_str, e),
        };

        let mut stores = self.stores.write().await;
        // Check again in case another thread opened it while we were awaiting IO
        if let Some(existing) = stores.get(&target_path) {
            Ok(Arc::clone(existing))
        } else {
            stores.insert(target_path, Arc::clone(&store));
            Ok(store)
        }
    }

    /// Convenience wrapper to retrieve a store specifically tracked by the "state" alias.
    pub async fn get_default(&self) -> Result<Arc<StateStore>> {
        self.open(&StoreSelector::Alias("state".to_string())).await
    }

    /// Ensure a path is absolute, resolving relative paths against the workspace root.
    fn resolve_path(&self, requested: &Path) -> PathBuf {
        if requested.is_absolute() {
            requested.to_path_buf()
        } else {
            self.workspace_root.join(requested)
        }
    }
}
