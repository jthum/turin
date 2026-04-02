//! Database connection pooling and dynamic Store routing.

mod cache_support;
mod path_support;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use tokio::sync::RwLock;

use super::state::StateStore;

/// Specifies how to locate or select a StateStore.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StoreSelector {
    /// Use a predefined alias (e.g., "state", "agent", "user")
    Alias(String),
    /// Use an explicit file path (relative to workspace or absolute)
    Path(String),
    /// Use an opaque runtime handle previously returned by `runtime.db.open(...)`.
    Handle(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorePathScope {
    WorkspaceOnly,
    AllowAny,
}

impl StorePathScope {
    pub fn from_policy(value: &str) -> Self {
        match value {
            "allow_any" => Self::AllowAny,
            _ => Self::WorkspaceOnly,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StoreHandleInfo {
    pub handle: String,
    pub path: PathBuf,
    pub alias: Option<String>,
    pub open_count: u64,
    pub idle_ms: u64,
}

struct StoreCacheEntry {
    store: Arc<StateStore>,
    last_access: Instant,
}

#[derive(Debug)]
struct HandleEntry {
    path: PathBuf,
    alias: Option<String>,
    open_count: u64,
    last_access: Instant,
}

/// Manages multiple independent StateStore instances,
/// enabling dynamic DB routing and connection pooling.
pub struct StoreManager {
    /// Active open stores, keyed by their canonical absolute path
    stores: RwLock<HashMap<PathBuf, StoreCacheEntry>>,

    /// Mapping of alias names to canonical paths
    aliases: RwLock<HashMap<String, PathBuf>>,

    /// Opaque runtime handle registry (handle id -> target path)
    handles: RwLock<HashMap<String, HandleEntry>>,

    /// The root directory for resolving relative paths safely
    workspace_root: PathBuf,

    /// Default root for alias-backed auxiliary stores.
    store_root: PathBuf,
}

impl StoreManager {
    /// Create a new StoreManager bound to a workspace root.
    pub fn new(workspace_root: impl Into<PathBuf>, store_root: impl Into<PathBuf>) -> Self {
        Self {
            stores: RwLock::new(HashMap::new()),
            aliases: RwLock::new(HashMap::new()),
            handles: RwLock::new(HashMap::new()),
            workspace_root: workspace_root.into(),
            store_root: store_root.into(),
        }
    }

    /// Register an alias mapped to a specific database path.
    pub async fn register_alias(
        &self,
        alias: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<()> {
        let alias = alias.into();
        let canonical_path = self.resolve_path_unchecked(path.as_ref())?;

        let mut aliases = self.aliases.write().await;
        aliases.insert(alias, canonical_path);
        Ok(())
    }

    /// Open or retrieve a cached database connection using a selector.
    ///
    /// This path does not enforce harness runtime policies. Harness-facing code should use
    /// [`StoreManager::open_with_path_scope`] or [`StoreManager::open_handle`].
    pub async fn open(&self, selector: &StoreSelector) -> Result<Arc<StateStore>> {
        self.open_with_path_scope(selector, StorePathScope::AllowAny)
            .await
    }

    /// Open or retrieve a cached database connection with path-scope enforcement.
    pub async fn open_with_path_scope(
        &self,
        selector: &StoreSelector,
        path_scope: StorePathScope,
    ) -> Result<Arc<StateStore>> {
        let target_path = self.resolve_selector_path(selector, path_scope).await?;
        self.open_path_cached(&target_path).await
    }

    pub async fn resolve_path_for_selector(
        &self,
        selector: &StoreSelector,
        path_scope: StorePathScope,
    ) -> Result<PathBuf> {
        self.resolve_selector_path(selector, path_scope).await
    }

    /// List registered aliases.
    pub async fn list_aliases(&self) -> Vec<String> {
        let aliases = self.aliases.read().await;
        let mut names = aliases.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    }

    /// Convenience wrapper to retrieve a store specifically tracked by the "state" alias.
    pub async fn get_default(&self) -> Result<Arc<StateStore>> {
        self.open(&StoreSelector::Alias("state".to_string())).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn store_manager_workspace_scope_rejects_parent_traversal() {
        let cwd = std::env::current_dir().unwrap();
        let mgr = StoreManager::new(cwd.clone(), cwd.join(".turin/data/stores"));
        let result = mgr
            .open_with_path_scope(
                &StoreSelector::Path("../outside.db".to_string()),
                StorePathScope::WorkspaceOnly,
            )
            .await;
        let err = match result {
            Ok(_) => panic!("expected workspace path scope to reject traversal"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("Path traversal")
                || err.to_string().contains("outside workspace")
        );
    }
}
