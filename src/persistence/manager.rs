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

    pub fn as_str(self) -> &'static str {
        match self {
            Self::AllowAny => "allow_any",
            Self::WorkspaceOnly => "workspace_only",
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

struct RawStoreCacheEntry {
    database: Arc<turso::Database>,
    last_access: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqlStoreKind {
    State,
    Raw,
}

#[derive(Debug)]
struct HandleEntry {
    path: PathBuf,
    alias: Option<String>,
    kind: SqlStoreKind,
    open_count: u64,
    last_access: Instant,
}

/// Manages multiple independent StateStore instances,
/// enabling dynamic DB routing and connection pooling.
pub struct StoreManager {
    /// Active open stores, keyed by their canonical absolute path
    stores: RwLock<HashMap<PathBuf, StoreCacheEntry>>,

    /// Raw SQL databases opened explicitly through `runtime.db` paths.
    raw_stores: RwLock<HashMap<PathBuf, RawStoreCacheEntry>>,

    /// Prevent one path from being opened with conflicting schema ownership.
    store_kinds: RwLock<HashMap<PathBuf, SqlStoreKind>>,

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
            raw_stores: RwLock::new(HashMap::new()),
            store_kinds: RwLock::new(HashMap::new()),
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

    #[tokio::test]
    async fn runtime_sql_paths_are_raw_and_cannot_be_reused_as_state_store_handles() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = StoreManager::new(tmp.path(), tmp.path().join("stores"));
        let handle = mgr
            .open_handle(
                &StoreSelector::Path("harness.db".to_string()),
                StorePathScope::WorkspaceOnly,
                8,
                300,
            )
            .await
            .unwrap();
        let selector = StoreSelector::Handle(handle.handle.clone());
        let (conn, kind) = mgr
            .open_sql_connection(&selector, StorePathScope::WorkspaceOnly)
            .await
            .unwrap();
        assert_eq!(kind, SqlStoreKind::Raw);
        conn.execute("CREATE TABLE custom_data (value TEXT)", ())
            .await
            .unwrap();

        let mut rows = conn
            .query(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name",
                (),
            )
            .await
            .unwrap();
        let mut tables = Vec::new();
        while let Some(row) = rows.next().await.unwrap() {
            tables.push(row.get::<String>(0).unwrap());
        }
        assert_eq!(tables, ["custom_data"]);

        let error = match mgr
            .open_with_path_scope(&selector, StorePathScope::WorkspaceOnly)
            .await
        {
            Ok(_) => panic!("raw SQL handles must not become semantic state stores"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("Raw SQL database handles"));
    }

    #[tokio::test]
    async fn state_and_raw_stores_share_one_retained_cache_budget() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = StoreManager::new(tmp.path(), tmp.path().join("stores"));
        mgr.register_alias("state", tmp.path().join("state.db"))
            .await
            .unwrap();

        let state = mgr.get_default().await.unwrap();
        drop(state);
        let raw = mgr
            .open_handle(
                &StoreSelector::Path("harness.db".to_string()),
                StorePathScope::WorkspaceOnly,
                8,
                300,
            )
            .await
            .unwrap();
        mgr.close_handle(&raw.handle).await.unwrap();

        mgr.trim_cache(1, 300).await;
        assert!(mgr.stores.read().await.len() + mgr.raw_stores.read().await.len() <= 1);
    }
}
