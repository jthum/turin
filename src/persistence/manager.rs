//! Database connection pooling and dynamic Store routing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

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
}

impl StoreManager {
    /// Create a new StoreManager bound to a workspace root.
    pub fn new(workspace_root: impl Into<PathBuf>) -> Self {
        Self {
            stores: RwLock::new(HashMap::new()),
            aliases: RwLock::new(HashMap::new()),
            handles: RwLock::new(HashMap::new()),
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

    /// Open a selector and register an opaque handle for later `runtime.db.query/exec/close`.
    pub async fn open_handle(
        &self,
        selector: &StoreSelector,
        path_scope: StorePathScope,
        max_open_handles: usize,
        idle_close_secs: u64,
    ) -> Result<StoreHandleInfo> {
        self.trim_cache(max_open_handles, idle_close_secs).await;

        let (target_path, alias) = self
            .resolve_selector_with_alias(selector, path_scope)
            .await?;

        {
            let handles = self.handles.read().await;
            if let Some((handle, entry)) = handles.iter().find(|(_, e)| e.path == target_path) {
                return Ok(StoreHandleInfo {
                    handle: handle.clone(),
                    path: entry.path.clone(),
                    alias: entry.alias.clone(),
                    open_count: entry.open_count,
                    idle_ms: entry.last_access.elapsed().as_millis() as u64,
                });
            }
            if handles.len() >= max_open_handles {
                anyhow::bail!(
                    "open handle limit reached ({}). Close handles or raise db.max_open_handles",
                    max_open_handles
                );
            }
        }

        let _ = self.open_path_cached(&target_path).await?;

        let handle_id = uuid::Uuid::now_v7().simple().to_string();
        let now = Instant::now();
        let mut handles = self.handles.write().await;
        let entry = HandleEntry {
            path: target_path.clone(),
            alias,
            open_count: 1,
            last_access: now,
        };
        handles.insert(handle_id.clone(), entry);

        Ok(StoreHandleInfo {
            handle: handle_id,
            path: target_path,
            alias: None,
            open_count: 1,
            idle_ms: 0,
        })
    }

    /// Close a runtime handle.
    pub async fn close_handle(&self, handle: &str) -> Result<bool> {
        let removed = self.handles.write().await.remove(handle).is_some();
        Ok(removed)
    }

    /// List active runtime handles.
    pub async fn list_handles(&self) -> Vec<StoreHandleInfo> {
        let handles = self.handles.read().await;
        handles
            .iter()
            .map(|(handle, entry)| StoreHandleInfo {
                handle: handle.clone(),
                path: entry.path.clone(),
                alias: entry.alias.clone(),
                open_count: entry.open_count,
                idle_ms: entry.last_access.elapsed().as_millis() as u64,
            })
            .collect()
    }

    /// List registered aliases.
    pub async fn list_aliases(&self) -> Vec<String> {
        let aliases = self.aliases.read().await;
        let mut names = aliases.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    }

    /// Trim idle/unreferenced cache entries. Returns number evicted.
    pub async fn trim_cache(&self, max_entries: usize, idle_close_secs: u64) -> usize {
        let idle_cutoff = std::time::Duration::from_secs(idle_close_secs);
        let protected_paths = {
            let handles = self.handles.read().await;
            handles.values().map(|h| h.path.clone()).collect::<Vec<_>>()
        };

        let mut stores = self.stores.write().await;
        let mut evicted = 0usize;

        // First pass: evict idle entries not referenced by active handles.
        let idle_candidates = stores
            .iter()
            .filter_map(|(path, entry)| {
                if protected_paths.iter().any(|p| p == path) {
                    return None;
                }
                if entry.last_access.elapsed() >= idle_cutoff
                    && Arc::strong_count(&entry.store) == 1
                {
                    Some(path.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        for path in idle_candidates {
            if stores.remove(&path).is_some() {
                evicted += 1;
            }
        }

        if stores.len() <= max_entries {
            return evicted;
        }

        let mut lru = stores
            .iter()
            .filter_map(|(path, entry)| {
                if protected_paths.iter().any(|p| p == path) {
                    return None;
                }
                if Arc::strong_count(&entry.store) > 1 {
                    return None;
                }
                Some((path.clone(), entry.last_access))
            })
            .collect::<Vec<_>>();
        lru.sort_by_key(|(_, last_access)| *last_access);

        for (path, _) in lru {
            if stores.len() <= max_entries {
                break;
            }
            if stores.remove(&path).is_some() {
                evicted += 1;
            }
        }

        evicted
    }

    /// Convenience wrapper to retrieve a store specifically tracked by the "state" alias.
    pub async fn get_default(&self) -> Result<Arc<StateStore>> {
        self.open(&StoreSelector::Alias("state".to_string())).await
    }

    fn default_alias_path(&self, alias: &str) -> PathBuf {
        let file_name = format!("{}.db", sanitize_alias(alias));
        self.workspace_root
            .join(".turin")
            .join("stores")
            .join(file_name)
    }

    async fn resolve_selector_with_alias(
        &self,
        selector: &StoreSelector,
        path_scope: StorePathScope,
    ) -> Result<(PathBuf, Option<String>)> {
        match selector {
            StoreSelector::Alias(alias) => {
                let path = self.resolve_alias_path(alias).await?;
                Ok((path, Some(alias.clone())))
            }
            StoreSelector::Path(p) => {
                let path = self.resolve_path_scoped(Path::new(p), path_scope)?;
                Ok((path, None))
            }
            StoreSelector::Handle(handle) => {
                let mut handles = self.handles.write().await;
                let entry = handles
                    .get_mut(handle)
                    .ok_or_else(|| anyhow::anyhow!("Unknown db handle '{}'", handle))?;
                entry.open_count = entry.open_count.saturating_add(1);
                entry.last_access = Instant::now();
                Ok((entry.path.clone(), entry.alias.clone()))
            }
        }
    }

    async fn resolve_selector_path(
        &self,
        selector: &StoreSelector,
        path_scope: StorePathScope,
    ) -> Result<PathBuf> {
        let (path, _) = self
            .resolve_selector_with_alias(selector, path_scope)
            .await?;
        Ok(path)
    }

    async fn resolve_alias_path(&self, alias: &str) -> Result<PathBuf> {
        if let Some(path) = self.aliases.read().await.get(alias).cloned() {
            return Ok(path);
        }

        let path = self.default_alias_path(alias);
        let path = self.resolve_path_unchecked(&path)?;
        let mut aliases = self.aliases.write().await;
        Ok(aliases
            .entry(alias.to_string())
            .or_insert_with(|| path.clone())
            .clone())
    }

    fn resolve_path_unchecked(&self, requested: &Path) -> Result<PathBuf> {
        if requested
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            anyhow::bail!("Path traversal (..) not allowed: {}", requested.display());
        }
        let resolved = if requested.is_absolute() {
            requested.to_path_buf()
        } else {
            self.workspace_root.join(requested)
        };
        Ok(resolved)
    }

    fn resolve_path_scoped(&self, requested: &Path, path_scope: StorePathScope) -> Result<PathBuf> {
        match path_scope {
            StorePathScope::AllowAny => self.resolve_path_unchecked(requested),
            StorePathScope::WorkspaceOnly => {
                crate::tools::is_safe_path(&self.workspace_root, requested)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))
            }
        }
    }

    async fn open_path_cached(&self, target_path: &PathBuf) -> Result<Arc<StateStore>> {
        {
            let mut stores = self.stores.write().await;
            if let Some(entry) = stores.get_mut(target_path) {
                entry.last_access = Instant::now();
                return Ok(Arc::clone(&entry.store));
            }
        }

        let db_path_str = target_path
            .to_str()
            .context("Database path contains invalid UTF-8")?;

        info!("Opening dynamic database store at: {}", db_path_str);

        let store = match StateStore::open(db_path_str).await {
            Ok(s) => Arc::new(s),
            Err(e) => anyhow::bail!("Failed to open store at {}: {}", db_path_str, e),
        };

        let mut stores = self.stores.write().await;
        if let Some(entry) = stores.get_mut(target_path) {
            entry.last_access = Instant::now();
            Ok(Arc::clone(&entry.store))
        } else {
            stores.insert(
                target_path.clone(),
                StoreCacheEntry {
                    store: Arc::clone(&store),
                    last_access: Instant::now(),
                },
            );
            Ok(store)
        }
    }
}

fn sanitize_alias(alias: &str) -> String {
    let mut out = String::with_capacity(alias.len());
    for ch in alias.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "store".to_string()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn store_manager_workspace_scope_rejects_parent_traversal() {
        let mgr = StoreManager::new(std::env::current_dir().unwrap());
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
