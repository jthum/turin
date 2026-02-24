use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use tracing::info;

use crate::persistence::manager::{
    HandleEntry, StoreCacheEntry, StoreHandleInfo, StoreManager, StorePathScope, StoreSelector,
};

use super::super::state::StateStore;

impl StoreManager {
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
                return Ok(handle_info_from_entry(handle, entry));
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
            .map(|(handle, entry)| handle_info_from_entry(handle, entry))
            .collect()
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

    pub(super) async fn open_path_cached(&self, target_path: &PathBuf) -> Result<Arc<StateStore>> {
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

fn handle_info_from_entry(handle: &str, entry: &HandleEntry) -> StoreHandleInfo {
    StoreHandleInfo {
        handle: handle.to_string(),
        path: entry.path.clone(),
        alias: entry.alias.clone(),
        open_count: entry.open_count,
        idle_ms: entry.last_access.elapsed().as_millis() as u64,
    }
}
