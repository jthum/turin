use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use tracing::info;

use crate::persistence::manager::{
    HandleEntry, RawStoreCacheEntry, SqlStoreKind, StoreCacheEntry, StoreHandleInfo, StoreManager,
    StorePathScope, StoreSelector,
};

use super::super::state::StateStore;

impl StoreManager {
    async fn claim_store_kind(&self, path: &PathBuf, requested: SqlStoreKind) -> Result<()> {
        let mut kinds = self.store_kinds.write().await;
        match kinds.get(path) {
            Some(existing) if *existing != requested => anyhow::bail!(
                "Database '{}' is already owned as a {:?} store",
                path.display(),
                existing
            ),
            Some(_) => Ok(()),
            None => {
                kinds.insert(path.clone(), requested);
                Ok(())
            }
        }
    }

    /// Open a selector and register an opaque handle for later `runtime.db.query/exec/close`.
    pub async fn open_handle(
        &self,
        selector: &StoreSelector,
        path_scope: StorePathScope,
        max_open_handles: usize,
        idle_close_seconds: u64,
    ) -> Result<StoreHandleInfo> {
        self.trim_cache(max_open_handles, idle_close_seconds).await;

        let (target_path, alias, kind) = self
            .resolve_selector_with_alias(selector, path_scope)
            .await?;

        {
            let handles = self.handles.read().await;
            if let Some((handle, entry)) = handles.iter().find(|(_, e)| e.path == target_path) {
                if entry.kind != kind {
                    anyhow::bail!(
                        "Database '{}' is already open as a {:?} store",
                        target_path.display(),
                        entry.kind
                    );
                }
                return Ok(handle_info_from_entry(handle, entry));
            }
            if handles.len() >= max_open_handles {
                anyhow::bail!(
                    "open handle limit reached ({}). Close handles or raise db.max_open_handles",
                    max_open_handles
                );
            }
        }

        match kind {
            SqlStoreKind::State => {
                let _ = self.open_path_cached(&target_path).await?;
            }
            SqlStoreKind::Raw => {
                let _ = self.open_raw_path_cached(&target_path).await?;
            }
        }

        let handle_id = uuid::Uuid::now_v7().simple().to_string();
        let now = Instant::now();
        let mut handles = self.handles.write().await;
        let entry = HandleEntry {
            path: target_path.clone(),
            alias,
            kind,
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
    pub async fn trim_cache(&self, max_entries: usize, idle_close_seconds: u64) -> usize {
        let idle_cutoff = std::time::Duration::from_secs(idle_close_seconds);
        let protected_paths = {
            let handles = self.handles.read().await;
            handles
                .values()
                .map(|h| h.path.clone())
                .collect::<HashSet<_>>()
        };

        let mut evicted = 0usize;
        let mut stores = self.stores.write().await;

        // First pass: evict idle entries not referenced by active handles.
        let idle_candidates = stores
            .iter()
            .filter_map(|(path, entry)| {
                if protected_paths.contains(path) {
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

        let mut raw_stores = self.raw_stores.write().await;
        let raw_idle_candidates = raw_stores
            .iter()
            .filter(|(path, entry)| {
                !protected_paths.contains(*path)
                    && entry.last_access.elapsed() >= idle_cutoff
                    && Arc::strong_count(&entry.database) == 1
            })
            .map(|(path, _)| path.clone())
            .collect::<Vec<_>>();
        for path in raw_idle_candidates {
            if raw_stores.remove(&path).is_some() {
                evicted += 1;
            }
        }

        // State and raw databases share one retained-store budget. Active handles and stores
        // borrowed by callers remain protected even when they temporarily exceed the cap.
        while stores.len() + raw_stores.len() > max_entries {
            let oldest_state = stores
                .iter()
                .filter(|(path, entry)| {
                    !protected_paths.contains(*path) && Arc::strong_count(&entry.store) == 1
                })
                .min_by_key(|(_, entry)| entry.last_access)
                .map(|(path, entry)| (path.clone(), entry.last_access));
            let oldest_raw = raw_stores
                .iter()
                .filter(|(path, entry)| {
                    !protected_paths.contains(*path) && Arc::strong_count(&entry.database) == 1
                })
                .min_by_key(|(_, entry)| entry.last_access)
                .map(|(path, entry)| (path.clone(), entry.last_access));

            let removed = match (oldest_state, oldest_raw) {
                (Some((state_path, state_access)), Some((raw_path, raw_access))) => {
                    if state_access <= raw_access {
                        stores.remove(&state_path).is_some()
                    } else {
                        raw_stores.remove(&raw_path).is_some()
                    }
                }
                (Some((path, _)), None) => stores.remove(&path).is_some(),
                (None, Some((path, _))) => raw_stores.remove(&path).is_some(),
                (None, None) => false,
            };
            if !removed {
                break;
            }
            evicted += 1;
        }

        evicted
    }

    pub async fn open_sql_connection(
        &self,
        selector: &StoreSelector,
        path_scope: StorePathScope,
    ) -> Result<(turso::Connection, SqlStoreKind)> {
        let (target_path, _, kind) = self
            .resolve_selector_with_alias(selector, path_scope)
            .await?;
        let connection = match kind {
            SqlStoreKind::State => {
                self.open_path_cached(&target_path)
                    .await?
                    .get_connection()
                    .await?
            }
            SqlStoreKind::Raw => self.open_raw_path_cached(&target_path).await?.connect()?,
        };
        if kind == SqlStoreKind::Raw {
            connection.execute("PRAGMA foreign_keys = ON;", ()).await?;
            connection
                .execute("PRAGMA busy_timeout = 5000;", ())
                .await
                .ok();
        }
        Ok((connection, kind))
    }

    pub(super) async fn open_path_cached(&self, target_path: &PathBuf) -> Result<Arc<StateStore>> {
        self.claim_store_kind(target_path, SqlStoreKind::State)
            .await?;
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

    async fn open_raw_path_cached(&self, target_path: &PathBuf) -> Result<Arc<turso::Database>> {
        self.claim_store_kind(target_path, SqlStoreKind::Raw)
            .await?;
        {
            let mut stores = self.raw_stores.write().await;
            if let Some(entry) = stores.get_mut(target_path) {
                entry.last_access = Instant::now();
                return Ok(Arc::clone(&entry.database));
            }
        }

        if let Some(parent) = target_path.parent()
            && !parent.exists()
        {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "Failed to create raw database directory: {}",
                    parent.display()
                )
            })?;
        }
        let db_path = target_path
            .to_str()
            .context("Database path contains invalid UTF-8")?;
        info!(path = %target_path.display(), "Opening raw SQL database");
        let database = Arc::new(
            turso::Builder::new_local(db_path)
                .build()
                .await
                .with_context(|| {
                    format!("Failed to open raw database: {}", target_path.display())
                })?,
        );

        let mut stores = self.raw_stores.write().await;
        if let Some(entry) = stores.get_mut(target_path) {
            entry.last_access = Instant::now();
            Ok(Arc::clone(&entry.database))
        } else {
            stores.insert(
                target_path.clone(),
                RawStoreCacheEntry {
                    database: Arc::clone(&database),
                    last_access: Instant::now(),
                },
            );
            Ok(database)
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
