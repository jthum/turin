use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use notify::Event;
use tokio::sync::{RwLock, broadcast};
use tracing::{error, info, warn};

use crate::daemon::protocol::EventEnvelope;
use crate::daemon::state::{DaemonState, DaemonStatus, DaemonWatchPaths};

use super::dispatch::{emit_event, emit_registry_issue_events};

pub(super) async fn start_daemon_watcher(
    state: Arc<RwLock<DaemonState>>,
    watcher_slot: Arc<std::sync::Mutex<Option<notify::RecommendedWatcher>>>,
    event_tx: broadcast::Sender<EventEnvelope>,
) -> Result<tokio::sync::mpsc::Sender<Vec<PathBuf>>> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<PathBuf>>(32);
    let watcher_tx = tx.clone();
    let task_watcher_tx = watcher_tx.clone();
    let state_for_task = Arc::clone(&state);
    let watcher_slot_for_task = Arc::clone(&watcher_slot);

    tokio::spawn(async move {
        while let Some(mut changed_paths) = rx.recv().await {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            while let Ok(mut more_paths) = rx.try_recv() {
                changed_paths.append(&mut more_paths);
            }

            let watch_paths = {
                let guard = state_for_task.read().await;
                guard.watch_paths()
            };

            if !should_rescan_daemon(&watch_paths, &changed_paths) {
                continue;
            }

            info!(
                ?changed_paths,
                "Daemon filesystem rescan triggered by file change"
            );

            if let Err(err) = rescan_and_refresh_watcher(
                Arc::clone(&state_for_task),
                Arc::clone(&watcher_slot_for_task),
                task_watcher_tx.clone(),
                event_tx.clone(),
            )
            .await
            {
                error!(error = %err, "Daemon filesystem rescan failed");
            }
        }
    });

    let watch_paths = {
        let guard = state.read().await;
        guard.watch_paths()
    };
    let watcher = build_daemon_watcher(&watch_paths, tx)?;
    let mut slot = watcher_slot
        .lock()
        .expect("daemon watcher mutex poisoned during startup");
    *slot = watcher;

    Ok(watcher_tx)
}

pub(super) async fn rescan_and_refresh_watcher(
    state: Arc<RwLock<DaemonState>>,
    watcher_slot: Arc<std::sync::Mutex<Option<notify::RecommendedWatcher>>>,
    tx: tokio::sync::mpsc::Sender<Vec<PathBuf>>,
    event_tx: broadcast::Sender<EventEnvelope>,
) -> Result<DaemonStatus> {
    let (status, watch_paths) = {
        let mut guard = state.write().await;
        let status = guard.rescan().await?;
        let watch_paths = guard.watch_paths();
        (status, watch_paths)
    };

    let watcher = build_daemon_watcher(&watch_paths, tx)?;
    let mut slot = watcher_slot
        .lock()
        .expect("daemon watcher mutex poisoned during refresh");
    *slot = watcher;

    emit_event(
        &event_tx,
        "runtime.rescanned",
        serde_json::json!(status.clone()),
    );
    emit_registry_issue_events(&event_tx, &status);
    Ok(status)
}

fn build_daemon_watcher(
    watch_paths: &DaemonWatchPaths,
    tx: tokio::sync::mpsc::Sender<Vec<PathBuf>>,
) -> Result<Option<notify::RecommendedWatcher>> {
    use notify::{RecursiveMode, Watcher};

    let roots = collect_daemon_watch_roots(watch_paths);
    if roots.is_empty() {
        warn!("No daemon watch roots available, skipping daemon watcher");
        return Ok(None);
    }

    let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| match res {
        Ok(event) => {
            if event.kind.is_modify() || event.kind.is_create() || event.kind.is_remove() {
                let _ = tx.blocking_send(event.paths.clone());
            }
        }
        Err(err) => error!(error = %err, "Daemon watcher channel error"),
    })?;

    for root in roots {
        if !root.path.exists() && root.recursive {
            continue;
        }

        let mode = if root.recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };
        watcher.watch(&root.path, mode)?;
        info!(
            path = %root.path.display(),
            recursive = matches!(mode, RecursiveMode::Recursive),
            "Watching daemon path"
        );
    }

    Ok(Some(watcher))
}

pub(super) fn should_rescan_daemon(
    watch_paths: &DaemonWatchPaths,
    changed_paths: &[PathBuf],
) -> bool {
    changed_paths.iter().any(|path| {
        path == &watch_paths.config_path
            || is_agent_toml(path, &watch_paths.agents_dir)
            || is_direct_child(path, &watch_paths.agents_dir)
            || is_direct_child(path, &watch_paths.harnesses_dir)
            || is_agent_harness_dir(path, &watch_paths.agents_dir)
            || path == &watch_paths.agents_dir
            || path == &watch_paths.harnesses_dir
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DaemonWatchRoot {
    path: PathBuf,
    recursive: bool,
}

fn collect_daemon_watch_roots(watch_paths: &DaemonWatchPaths) -> Vec<DaemonWatchRoot> {
    let mut roots = Vec::new();
    push_watch_root(
        &mut roots,
        watch_paths
            .config_path
            .parent()
            .unwrap_or_else(|| Path::new(".")),
        false,
    );
    push_watch_root(
        &mut roots,
        watch_paths
            .agents_dir
            .parent()
            .unwrap_or_else(|| Path::new(".")),
        false,
    );
    push_watch_root(
        &mut roots,
        watch_paths
            .harnesses_dir
            .parent()
            .unwrap_or_else(|| Path::new(".")),
        false,
    );
    if watch_paths.agents_dir.exists() {
        push_watch_root(&mut roots, &watch_paths.agents_dir, true);
    }
    if watch_paths.harnesses_dir.exists() {
        push_watch_root(&mut roots, &watch_paths.harnesses_dir, true);
    }
    roots
}

fn push_watch_root(roots: &mut Vec<DaemonWatchRoot>, path: &Path, recursive: bool) {
    let root = DaemonWatchRoot {
        path: path.to_path_buf(),
        recursive,
    };
    if !roots.contains(&root) {
        roots.push(root);
    }
}

fn is_agent_toml(path: &Path, agents_dir: &Path) -> bool {
    path.file_name().and_then(|name| name.to_str()) == Some("agent.toml")
        && path.starts_with(agents_dir)
}

fn is_direct_child(path: &Path, parent: &Path) -> bool {
    path.parent() == Some(parent)
}

fn is_agent_harness_dir(path: &Path, agents_dir: &Path) -> bool {
    path.file_name().and_then(|name| name.to_str()) == Some("harness")
        && path
            .parent()
            .and_then(Path::parent)
            .is_some_and(|grandparent| grandparent == agents_dir)
}
