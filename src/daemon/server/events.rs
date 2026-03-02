use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use serde_json::json;
use tokio::io::AsyncWriteExt;
use tokio::net::unix::OwnedWriteHalf;
use tokio::sync::{Mutex, broadcast, watch};

use crate::daemon::protocol::{EventEnvelope, RequestEnvelope, ResponseEnvelope};
use crate::daemon::state::{DaemonState, DaemonStatus};

use super::dispatch::{classify_registry_issue, emit_event};

pub(super) async fn stream_events(
    request: RequestEnvelope,
    state: Arc<Mutex<DaemonState>>,
    mut event_rx: broadcast::Receiver<EventEnvelope>,
    mut shutdown_rx: watch::Receiver<bool>,
    writer: &mut OwnedWriteHalf,
) -> Result<()> {
    let ack = ResponseEnvelope::ok(request.id, json!({ "subscribed": true }));
    writer
        .write_all(serde_json::to_string(&ack)?.as_bytes())
        .await?;
    writer.write_all(b"\n").await?;

    let snapshot = {
        let guard = state.lock().await;
        serde_json::to_value(guard.status().await)?
    };
    let snapshot_event = EventEnvelope::new("runtime.snapshot", snapshot);
    writer
        .write_all(serde_json::to_string(&snapshot_event)?.as_bytes())
        .await?;
    writer.write_all(b"\n").await?;

    let status: DaemonStatus = {
        let guard = state.lock().await;
        guard.status().await
    };
    for issue in &status.registry.issues {
        if let Some((event_name, data)) = classify_registry_issue(&status, issue) {
            let event = EventEnvelope::new(event_name, data);
            writer
                .write_all(serde_json::to_string(&event)?.as_bytes())
                .await?;
            writer.write_all(b"\n").await?;
        }
    }

    loop {
        tokio::select! {
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    break;
                }
            }
            event = event_rx.recv() => {
                match event {
                    Ok(event) => {
                        writer
                            .write_all(serde_json::to_string(&event)?.as_bytes())
                            .await?;
                        writer.write_all(b"\n").await?;
                    }
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        let lagged = EventEnvelope::new("runtime.events_lagged", json!({ "skipped": skipped }));
                        writer
                            .write_all(serde_json::to_string(&lagged)?.as_bytes())
                            .await?;
                        writer.write_all(b"\n").await?;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        }
    }

    Ok(())
}

pub(super) fn start_task_event_poller(
    state: Arc<Mutex<DaemonState>>,
    event_tx: broadcast::Sender<EventEnvelope>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    tokio::spawn(async move {
        let mut seen: HashMap<String, serde_json::Value> = HashMap::new();

        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        break;
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(250)) => {
                    let tasks = {
                        let guard = state.lock().await;
                        guard.list_tasks().await
                    };

                    for task in tasks {
                        let Ok(value) = serde_json::to_value(&task) else {
                            continue;
                        };
                        let changed = seen
                            .get(&task.request_id)
                            .map(|previous| previous != &value)
                            .unwrap_or(true);
                        if changed {
                            emit_event(&event_tx, "task.updated", value.clone());
                            seen.insert(task.request_id.clone(), value);
                        }
                    }
                }
            }
        }
    });
}
