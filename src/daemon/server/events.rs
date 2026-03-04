use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use serde_json::json;
use tokio::io::AsyncWriteExt;
use tokio::net::unix::OwnedWriteHalf;
use tokio::sync::{RwLock, broadcast, watch};

use crate::daemon::protocol::{
    DaemonRequest, EventEnvelope, RequestEnvelope, ResponseEnvelope, RuntimeEventsSubscribeParams,
};
use crate::daemon::state::{DaemonState, DaemonStatus};

use super::dispatch::{classify_registry_issue, emit_event};

#[derive(Debug, Clone, Default)]
struct EventFilter {
    agent_id: Option<String>,
    session_id: Option<String>,
}

pub(super) async fn stream_events(
    request: RequestEnvelope,
    state: Arc<RwLock<DaemonState>>,
    mut event_rx: broadcast::Receiver<EventEnvelope>,
    mut shutdown_rx: watch::Receiver<bool>,
    writer: &mut OwnedWriteHalf,
) -> Result<()> {
    let filter = EventFilter::from_request(&request);
    let ack = ResponseEnvelope::ok(request.id, json!({ "subscribed": true }));
    writer
        .write_all(serde_json::to_string(&ack)?.as_bytes())
        .await?;
    writer.write_all(b"\n").await?;

    let snapshot = {
        let guard = state.read().await;
        serde_json::to_value(guard.status().await)?
    };
    let snapshot_event = EventEnvelope::new("runtime.snapshot", snapshot);
    writer
        .write_all(serde_json::to_string(&snapshot_event)?.as_bytes())
        .await?;
    writer.write_all(b"\n").await?;

    let status: DaemonStatus = {
        let guard = state.read().await;
        guard.status().await
    };
    for issue in &status.registry.issues {
        if let Some((event_name, data)) = classify_registry_issue(&status, issue) {
            let event = EventEnvelope::new(event_name, data);
            if filter.matches(&event) {
                writer
                    .write_all(serde_json::to_string(&event)?.as_bytes())
                    .await?;
                writer.write_all(b"\n").await?;
            }
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
                        if filter.matches(&event) {
                            writer
                                .write_all(serde_json::to_string(&event)?.as_bytes())
                                .await?;
                            writer.write_all(b"\n").await?;
                        }
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

impl EventFilter {
    fn from_request(request: &RequestEnvelope) -> Self {
        match &request.request {
            DaemonRequest::RuntimeEventsSubscribe(RuntimeEventsSubscribeParams {
                agent_id,
                session_id,
            }) => Self {
                agent_id: agent_id.clone(),
                session_id: session_id.clone(),
            },
            _ => Self::default(),
        }
    }

    fn matches(&self, event: &EventEnvelope) -> bool {
        self.matches_agent(event) && self.matches_session(event)
    }

    fn matches_agent(&self, event: &EventEnvelope) -> bool {
        let Some(expected) = self.agent_id.as_deref() else {
            return true;
        };

        event
            .data
            .get("agent_id")
            .and_then(|value| value.as_str())
            .or_else(|| event.data.get("id").and_then(|value| value.as_str()))
            == Some(expected)
    }

    fn matches_session(&self, event: &EventEnvelope) -> bool {
        let Some(expected) = self.session_id.as_deref() else {
            return true;
        };

        event
            .data
            .get("session_id")
            .and_then(|value| value.as_str())
            == Some(expected)
    }
}

pub(super) fn start_task_event_poller(
    state: Arc<RwLock<DaemonState>>,
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
                        let guard = state.read().await;
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
