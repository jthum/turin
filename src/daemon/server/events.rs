mod filter;
mod scope;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use serde_json::json;
use tokio::io::AsyncWriteExt;
use tokio::sync::{RwLock, broadcast, watch};
use turin_local_ipc::LocalIpcWriteHalf;

use crate::daemon::channels::ChannelRuntimeManager;
use crate::daemon::protocol::{EventEnvelope, RequestEnvelope, ResponseEnvelope};
use crate::daemon::state::{DaemonState, DaemonStatus};
use crate::kernel::agent_manager::SessionEventReceiver;
use crate::kernel::event::KernelEvent;

use super::dispatch::{build_runtime_snapshot, classify_registry_issue, emit_event};
use filter::EventFilter;
use scope::{scope_runtime_snapshot, scoped_snapshot_is_empty};

type ScopedSessionEventStream = (String, String, String, SessionEventReceiver);

pub(super) async fn stream_events(
    request: RequestEnvelope,
    state: Arc<RwLock<DaemonState>>,
    channel_runtimes: Arc<ChannelRuntimeManager>,
    mut event_rx: broadcast::Receiver<EventEnvelope>,
    mut shutdown_rx: watch::Receiver<bool>,
    writer: &mut LocalIpcWriteHalf,
) -> Result<()> {
    let filter = EventFilter::from_request(&request);
    let mut session_event_rx = if let Some(session_id) = filter.session_id.as_deref() {
        let guard = state.read().await;
        guard
            .subscribe_live_session_events(session_id, filter.slot_id.as_deref())
            .await
            .map(|(agent_id, slot_id, receiver)| {
                (agent_id, session_id.to_string(), slot_id, receiver)
            })
    } else {
        None
    };
    let ack = ResponseEnvelope::ok(request.id, json!({ "subscribed": true }));
    writer
        .write_all(serde_json::to_string(&ack)?.as_bytes())
        .await?;
    writer.write_all(b"\n").await?;

    write_runtime_snapshot_event(
        "runtime.snapshot",
        &state,
        &channel_runtimes,
        &filter,
        false,
        writer,
    )
    .await?;

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
                        if event.event == "runtime.rescanned" {
                            write_runtime_snapshot_event(
                                "runtime.rescanned",
                                &state,
                                &channel_runtimes,
                                &filter,
                                true,
                                writer,
                            )
                            .await?;
                        } else if filter.matches(&event) {
                            write_event(writer, &event).await?;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        let lagged = EventEnvelope::new("runtime.events_lagged", json!({ "skipped": skipped }));
                        write_event(writer, &lagged).await?;
                        write_runtime_snapshot_event(
                            "runtime.snapshot",
                            &state,
                            &channel_runtimes,
                            &filter,
                            false,
                            writer,
                        )
                        .await?;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            session_event = next_session_kernel_event(&mut session_event_rx), if session_event_rx.is_some() => {
                match session_event {
                    Some(Ok(event)) => {
                        if filter.matches(&event) {
                            write_event(writer, &event).await?;
                        }
                    }
                    Some(Err(broadcast::error::RecvError::Lagged(skipped))) => {
                        let lagged = EventEnvelope::new("session.events_lagged", json!({ "skipped": skipped }));
                        write_event(writer, &lagged).await?;
                    }
                    Some(Err(broadcast::error::RecvError::Closed)) | None => {
                        session_event_rx = None;
                    }
                }
            }
        }
    }

    Ok(())
}

async fn write_runtime_snapshot_event(
    event_name: &str,
    state: &Arc<RwLock<DaemonState>>,
    channel_runtimes: &Arc<ChannelRuntimeManager>,
    filter: &EventFilter,
    skip_empty_scoped: bool,
    writer: &mut LocalIpcWriteHalf,
) -> Result<()> {
    let snapshot = build_runtime_snapshot(state, channel_runtimes).await;
    let scoped = scope_runtime_snapshot(snapshot, filter);
    if skip_empty_scoped && filter.has_scope() && scoped_snapshot_is_empty(&scoped) {
        return Ok(());
    }
    let event = EventEnvelope::new(event_name, serde_json::to_value(scoped)?);
    write_event(writer, &event).await
}

async fn write_event(writer: &mut LocalIpcWriteHalf, event: &EventEnvelope) -> Result<()> {
    writer
        .write_all(serde_json::to_string(event)?.as_bytes())
        .await?;
    writer.write_all(b"\n").await?;
    Ok(())
}

async fn next_session_kernel_event(
    session_event_rx: &mut Option<ScopedSessionEventStream>,
) -> Option<std::result::Result<EventEnvelope, broadcast::error::RecvError>> {
    let (agent_id, session_id, slot_id, rx) = session_event_rx.as_mut()?;
    Some(
        rx.recv()
            .await
            .map(|(_, event)| kernel_event_envelope(agent_id, session_id, slot_id, &event)),
    )
}

fn kernel_event_envelope(
    agent_id: &str,
    session_id: &str,
    slot_id: &str,
    event: &KernelEvent,
) -> EventEnvelope {
    let mut data = serde_json::to_value(event).unwrap_or_else(|_| json!({}));
    if let serde_json::Value::Object(ref mut map) = data {
        map.insert("agent_id".to_string(), json!(agent_id));
        map.insert("session_id".to_string(), json!(session_id));
        map.insert("slot_id".to_string(), json!(slot_id));
    }
    EventEnvelope::new(event.event_type(), data)
}

pub(super) fn start_task_event_poller(
    state: Arc<RwLock<DaemonState>>,
    event_tx: broadcast::Sender<EventEnvelope>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    tokio::spawn(async move {
        let mut seen = HashMap::new();

        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        break;
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(250)) => {
                    let changed_tasks = {
                        let guard = state.read().await;
                        let fingerprints = guard.list_task_fingerprints().await;
                        let changed_fingerprints: Vec<_> = fingerprints
                            .into_iter()
                            .filter_map(|fingerprint| {
                                let changed = seen
                                    .get(&fingerprint.request_id)
                                    .map(|previous| previous != &fingerprint)
                                    .unwrap_or(true);
                                changed.then(|| {
                                    let request_id = fingerprint.request_id.clone();
                                    (request_id, fingerprint)
                                })
                            })
                            .collect();

                        let mut tasks = Vec::with_capacity(changed_fingerprints.len());
                        for (request_id, fingerprint) in changed_fingerprints {
                            if let Some(task) = guard.get_task(&request_id).await {
                                tasks.push((fingerprint, task));
                            }
                        }
                        tasks
                    };

                    for (fingerprint, task) in changed_tasks {
                        let Ok(value) = serde_json::to_value(&task) else {
                            continue;
                        };
                        emit_event(&event_tx, "task.updated", value);
                        seen.insert(fingerprint.request_id.clone(), fingerprint);
                    }
                }
            }
        }
    });
}
