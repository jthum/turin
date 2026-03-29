use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use serde_json::json;
use tokio::io::AsyncWriteExt;
use tokio::sync::{RwLock, broadcast, watch};
use turin_local_ipc::LocalIpcWriteHalf;

use crate::daemon::channels::ChannelRuntimeManager;
use crate::daemon::protocol::{
    DaemonRequest, EventEnvelope, RequestEnvelope, ResponseEnvelope, RuntimeEventsSubscribeParams,
};
use crate::daemon::state::{DaemonRuntimeSnapshot, DaemonState, DaemonStatus};
use crate::kernel::agent_manager::SessionEventReceiver;
use crate::kernel::event::KernelEvent;

use super::dispatch::{build_runtime_snapshot, classify_registry_issue, emit_event};

#[derive(Debug, Clone, Default)]
struct EventFilter {
    agent_id: Option<String>,
    session_id: Option<String>,
}

type ScopedSessionEventStream = (String, String, SessionEventReceiver);

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
            .subscribe_live_session_events(session_id)
            .await
            .map(|(agent_id, receiver)| (agent_id, session_id.to_string(), receiver))
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
        if self.should_always_deliver(&event.event) {
            return true;
        }
        self.matches_agent(event) && self.matches_session(event)
    }

    fn should_always_deliver(&self, event_name: &str) -> bool {
        matches!(event_name, "runtime.rescan_failed" | "runtime.rescanned")
    }

    fn has_scope(&self) -> bool {
        self.agent_id.is_some() || self.session_id.is_some()
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
    let (agent_id, session_id, rx) = session_event_rx.as_mut()?;
    Some(
        rx.recv()
            .await
            .map(|(_, event)| kernel_event_envelope(agent_id, session_id, &event)),
    )
}

fn kernel_event_envelope(agent_id: &str, session_id: &str, event: &KernelEvent) -> EventEnvelope {
    let mut data = serde_json::to_value(event).unwrap_or_else(|_| json!({}));
    if let serde_json::Value::Object(ref mut map) = data {
        map.insert("agent_id".to_string(), json!(agent_id));
        map.insert(
            "session_id".to_string(),
            json!(kernel_event_session_id(event).unwrap_or(session_id)),
        );
    }
    EventEnvelope::new(event.event_type(), data)
}

fn kernel_event_session_id(event: &KernelEvent) -> Option<&str> {
    match event {
        KernelEvent::Lifecycle(lifecycle) => match lifecycle {
            crate::kernel::event::LifecycleEvent::SessionStart { identity }
            | crate::kernel::event::LifecycleEvent::SessionResume { identity }
            | crate::kernel::event::LifecycleEvent::SessionEnd { identity, .. }
            | crate::kernel::event::LifecycleEvent::TaskStart { identity, .. }
            | crate::kernel::event::LifecycleEvent::TaskComplete { identity, .. }
            | crate::kernel::event::LifecycleEvent::PlanComplete { identity, .. }
            | crate::kernel::event::LifecycleEvent::AllTasksComplete { identity }
            | crate::kernel::event::LifecycleEvent::TurnStart { identity, .. }
            | crate::kernel::event::LifecycleEvent::TurnPrepare { identity, .. }
            | crate::kernel::event::LifecycleEvent::TurnEnd { identity, .. } => {
                Some(identity.session_id())
            }
        },
        _ => None,
    }
}

fn scope_runtime_snapshot(
    mut snapshot: DaemonRuntimeSnapshot,
    filter: &EventFilter,
) -> DaemonRuntimeSnapshot {
    if filter.agent_id.is_none() && filter.session_id.is_none() {
        return snapshot;
    }

    let mut visible_agents = filter.agent_id.iter().cloned().collect::<HashSet<_>>();
    if let Some(session_id) = filter.session_id.as_deref() {
        let session_agents = snapshot
            .agent_runtimes
            .iter()
            .filter(|runtime| runtime.current_session_id.as_deref() == Some(session_id))
            .map(|runtime| runtime.agent_id.clone())
            .collect::<HashSet<_>>();
        if visible_agents.is_empty() {
            visible_agents = session_agents;
        } else {
            visible_agents.retain(|agent_id| session_agents.contains(agent_id));
        }
    }

    snapshot
        .registry
        .agents
        .retain(|agent| visible_agents.contains(&agent.id));
    snapshot.agent_runtimes.retain(|runtime| {
        visible_agents.contains(&runtime.agent_id)
            && filter
                .session_id
                .as_deref()
                .is_none_or(|session_id| runtime.current_session_id.as_deref() == Some(session_id))
    });
    snapshot.harnesses.retain(|harness| {
        harness
            .bound_agents
            .iter()
            .any(|agent_id| visible_agents.contains(agent_id))
    });

    let visible_shared_harness_ids = snapshot
        .registry
        .agents
        .iter()
        .filter(|agent| agent.harness_kind == "shared")
        .map(|agent| agent.harness_ref.clone())
        .collect::<HashSet<_>>();
    snapshot
        .registry
        .shared_harnesses
        .retain(|harness| visible_shared_harness_ids.contains(&harness.id));

    snapshot
        .registry
        .channels
        .retain(|channel| visible_agents.contains(&channel.agent_id));
    snapshot
        .channel_runtimes
        .retain(|channel| visible_agents.contains(&channel.agent_id));

    let visible_channel_ids = snapshot
        .registry
        .channels
        .iter()
        .map(|channel| channel.id.clone())
        .collect::<HashSet<_>>();
    let agents_dir = snapshot.registry.agents_dir.clone();
    let harnesses_dir = snapshot.registry.harnesses_dir.clone();
    let channels_dir = snapshot.registry.channels_dir.clone();
    snapshot.registry.issues.retain(|issue| {
        issue_matches_scope(
            issue,
            &agents_dir,
            &harnesses_dir,
            &channels_dir,
            &visible_agents,
            &visible_shared_harness_ids,
            &visible_channel_ids,
        )
    });

    snapshot
}

fn scoped_snapshot_is_empty(snapshot: &DaemonRuntimeSnapshot) -> bool {
    snapshot.registry.agents.is_empty()
        && snapshot.registry.shared_harnesses.is_empty()
        && snapshot.registry.channels.is_empty()
        && snapshot.registry.issues.is_empty()
        && snapshot.harnesses.is_empty()
        && snapshot.agent_runtimes.is_empty()
        && snapshot.channel_runtimes.is_empty()
}

fn issue_matches_scope(
    issue: &crate::daemon::registry::RegistryIssue,
    agents_dir: &str,
    harnesses_dir: &str,
    channels_dir: &str,
    visible_agents: &HashSet<String>,
    visible_harness_ids: &HashSet<String>,
    visible_channel_ids: &HashSet<String>,
) -> bool {
    let issue_path = Path::new(&issue.path);
    if let Ok(relative) = issue_path.strip_prefix(Path::new(agents_dir))
        && let Some(agent_id) = relative.components().next()
    {
        return visible_agents.contains(&agent_id.as_os_str().to_string_lossy().to_string());
    }
    if let Ok(relative) = issue_path.strip_prefix(Path::new(harnesses_dir))
        && let Some(harness_id) = relative.components().next()
    {
        return visible_harness_ids.contains(&harness_id.as_os_str().to_string_lossy().to_string());
    }
    if let Ok(relative) = issue_path.strip_prefix(Path::new(channels_dir))
        && let Some(channel_id) = relative.components().next()
    {
        return visible_channel_ids.contains(&channel_id.as_os_str().to_string_lossy().to_string());
    }
    false
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use crate::daemon::channels::ChannelRuntimeSnapshot;
    use crate::daemon::protocol::RuntimeEventsSubscribeParams;
    use crate::daemon::registry::{
        AgentSummary, ChannelSummary, RegistryIssue, RegistrySnapshot, SharedHarnessSummary,
    };
    use crate::kernel::HarnessRuntimeSnapshot;
    use crate::kernel::agent_manager::AgentStatusSnapshot;

    #[test]
    fn scoped_snapshot_filters_agent_related_state() {
        let snapshot = DaemonRuntimeSnapshot {
            config_path: "turin.toml".into(),
            workspace_root: "/tmp/work".into(),
            endpoint: "/tmp/work/.turin/daemon.sock".into(),
            registry: RegistrySnapshot {
                agents_dir: "/tmp/work/agents".into(),
                harnesses_dir: "/tmp/work/harnesses".into(),
                channels_dir: "/tmp/work/channels".into(),
                agents: vec![
                    AgentSummary {
                        id: "default".into(),
                        directory: "/tmp/work/agents/default".into(),
                        enabled: true,
                        provider: "mock".into(),
                        model: "mock-model".into(),
                        mode: "interactive".into(),
                        harness_kind: "local".into(),
                        harness_ref: "default".into(),
                    },
                    AgentSummary {
                        id: "writer".into(),
                        directory: "/tmp/work/agents/writer".into(),
                        enabled: true,
                        provider: "mock".into(),
                        model: "mock-model".into(),
                        mode: "interactive".into(),
                        harness_kind: "shared".into(),
                        harness_ref: "reviewer".into(),
                    },
                ],
                shared_harnesses: vec![
                    SharedHarnessSummary {
                        id: "reviewer".into(),
                        directory: "/tmp/work/harnesses/reviewer".into(),
                    },
                    SharedHarnessSummary {
                        id: "other".into(),
                        directory: "/tmp/work/harnesses/other".into(),
                    },
                ],
                channels: vec![
                    ChannelSummary {
                        id: "default-fs".into(),
                        directory: "/tmp/work/channels/default-fs".into(),
                        enabled: true,
                        kind: "fs".into(),
                        agent_id: "default".into(),
                        idle_ttl_secs: Some(300),
                    },
                    ChannelSummary {
                        id: "writer-fs".into(),
                        directory: "/tmp/work/channels/writer-fs".into(),
                        enabled: true,
                        kind: "fs".into(),
                        agent_id: "writer".into(),
                        idle_ttl_secs: Some(300),
                    },
                ],
                issues: vec![
                    RegistryIssue {
                        path: "/tmp/work/agents/default/agent.toml".into(),
                        message: "default issue".into(),
                    },
                    RegistryIssue {
                        path: "/tmp/work/harnesses/reviewer/main.lua".into(),
                        message: "reviewer issue".into(),
                    },
                    RegistryIssue {
                        path: "/tmp/work/channels/writer-fs/channel.toml".into(),
                        message: "writer channel issue".into(),
                    },
                ],
            },
            harnesses: vec![
                HarnessRuntimeSnapshot {
                    harness_id: "default".into(),
                    directory: "/tmp/work/agents/default/harness".into(),
                    bound_agents: vec!["default".into()],
                    watched_roots: Vec::new(),
                    loaded_scripts: Vec::new(),
                },
                HarnessRuntimeSnapshot {
                    harness_id: "reviewer".into(),
                    directory: "/tmp/work/harnesses/reviewer".into(),
                    bound_agents: vec!["writer".into()],
                    watched_roots: Vec::new(),
                    loaded_scripts: Vec::new(),
                },
            ],
            agent_runtimes: vec![
                AgentStatusSnapshot {
                    agent_id: "default".into(),
                    running: true,
                    active_tasks: 0,
                    queued_tasks: 0,
                    awaiting_results: 0,
                    current_session_id: Some("sess-default".into()),
                    current_request_id: None,
                },
                AgentStatusSnapshot {
                    agent_id: "writer".into(),
                    running: true,
                    active_tasks: 1,
                    queued_tasks: 0,
                    awaiting_results: 0,
                    current_session_id: Some("sess-writer".into()),
                    current_request_id: Some("req-writer".into()),
                },
            ],
            channel_runtimes: vec![
                ChannelRuntimeSnapshot {
                    id: "default-fs".into(),
                    kind: "fs".into(),
                    agent_id: "default".into(),
                    directory: "/tmp/work/channels/default-fs".into(),
                    state: "running".into(),
                    last_error: None,
                    last_error_code: None,
                    start_count: 1,
                    restart_count: 0,
                    failure_count: 0,
                    last_transition_unix_ms: 1,
                    last_started_unix_ms: Some(1),
                    last_stopped_unix_ms: None,
                    handshake: None,
                },
                ChannelRuntimeSnapshot {
                    id: "writer-fs".into(),
                    kind: "fs".into(),
                    agent_id: "writer".into(),
                    directory: "/tmp/work/channels/writer-fs".into(),
                    state: "running".into(),
                    last_error: None,
                    last_error_code: None,
                    start_count: 1,
                    restart_count: 0,
                    failure_count: 0,
                    last_transition_unix_ms: 1,
                    last_started_unix_ms: Some(1),
                    last_stopped_unix_ms: None,
                    handshake: None,
                },
            ],
        };

        let scoped = scope_runtime_snapshot(
            snapshot,
            &EventFilter::from_request(&RequestEnvelope::new(
                None,
                DaemonRequest::RuntimeEventsSubscribe(RuntimeEventsSubscribeParams {
                    agent_id: Some("writer".into()),
                    session_id: None,
                }),
            )),
        );

        assert_eq!(scoped.registry.agents.len(), 1);
        assert_eq!(scoped.registry.agents[0].id, "writer");
        assert_eq!(scoped.registry.shared_harnesses.len(), 1);
        assert_eq!(scoped.registry.shared_harnesses[0].id, "reviewer");
        assert_eq!(scoped.harnesses.len(), 1);
        assert_eq!(scoped.harnesses[0].harness_id, "reviewer");
        assert_eq!(scoped.registry.channels.len(), 1);
        assert_eq!(scoped.registry.channels[0].id, "writer-fs");
        assert_eq!(scoped.channel_runtimes.len(), 1);
        assert_eq!(scoped.channel_runtimes[0].id, "writer-fs");
        assert_eq!(scoped.agent_runtimes.len(), 1);
        assert_eq!(scoped.agent_runtimes[0].agent_id, "writer");
        assert_eq!(scoped.registry.issues.len(), 2);
        assert!(
            scoped
                .registry
                .issues
                .iter()
                .all(|issue| !issue.message.contains("default"))
        );
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
