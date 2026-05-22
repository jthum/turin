use std::collections::HashSet;
use std::path::Path;

use crate::daemon::state::DaemonRuntimeSnapshot;

use super::filter::EventFilter;

pub(super) fn scope_runtime_snapshot(
    mut snapshot: DaemonRuntimeSnapshot,
    filter: &EventFilter,
) -> DaemonRuntimeSnapshot {
    if !filter.has_scope() {
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
    snapshot.live_sessions.retain(|session| {
        visible_agents.contains(&session.agent_id)
            && filter
                .session_id
                .as_deref()
                .is_none_or(|session_id| session.session_id == session_id)
            && filter
                .slot_id
                .as_deref()
                .is_none_or(|slot_id| session.slot_id == slot_id)
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

pub(super) fn scoped_snapshot_is_empty(snapshot: &DaemonRuntimeSnapshot) -> bool {
    snapshot.registry.agents.is_empty()
        && snapshot.registry.shared_harnesses.is_empty()
        && snapshot.registry.channels.is_empty()
        && snapshot.registry.issues.is_empty()
        && snapshot.harnesses.is_empty()
        && snapshot.agent_runtimes.is_empty()
        && snapshot.live_sessions.is_empty()
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
mod tests {
    use super::*;
    use crate::daemon::channels::ChannelRuntimeSnapshot;
    use crate::daemon::protocol::{DaemonRequest, RequestEnvelope, RuntimeEventsSubscribeParams};
    use crate::daemon::registry::{
        AgentSummary, ChannelSummary, RegistryIssue, RegistrySnapshot, SharedHarnessSummary,
    };
    use crate::kernel::HarnessRuntimeSnapshot;
    use crate::kernel::agent_manager::{AgentStatusSnapshot, LiveSessionSnapshot};
    use crate::kernel::session::ExecutionConflictPolicy;

    #[test]
    fn scoped_snapshot_filters_agent_related_state() {
        let snapshot = DaemonRuntimeSnapshot {
            config_path: ".turin/config.toml".into(),
            workspace_root: "/tmp/work".into(),
            endpoint: "/tmp/work/.turin/daemon.sock".into(),
            registry: RegistrySnapshot {
                agents_dir: "/tmp/work/.turin/runtime/agents".into(),
                harnesses_dir: "/tmp/work/.turin/harnesses".into(),
                channels_dir: "/tmp/work/.turin/runtime/channels".into(),
                agents: vec![
                    AgentSummary {
                        id: "default".into(),
                        directory: "/tmp/work/.turin/runtime/agents/default".into(),
                        enabled: true,
                        provider: "mock".into(),
                        model: "mock-model".into(),
                        idle_timeout_seconds: Some(20),
                        harness_kind: "local".into(),
                        harness_ref: "default".into(),
                    },
                    AgentSummary {
                        id: "writer".into(),
                        directory: "/tmp/work/.turin/runtime/agents/writer".into(),
                        enabled: true,
                        provider: "mock".into(),
                        model: "mock-model".into(),
                        idle_timeout_seconds: Some(20),
                        harness_kind: "shared".into(),
                        harness_ref: "reviewer".into(),
                    },
                ],
                shared_harnesses: vec![
                    SharedHarnessSummary {
                        id: "reviewer".into(),
                        directory: "/tmp/work/.turin/harnesses/reviewer".into(),
                    },
                    SharedHarnessSummary {
                        id: "other".into(),
                        directory: "/tmp/work/.turin/harnesses/other".into(),
                    },
                ],
                channels: vec![
                    ChannelSummary {
                        id: "default-fs".into(),
                        directory: "/tmp/work/.turin/runtime/channels/default-fs".into(),
                        enabled: true,
                        kind: "fs".into(),
                        agent_id: "default".into(),
                        idle_timeout_seconds: Some(300),
                    },
                    ChannelSummary {
                        id: "writer-fs".into(),
                        directory: "/tmp/work/.turin/runtime/channels/writer-fs".into(),
                        enabled: true,
                        kind: "fs".into(),
                        agent_id: "writer".into(),
                        idle_timeout_seconds: Some(300),
                    },
                ],
                issues: vec![
                    RegistryIssue {
                        path: "/tmp/work/.turin/runtime/agents/default/config.toml".into(),
                        message: "default issue".into(),
                    },
                    RegistryIssue {
                        path: "/tmp/work/.turin/harnesses/reviewer/main.lua".into(),
                        message: "reviewer issue".into(),
                    },
                    RegistryIssue {
                        path: "/tmp/work/.turin/runtime/channels/writer-fs/config.toml".into(),
                        message: "writer channel issue".into(),
                    },
                ],
            },
            harnesses: vec![
                HarnessRuntimeSnapshot {
                    harness_id: "default".into(),
                    directory: "/tmp/work/.turin/runtime/agents/default/harness".into(),
                    bound_agents: vec!["default".into()],
                    watched_roots: Vec::new(),
                    loaded_scripts: Vec::new(),
                },
                HarnessRuntimeSnapshot {
                    harness_id: "reviewer".into(),
                    directory: "/tmp/work/.turin/harnesses/reviewer".into(),
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
            live_sessions: vec![
                LiveSessionSnapshot {
                    agent_id: "default".into(),
                    slot_id: "default".into(),
                    session_id: "sess-default".into(),
                    running: true,
                    active_tasks: 0,
                    queued_tasks: 0,
                    current_request_id: None,
                    execution: crate::kernel::agent_manager::ExecutionStatusSnapshot {
                        execution_id: "ex-default".into(),
                        context_target:
                            crate::kernel::session::ExecutionContextTarget::BranchHead {
                                branch_head_id: Some(1),
                            },
                        visibility: crate::kernel::session::ExecutionVisibility::Visible,
                        durability: crate::kernel::session::ExecutionDurability::Durable,
                        write_policy:
                            crate::kernel::session::ExecutionWritePolicy::AdvanceBranchHead,
                    },
                    conflict_policy: ExecutionConflictPolicy::Reject,
                },
                LiveSessionSnapshot {
                    agent_id: "writer".into(),
                    slot_id: "writer-slot".into(),
                    session_id: "sess-writer".into(),
                    running: true,
                    active_tasks: 1,
                    queued_tasks: 0,
                    current_request_id: Some("req-writer".into()),
                    execution: crate::kernel::agent_manager::ExecutionStatusSnapshot {
                        execution_id: "ex-writer".into(),
                        context_target: crate::kernel::session::ExecutionContextTarget::TurnId {
                            turn_id: 42,
                        },
                        visibility: crate::kernel::session::ExecutionVisibility::Hidden,
                        durability: crate::kernel::session::ExecutionDurability::Ephemeral,
                        write_policy: crate::kernel::session::ExecutionWritePolicy::Detached,
                    },
                    conflict_policy: ExecutionConflictPolicy::Detached,
                },
            ],
            channel_runtimes: vec![
                ChannelRuntimeSnapshot {
                    id: "default-fs".into(),
                    kind: "fs".into(),
                    agent_id: "default".into(),
                    directory: "/tmp/work/.turin/runtime/channels/default-fs".into(),
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
                    directory: "/tmp/work/.turin/runtime/channels/writer-fs".into(),
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
                    slot_id: None,
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
        assert_eq!(scoped.live_sessions.len(), 1);
        assert_eq!(scoped.live_sessions[0].agent_id, "writer");
        assert_eq!(
            scoped.live_sessions[0].conflict_policy,
            ExecutionConflictPolicy::Detached
        );
        assert_eq!(scoped.live_sessions[0].execution.execution_id, "ex-writer");
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
