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

    let agents_dir = snapshot.registry.agents_dir.clone();
    let harnesses_dir = snapshot.registry.harnesses_dir.clone();
    snapshot.registry.issues.retain(|issue| {
        issue_matches_scope(
            issue,
            &agents_dir,
            &harnesses_dir,
            &visible_agents,
            &visible_shared_harness_ids,
        )
    });

    snapshot
}

pub(super) fn scoped_snapshot_is_empty(snapshot: &DaemonRuntimeSnapshot) -> bool {
    snapshot.registry.agents.is_empty()
        && snapshot.registry.shared_harnesses.is_empty()
        && snapshot.registry.issues.is_empty()
        && snapshot.harnesses.is_empty()
        && snapshot.agent_runtimes.is_empty()
        && snapshot.live_sessions.is_empty()
}

fn issue_matches_scope(
    issue: &crate::daemon::registry::RegistryIssue,
    agents_dir: &str,
    harnesses_dir: &str,
    visible_agents: &HashSet<String>,
    visible_harness_ids: &HashSet<String>,
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
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daemon::registry::RegistryIssue;

    #[test]
    fn issue_scope_matches_only_visible_agent_and_shared_harness_paths() {
        let visible_agents = HashSet::from(["writer".to_string()]);
        let visible_harnesses = HashSet::from(["reviewer".to_string()]);

        let writer_issue = RegistryIssue {
            path: "/tmp/work/.turin/runtime/agents/writer/config.toml".into(),
            message: "writer issue".into(),
        };
        let other_issue = RegistryIssue {
            path: "/tmp/work/.turin/runtime/agents/default/config.toml".into(),
            message: "default issue".into(),
        };
        let harness_issue = RegistryIssue {
            path: "/tmp/work/.turin/harnesses/reviewer/main.lua".into(),
            message: "reviewer issue".into(),
        };
        let unrelated_issue = RegistryIssue {
            path: "/tmp/work/.turin/relays/telegram/config.toml".into(),
            message: "relay issue".into(),
        };

        let matches = |issue| {
            issue_matches_scope(
                issue,
                "/tmp/work/.turin/runtime/agents",
                "/tmp/work/.turin/harnesses",
                &visible_agents,
                &visible_harnesses,
            )
        };

        assert!(matches(&writer_issue));
        assert!(!matches(&other_issue));
        assert!(matches(&harness_issue));
        assert!(!matches(&unrelated_issue));
    }
}
