use std::path::PathBuf;

use crate::daemon::registry::{RegistryIssue, RegistrySnapshot};
use crate::daemon::state::{DaemonStatus, DaemonWatchPaths};

use super::dispatch::classify_registry_issue;
use super::watch::should_rescan_daemon;

#[test]
fn rescan_filter_ignores_harness_script_edits_but_tracks_registry_changes() {
    let watch_paths = DaemonWatchPaths {
        config_path: PathBuf::from("/tmp/turin/turin.toml"),
        agents_dir: PathBuf::from("/tmp/turin/agents"),
        harnesses_dir: PathBuf::from("/tmp/turin/harnesses"),
    };

    assert!(should_rescan_daemon(
        &watch_paths,
        &[PathBuf::from("/tmp/turin/turin.toml")]
    ));
    assert!(should_rescan_daemon(
        &watch_paths,
        &[PathBuf::from("/tmp/turin/agents/docs/agent.toml")]
    ));
    assert!(should_rescan_daemon(
        &watch_paths,
        &[PathBuf::from("/tmp/turin/agents/docs")]
    ));
    assert!(should_rescan_daemon(
        &watch_paths,
        &[PathBuf::from("/tmp/turin/agents/docs/harness")]
    ));
    assert!(should_rescan_daemon(
        &watch_paths,
        &[PathBuf::from("/tmp/turin/harnesses/reviewer")]
    ));

    assert!(!should_rescan_daemon(
        &watch_paths,
        &[PathBuf::from("/tmp/turin/agents/docs/harness/main.lua")]
    ));
    assert!(!should_rescan_daemon(
        &watch_paths,
        &[PathBuf::from("/tmp/turin/harnesses/reviewer/main.lua")]
    ));
}

#[test]
fn classify_registry_issue_recognizes_agent_and_harness_paths() {
    let status = DaemonStatus {
        config_path: "turin.toml".to_string(),
        workspace_root: ".".to_string(),
        socket_path: ".turin/daemon.sock".to_string(),
        registry: RegistrySnapshot {
            agents_dir: "/tmp/work/agents".to_string(),
            harnesses_dir: "/tmp/work/harnesses".to_string(),
            agents: Vec::new(),
            shared_harnesses: Vec::new(),
            issues: Vec::new(),
        },
        harnesses: Vec::new(),
        agent_runtimes: Vec::new(),
    };

    let agent_issue = RegistryIssue {
        path: "/tmp/work/agents/docs-reviewer/agent.toml".to_string(),
        message: "bad toml".to_string(),
    };
    let harness_issue = RegistryIssue {
        path: "/tmp/work/harnesses/reviewer/main.lua".to_string(),
        message: "bad lua".to_string(),
    };

    let (agent_event, agent_data) =
        classify_registry_issue(&status, &agent_issue).expect("agent issue classified");
    assert_eq!(agent_event, "agent.load_failed");
    assert_eq!(agent_data["agent_id"], "docs-reviewer");

    let (harness_event, harness_data) =
        classify_registry_issue(&status, &harness_issue).expect("harness issue classified");
    assert_eq!(harness_event, "harness.load_failed");
    assert_eq!(harness_data["harness_id"], "reviewer");
}
