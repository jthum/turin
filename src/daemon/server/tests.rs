use std::path::PathBuf;

use anyhow::anyhow;

use crate::daemon::registry::{RegistryIssue, RegistrySnapshot};
use crate::daemon::state::{DaemonStatus, DaemonWatchPaths};

use super::dispatch::classify_registry_issue;
use super::is_expected_client_disconnect;
use super::watch::should_rescan_daemon;

#[test]
fn rescan_filter_ignores_harness_script_edits_but_tracks_registry_changes() {
    let watch_paths = DaemonWatchPaths {
        config_path: PathBuf::from("/tmp/turin/.turin/config.toml"),
        agents_dir: PathBuf::from("/tmp/turin/.turin/agents"),
        harnesses_dir: PathBuf::from("/tmp/turin/.turin/harnesses"),
        channels_dir: PathBuf::from("/tmp/turin/.turin/channels"),
    };

    assert!(should_rescan_daemon(
        &watch_paths,
        &[PathBuf::from("/tmp/turin/.turin/config.toml")]
    ));
    assert!(should_rescan_daemon(
        &watch_paths,
        &[PathBuf::from("/tmp/turin/.turin/agents/docs/config.toml")]
    ));
    assert!(should_rescan_daemon(
        &watch_paths,
        &[PathBuf::from("/tmp/turin/.turin/agents/docs")]
    ));
    assert!(should_rescan_daemon(
        &watch_paths,
        &[PathBuf::from("/tmp/turin/.turin/agents/docs/harness")]
    ));
    assert!(should_rescan_daemon(
        &watch_paths,
        &[PathBuf::from("/tmp/turin/.turin/harnesses/reviewer")]
    ));
    assert!(should_rescan_daemon(
        &watch_paths,
        &[PathBuf::from("/tmp/turin/.turin/channels/discord")]
    ));
    assert!(should_rescan_daemon(
        &watch_paths,
        &[PathBuf::from(
            "/tmp/turin/.turin/channels/discord/config.toml"
        )]
    ));

    assert!(!should_rescan_daemon(
        &watch_paths,
        &[PathBuf::from(
            "/tmp/turin/.turin/agents/docs/harness/main.lua"
        )]
    ));
    assert!(!should_rescan_daemon(
        &watch_paths,
        &[PathBuf::from(
            "/tmp/turin/.turin/harnesses/reviewer/main.lua"
        )]
    ));
}

#[test]
fn classify_registry_issue_recognizes_agent_and_harness_paths() {
    let status = DaemonStatus {
        config_path: ".turin/config.toml".to_string(),
        workspace_root: ".".to_string(),
        endpoint: ".turin/daemon.sock".to_string(),
        registry: RegistrySnapshot {
            agents_dir: "/tmp/work/.turin/agents".to_string(),
            harnesses_dir: "/tmp/work/.turin/harnesses".to_string(),
            channels_dir: "/tmp/work/.turin/channels".to_string(),
            agents: Vec::new(),
            shared_harnesses: Vec::new(),
            channels: Vec::new(),
            issues: Vec::new(),
        },
        harnesses: Vec::new(),
        agent_runtimes: Vec::new(),
        live_sessions: Vec::new(),
    };

    let agent_issue = RegistryIssue {
        path: "/tmp/work/.turin/agents/docs-reviewer/config.toml".to_string(),
        message: "bad toml".to_string(),
    };
    let harness_issue = RegistryIssue {
        path: "/tmp/work/.turin/harnesses/reviewer/main.lua".to_string(),
        message: "bad lua".to_string(),
    };
    let channel_issue = RegistryIssue {
        path: "/tmp/work/.turin/channels/discord/config.toml".to_string(),
        message: "bad toml".to_string(),
    };

    let (agent_event, agent_data) =
        classify_registry_issue(&status, &agent_issue).expect("agent issue classified");
    assert_eq!(agent_event, "agent.load_failed");
    assert_eq!(agent_data["agent_id"], "docs-reviewer");

    let (harness_event, harness_data) =
        classify_registry_issue(&status, &harness_issue).expect("harness issue classified");
    assert_eq!(harness_event, "harness.load_failed");
    assert_eq!(harness_data["harness_id"], "reviewer");

    let (channel_event, channel_data) =
        classify_registry_issue(&status, &channel_issue).expect("channel issue classified");
    assert_eq!(channel_event, "channel.load_failed");
    assert_eq!(channel_data["channel_id"], "discord");
}

#[test]
fn expected_client_disconnect_recognizes_broken_pipe_and_reset() {
    let broken_pipe = anyhow!(std::io::Error::new(
        std::io::ErrorKind::BrokenPipe,
        "broken pipe"
    ));
    assert!(is_expected_client_disconnect(&broken_pipe));

    let connection_reset = anyhow!(std::io::Error::new(
        std::io::ErrorKind::ConnectionReset,
        "connection reset"
    ));
    assert!(is_expected_client_disconnect(&connection_reset));

    let other = anyhow!(std::io::Error::other("other"));
    assert!(!is_expected_client_disconnect(&other));
}
