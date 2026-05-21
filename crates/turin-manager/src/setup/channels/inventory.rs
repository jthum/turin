use std::collections::{BTreeMap, BTreeSet};

use anyhow::Result;
use turin_control_client::{ChannelRuntime, ConnectionSpec, ControlClient};

use crate::files::{ConfiguredChannel, load_configured_channels};
use crate::runner::{describe_external_runner, discover_external_runner_kinds};

use super::super::{ChannelsListArgs, ChannelsStatusArgs};

pub(crate) async fn run_channels_list(args: ChannelsListArgs) -> Result<()> {
    let configured_channels = load_configured_channels(&args.config)?;
    let configured_by_kind = configured_channels_by_kind(&configured_channels);

    let mut discovered_manifests = BTreeMap::new();
    for kind in discover_external_runner_kinds() {
        if let Ok(manifest) = describe_external_runner(&kind) {
            discovered_manifests.insert(kind, manifest);
        }
    }

    let mut all_kinds: BTreeSet<String> = discovered_manifests.keys().cloned().collect();
    all_kinds.extend(configured_by_kind.keys().cloned());

    if all_kinds.is_empty() {
        println!("No channels discovered.");
        println!(
            "Install or place a `turin-channel-<kind>` sidecar where Turin can resolve it, then run `turin-manager channels configure <kind>`."
        );
        return Ok(());
    }

    let mut rows = Vec::new();
    rows.push(vec![
        "KIND".to_string(),
        "NAME".to_string(),
        "INSTALLED".to_string(),
        "CONFIGURED".to_string(),
        "CHANNEL IDS".to_string(),
    ]);

    for kind in all_kinds {
        let configured_ids = configured_by_kind.get(&kind).cloned().unwrap_or_default();
        let manifest = discovered_manifests
            .get(&kind)
            .cloned()
            .or_else(|| describe_external_runner(&kind).ok());
        let display_name = manifest
            .as_ref()
            .map(|manifest| manifest.display_name_or_kind().to_string())
            .unwrap_or_else(|| kind.clone());
        rows.push(vec![
            kind,
            display_name,
            yes_no(manifest.is_some()),
            yes_no(!configured_ids.is_empty()),
            if configured_ids.is_empty() {
                "-".to_string()
            } else {
                configured_ids.join(", ")
            },
        ]);
    }

    print_table(&rows);
    Ok(())
}

pub(crate) async fn run_channels_status(args: ChannelsStatusArgs) -> Result<()> {
    let configured_channels = load_configured_channels(&args.config)?;
    if configured_channels.is_empty() {
        println!("No configured channels found.");
        println!("Use `turin-manager channels configure <kind>` to add one.");
        return Ok(());
    }

    let mut runtimes_by_id = BTreeMap::new();
    let daemon_note = match ControlClient::connect(&ConnectionSpec::LocalConfig {
        config_path: args.config.clone(),
    })
    .await
    {
        Ok(client) => match client.status().await {
            Ok(status) => {
                for runtime in status.channel_runtimes {
                    runtimes_by_id.insert(runtime.id.clone(), runtime);
                }
                None
            }
            Err(err) => Some(format!("Daemon status unavailable: {err}")),
        },
        Err(err) => Some(format!("Daemon not reachable: {err}")),
    };

    if let Some(note) = &daemon_note {
        println!("{note}");
        println!(
            "Showing configured channels only. Start Turin with `turin daemon start --config {}` for runtime state.",
            args.config.display()
        );
        println!();
    }

    let mut rows = Vec::new();
    rows.push(vec![
        "CHANNEL".to_string(),
        "KIND".to_string(),
        "ENABLED".to_string(),
        "AGENT".to_string(),
        "STATE".to_string(),
        "ERROR".to_string(),
    ]);

    for channel in configured_channels {
        let runtime = runtimes_by_id.get(&channel.id);
        rows.push(channel_status_row(channel, runtime));
    }

    print_table(&rows);
    Ok(())
}

fn channel_status_row(channel: ConfiguredChannel, runtime: Option<&ChannelRuntime>) -> Vec<String> {
    vec![
        channel.id,
        channel.kind,
        yes_no(channel.enabled),
        channel.agent_id.unwrap_or_else(|| "-".to_string()),
        runtime
            .map(|runtime| runtime.state.clone())
            .unwrap_or_else(|| "unknown".to_string()),
        runtime
            .and_then(|runtime| {
                runtime
                    .last_error_code
                    .clone()
                    .or_else(|| runtime.last_error.clone())
            })
            .unwrap_or_else(|| "-".to_string()),
    ]
}

fn configured_channels_by_kind(channels: &[ConfiguredChannel]) -> BTreeMap<String, Vec<String>> {
    let mut by_kind = BTreeMap::new();
    for channel in channels {
        by_kind
            .entry(channel.kind.clone())
            .or_insert_with(Vec::new)
            .push(channel.id.clone());
    }
    by_kind
}

fn yes_no(value: bool) -> String {
    if value {
        "yes".to_string()
    } else {
        "no".to_string()
    }
}

fn print_table(rows: &[Vec<String>]) {
    if rows.is_empty() {
        return;
    }

    let cols = rows[0].len();
    let mut widths = vec![0usize; cols];
    for row in rows {
        for (idx, cell) in row.iter().enumerate() {
            widths[idx] = widths[idx].max(cell.len());
        }
    }

    for (row_idx, row) in rows.iter().enumerate() {
        let line = row
            .iter()
            .enumerate()
            .map(|(idx, cell)| format!("{:width$}", cell, width = widths[idx]))
            .collect::<Vec<_>>()
            .join("  ");
        println!("{line}");
        if row_idx == 0 {
            let sep = widths
                .iter()
                .map(|width| "-".repeat(*width))
                .collect::<Vec<_>>()
                .join("  ");
            println!("{sep}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_configured_channels_by_kind() {
        let channels = vec![
            ConfiguredChannel {
                id: "telegram-main".to_string(),
                kind: "telegram".to_string(),
                enabled: true,
                agent_id: Some("default".to_string()),
            },
            ConfiguredChannel {
                id: "telegram-ops".to_string(),
                kind: "telegram".to_string(),
                enabled: true,
                agent_id: Some("default".to_string()),
            },
            ConfiguredChannel {
                id: "discord".to_string(),
                kind: "discord".to_string(),
                enabled: true,
                agent_id: Some("default".to_string()),
            },
        ];

        let grouped = configured_channels_by_kind(&channels);
        assert_eq!(
            grouped.get("telegram"),
            Some(&vec![
                "telegram-main".to_string(),
                "telegram-ops".to_string()
            ])
        );
        assert_eq!(grouped.get("discord"), Some(&vec!["discord".to_string()]));
    }

    #[test]
    fn channel_status_row_prefers_error_code() {
        let channel = ConfiguredChannel {
            id: "telegram-main".to_string(),
            kind: "telegram".to_string(),
            enabled: true,
            agent_id: Some("default".to_string()),
        };
        let runtime = ChannelRuntime {
            id: "telegram-main".to_string(),
            kind: "telegram".to_string(),
            agent_id: "default".to_string(),
            directory: "/tmp/channel".to_string(),
            state: "failed".to_string(),
            last_error: Some("boom".to_string()),
            last_error_code: Some("channel_failed".to_string()),
            start_count: 1,
            restart_count: 0,
            failure_count: 1,
            last_transition_unix_ms: 10,
            last_started_unix_ms: Some(5),
            last_stopped_unix_ms: None,
            handshake: None,
        };

        let row = channel_status_row(channel, Some(&runtime));
        assert_eq!(row[4], "failed");
        assert_eq!(row[5], "channel_failed");
    }
}
