use std::collections::{BTreeMap, BTreeSet};

use crate::files::{ConfiguredChannel, load_configured_channels};
use crate::runner::{describe_external_runner, discover_external_runner_kinds};
use anyhow::Result;

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

    let mut rows = Vec::new();
    rows.push(vec![
        "CHANNEL".to_string(),
        "KIND".to_string(),
        "ENABLED".to_string(),
        "AGENT".to_string(),
        "CONFIG".to_string(),
    ]);

    for channel in configured_channels {
        rows.push(channel_status_row(channel));
    }

    print_table(&rows);
    Ok(())
}

fn channel_status_row(channel: ConfiguredChannel) -> Vec<String> {
    vec![
        channel.id,
        channel.kind,
        yes_no(channel.enabled),
        channel.agent_id.unwrap_or_else(|| "-".to_string()),
        "configured".to_string(),
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
    fn channel_status_row_reports_configuration() {
        let channel = ConfiguredChannel {
            id: "telegram-main".to_string(),
            kind: "telegram".to_string(),
            enabled: true,
            agent_id: Some("default".to_string()),
        };
        let row = channel_status_row(channel);
        assert_eq!(row[4], "configured");
    }
}
