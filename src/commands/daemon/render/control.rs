use anyhow::Result;

use super::common::yes_no;
use super::types::{DaemonHealthReport, DaemonStartReport};

pub(in crate::commands::daemon) fn print_health_report(
    report: &DaemonHealthReport,
    json_output: bool,
) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string_pretty(report)?);
        return Ok(());
    }

    println!("State:     {}", report.state);
    println!("Ready:     {}", yes_no(report.ready));
    println!("Endpoint:  {}", report.endpoint);
    if let Some(error) = &report.error {
        println!("Error:     {}", error);
        return Ok(());
    }
    if let Some(version) = &report.version {
        println!("Version:   {}", version);
    }
    if let Some(protocol_version) = report.protocol_version {
        println!("Protocol:  {}", protocol_version);
    }
    if let Some(transport) = &report.transport {
        println!("Transport: {}", transport);
    }
    println!(
        "Counts:    {} agents, {} shared harnesses, {} issues",
        report.agent_count, report.harness_count, report.issue_count
    );
    println!(
        "Load:      {} running agents, {} active tasks, {} queued tasks",
        report.running_agent_count, report.active_task_count, report.queued_task_count
    );
    Ok(())
}

pub(in crate::commands::daemon) fn print_start_report(
    report: DaemonStartReport,
    json_output: bool,
) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    if report.started {
        println!("Daemon started in the background.");
    } else {
        println!("Daemon already running.");
    }
    println!("Endpoint:  {}", report.endpoint);
    println!("Logs:      {}", report.log_path);
    println!();
    print_health_report(&report.health, false)
}
