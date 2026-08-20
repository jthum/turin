use std::collections::{HashMap, HashSet};

use super::common::{print_indented, print_issue_list, print_table, yes_no};
use super::types::{
    AgentDetailView, AgentRuntimeView, DaemonStatusView, HarnessDetailView, HarnessRuntimeView,
};

pub(in crate::commands::daemon) fn print_daemon_status(status: DaemonStatusView) {
    println!("Config:    {}", status.config_path);
    println!("Workspace: {}", status.workspace_root);
    println!("Endpoint:  {}", status.endpoint);
    println!(
        "Agents:    {} daemon-managed, {} shared harnesses, {} issues",
        status.registry.agents.len(),
        status.registry.shared_harnesses.len(),
        status.registry.issues.len()
    );

    if !status.agent_runtimes.is_empty() {
        println!("\nAgent Runtimes");
        let mut runtime_by_agent: HashMap<_, _> = status
            .agent_runtimes
            .into_iter()
            .map(|runtime| (runtime.agent_id.clone(), runtime))
            .collect();

        let mut rows = Vec::new();
        rows.push(vec![
            "AGENT".to_string(),
            "ENABLED".to_string(),
            "RUNNING".to_string(),
            "ACTIVE".to_string(),
            "QUEUED".to_string(),
            "AWAIT".to_string(),
            "SESSION".to_string(),
            "HARNESS".to_string(),
            "MODEL".to_string(),
        ]);

        for agent in &status.registry.agents {
            let runtime = runtime_by_agent
                .remove(&agent.id)
                .unwrap_or(AgentRuntimeView {
                    agent_id: agent.id.clone(),
                    running: false,
                    active_tasks: 0,
                    queued_tasks: 0,
                    awaiting_results: 0,
                    current_session_id: None,
                    current_request_id: None,
                });
            rows.push(vec![
                agent.id.clone(),
                yes_no(agent.enabled),
                yes_no(runtime.running),
                runtime.active_tasks.to_string(),
                runtime.queued_tasks.to_string(),
                runtime.awaiting_results.to_string(),
                runtime
                    .current_session_id
                    .clone()
                    .unwrap_or_else(|| "-".to_string()),
                agent.harness_ref.clone(),
                agent.model.clone(),
            ]);
        }

        for (agent_id, runtime) in runtime_by_agent {
            rows.push(vec![
                agent_id,
                "bootstrap".to_string(),
                yes_no(runtime.running),
                runtime.active_tasks.to_string(),
                runtime.queued_tasks.to_string(),
                runtime.awaiting_results.to_string(),
                runtime
                    .current_session_id
                    .unwrap_or_else(|| "-".to_string()),
                "default".to_string(),
                "-".to_string(),
            ]);
        }

        print_table(&rows);
    }

    if !status.harnesses.is_empty() {
        println!("\nHarness Runtimes");
        print_harness_runtime_table(status.harnesses);
    }

    if !status.registry.issues.is_empty() {
        println!();
        print_issue_list("Runtime issues", &status.registry.issues);
    }
}

pub(in crate::commands::daemon) fn print_agent_list(status: DaemonStatusView) {
    let runtime_by_agent: HashMap<_, _> = status
        .agent_runtimes
        .into_iter()
        .map(|runtime| (runtime.agent_id.clone(), runtime))
        .collect();

    let mut rows = Vec::new();
    rows.push(vec![
        "AGENT".to_string(),
        "ENABLED".to_string(),
        "RUNNING".to_string(),
        "ACTIVE".to_string(),
        "QUEUED".to_string(),
        "AWAIT".to_string(),
        "SESSION".to_string(),
        "HARNESS".to_string(),
        "PROVIDER".to_string(),
        "MODEL".to_string(),
    ]);

    for agent in status.registry.agents {
        let runtime = runtime_by_agent.get(&agent.id);
        rows.push(vec![
            agent.id,
            yes_no(agent.enabled),
            yes_no(runtime.map(|r| r.running).unwrap_or(false)),
            runtime.map(|r| r.active_tasks).unwrap_or(0).to_string(),
            runtime.map(|r| r.queued_tasks).unwrap_or(0).to_string(),
            runtime.map(|r| r.awaiting_results).unwrap_or(0).to_string(),
            runtime
                .and_then(|r| r.current_session_id.clone())
                .unwrap_or_else(|| "-".to_string()),
            agent.harness_ref,
            agent.provider,
            agent.model,
        ]);
    }

    print_table(&rows);
}

pub(in crate::commands::daemon) fn print_agent_detail(agent: AgentDetailView) {
    println!("Agent");
    println!("  id:                {}", agent.id);
    println!("  enabled:           {}", yes_no(agent.enabled));
    println!("  provider:          {}", agent.provider);
    println!("  model:             {}", agent.model);
    println!(
        "  harness:           {}",
        agent.harness.unwrap_or_else(|| "local".to_string())
    );
    println!("  local_harness:     {}", yes_no(agent.has_local_harness));
    println!("  directory:         {}", agent.directory);
    let runtime_idle = agent
        .idle_timeout_seconds
        .map(|secs| secs.to_string())
        .unwrap_or_else(|| "never".to_string());
    println!("  idle_timeout_seconds: {}", runtime_idle);
    if let Some(system_prompt) = &agent.system_prompt {
        println!("  system_prompt:");
        print_indented(system_prompt);
    }
}

pub(in crate::commands::daemon) fn print_agent_runtime_status(status: AgentRuntimeView) {
    println!("Agent Runtime");
    println!("  agent:           {}", status.agent_id);
    println!("  running:         {}", yes_no(status.running));
    println!("  active_tasks:    {}", status.active_tasks);
    println!("  queued_tasks:    {}", status.queued_tasks);
    println!("  awaiting_results: {}", status.awaiting_results);
    println!(
        "  current_session: {}",
        status.current_session_id.unwrap_or_else(|| "-".to_string())
    );
    println!(
        "  current_request: {}",
        status.current_request_id.unwrap_or_else(|| "-".to_string())
    );
}

pub(in crate::commands::daemon) fn print_harness_list(status: DaemonStatusView) {
    let shared_ids: HashSet<_> = status
        .registry
        .shared_harnesses
        .into_iter()
        .map(|shared| shared.id)
        .collect();

    let mut rows = Vec::new();
    rows.push(vec![
        "HARNESS".to_string(),
        "KIND".to_string(),
        "BOUND".to_string(),
        "SCRIPTS".to_string(),
        "WATCHED".to_string(),
    ]);

    for harness in status.harnesses {
        let kind = if harness.harness_id == "default" {
            "default"
        } else if shared_ids.contains(&harness.harness_id) {
            "shared"
        } else {
            "local"
        };
        rows.push(vec![
            harness.harness_id,
            kind.to_string(),
            harness.bound_agents.len().to_string(),
            harness.loaded_scripts.len().to_string(),
            harness.watched_roots.len().to_string(),
        ]);
    }

    print_table(&rows);
}

pub(in crate::commands::daemon) fn print_harness_detail(harness: HarnessDetailView) {
    println!("Harness");
    println!("  harness_id:   {}", harness.harness_id);
    println!("  directory:    {}", harness.directory);
    println!("  bound_agents: {}", harness.bound_agents.len());
    if !harness.bound_agents.is_empty() {
        println!("    {}", harness.bound_agents.join(", "));
    }
    println!("  watched_roots: {}", harness.watched_roots.len());
    for root in harness.watched_roots {
        println!("    {}", root);
    }
    println!("  loaded_scripts: {}", harness.loaded_scripts.len());
    for script in harness.loaded_scripts {
        println!("    {}", script);
    }
}

fn print_harness_runtime_table(harnesses: Vec<HarnessRuntimeView>) {
    let mut rows = Vec::new();
    rows.push(vec![
        "HARNESS".to_string(),
        "KIND".to_string(),
        "BOUND".to_string(),
        "SCRIPTS".to_string(),
        "WATCHED".to_string(),
    ]);
    for harness in harnesses {
        let kind = if harness.harness_id == "default" {
            "default"
        } else if harness.harness_id.starts_with("agent::") {
            "local"
        } else {
            "shared"
        };
        rows.push(vec![
            harness.harness_id,
            kind.to_string(),
            harness.bound_agents.len().to_string(),
            harness.loaded_scripts.len().to_string(),
            harness.watched_roots.len().to_string(),
        ]);
    }
    print_table(&rows);
}
