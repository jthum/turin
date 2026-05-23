use super::common::{print_indented, print_table, yes_no};
use super::types::{ChannelDetailView, ChannelRuntimeView, DaemonStatusView};

pub(in crate::commands::daemon) fn print_channel_list(status: DaemonStatusView) {
    let mut rows = Vec::new();
    rows.push(vec![
        "CHANNEL".to_string(),
        "ENABLED".to_string(),
        "KIND".to_string(),
        "AGENT".to_string(),
    ]);

    for channel in status.registry.channels {
        rows.push(vec![
            channel.id,
            yes_no(channel.enabled),
            channel.kind,
            channel.agent_id,
        ]);
    }

    print_table(&rows);
}

pub(in crate::commands::daemon) fn print_channel_detail(channel: ChannelDetailView) {
    println!("Channel");
    println!("  id:            {}", channel.id);
    println!("  kind:          {}", channel.kind);
    println!("  agent:         {}", channel.agent_id);
    println!("  enabled:       {}", yes_no(channel.enabled));
    println!("  directory:     {}", channel.directory);
    if let Some(idle_timeout_seconds) = channel.idle_timeout_seconds {
        println!("  idle_timeout_seconds: {}", idle_timeout_seconds);
    }
    if let Some(adapter) = &channel.adapter {
        println!("  adapter:");
        println!("    kind: {}", adapter.kind);
        println!("    display_name: {}", adapter.display_name_or_kind());
        println!("    protocol_version: {}", adapter.protocol_version);
        if !adapter.runtime.session_scopes.is_empty() {
            println!(
                "    session_scopes: {}",
                adapter.runtime.session_scopes.join(", ")
            );
        }
        if !adapter.runtime.enum_settings.is_empty() {
            println!("    enum_settings:");
            for setting in &adapter.runtime.enum_settings {
                println!(
                    "      {}: {}",
                    setting.key,
                    if setting.options.is_empty() {
                        "<none>".to_string()
                    } else {
                        setting.options.join(", ")
                    }
                );
            }
        }
        let capabilities = &adapter.runtime.capabilities;
        if capabilities.dm
            || capabilities.groups
            || capabilities.threads
            || capabilities.attachments
            || capabilities.streaming
        {
            println!(
                "    capabilities: dm={}, groups={}, threads={}, attachments={}, streaming={}",
                yes_no(capabilities.dm),
                yes_no(capabilities.groups),
                yes_no(capabilities.threads),
                yes_no(capabilities.attachments),
                yes_no(capabilities.streaming),
            );
        }
        if let Some(setup) = &adapter.setup {
            println!(
                "    setup: secrets={}, fields={}, validations={}",
                setup.required_secrets.len(),
                setup.config_fields.len(),
                setup.validation_checks.len()
            );
        }
        if let Some(install) = &adapter.install
            && let Some(binary_name) = &install.binary_name
        {
            println!("    binary: {}", binary_name);
        }
    }
    if channel.settings.is_object()
        && !channel
            .settings
            .as_object()
            .is_some_and(|map| map.is_empty())
    {
        println!("  settings:");
        print_indented(&serde_json::to_string_pretty(&channel.settings).unwrap_or_default());
    }
}

pub(in crate::commands::daemon) fn print_channel_runtime(channel: ChannelRuntimeView) {
    println!("Channel Runtime:");
    println!("  id:            {}", channel.id);
    println!("  kind:          {}", channel.kind);
    println!("  agent_id:      {}", channel.agent_id);
    println!("  directory:     {}", channel.directory);
    println!("  state:         {}", channel.state);
    println!("  start_count:   {}", channel.start_count);
    println!("  restart_count: {}", channel.restart_count);
    println!("  failure_count: {}", channel.failure_count);
    println!("  transitioned:  {}", channel.last_transition_unix_ms);
    if let Some(last_started) = channel.last_started_unix_ms {
        println!("  last_started:  {}", last_started);
    }
    if let Some(last_stopped) = channel.last_stopped_unix_ms {
        println!("  last_stopped:  {}", last_stopped);
    }
    if let Some(handshake) = channel.handshake {
        println!("  runner:");
        println!("    display_name: {}", handshake.display_name);
        println!("    protocol_version: {}", handshake.protocol_version);
        if let Some(binary) = handshake.runner_binary {
            println!("    binary: {}", binary);
        }
        if let Some(version) = handshake.runner_version {
            println!("    version: {}", version);
        }
        if let Some(pid) = handshake.pid {
            println!("    pid: {}", pid);
        }
        println!("    last_hello: {}", handshake.last_handshake_unix_ms);
    }
    if let Some(code) = channel.last_error_code {
        println!("  error_code:    {}", code);
    }
    if let Some(error) = channel.last_error {
        println!("  last_error:    {}", error);
    }
}
