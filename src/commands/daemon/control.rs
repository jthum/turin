use super::*;

pub async fn run_start(
    config_path: &Path,
    background: bool,
    wait_timeout_ms: u64,
    json_output: bool,
    log_level: &str,
    log_file_override: Option<&Path>,
) -> Result<()> {
    if !background {
        return crate::composition::serve_daemon(config_path).await;
    }

    let report = ensure_background_daemon(
        config_path,
        Duration::from_millis(wait_timeout_ms),
        Duration::from_millis(100),
        log_level,
        log_file_override,
    )
    .await?;
    print_start_report(report, json_output)
}

pub async fn run_health(config_path: &Path, json_output: bool) -> Result<()> {
    let report = daemon_health_report(config_path).await?;
    print_health_report(&report, json_output)
}

pub async fn run_wait(
    config_path: &Path,
    timeout_ms: u64,
    poll_interval_ms: u64,
    json_output: bool,
) -> Result<()> {
    let client = daemon_client_from_config(config_path)?;
    client
        .wait_until_ready(
            Duration::from_millis(timeout_ms),
            Duration::from_millis(poll_interval_ms),
        )
        .await
        .map_err(|err| wrap_daemon_client_error(config_path, err))?;
    let report = daemon_health_report(config_path).await?;
    print_health_report(&report, json_output)
}

pub async fn run_ensure(
    config_path: &Path,
    timeout_ms: u64,
    poll_interval_ms: u64,
    json_output: bool,
    log_level: &str,
    log_file_override: Option<&Path>,
) -> Result<()> {
    let report = ensure_background_daemon(
        config_path,
        Duration::from_millis(timeout_ms),
        Duration::from_millis(poll_interval_ms),
        log_level,
        log_file_override,
    )
    .await?;
    print_start_report(report, json_output)
}

pub async fn run_logs(
    config_path: &Path,
    lines: usize,
    path_only: bool,
    json_output: bool,
    log_file_override: Option<&Path>,
) -> Result<()> {
    let log_path = resolve_daemon_log_path(config_path, log_file_override)?;
    let exists = log_path.exists();
    let rendered_lines = if path_only || !exists {
        Vec::new()
    } else {
        tail_lines(&log_path, lines)?
    };
    let report = DaemonLogReport {
        path: log_path.display().to_string(),
        exists,
        lines: rendered_lines,
    };

    if json_output {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    if path_only {
        println!("{}", report.path);
        return Ok(());
    }

    println!("Log Path: {}", report.path);
    println!("Exists:   {}", yes_no(report.exists));
    if !report.exists {
        println!("No daemon log file exists yet.");
        return Ok(());
    }
    if report.lines.is_empty() {
        println!("No log lines available.");
        return Ok(());
    }
    println!();
    for line in report.lines {
        println!("{}", line);
    }
    Ok(())
}

pub async fn run_ping(config_path: &Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "daemon.ping", json!({})).await?;
    print_response(response, json_output)
}

pub async fn run_status(config_path: &Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "daemon.status", json!({})).await?;
    if json_output {
        return print_response(response, true);
    }

    let status: DaemonStatusView = decode_result(response)?;
    print_daemon_status(status);
    Ok(())
}

pub async fn run_rescan(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "runtime.rescan", json!({})).await?;
    print_response(response, json_output)
}

pub async fn run_reload(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "runtime.reload", json!({})).await?;
    print_response(response, json_output)
}

pub async fn run_events(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let endpoint = resolve_endpoint_path(config_path)?;
    let stream = connect_local_ipc(&endpoint)
        .await
        .with_context(|| {
            format!(
                "Failed to connect to daemon endpoint '{}'",
                endpoint.display()
            )
        })
        .map_err(|err| wrap_daemon_client_error(config_path, err))?;

    let (reader, mut writer) = split_local_ipc(stream);
    let request = RequestEnvelope::new(
        Some(format!("req-{}", uuid::Uuid::new_v4())),
        DaemonRequest::RuntimeEventsSubscribe(RuntimeEventsSubscribeParams::default()),
    );

    writer
        .write_all(serde_json::to_string(&request)?.as_bytes())
        .await?;
    writer.write_all(b"\n").await?;

    let mut lines = BufReader::new(reader).lines();
    let Some(line) = lines.next_line().await? else {
        anyhow::bail!("Daemon closed event stream before acknowledging subscription");
    };
    let response: ResponseEnvelope =
        serde_json::from_str(&line).with_context(|| "Failed to parse daemon subscription ack")?;
    if !response.ok {
        let error = response.error.unwrap_or(ErrorEnvelope {
            code: ErrorCode::InternalError,
            message: "Unknown daemon error".to_string(),
            details: None,
        });
        anyhow::bail!("{}: {}", error.code, error.message);
    }

    while let Some(line) = lines.next_line().await? {
        let event: EventEnvelope =
            serde_json::from_str(&line).with_context(|| "Failed to parse daemon event")?;
        if json_output {
            println!("{}", serde_json::to_string(&event)?);
        } else if event.data.is_null()
            || event.data == serde_json::Value::Object(Default::default())
        {
            println!("{}", event.event);
        } else {
            println!(
                "{} {}",
                event.event,
                serde_json::to_string_pretty(&event.data)?
            );
        }
    }

    Ok(())
}

pub async fn run_stop(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "daemon.stop", json!({})).await?;
    print_response(response, json_output)
}

pub async fn run_runtime_errors(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "runtime.errors", json!({})).await?;
    if json_output {
        return print_response(response, true);
    }

    let issues: IssueListView = decode_result(response)?;
    print_issue_list("Runtime issues", &issues.issues);
    Ok(())
}
