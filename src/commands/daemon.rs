use anyhow::{Context, Result};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

use turin::daemon::protocol::{ErrorEnvelope, EventEnvelope, RequestEnvelope, ResponseEnvelope};
use turin::kernel::config::TurinConfig;

pub async fn run_start(config_path: &std::path::Path) -> Result<()> {
    turin::daemon::server::serve(config_path).await
}

pub async fn run_ping(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "daemon.ping", json!({})).await?;
    print_response(response, json_output)
}

pub async fn run_status(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "daemon.status", json!({})).await?;
    print_response(response, json_output)
}

pub async fn run_rescan(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "runtime.rescan", json!({})).await?;
    print_response(response, json_output)
}

pub async fn run_agent_list(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "agent.list", json!({})).await?;
    print_response(response, json_output)
}

pub async fn run_agent_get(
    config_path: &std::path::Path,
    agent_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "agent.get", json!({ "id": agent_id })).await?;
    print_response(response, json_output)
}

pub async fn run_agent_status(
    config_path: &std::path::Path,
    agent_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "agent.status", json!({ "id": agent_id })).await?;
    print_response(response, json_output)
}

pub async fn run_agent_create(
    config_path: &std::path::Path,
    params: Value,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "agent.create", params).await?;
    print_response(response, json_output)
}

pub async fn run_agent_enable(
    config_path: &std::path::Path,
    agent_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "agent.enable", json!({ "id": agent_id })).await?;
    print_response(response, json_output)
}

pub async fn run_agent_disable(
    config_path: &std::path::Path,
    agent_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "agent.disable", json!({ "id": agent_id })).await?;
    print_response(response, json_output)
}

pub async fn run_agent_update(
    config_path: &std::path::Path,
    params: Value,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "agent.update", params).await?;
    print_response(response, json_output)
}

pub async fn run_agent_bind_harness(
    config_path: &std::path::Path,
    agent_id: &str,
    harness_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "agent.bind_harness",
        json!({ "id": agent_id, "harness_id": harness_id }),
    )
    .await?;
    print_response(response, json_output)
}

pub async fn run_agent_use_local_harness(
    config_path: &std::path::Path,
    agent_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "agent.use_local_harness",
        json!({ "id": agent_id }),
    )
    .await?;
    print_response(response, json_output)
}

pub async fn run_agent_delete(
    config_path: &std::path::Path,
    agent_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "agent.delete", json!({ "id": agent_id })).await?;
    print_response(response, json_output)
}

pub async fn run_task_submit(
    config_path: &std::path::Path,
    agent_id: &str,
    prompt: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "task.submit",
        json!({ "agent_id": agent_id, "prompt": prompt }),
    )
    .await?;
    print_response(response, json_output)
}

pub async fn run_task_get(
    config_path: &std::path::Path,
    request_id: &str,
    json_output: bool,
) -> Result<()> {
    let response =
        send_request(config_path, "task.get", json!({ "request_id": request_id })).await?;
    print_response(response, json_output)
}

pub async fn run_task_list(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "task.list", json!({})).await?;
    print_response(response, json_output)
}

pub async fn run_session_list(
    config_path: &std::path::Path,
    limit: usize,
    offset: usize,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "session.list",
        json!({ "limit": limit, "offset": offset }),
    )
    .await?;
    print_response(response, json_output)
}

pub async fn run_session_get(
    config_path: &std::path::Path,
    session_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "session.get",
        json!({ "session_id": session_id }),
    )
    .await?;
    print_response(response, json_output)
}

pub async fn run_harness_list(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "harness.list", json!({})).await?;
    print_response(response, json_output)
}

pub async fn run_harness_create(
    config_path: &std::path::Path,
    harness_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "harness.create", json!({ "id": harness_id })).await?;
    print_response(response, json_output)
}

pub async fn run_harness_get(
    config_path: &std::path::Path,
    harness_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "harness.get", json!({ "id": harness_id })).await?;
    print_response(response, json_output)
}

pub async fn run_harness_reload(
    config_path: &std::path::Path,
    harness_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "harness.reload", json!({ "id": harness_id })).await?;
    print_response(response, json_output)
}

pub async fn run_harness_validate(
    config_path: &std::path::Path,
    harness_id: &str,
    json_output: bool,
) -> Result<()> {
    let response =
        send_request(config_path, "harness.validate", json!({ "id": harness_id })).await?;
    print_response(response, json_output)
}

pub async fn run_harness_delete(
    config_path: &std::path::Path,
    harness_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "harness.delete", json!({ "id": harness_id })).await?;
    print_response(response, json_output)
}

pub async fn run_events(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let socket_path = resolve_socket_path(config_path)?;
    let stream = UnixStream::connect(&socket_path).await.with_context(|| {
        format!(
            "Failed to connect to daemon socket '{}'",
            socket_path.display()
        )
    })?;

    let (reader, mut writer) = stream.into_split();
    let request = RequestEnvelope {
        id: Some(format!("req-{}", uuid::Uuid::new_v4())),
        op: "runtime.events.subscribe".to_string(),
        params: json!({}),
    };

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
            code: "unknown_error".to_string(),
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
    print_response(response, json_output)
}

fn print_response(response: ResponseEnvelope, json_output: bool) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string_pretty(&response)?);
        return Ok(());
    }

    if response.ok {
        if let Some(result) = response.result {
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            println!("ok");
        }
        Ok(())
    } else {
        let error = response.error.unwrap_or(ErrorEnvelope {
            code: "unknown_error".to_string(),
            message: "Unknown daemon error".to_string(),
            details: None,
        });
        anyhow::bail!("{}: {}", error.code, error.message);
    }
}

async fn send_request(
    config_path: &std::path::Path,
    op: &str,
    params: Value,
) -> Result<ResponseEnvelope> {
    let socket_path = resolve_socket_path(config_path)?;
    let stream = UnixStream::connect(&socket_path).await.with_context(|| {
        format!(
            "Failed to connect to daemon socket '{}'",
            socket_path.display()
        )
    })?;

    let (reader, mut writer) = stream.into_split();
    let request = RequestEnvelope {
        id: Some(format!("req-{}", uuid::Uuid::new_v4())),
        op: op.to_string(),
        params,
    };

    writer
        .write_all(serde_json::to_string(&request)?.as_bytes())
        .await?;
    writer.write_all(b"\n").await?;

    let mut lines = BufReader::new(reader).lines();
    let Some(line) = lines.next_line().await? else {
        anyhow::bail!("Daemon closed connection without sending a response");
    };

    let response: ResponseEnvelope =
        serde_json::from_str(&line).with_context(|| "Failed to parse daemon response")?;
    Ok(response)
}

fn resolve_socket_path(config_path: &std::path::Path) -> Result<std::path::PathBuf> {
    let config = TurinConfig::from_file(config_path)?;
    let config_base = config_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    Ok(config.resolve_daemon_socket_path(config_base))
}
