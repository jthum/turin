use anyhow::{Context, Result};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

use turin::daemon::protocol::{ErrorEnvelope, RequestEnvelope, ResponseEnvelope};
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

pub async fn run_stop(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "daemon.stop", json!({})).await?;
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
