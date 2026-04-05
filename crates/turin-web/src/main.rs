use std::io::{Read, Write};

use anyhow::{Context, Result, bail};
use turin_types::web_tools::{WebToolErrorKind, WebToolRequest, WebToolResponse};

#[tokio::main]
async fn main() -> Result<()> {
    match std::env::args().nth(1).as_deref() {
        Some("run-json") => run_json().await,
        Some(other) => bail!("unknown command '{other}'"),
        None => bail!("expected command (supported: run-json)"),
    }
}

async fn run_json() -> Result<()> {
    let mut request_json = Vec::new();
    std::io::stdin()
        .read_to_end(&mut request_json)
        .context("failed to read request JSON from stdin")?;

    let request: WebToolRequest = match serde_json::from_slice(&request_json) {
        Ok(request) => request,
        Err(error) => {
            write_response(&WebToolResponse::Error {
                kind: WebToolErrorKind::InvalidParams,
                message: format!("failed to decode web tool request: {error}"),
            })?;
            return Ok(());
        }
    };

    let response = turin_web::handle_request(request).await;
    write_response(&response)
}

fn write_response(response: &WebToolResponse) -> Result<()> {
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    serde_json::to_writer(&mut lock, response).context("failed to encode response JSON")?;
    lock.write_all(b"\n")
        .context("failed to flush response newline")?;
    Ok(())
}
