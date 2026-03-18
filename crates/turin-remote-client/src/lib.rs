use anyhow::{Context, Result, anyhow};
use bytes::Bytes;
use futures::StreamExt;
use futures::stream::BoxStream;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
use tokio::time::sleep;
use turin_daemon_protocol::{
    DAEMON_PROTOCOL_VERSION, DaemonHandshake, DaemonRequest, EventEnvelope, NoParams,
    RequestEnvelope, ResponseEnvelope, RuntimeEventsSubscribeParams,
};

#[derive(Debug, Clone)]
pub struct RemoteClient {
    base_url: String,
    auth_token: String,
    http: reqwest::Client,
}

#[derive(Debug, Clone, Copy)]
pub struct ManagedSubscribeOptions {
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
}

impl Default for ManagedSubscribeOptions {
    fn default() -> Self {
        Self {
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(1),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct RemoteErrorPayload {
    error: RemoteErrorEnvelope,
}

#[derive(Debug, Deserialize, Serialize)]
struct RemoteErrorEnvelope {
    code: String,
    message: String,
    #[serde(default)]
    details: Option<Value>,
}

impl RemoteClient {
    pub fn new(base_url: impl Into<String>, auth_token: impl Into<String>) -> Self {
        Self {
            base_url: normalize_base_url(base_url.into()),
            auth_token: auth_token.into(),
            http: reqwest::Client::new(),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub async fn send(&self, request: RequestEnvelope) -> Result<ResponseEnvelope> {
        let response = self
            .http
            .post(format!("{}/v1/daemon/request", self.base_url))
            .bearer_auth(&self.auth_token)
            .header(CONTENT_TYPE, "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send remote daemon request")?;
        decode_json_response(response).await
    }

    pub async fn request(
        &self,
        id: Option<String>,
        request: DaemonRequest,
    ) -> Result<ResponseEnvelope> {
        self.send(RequestEnvelope::new(id, request)).await
    }

    pub async fn request_ok<T: DeserializeOwned>(
        &self,
        id: Option<String>,
        request: DaemonRequest,
    ) -> Result<T> {
        let response = self.request(id, request).await?;
        decode_ok(response)
    }

    pub async fn handshake(&self) -> Result<DaemonHandshake> {
        let handshake: DaemonHandshake = self
            .request_ok(None, DaemonRequest::DaemonPing(NoParams::default()))
            .await?;
        ensure_compatible_remote_handshake(&handshake)?;
        Ok(handshake)
    }

    pub async fn subscribe(
        &self,
        filter: RuntimeEventsSubscribeParams,
    ) -> Result<RemoteEventStream> {
        let mut url = reqwest::Url::parse(&format!("{}/v1/events", self.base_url))
            .context("Failed to construct remote SSE URL")?;
        if let Some(agent_id) = filter.agent_id {
            url.query_pairs_mut().append_pair("agent_id", &agent_id);
        }
        if let Some(session_id) = filter.session_id {
            url.query_pairs_mut().append_pair("session_id", &session_id);
        }

        let response = self
            .http
            .get(url)
            .header(AUTHORIZATION, format!("Bearer {}", self.auth_token))
            .send()
            .await
            .context("Failed to subscribe to remote SSE events")?;
        let response = ensure_success(response, "remote SSE subscription failed").await?;
        Ok(RemoteEventStream {
            stream: Box::pin(response.bytes_stream()),
            buffer: String::new(),
        })
    }

    pub async fn subscribe_managed(
        &self,
        filter: RuntimeEventsSubscribeParams,
    ) -> Result<ManagedRemoteEventStream> {
        self.subscribe_managed_with_options(filter, ManagedSubscribeOptions::default())
            .await
    }

    pub async fn subscribe_managed_with_options(
        &self,
        filter: RuntimeEventsSubscribeParams,
        options: ManagedSubscribeOptions,
    ) -> Result<ManagedRemoteEventStream> {
        let stream = self.subscribe(filter.clone()).await?;
        Ok(ManagedRemoteEventStream {
            client: self.clone(),
            filter,
            options,
            stream: Some(stream),
        })
    }
}

pub struct RemoteEventStream {
    stream: BoxStream<'static, std::result::Result<Bytes, reqwest::Error>>,
    buffer: String,
}

impl RemoteEventStream {
    pub async fn next(&mut self) -> Result<Option<EventEnvelope>> {
        loop {
            if let Some(event) = parse_next_sse_event(&mut self.buffer)? {
                return Ok(Some(event));
            }

            let Some(chunk) = self.stream.next().await else {
                return Ok(None);
            };
            let chunk = chunk.context("Failed to read remote SSE stream chunk")?;
            self.buffer.push_str(
                std::str::from_utf8(&chunk).context("Remote SSE payload was not valid UTF-8")?,
            );
        }
    }
}

pub struct ManagedRemoteEventStream {
    client: RemoteClient,
    filter: RuntimeEventsSubscribeParams,
    options: ManagedSubscribeOptions,
    stream: Option<RemoteEventStream>,
}

impl ManagedRemoteEventStream {
    pub async fn next_event(&mut self) -> Result<EventEnvelope> {
        loop {
            if self.stream.is_none() {
                self.stream =
                    Some(reconnect(self.client.clone(), self.filter.clone(), self.options).await?);
            }

            match self
                .stream
                .as_mut()
                .expect("managed remote stream is set before polling")
                .next()
                .await
            {
                Ok(Some(event)) => return Ok(event),
                Ok(None) => {
                    self.stream = None;
                }
                Err(err) if is_recoverable_subscription_error(&err) => {
                    self.stream = None;
                }
                Err(err) => return Err(err),
            }
        }
    }
}

async fn reconnect(
    client: RemoteClient,
    filter: RuntimeEventsSubscribeParams,
    options: ManagedSubscribeOptions,
) -> Result<RemoteEventStream> {
    let mut delay = options.initial_backoff;
    loop {
        match client.handshake().await {
            Ok(_) => {}
            Err(err) if is_recoverable_subscription_error(&err) => {
                sleep(delay).await;
                delay = next_backoff(delay, options.max_backoff);
                continue;
            }
            Err(err) => return Err(err),
        }

        match client.subscribe(filter.clone()).await {
            Ok(stream) => return Ok(stream),
            Err(err) if is_recoverable_subscription_error(&err) => {
                sleep(delay).await;
                delay = next_backoff(delay, options.max_backoff);
            }
            Err(err) => return Err(err),
        }
    }
}

pub fn decode_ok<T: DeserializeOwned>(response: ResponseEnvelope) -> Result<T> {
    if !response.ok {
        return Err(anyhow!(format_error(&response)));
    }
    let result = response
        .result
        .ok_or_else(|| anyhow!("Daemon response missing result payload"))?;
    serde_json::from_value(result).context("Failed to decode daemon result payload")
}

pub fn ensure_compatible_remote_handshake(handshake: &DaemonHandshake) -> Result<()> {
    if handshake.protocol_version != DAEMON_PROTOCOL_VERSION {
        return Err(anyhow!(
            "Unsupported daemon protocol version {} (client expects {})",
            handshake.protocol_version,
            DAEMON_PROTOCOL_VERSION
        ));
    }
    Ok(())
}

fn normalize_base_url(base_url: String) -> String {
    base_url.trim_end_matches('/').to_string()
}

async fn decode_json_response(response: reqwest::Response) -> Result<ResponseEnvelope> {
    let response = ensure_success(response, "remote daemon request failed").await?;
    response
        .json::<ResponseEnvelope>()
        .await
        .context("Failed to decode remote daemon response")
}

async fn ensure_success(response: reqwest::Response, context: &str) -> Result<reqwest::Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    Err(build_remote_error(status, response, context).await)
}

async fn build_remote_error(
    status: reqwest::StatusCode,
    response: reqwest::Response,
    context: &str,
) -> anyhow::Error {
    let body = response.text().await.unwrap_or_default();
    if let Ok(payload) = serde_json::from_str::<RemoteErrorPayload>(&body) {
        return anyhow!(
            "{} ({}): {}: {}",
            context,
            status,
            payload.error.code,
            payload.error.message
        );
    }
    if body.trim().is_empty() {
        anyhow!("{} ({})", context, status)
    } else {
        anyhow!("{} ({}): {}", context, status, body.trim())
    }
}

fn parse_next_sse_event(buffer: &mut String) -> Result<Option<EventEnvelope>> {
    loop {
        let Some(index) = buffer.find("\n\n") else {
            return Ok(None);
        };
        let chunk = buffer[..index].to_string();
        *buffer = buffer[index + 2..].to_string();
        if chunk.trim().is_empty() || chunk.starts_with(':') {
            continue;
        }

        let mut event_name = None;
        let mut data = None;
        for line in chunk.lines() {
            if let Some(value) = line.strip_prefix("event: ") {
                event_name = Some(value.to_string());
            } else if let Some(value) = line.strip_prefix("data: ") {
                data = Some(
                    serde_json::from_str::<Value>(value)
                        .context("Failed to decode remote SSE event JSON")?,
                );
            }
        }

        let event_name = event_name.ok_or_else(|| anyhow!("Remote SSE event missing name"))?;
        let data = data.ok_or_else(|| anyhow!("Remote SSE event missing data"))?;
        if event_name == "remote.error" {
            let message = data
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("remote error");
            return Err(anyhow!(message.to_string()));
        }

        return Ok(Some(EventEnvelope::new(event_name, data)));
    }
}

fn is_recoverable_subscription_error(err: &anyhow::Error) -> bool {
    let message = err.to_string();
    message.contains("Failed to subscribe to remote SSE events")
        || message.contains("remote SSE subscription failed")
        || message.contains("Remote SSE payload")
        || message.contains("remote error")
        || err.chain().any(|cause| cause.is::<reqwest::Error>())
}

fn next_backoff(current: Duration, max: Duration) -> Duration {
    current.checked_mul(2).unwrap_or(max).min(max)
}

fn format_error(response: &ResponseEnvelope) -> String {
    match &response.error {
        Some(err) => match &err.details {
            Some(details) => format!("{}: {} ({})", err.code, err.message, details),
            None => format!("{}: {}", err.code, err.message),
        },
        None => "daemon request failed without error envelope".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turin_daemon_protocol::DaemonCapabilities;

    #[test]
    fn normalize_base_url_trims_trailing_slash() {
        assert_eq!(
            normalize_base_url("http://127.0.0.1:9324/".into()),
            "http://127.0.0.1:9324"
        );
    }

    #[test]
    fn parse_next_sse_event_decodes_event() {
        let mut buffer = "event: runtime.snapshot\ndata: {\"ready\":true}\n\n".to_string();
        let event = parse_next_sse_event(&mut buffer)
            .expect("parse succeeds")
            .expect("event present");
        assert_eq!(event.event, "runtime.snapshot");
        assert_eq!(event.data["ready"], true);
        assert!(buffer.is_empty());
    }

    #[test]
    fn remote_error_events_become_errors() {
        let mut buffer =
            "event: remote.error\ndata: {\"message\":\"daemon closed\"}\n\n".to_string();
        let err = parse_next_sse_event(&mut buffer).expect_err("remote error becomes err");
        assert!(err.to_string().contains("daemon closed"));
    }

    #[test]
    fn remote_handshake_requires_matching_protocol_version() {
        let handshake = DaemonHandshake {
            pong: true,
            version: env!("CARGO_PKG_VERSION").into(),
            protocol_version: DAEMON_PROTOCOL_VERSION + 1,
            transport: "unix".into(),
            wire_format: "ndjson".into(),
            capabilities: DaemonCapabilities {
                runtime_snapshot_v1: true,
                scoped_event_snapshots: true,
                lag_resnapshot: true,
                watcher_rescan_failed_events: true,
                channels: true,
            },
        };
        let err =
            ensure_compatible_remote_handshake(&handshake).expect_err("protocol mismatch rejected");
        assert!(
            err.to_string()
                .contains("Unsupported daemon protocol version")
        );
    }
}
