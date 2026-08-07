use anyhow::Result;
use base64::Engine;
use bytes::Bytes;
use futures::{SinkExt, StreamExt, stream};
use http::header::{
    AUTHORIZATION, CACHE_CONTROL, CONNECTION, CONTENT_TYPE, SEC_WEBSOCKET_ACCEPT,
    SEC_WEBSOCKET_KEY, SEC_WEBSOCKET_VERSION, UPGRADE, WWW_AUTHENTICATE,
};
use http::{Method, Request, Response, StatusCode};
use http_body_util::{BodyExt, Empty, Full, StreamBody, combinators::UnsyncBoxBody};
use hyper::body::{Frame, Incoming};
use hyper::upgrade::Upgraded;
use hyper_util::rt::TokioIo;
use serde::Serialize;
use serde_json::{Value, json};
use sha1::{Digest, Sha1};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::{MissedTickBehavior, interval};
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::protocol::{Message, Role};
use tracing::warn;
use turin_daemon_client::{DaemonClient, EventStream};
use turin_daemon_protocol::{EventEnvelope, RequestEnvelope, RuntimeEventsSubscribeParams};
use url::form_urlencoded;

pub(crate) type RemoteBody = UnsyncBoxBody<Bytes, Infallible>;

#[derive(Clone)]
pub(crate) struct RemoteState {
    pub(crate) bind: String,
    pub(crate) daemon_endpoint: String,
    pub(crate) auth_token: Arc<str>,
    pub(crate) event_keepalive: Duration,
    pub(crate) client: DaemonClient,
}

#[derive(Debug, Clone)]
struct RemoteError {
    status: StatusCode,
    code: &'static str,
    message: String,
    details: Option<Value>,
    include_auth_challenge: bool,
}

#[derive(Debug, Serialize)]
struct RemoteHealthz {
    ok: bool,
    version: String,
}

#[derive(Debug, Serialize)]
struct RemoteHealthReport {
    remote: RemoteRuntimeReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    daemon: Option<turin_daemon_client::DaemonHealth>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct RemoteRuntimeReport {
    ready: bool,
    version: String,
    bind: String,
    daemon_endpoint: String,
    auth_mode: &'static str,
    event_streams: RemoteEventStreams,
}

#[derive(Debug, Serialize)]
struct RemoteEventStreams {
    sse: bool,
    websocket: bool,
}

#[derive(Debug, Serialize)]
struct RemoteErrorPayload {
    error: RemoteErrorEnvelope,
}

#[derive(Debug, Serialize)]
struct RemoteErrorEnvelope {
    code: &'static str,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<Value>,
}

struct SseState {
    events: EventStream,
    keepalive: tokio::time::Interval,
    closed: bool,
}

impl RemoteError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            details: None,
            include_auth_challenge: false,
        }
    }

    fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            code: "unauthorized",
            message: "Missing or invalid bearer token".to_string(),
            details: None,
            include_auth_challenge: true,
        }
    }

    fn with_details(mut self, details: Value) -> Self {
        self.details = Some(details);
        self
    }

    fn into_response(self) -> Response<RemoteBody> {
        let body = RemoteErrorPayload {
            error: RemoteErrorEnvelope {
                code: self.code,
                message: self.message,
                details: self.details,
            },
        };
        let mut response = json_response(self.status, &body);
        if self.include_auth_challenge {
            response.headers_mut().insert(
                WWW_AUTHENTICATE,
                http::HeaderValue::from_static("Bearer realm=\"turin-remote\""),
            );
        }
        response
    }
}

pub(crate) async fn handle_http(
    req: Request<Incoming>,
    state: Arc<RemoteState>,
) -> std::result::Result<Response<RemoteBody>, Infallible> {
    let response = match route_request(req, state).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    };
    Ok(response)
}

async fn route_request(
    req: Request<Incoming>,
    state: Arc<RemoteState>,
) -> std::result::Result<Response<RemoteBody>, RemoteError> {
    let method = req.method().clone();
    let path = req.uri().path().trim_end_matches('/');
    let path = if path.is_empty() { "/" } else { path };

    match (method, path) {
        (Method::GET, "/healthz") => Ok(json_response(
            StatusCode::OK,
            &RemoteHealthz {
                ok: true,
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
        )),
        (Method::GET, "/v1/health") => {
            authorize(&req, state.auth_token.as_ref())?;
            Ok(remote_health_response(&state).await)
        }
        (Method::POST, "/v1/daemon/request") => {
            authorize(&req, state.auth_token.as_ref())?;
            handle_daemon_request(req, &state).await
        }
        (Method::GET, "/v1/events") => {
            authorize(&req, state.auth_token.as_ref())?;
            handle_sse_events(req, &state).await
        }
        (Method::GET, "/v1/events/ws") => {
            authorize(&req, state.auth_token.as_ref())?;
            handle_websocket_events(req, &state).await
        }
        _ => Err(RemoteError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            format!("No turin-remote route matches '{}'", path),
        )),
    }
}

fn authorize(
    req: &Request<Incoming>,
    expected_token: &str,
) -> std::result::Result<(), RemoteError> {
    let Some(value) = req.headers().get(AUTHORIZATION) else {
        return Err(RemoteError::unauthorized());
    };
    let Ok(text) = value.to_str() else {
        return Err(RemoteError::unauthorized());
    };
    let Some(token) = text.strip_prefix("Bearer ") else {
        return Err(RemoteError::unauthorized());
    };
    if token != expected_token {
        return Err(RemoteError::unauthorized());
    }
    Ok(())
}

async fn remote_health_response(state: &RemoteState) -> Response<RemoteBody> {
    let remote = RemoteRuntimeReport {
        ready: false,
        version: env!("CARGO_PKG_VERSION").to_string(),
        bind: state.bind.clone(),
        daemon_endpoint: state.daemon_endpoint.clone(),
        auth_mode: "bearer",
        event_streams: RemoteEventStreams {
            sse: true,
            websocket: true,
        },
    };

    match state.client.health().await {
        Ok(daemon) => json_response(
            StatusCode::OK,
            &RemoteHealthReport {
                remote: RemoteRuntimeReport {
                    ready: true,
                    ..remote
                },
                daemon: Some(daemon),
                error: None,
            },
        ),
        Err(err) => json_response(
            StatusCode::SERVICE_UNAVAILABLE,
            &RemoteHealthReport {
                remote,
                daemon: None,
                error: Some(err.to_string()),
            },
        ),
    }
}

async fn handle_daemon_request(
    req: Request<Incoming>,
    state: &RemoteState,
) -> std::result::Result<Response<RemoteBody>, RemoteError> {
    let body = req
        .into_body()
        .collect()
        .await
        .map_err(|err| {
            RemoteError::new(
                StatusCode::BAD_REQUEST,
                "invalid_request_body",
                format!("Failed to read request body: {}", err),
            )
        })?
        .to_bytes();
    let request: RequestEnvelope = serde_json::from_slice(&body).map_err(|err| {
        RemoteError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_body",
            format!("Failed to decode daemon request JSON: {}", err),
        )
    })?;
    let response = state.client.send(request).await.map_err(|err| {
        RemoteError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "daemon_unavailable",
            format!("Failed to reach daemon: {}", err),
        )
        .with_details(json!({
            "daemon_endpoint": state.daemon_endpoint,
        }))
    })?;
    Ok(json_response(StatusCode::OK, &response))
}

async fn handle_sse_events(
    req: Request<Incoming>,
    state: &RemoteState,
) -> std::result::Result<Response<RemoteBody>, RemoteError> {
    let filter = parse_event_filter(req.uri().query())?;
    let events = state.client.subscribe(None, filter).await.map_err(|err| {
        RemoteError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "daemon_unavailable",
            format!("Failed to subscribe to daemon events: {}", err),
        )
    })?;
    let mut keepalive = interval(state.event_keepalive);
    keepalive.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let sse_state = SseState {
        events,
        keepalive,
        closed: false,
    };
    let stream = stream::unfold(sse_state, |mut state| async move {
        if state.closed {
            return None;
        }
        tokio::select! {
            _ = state.keepalive.tick() => {
                Some((
                    Ok::<Frame<Bytes>, Infallible>(Frame::data(Bytes::from_static(b": keep-alive\n\n"))),
                    state,
                ))
            }
            result = state.events.next() => {
                match result {
                    Ok(Some(event)) => Some((Ok(Frame::data(Bytes::from(format_sse_event(&event)))), state)),
                    Ok(None) => {
                        state.closed = true;
                        Some((Ok(Frame::data(Bytes::from(format_sse_error("Daemon event stream closed")))), state))
                    }
                    Err(err) => {
                        state.closed = true;
                        Some((Ok(Frame::data(Bytes::from(format_sse_error(&err.to_string())))), state))
                    }
                }
            }
        }
    });
    let body = http_body_util::BodyExt::boxed_unsync(StreamBody::new(stream));
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/event-stream")
        .header(CACHE_CONTROL, "no-store")
        .header(CONNECTION, "keep-alive")
        .body(body)
        .expect("SSE response builds"))
}

async fn handle_websocket_events(
    req: Request<Incoming>,
    state: &RemoteState,
) -> std::result::Result<Response<RemoteBody>, RemoteError> {
    validate_websocket_request(&req)?;
    let filter = parse_event_filter(req.uri().query())?;
    let key = req
        .headers()
        .get(SEC_WEBSOCKET_KEY)
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| {
            RemoteError::new(
                StatusCode::BAD_REQUEST,
                "invalid_websocket_request",
                "Missing Sec-WebSocket-Key header",
            )
        })?
        .to_string();
    let accept = websocket_accept_value(&key);
    let on_upgrade = hyper::upgrade::on(req);
    let state = state.clone();
    tokio::spawn(async move {
        match on_upgrade.await {
            Ok(upgraded) => {
                if let Err(err) = run_websocket_stream(upgraded, state, filter).await {
                    warn!(error = %err, "turin-remote websocket stream failed");
                }
            }
            Err(err) => {
                warn!(error = %err, "turin-remote websocket upgrade failed");
            }
        }
    });

    Ok(Response::builder()
        .status(StatusCode::SWITCHING_PROTOCOLS)
        .header(CONNECTION, "Upgrade")
        .header(UPGRADE, "websocket")
        .header(SEC_WEBSOCKET_ACCEPT, accept)
        .body(empty_body())
        .expect("websocket upgrade response builds"))
}

async fn run_websocket_stream(
    upgraded: Upgraded,
    state: RemoteState,
    filter: RuntimeEventsSubscribeParams,
) -> Result<()> {
    let io = TokioIo::new(upgraded);
    let mut websocket = WebSocketStream::from_raw_socket(io, Role::Server, None).await;
    let mut events = state.client.subscribe(None, filter).await?;
    let mut keepalive = interval(state.event_keepalive);
    keepalive.set_missed_tick_behavior(MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            _ = keepalive.tick() => {
                websocket.send(Message::Ping(Vec::new().into())).await?;
            }
            incoming = websocket.next() => {
                match incoming {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Ping(payload))) => {
                        websocket.send(Message::Pong(payload)).await?;
                    }
                    Some(Ok(Message::Text(_)))
                    | Some(Ok(Message::Binary(_)))
                    | Some(Ok(Message::Pong(_))) => {}
                    Some(Ok(Message::Frame(_))) => {}
                    Some(Err(err)) => return Err(err.into()),
                }
            }
            event = events.next() => {
                match event {
                    Ok(Some(event)) => {
                        websocket
                            .send(Message::Text(serde_json::to_string(&event)?.into()))
                            .await?;
                    }
                    Ok(None) => break,
                    Err(err) => {
                        let payload =
                            EventEnvelope::new("remote.error", json!({ "message": err.to_string() }));
                        websocket
                            .send(Message::Text(serde_json::to_string(&payload)?.into()))
                            .await?;
                        break;
                    }
                }
            }
        }
    }

    Ok(())
}

fn validate_websocket_request(req: &Request<Incoming>) -> std::result::Result<(), RemoteError> {
    let upgrade = req
        .headers()
        .get(UPGRADE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    let connection = req
        .headers()
        .get(CONNECTION)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    let version = req
        .headers()
        .get(SEC_WEBSOCKET_VERSION)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    if !upgrade.eq_ignore_ascii_case("websocket")
        || !connection
            .split(',')
            .any(|part| part.trim().eq_ignore_ascii_case("upgrade"))
        || version != "13"
    {
        return Err(RemoteError::new(
            StatusCode::BAD_REQUEST,
            "invalid_websocket_request",
            "Request is not a valid websocket upgrade",
        ));
    }
    Ok(())
}

fn websocket_accept_value(key: &str) -> String {
    let mut hasher = Sha1::new();
    hasher.update(key.as_bytes());
    hasher.update(b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    base64::engine::general_purpose::STANDARD.encode(hasher.finalize())
}

fn parse_event_filter(
    query: Option<&str>,
) -> std::result::Result<RuntimeEventsSubscribeParams, RemoteError> {
    let mut filter = RuntimeEventsSubscribeParams::default();
    if let Some(query) = query {
        for (key, value) in form_urlencoded::parse(query.as_bytes()) {
            match key.as_ref() {
                "agent_id" => {
                    if !value.is_empty() {
                        filter.agent_id = Some(value.into_owned());
                    }
                }
                "session_id" => {
                    if !value.is_empty() {
                        filter.session_id = Some(value.into_owned());
                    }
                }
                "slot_id" => {
                    if !value.is_empty() {
                        filter.slot_id = Some(value.into_owned());
                    }
                }
                other => {
                    return Err(RemoteError::new(
                        StatusCode::BAD_REQUEST,
                        "invalid_query",
                        format!("Unsupported query parameter '{}'", other),
                    ));
                }
            }
        }
    }
    Ok(filter)
}

fn format_sse_event(event: &EventEnvelope) -> String {
    let payload = serde_json::to_string(&event.data).expect("event payload serializes");
    format!("event: {}\ndata: {}\n\n", event.event, payload)
}

fn format_sse_error(message: &str) -> String {
    format!(
        "event: remote.error\ndata: {}\n\n",
        serde_json::to_string(&json!({ "message": message })).expect("remote error serializes")
    )
}

fn json_response<T: Serialize>(status: StatusCode, value: &T) -> Response<RemoteBody> {
    let bytes = serde_json::to_vec(value).expect("JSON response serializes");
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "application/json")
        .body(full_body(bytes))
        .expect("JSON response builds")
}

fn full_body(data: impl Into<Bytes>) -> RemoteBody {
    http_body_util::BodyExt::boxed_unsync(Full::new(data.into()))
}

fn empty_body() -> RemoteBody {
    http_body_util::BodyExt::boxed_unsync(Empty::<Bytes>::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn websocket_accept_value_matches_rfc_example() {
        assert_eq!(
            websocket_accept_value("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }

    #[test]
    fn parse_event_filter_rejects_unknown_query_key() {
        let err = parse_event_filter(Some("bad=value")).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.code, "invalid_query");
    }
}
