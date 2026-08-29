use std::collections::HashMap;
use std::convert::Infallible;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use bytes::Bytes;
use futures::stream;
use http_body_util::{BodyExt, Limited, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::header::{CACHE_CONTROL, CONNECTION, CONTENT_TYPE};
use hyper::{Method, Request, Response, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::{MissedTickBehavior, interval};
use turin_client::{ManagedEventStream, SessionMessageDetail, SessionSummary};
use turin_daemon_protocol::{EventEnvelope, RuntimeEventsSubscribeParams};
use url::form_urlencoded;

use super::{WebBody, WebState, json_response, text_response};

const DEFAULT_SESSION_LIMIT: usize = 50;
const MAX_SESSION_LIMIT: usize = 100;
const DEFAULT_MESSAGE_LIMIT: usize = 80;
const MAX_MESSAGE_LIMIT: usize = 200;
const MAX_REQUEST_BYTES: usize = 64 * 1024;

#[derive(Serialize)]
struct AgentList {
    agents: Vec<WebAgent>,
}

#[derive(Serialize)]
struct WebAgent {
    id: String,
    name: String,
    provider: String,
    model: String,
    enabled: bool,
}

#[derive(Clone, Serialize)]
struct WebSession {
    id: String,
    title: String,
    agent_id: String,
    created_at: String,
    message_count: Option<usize>,
}

#[derive(Serialize)]
struct SessionPage {
    sessions: Vec<WebSession>,
    offset: usize,
    has_more: bool,
}

#[derive(Serialize)]
struct SessionResponse {
    session: WebSession,
}

#[derive(Serialize)]
struct WebMessage {
    id: String,
    turn_id: String,
    role: String,
    content: String,
    created_at: String,
    token_count: Option<u64>,
}

#[derive(Serialize)]
struct MessagePage {
    messages: Vec<WebMessage>,
    offset: usize,
    total: usize,
    has_more: bool,
}

#[derive(Deserialize)]
struct CreateSessionRequest {
    agent_id: String,
}

#[derive(Deserialize)]
struct RenameSessionRequest {
    title: String,
}

#[derive(Deserialize)]
struct SubmitMessageRequest {
    content: String,
}

#[derive(Serialize)]
struct SubmittedTask {
    request_id: String,
    session_id: String,
}

pub(super) async fn list_agents(state: &WebState) -> Result<Response<WebBody>> {
    let status = state.client.status().await?;
    let agents = status
        .registry
        .agents
        .into_iter()
        .map(|agent| WebAgent {
            name: display_name(&agent.id),
            id: agent.id,
            provider: agent.provider,
            model: agent.model,
            enabled: agent.enabled,
        })
        .collect();
    Ok(json_response(StatusCode::OK, &AgentList { agents }))
}

pub(super) async fn list_sessions(
    request: &Request<Incoming>,
    state: &WebState,
) -> Result<Response<WebBody>> {
    let query = query_values(request.uri().query());
    let limit = bounded_usize(&query, "limit", DEFAULT_SESSION_LIMIT, MAX_SESSION_LIMIT)?;
    let offset = bounded_usize(&query, "offset", 0, usize::MAX)?;
    let mut sessions = state.client.list_sessions(limit + 1, offset).await?;
    let has_more = sessions.len() > limit;
    sessions.truncate(limit);
    Ok(json_response(
        StatusCode::OK,
        &SessionPage {
            sessions: sessions.into_iter().map(web_session).collect(),
            offset,
            has_more,
        },
    ))
}

pub(super) async fn create_session(
    request: Request<Incoming>,
    state: &WebState,
) -> Result<Response<WebBody>> {
    let input: CreateSessionRequest = read_json(request).await?;
    let agent_id = required(&input.agent_id, "agent_id")?;
    let live = state.client.open_session(agent_id, None).await?;
    let detail = state.client.get_session_window(&live.session_id, 1).await?;
    Ok(json_response(
        StatusCode::CREATED,
        &SessionResponse {
            session: web_session(detail.session),
        },
    ))
}

pub(super) async fn session_route(
    request: Request<Incoming>,
    state: &WebState,
) -> Result<Response<WebBody>> {
    let (session_id, messages) = parse_session_path(request.uri().path())?;
    match (request.method(), messages) {
        (&Method::GET, true) => get_messages(&request, state, &session_id).await,
        (&Method::POST, true) => submit_message(request, state, &session_id).await,
        (&Method::PATCH, false) => rename_session(request, state, &session_id).await,
        (&Method::DELETE, false) => delete_session(state, &session_id).await,
        _ => Ok(text_response(
            StatusCode::METHOD_NOT_ALLOWED,
            "Method not allowed",
        )),
    }
}

async fn get_messages(
    request: &Request<Incoming>,
    state: &WebState,
    session_id: &str,
) -> Result<Response<WebBody>> {
    let query = query_values(request.uri().query());
    let limit = bounded_usize(&query, "limit", DEFAULT_MESSAGE_LIMIT, MAX_MESSAGE_LIMIT)?;
    let offset = bounded_usize(&query, "offset", 0, usize::MAX)?;
    let detail = state
        .client
        .get_session_window_at(session_id, limit, Some(offset))
        .await?;
    let total = detail
        .message_window
        .as_ref()
        .map_or(detail.messages.len(), |window| window.total);
    let loaded = detail.messages.len();
    Ok(json_response(
        StatusCode::OK,
        &MessagePage {
            messages: detail.messages.into_iter().map(web_message).collect(),
            offset,
            total,
            has_more: offset + loaded < total,
        },
    ))
}

async fn rename_session(
    request: Request<Incoming>,
    state: &WebState,
    session_id: &str,
) -> Result<Response<WebBody>> {
    let input: RenameSessionRequest = read_json(request).await?;
    let title = required(&input.title, "title")?;
    let session = state
        .client
        .set_session_title(session_id, Some(title.to_string()))
        .await?;
    Ok(json_response(
        StatusCode::OK,
        &SessionResponse {
            session: web_session(session),
        },
    ))
}

async fn delete_session(state: &WebState, session_id: &str) -> Result<Response<WebBody>> {
    state.client.delete_session(session_id).await?;
    Ok(Response::builder()
        .status(StatusCode::NO_CONTENT)
        .body(http_body_util::Empty::new().boxed_unsync())
        .expect("empty response is valid"))
}

async fn submit_message(
    request: Request<Incoming>,
    state: &WebState,
    session_id: &str,
) -> Result<Response<WebBody>> {
    let input: SubmitMessageRequest = read_json(request).await?;
    let content = required(&input.content, "content")?;
    let slot_id = match state
        .client
        .list_live_sessions()
        .await?
        .into_iter()
        .find(|session| session.session_id == session_id)
    {
        Some(session) => session.slot_id,
        None => state.client.resume_session(session_id, None).await?.slot_id,
    };
    let task = state
        .client
        .submit_task_in_slot(
            None,
            Some(session_id.to_string()),
            Some(slot_id),
            content.to_string(),
        )
        .await?;
    Ok(json_response(
        StatusCode::ACCEPTED,
        &SubmittedTask {
            request_id: task.request_id,
            session_id: session_id.to_string(),
        },
    ))
}

pub(super) async fn stream_events(
    request: &Request<Incoming>,
    state: &WebState,
) -> Result<Response<WebBody>> {
    let query = query_values(request.uri().query());
    let session_id = required(
        query.get("session_id").map(String::as_str).unwrap_or(""),
        "session_id",
    )?;
    if !state
        .client
        .list_live_sessions()
        .await?
        .iter()
        .any(|session| session.session_id == session_id)
    {
        state.client.resume_session(session_id, None).await?;
    }
    let events = state
        .client
        .subscribe_managed(RuntimeEventsSubscribeParams {
            session_id: Some(session_id.to_string()),
            ..Default::default()
        })
        .await?;
    let mut keepalive = interval(Duration::from_secs(15));
    keepalive.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let stream = stream::unfold(
        EventStreamState {
            events,
            keepalive,
            active_task: None,
            session_id: session_id.to_string(),
        },
        |mut state| async move {
            loop {
                tokio::select! {
                    _ = state.keepalive.tick() => {
                        return Some((Ok::<Frame<Bytes>, Infallible>(Frame::data(Bytes::from_static(b": keep-alive\n\n"))), state));
                    }
                    event = state.events.next_event() => {
                        match event {
                            Ok(event) => {
                                if let Some(frame) = browser_event(
                                    &state.session_id,
                                    &mut state.active_task,
                                    event,
                                ) {
                                    return Some((Ok(Frame::data(Bytes::from(frame))), state));
                                }
                            }
                            Err(error) => {
                                let data = serde_json::json!({
                                    "request_id": state.active_task.as_ref().map_or("", |task| task.request_id.as_str()),
                                    "session_id": state.session_id,
                                    "message": error.to_string(),
                                    "retryable": true,
                                });
                                return Some((Ok(Frame::data(Bytes::from(sse("conversation.task.failed", &data)))), state));
                            }
                        }
                    }
                }
            }
        },
    );
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/event-stream")
        .header(CACHE_CONTROL, "no-store")
        .header(CONNECTION, "keep-alive")
        .body(StreamBody::new(stream).boxed_unsync())
        .expect("SSE response is valid"))
}

struct EventStreamState {
    events: ManagedEventStream,
    keepalive: tokio::time::Interval,
    active_task: Option<ActiveTask>,
    session_id: String,
}

struct ActiveTask {
    request_id: String,
    message_sequence: usize,
    message_id: Option<String>,
}

fn browser_event(
    session_id: &str,
    active_task: &mut Option<ActiveTask>,
    event: EventEnvelope,
) -> Option<String> {
    match event.event.as_str() {
        "task_start" => {
            let request_id = event.data.get("task_id")?.as_str()?.to_string();
            let agent_id = event.data.get("agent_id")?.as_str()?;
            *active_task = Some(ActiveTask {
                request_id: request_id.clone(),
                message_sequence: 0,
                message_id: None,
            });
            Some(sse(
                "conversation.task.started",
                &serde_json::json!({
                    "request_id": request_id,
                    "session_id": session_id,
                    "agent_id": agent_id,
                }),
            ))
        }
        "message_start" => {
            let task = active_task.as_mut()?;
            task.message_sequence += 1;
            let message_id = format!("stream-{}-{}", task.request_id, task.message_sequence);
            task.message_id = Some(message_id.clone());
            Some(sse(
                "conversation.message.started",
                &serde_json::json!({
                    "request_id": task.request_id,
                    "session_id": session_id,
                    "message_id": message_id,
                }),
            ))
        }
        "message_delta" => {
            let task = active_task.as_mut()?;
            if task.message_id.is_none() {
                task.message_sequence += 1;
                task.message_id = Some(format!(
                    "stream-{}-{}",
                    task.request_id, task.message_sequence
                ));
            }
            let message_id = task.message_id.as_ref()?;
            let delta = event.data.get("content_delta")?.as_str()?;
            Some(sse(
                "conversation.message.delta",
                &serde_json::json!({
                    "request_id": task.request_id,
                    "session_id": session_id,
                    "message_id": message_id,
                    "delta": delta,
                }),
            ))
        }
        "task_complete" => {
            let task = active_task.take()?;
            let status = event
                .data
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("error");
            if status == "success" {
                Some(sse(
                    "conversation.task.completed",
                    &serde_json::json!({
                        "request_id": task.request_id,
                        "session_id": session_id,
                    }),
                ))
            } else {
                Some(sse(
                    "conversation.task.failed",
                    &serde_json::json!({
                        "request_id": task.request_id,
                        "session_id": session_id,
                        "message": event.data.get("error").and_then(Value::as_str).unwrap_or("The task did not complete."),
                        "retryable": matches!(status, "error" | "timed_out"),
                    }),
                ))
            }
        }
        _ => None,
    }
}

fn web_session(session: SessionSummary) -> WebSession {
    let title = session
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("title"))
        .and_then(Value::as_str)
        .filter(|title| !title.trim().is_empty())
        .unwrap_or("New conversation")
        .to_string();
    WebSession {
        id: session.session_id,
        title,
        agent_id: session.agent_id,
        created_at: session.created_at,
        message_count: None,
    }
}

fn web_message(message: SessionMessageDetail) -> WebMessage {
    WebMessage {
        id: message.id.to_string(),
        turn_id: message.turn_id.to_string(),
        role: message.role,
        content: text_content(&message.content),
        created_at: message.created_at,
        token_count: message
            .token_count
            .or(message.estimated_token_count.map(u64::from)),
    }
}

fn text_content(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .map(text_content)
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(object) => object
            .get("text")
            .or_else(|| object.get("content"))
            .map(text_content)
            .unwrap_or_default(),
        _ => String::new(),
    }
}

fn display_name(id: &str) -> String {
    id.split(['-', '_'])
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            chars.next().map_or_else(String::new, |first| {
                first.to_uppercase().collect::<String>() + chars.as_str()
            })
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn parse_session_path(path: &str) -> Result<(String, bool)> {
    let suffix = path
        .strip_prefix("/api/sessions/")
        .context("invalid session path")?;
    let (encoded, messages) = suffix
        .strip_suffix("/messages")
        .map_or((suffix, false), |id| (id, true));
    if encoded.is_empty() || encoded.contains('/') {
        bail!("invalid session path");
    }
    let session_id = form_urlencoded::parse(format!("id={encoded}").as_bytes())
        .next()
        .map(|(_, value)| value.into_owned())
        .context("invalid session id")?;
    Ok((session_id, messages))
}

fn query_values(query: Option<&str>) -> HashMap<String, String> {
    form_urlencoded::parse(query.unwrap_or_default().as_bytes())
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect()
}

fn bounded_usize(
    values: &HashMap<String, String>,
    key: &str,
    default: usize,
    maximum: usize,
) -> Result<usize> {
    let value = values
        .get(key)
        .map_or(Ok(default), |value| value.parse::<usize>())?;
    if value > maximum {
        bail!("{key} must not exceed {maximum}");
    }
    Ok(value)
}

fn required<'a>(value: &'a str, field: &str) -> Result<&'a str> {
    let value = value.trim();
    if value.is_empty() {
        bail!("{field} must not be empty");
    }
    Ok(value)
}

async fn read_json<T: for<'de> Deserialize<'de>>(request: Request<Incoming>) -> Result<T> {
    let bytes = Limited::new(request.into_body(), MAX_REQUEST_BYTES)
        .collect()
        .await
        .map_err(|error| anyhow::anyhow!("request body exceeds the allowed size: {error}"))?
        .to_bytes();
    serde_json::from_slice(&bytes).context("request body is not valid JSON")
}

fn sse(event: &str, data: &Value) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_paths_decode_opaque_ids() {
        let (session_id, messages) =
            parse_session_path("/api/sessions/id%40%2Ftmp%2Fstate/messages").unwrap();
        assert_eq!(session_id, "id@/tmp/state");
        assert!(messages);
    }

    #[test]
    fn message_content_projects_text_parts() {
        assert_eq!(
            text_content(&serde_json::json!([
                { "type": "text", "text": "first" },
                { "type": "image", "url": "image.png" },
                { "type": "text", "text": "second" }
            ])),
            "first\nsecond"
        );
    }

    #[test]
    fn kernel_stream_events_are_projected_into_browser_events() {
        let mut active_task = None;
        let started = browser_event(
            "session-1",
            &mut active_task,
            EventEnvelope::new(
                "task_start",
                serde_json::json!({ "task_id": "task-1", "agent_id": "default" }),
            ),
        )
        .unwrap();
        assert!(started.contains("conversation.task.started"));

        let message = browser_event(
            "session-1",
            &mut active_task,
            EventEnvelope::new("message_start", serde_json::json!({})),
        )
        .unwrap();
        assert!(message.contains("conversation.message.started"));

        let delta = browser_event(
            "session-1",
            &mut active_task,
            EventEnvelope::new(
                "message_delta",
                serde_json::json!({ "content_delta": "Hi" }),
            ),
        )
        .unwrap();
        assert!(delta.contains("\"delta\":\"Hi\""));
    }
}
