use std::collections::HashMap;
use std::convert::Infallible;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
use turin_client::{
    ManagedEventStream, SessionBranchDetail, SessionEfficiencyDetail, SessionMessageDetail,
    SessionSummary,
};
use turin_daemon_protocol::{
    EventEnvelope, MemoryListParams, RuntimeEventsSubscribeParams, SessionSearchHitKind,
    SessionSearchScope, WorkItemControlAction, WorkItemControlParams, WorkItemDetail,
    WorklistItemsParams, WorklistListParams,
};
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
struct HarnessList {
    harnesses: Vec<WebHarness>,
}

#[derive(Serialize)]
struct WebHarness {
    id: String,
    name: String,
    bound_agent_ids: Vec<String>,
    has_ui: bool,
}

#[derive(Serialize)]
struct WebAgent {
    id: String,
    name: String,
    provider: String,
    model: String,
    harness_id: String,
    enabled: bool,
}

#[derive(Serialize)]
struct WebWorklist {
    id: String,
    name: String,
    scope: String,
    created_at: String,
    updated_at: String,
}

#[derive(Serialize)]
struct WebWorklistList {
    worklists: Vec<WebWorklist>,
}

#[derive(Serialize)]
struct WebWorkItem {
    id: String,
    worklist_id: String,
    parent_id: Option<String>,
    title: String,
    kind: String,
    prompt: Option<String>,
    action_name: Option<String>,
    status: String,
    priority: i64,
    paused: bool,
    pause_reason: Option<String>,
    pause_until_unix_ms: Option<i64>,
    after: Vec<String>,
    claim_agent_id: Option<String>,
    claim_session_id: Option<String>,
    claim_execution_id: Option<String>,
    claim_heartbeat_unix_ms: Option<i64>,
    claimed_at: Option<String>,
    completed_at: Option<String>,
    failure_reason: Option<String>,
    created_at: String,
    updated_at: String,
}

#[derive(Serialize)]
struct WebWorkItemList {
    worklist_id: String,
    items: Vec<WebWorkItem>,
}

#[derive(Deserialize)]
struct WorkItemControlRequest {
    action: WorkItemControlAction,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    stale_after_ms: Option<u64>,
}

#[derive(Serialize)]
struct WebMemory {
    id: String,
    scope_kind: String,
    scope_key: String,
    content: String,
    storage: String,
    weight: f64,
    retrieval_count: u64,
    created_at: String,
}

#[derive(Serialize)]
struct WebMemoryList {
    memories: Vec<WebMemory>,
    total: u64,
    offset: u32,
    limit: u32,
}

#[derive(Serialize)]
struct WebSearchHit {
    kind: SessionSearchHitKind,
    session_id: String,
    agent_id: String,
    title: Option<String>,
    created_at: String,
    turn_id: Option<String>,
    turn_index: Option<u32>,
    role: Option<String>,
    tool_name: Option<String>,
    event_type: Option<String>,
    snippet: String,
}

#[derive(Serialize)]
struct WebSearchResults {
    hits: Vec<WebSearchHit>,
}

#[derive(Clone, Serialize)]
struct WebSession {
    id: String,
    title: String,
    agent_id: String,
    created_at: String,
    message_count: Option<usize>,
    visibility: String,
    relation_kind: Option<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    metrics: Option<WebMessageMetrics>,
}

#[derive(Clone, Serialize)]
struct WebMessageMetrics {
    input_tokens: u64,
    output_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_read_input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_creation_input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
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

#[derive(Deserialize)]
struct CreateBranchRequest {
    turn_id: String,
    #[serde(default)]
    activate: bool,
}

#[derive(Serialize)]
struct BranchResponse {
    branch: SessionBranchDetail,
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
            name: agent_display_name(&agent.id),
            id: agent.id,
            provider: agent.provider,
            model: agent.model,
            harness_id: agent.harness_ref,
            enabled: agent.enabled,
        })
        .collect();
    Ok(json_response(StatusCode::OK, &AgentList { agents }))
}

pub(super) async fn list_harnesses(state: &WebState) -> Result<Response<WebBody>> {
    let harnesses = state
        .client
        .list_harnesses()
        .await?
        .into_iter()
        .map(|harness| WebHarness {
            name: harness_display_name(&harness.harness_id),
            id: harness.harness_id,
            bound_agent_ids: harness.bound_agents,
            has_ui: !harness.ui_intents.is_empty(),
        })
        .collect();
    Ok(json_response(StatusCode::OK, &HarnessList { harnesses }))
}

pub(super) async fn list_worklists(state: &WebState) -> Result<Response<WebBody>> {
    let worklists = state
        .client
        .list_worklists(WorklistListParams {
            persistence: None,
            name: None,
            scope: None,
        })
        .await?
        .into_iter()
        .map(|worklist| WebWorklist {
            id: worklist.public_id,
            name: worklist.name,
            scope: worklist.scope_ref,
            created_at: worklist.created_at,
            updated_at: worklist.updated_at,
        })
        .collect();
    Ok(json_response(
        StatusCode::OK,
        &WebWorklistList { worklists },
    ))
}

pub(super) async fn list_worklist_items(
    request: &Request<Incoming>,
    state: &WebState,
) -> Result<Response<WebBody>> {
    let encoded = request
        .uri()
        .path()
        .strip_prefix("/api/worklists/")
        .and_then(|path| path.strip_suffix("/items"))
        .filter(|path| !path.is_empty() && !path.contains('/'))
        .context("invalid worklist items path")?;
    let worklist_id = form_urlencoded::parse(format!("id={encoded}").as_bytes())
        .next()
        .map(|(_, value)| value.into_owned())
        .context("invalid worklist id")?;
    let result = state
        .client
        .list_worklist_items(WorklistItemsParams {
            id: worklist_id,
            persistence: None,
            status: None,
            parent_id: None,
            r#where: None,
            claimed_only: false,
            paused_only: false,
            due_only: false,
            limit: Some(200),
        })
        .await?;
    let items = result.items.into_iter().map(web_work_item).collect();
    Ok(json_response(
        StatusCode::OK,
        &WebWorkItemList {
            worklist_id: result.worklist_id,
            items,
        },
    ))
}

pub(super) async fn work_item_route(
    request: Request<Incoming>,
    state: &WebState,
) -> Result<Response<WebBody>> {
    let (work_item_id, resource) = parse_work_item_path(request.uri().path())?;
    match (request.method(), resource) {
        (&Method::GET, None) => {
            let item = state.client.get_workitem(work_item_id, None).await?;
            Ok(json_response(StatusCode::OK, &web_work_item(item)))
        }
        (&Method::POST, Some("control")) => {
            let input: WorkItemControlRequest = read_json(request).await?;
            let item = match state
                .client
                .control_workitem(WorkItemControlParams {
                    id: work_item_id.clone(),
                    action: input.action,
                    reason: input.reason,
                    stale_after_ms: input.stale_after_ms,
                    persistence: None,
                })
                .await
            {
                Ok(item) => item,
                Err(error) => {
                    tracing::info!(%error, %work_item_id, "work item control was rejected");
                    return Ok(json_response(
                        StatusCode::CONFLICT,
                        &serde_json::json!({
                            "error": "The work item changed or is not eligible for that operation. Refresh and try again."
                        }),
                    ));
                }
            };
            Ok(json_response(StatusCode::OK, &web_work_item(item)))
        }
        _ => Ok(text_response(StatusCode::NOT_FOUND, "API route not found")),
    }
}

fn web_work_item(item: WorkItemDetail) -> WebWorkItem {
    WebWorkItem {
        id: item.public_id,
        worklist_id: item.worklist_id,
        parent_id: item.parent_id,
        title: item.title,
        kind: item.kind,
        prompt: item.prompt,
        action_name: item.action.map(|action| action.name),
        status: item.status,
        priority: item.priority,
        paused: item.paused,
        pause_reason: item.pause_reason,
        pause_until_unix_ms: item.pause_until_unix_ms,
        after: item.after.unwrap_or_default(),
        claim_agent_id: item.claim_agent_id,
        claim_session_id: item.claim_session_id,
        claim_execution_id: item.claim_execution_id,
        claim_heartbeat_unix_ms: item.claim_heartbeat_unix_ms,
        claimed_at: item.claimed_at,
        completed_at: item.completed_at,
        failure_reason: item.failure_reason,
        created_at: item.created_at,
        updated_at: item.updated_at,
    }
}

fn parse_work_item_path(path: &str) -> Result<(String, Option<&str>)> {
    let path = path
        .strip_prefix("/api/work-items/")
        .context("invalid work item path")?;
    let mut segments = path.split('/');
    let encoded = segments
        .next()
        .filter(|value| !value.is_empty())
        .context("missing work item id")?;
    let resource = segments.next();
    if segments.next().is_some() {
        bail!("invalid work item path");
    }
    let id = form_urlencoded::parse(format!("id={encoded}").as_bytes())
        .next()
        .map(|(_, value)| value.into_owned())
        .context("invalid work item id")?;
    Ok((id, resource))
}

pub(super) async fn list_memories(
    request: &Request<Incoming>,
    state: &WebState,
) -> Result<Response<WebBody>> {
    let query = query_values(request.uri().query());
    let limit = bounded_usize(&query, "limit", 100, 200)? as u32;
    let offset = bounded_usize(&query, "offset", 0, u32::MAX as usize)? as u32;
    let result = state
        .client
        .list_memories(MemoryListParams {
            persistence: None,
            scope_kind: None,
            scope_key: None,
            include_superseded: false,
            limit: Some(limit),
            offset: Some(offset),
        })
        .await?;
    let memories = result
        .memories
        .into_iter()
        .map(|memory| WebMemory {
            id: memory.public_id,
            scope_kind: memory.scope_kind,
            scope_key: memory.scope_key,
            content: memory.content,
            storage: memory.storage,
            weight: memory.weight,
            retrieval_count: memory.retrieval_count,
            created_at: memory.created_at,
        })
        .collect();
    Ok(json_response(
        StatusCode::OK,
        &WebMemoryList {
            memories,
            total: result.total,
            offset: result.offset,
            limit: result.limit,
        },
    ))
}

pub(super) async fn search_sessions(
    request: &Request<Incoming>,
    state: &WebState,
) -> Result<Response<WebBody>> {
    let query = query_values(request.uri().query());
    let query = required(query.get("q").map(String::as_str).unwrap_or(""), "q")?;
    let hits = state
        .client
        .search_sessions(query, SessionSearchScope::Sessions, 50, 0)
        .await?;
    Ok(json_response(
        StatusCode::OK,
        &WebSearchResults {
            hits: hits.into_iter().map(web_search_hit).collect(),
        },
    ))
}

pub(super) async fn search_workspace(
    request: &Request<Incoming>,
    state: &WebState,
) -> Result<Response<WebBody>> {
    let query = query_values(request.uri().query());
    let query = required(query.get("q").map(String::as_str).unwrap_or(""), "q")?;
    let hits = state
        .client
        .search_sessions(query, SessionSearchScope::All, 50, 0)
        .await?;
    Ok(json_response(
        StatusCode::OK,
        &WebSearchResults {
            hits: hits.into_iter().map(web_search_hit).collect(),
        },
    ))
}

async fn search_session_messages(
    request: &Request<Incoming>,
    state: &WebState,
    session_id: &str,
) -> Result<Response<WebBody>> {
    let query = query_values(request.uri().query());
    let query = required(query.get("q").map(String::as_str).unwrap_or(""), "q")?;
    let hits = state
        .client
        .search_session_messages(session_id, query, 50, 0)
        .await?;
    Ok(json_response(
        StatusCode::OK,
        &WebSearchResults {
            hits: hits.into_iter().map(web_search_hit).collect(),
        },
    ))
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
    let (session_id, resource) = parse_session_path(request.uri().path())?;
    match (request.method(), resource) {
        (&Method::GET, SessionResource::Messages) => {
            get_messages(&request, state, &session_id).await
        }
        (&Method::POST, SessionResource::Messages) => {
            submit_message(request, state, &session_id).await
        }
        (&Method::POST, SessionResource::Branches) => {
            create_branch(request, state, &session_id).await
        }
        (&Method::GET, SessionResource::Search) => {
            search_session_messages(&request, state, &session_id).await
        }
        (&Method::PATCH, SessionResource::Session) => {
            rename_session(request, state, &session_id).await
        }
        (&Method::DELETE, SessionResource::Session) => delete_session(state, &session_id).await,
        _ => Ok(text_response(
            StatusCode::METHOD_NOT_ALLOWED,
            "Method not allowed",
        )),
    }
}

async fn create_branch(
    request: Request<Incoming>,
    state: &WebState,
    session_id: &str,
) -> Result<Response<WebBody>> {
    let input: CreateBranchRequest = read_json(request).await?;
    let turn_id = required(&input.turn_id, "turn_id")?
        .parse::<i64>()
        .context("turn_id must identify a durable turn")?;
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let branch = state
        .client
        .create_session_branch_from_turn_id(
            session_id,
            None,
            &format!("fork-{turn_id}-{suffix}"),
            turn_id,
            input.activate,
        )
        .await?;
    Ok(json_response(
        StatusCode::CREATED,
        &BranchResponse { branch },
    ))
}

async fn get_messages(
    request: &Request<Incoming>,
    state: &WebState,
    session_id: &str,
) -> Result<Response<WebBody>> {
    let query = query_values(request.uri().query());
    let limit = bounded_usize(&query, "limit", DEFAULT_MESSAGE_LIMIT, MAX_MESSAGE_LIMIT)?;
    let offset_from_end = bounded_usize(&query, "offset", 0, usize::MAX)?;
    let anchor_turn_id = query
        .get("turn_id")
        .map(|value| {
            value
                .parse::<i64>()
                .context("turn_id must identify a durable turn")
        })
        .transpose()?;
    let detail = if let Some(turn_id) = anchor_turn_id {
        state
            .client
            .get_session_window_around_turn(session_id, turn_id, limit)
            .await
            .with_context(|| {
                format!(
                    "failed to load message window around turn '{turn_id}' for session '{session_id}'"
                )
            })?
    } else if offset_from_end == 0 {
        state
            .client
            .get_session_window(session_id, limit)
            .await
            .with_context(|| {
                format!("failed to load latest message window for session '{session_id}'")
            })?
    } else {
        let total_hint = match query.get("total") {
            Some(value) => value
                .parse::<usize>()
                .context("total must be a non-negative integer")?,
            None => {
                let recent = state.client.get_session_window(session_id, 1).await?;
                recent
                    .message_window
                    .as_ref()
                    .map_or(recent.messages.len(), |window| window.total)
            }
        };
        let start = oldest_first_window_start(total_hint, offset_from_end, limit);
        state
            .client
            .get_session_window_at(session_id, limit, Some(start))
			.await
			.with_context(|| {
				format!(
					"failed to load message window for session '{session_id}' (offset_from_end={offset_from_end}, oldest_first_start={start}, limit={limit}, total_hint={total_hint})"
				)
			})?
    };
    let total = detail
        .message_window
        .as_ref()
        .map_or(detail.messages.len(), |window| window.total);
    let loaded = detail.messages.len();
    let persisted_start = detail
        .message_window
        .as_ref()
        .map_or(0, |window| window.offset);
    let resolved_offset = newest_first_window_offset(total, persisted_start, loaded);
    let metrics = message_metrics(detail.efficiency.as_ref());
    Ok(json_response(
        StatusCode::OK,
        &MessagePage {
            messages: detail
                .messages
                .into_iter()
                .map(|message| {
                    let turn_metrics = (message.role == "assistant")
                        .then(|| metrics.get(&message.turn_index).cloned())
                        .flatten();
                    web_message(message, turn_metrics)
                })
                .collect(),
            offset: resolved_offset,
            total,
            has_more: resolved_offset.saturating_add(loaded) < total,
        },
    ))
}

fn oldest_first_window_start(total: usize, offset_from_end: usize, limit: usize) -> usize {
    total.saturating_sub(offset_from_end.saturating_add(limit))
}

fn newest_first_window_offset(total: usize, start: usize, loaded: usize) -> usize {
    total.saturating_sub(start.saturating_add(loaded))
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
        visibility: session.visibility,
        relation_kind: session.relation_kind,
    }
}

fn web_search_hit(hit: turin_client::SessionSearchHit) -> WebSearchHit {
    WebSearchHit {
        kind: hit.kind,
        session_id: hit.session_id,
        agent_id: hit.agent_id,
        title: hit.title,
        created_at: hit.created_at,
        turn_id: hit.turn_id.map(|turn_id| turn_id.to_string()),
        turn_index: hit.turn_index,
        role: hit.role,
        tool_name: hit.tool_name,
        event_type: hit.event_type,
        snippet: hit.snippet,
    }
}

fn message_metrics(
    efficiency: Option<&SessionEfficiencyDetail>,
) -> HashMap<u32, WebMessageMetrics> {
    let Some(efficiency) = efficiency else {
        return HashMap::new();
    };
    efficiency
        .turns
        .iter()
        .map(|turn| {
            let latest = turn.requests.last();
            let cache_read_input_tokens = efficiency.provider_cache_metrics_available.then(|| {
                turn.requests
                    .iter()
                    .filter_map(|request| request.cache_read_input_tokens)
                    .sum()
            });
            let cache_creation_input_tokens =
                efficiency.provider_cache_metrics_available.then(|| {
                    turn.requests
                        .iter()
                        .filter_map(|request| request.cache_creation_input_tokens)
                        .sum()
                });
            (
                turn.turn_index,
                WebMessageMetrics {
                    input_tokens: turn.input_tokens,
                    output_tokens: turn.output_tokens,
                    cache_read_input_tokens,
                    cache_creation_input_tokens,
                    provider: latest
                        .and_then(|request| request.metrics.as_ref())
                        .map(|metrics| metrics.provider.clone()),
                    model: latest
                        .and_then(|request| request.metrics.as_ref())
                        .map(|metrics| metrics.model.clone()),
                },
            )
        })
        .collect()
}

fn web_message(message: SessionMessageDetail, metrics: Option<WebMessageMetrics>) -> WebMessage {
    WebMessage {
        id: message.id.to_string(),
        turn_id: message.turn_id.to_string(),
        role: message.role,
        content: text_content(&message.content),
        created_at: message.created_at,
        token_count: message
            .token_count
            .or(message.estimated_token_count.map(u64::from)),
        metrics,
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

fn agent_display_name(id: &str) -> String {
    match id {
        "default" => "Turin".to_owned(),
        _ => display_name(id),
    }
}

fn harness_display_name(id: &str) -> String {
    match id {
        "default" => "General".to_owned(),
        _ => display_name(id),
    }
}

#[derive(Debug, PartialEq, Eq)]
enum SessionResource {
    Session,
    Messages,
    Branches,
    Search,
}

fn parse_session_path(path: &str) -> Result<(String, SessionResource)> {
    let suffix = path
        .strip_prefix("/api/sessions/")
        .context("invalid session path")?;
    let (encoded, resource) = if let Some(id) = suffix.strip_suffix("/messages") {
        (id, SessionResource::Messages)
    } else if let Some(id) = suffix.strip_suffix("/branches") {
        (id, SessionResource::Branches)
    } else if let Some(id) = suffix.strip_suffix("/search") {
        (id, SessionResource::Search)
    } else {
        (suffix, SessionResource::Session)
    };
    if encoded.is_empty() || encoded.contains('/') {
        bail!("invalid session path");
    }
    let session_id = form_urlencoded::parse(format!("id={encoded}").as_bytes())
        .next()
        .map(|(_, value)| value.into_owned())
        .context("invalid session id")?;
    Ok((session_id, resource))
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
        let (session_id, resource) =
            parse_session_path("/api/sessions/id%40%2Ftmp%2Fstate/messages").unwrap();
        assert_eq!(session_id, "id@/tmp/state");
        assert_eq!(resource, SessionResource::Messages);

        let (_, resource) = parse_session_path("/api/sessions/session-1/branches").unwrap();
        assert_eq!(resource, SessionResource::Branches);
    }

    #[test]
    fn work_item_paths_decode_id_and_control_resource() {
        assert_eq!(
            parse_work_item_path("/api/work-items/item%2Fone/control").unwrap(),
            ("item/one".to_string(), Some("control"))
        );
        assert!(parse_work_item_path("/api/work-items/item/unknown/extra").is_err());
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

    #[test]
    fn browser_message_offsets_are_translated_at_the_web_boundary() {
        assert_eq!(oldest_first_window_start(1_000, 0, 80), 920);
        assert_eq!(oldest_first_window_start(1_000, 400, 80), 520);
        assert_eq!(oldest_first_window_start(30, 900, 80), 0);
        assert_eq!(newest_first_window_offset(1_000, 520, 80), 400);
        assert_eq!(newest_first_window_offset(30, 0, 30), 0);
    }

    #[test]
    fn bootstrap_ids_use_product_facing_names() {
        assert_eq!(agent_display_name("default"), "Turin");
        assert_eq!(harness_display_name("default"), "General");
        assert_eq!(agent_display_name("release_operator"), "Release Operator");
    }
}
