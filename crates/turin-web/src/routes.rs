use std::convert::Infallible;
use std::error::Error;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures::stream;
use http::header::{CACHE_CONTROL, CONNECTION, CONTENT_TYPE};
use http::{Method, Request, Response, StatusCode};
use http_body_util::{
    BodyExt, Full, LengthLimitError, Limited, StreamBody, combinators::UnsyncBoxBody,
};
use hyper::body::{Frame, Incoming};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use tokio::time::{MissedTickBehavior, interval};
use turin_control_client::{
    ControlClient, DaemonStatus, LiveSession, ManagedEventStream, SessionBranchDetail,
    SessionDetail, SessionGraphDetail, SessionSummary, TaskStatus,
};
use turin_daemon_protocol::{
    DaemonRequest, EventEnvelope, HarnessActionRunParams, HarnessActionRunResult, MemoryList,
    MemoryListParams, RuntimeEventsSubscribeParams, SidestepContextTargetParams,
    SidestepModeParams, SubmitTaskParams, WorkItemList, WorklistDetail, WorklistItemsParams,
    WorklistListParams,
};
use turin_ui_core::{
    DashboardSnapshot, DashboardState, UiAppRecord, UiListRequest, UiRegistry,
    UiWorklistSourceError, ui_worklist_name_from_source as core_worklist_name_from_source,
};
use url::form_urlencoded;

const MAX_JSON_BODY_BYTES: usize = 1024 * 1024;
const DEFAULT_MESSAGE_LIMIT: usize = 48;
const MAX_MESSAGE_LIMIT: usize = 256;
const DEFAULT_DATA_LIMIT: u32 = 100;
const MAX_DATA_LIMIT: u32 = 250;
const EVENT_KEEPALIVE: Duration = Duration::from_secs(15);
const INDEX_HTML: &str = include_str!("../static/index.html");
const APP_CSS: &str = include_str!("../static/assets/app.css");
const APP_JS: &str = include_str!("../static/assets/app.js");

pub(crate) type WebBody = UnsyncBoxBody<Bytes, Infallible>;

#[derive(Clone)]
pub(crate) struct WebState {
    pub(crate) bind: String,
    pub(crate) client: ControlClient,
}

#[derive(Debug, Clone)]
struct WebError {
    status: StatusCode,
    code: &'static str,
    message: String,
    details: Option<Value>,
}

#[derive(Debug, Serialize)]
struct WebHealthz {
    ok: bool,
    version: String,
}

#[derive(Debug, Serialize)]
struct WebStatusResponse {
    web: WebRuntimeReport,
    snapshot: DashboardSnapshot,
    ui: UiRegistry,
}

#[derive(Debug, Serialize)]
struct WebRuntimeReport {
    ready: bool,
    version: String,
    bind: String,
    connection_kind: turin_control_client::ConnectionKind,
    connection_target: String,
}

#[derive(Debug, Serialize)]
struct WebAppsResponse {
    apps: Vec<UiAppRecord>,
}

#[derive(Debug, Serialize)]
struct WebAppResponse {
    app: UiAppRecord,
}

#[derive(Debug, Serialize)]
struct WebListResponse {
    request: UiListRequest,
    list: WorkItemList,
}

#[derive(Debug, Serialize)]
struct WebWorklistsResponse {
    worklists: Vec<WorklistDetail>,
}

#[derive(Debug, Serialize)]
struct WebWorklistItemsResponse {
    list: WorkItemList,
}

#[derive(Debug, Serialize)]
struct WebMemoriesResponse {
    list: MemoryList,
}

#[derive(Debug, Serialize)]
struct WebActionResponse {
    result: HarnessActionRunResult,
}

#[derive(Debug, Deserialize)]
struct WebSessionOpenRequest {
    agent_id: String,
    #[serde(default)]
    slot_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WebSessionResumeRequest {
    session_id: String,
    #[serde(default)]
    slot_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WebSessionTitleRequest {
    session_id: String,
    title: String,
}

#[derive(Debug, Deserialize)]
struct WebBranchCreateRequest {
    session_id: String,
    #[serde(default)]
    slot_id: Option<String>,
    name: String,
    from_turn_id: i64,
    #[serde(default)]
    activate: bool,
}

#[derive(Debug, Deserialize)]
struct WebBranchCheckoutRequest {
    session_id: String,
    #[serde(default)]
    slot_id: Option<String>,
    branch: String,
}

#[derive(Debug, Deserialize)]
struct WebSidestepRequest {
    session_id: String,
    #[serde(default)]
    slot_id: Option<String>,
    prompt: String,
    #[serde(default)]
    mode: SidestepModeParams,
    turn_id: i64,
    #[serde(default)]
    timeout_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct WebPromoteRequest {
    request_id: String,
    #[serde(default)]
    branch_name: Option<String>,
}

#[derive(Debug, Serialize)]
struct WebSessionResponse {
    session: LiveSession,
}

#[derive(Debug, Serialize)]
struct WebSessionDetailResponse {
    detail: SessionDetail,
}

#[derive(Debug, Serialize)]
struct WebSessionGraphResponse {
    graph: SessionGraphDetail,
}

#[derive(Debug, Serialize)]
struct WebSessionBranchResponse {
    branch: SessionBranchDetail,
}

#[derive(Debug, Serialize)]
struct WebSessionTitleResponse {
    session: SessionSummary,
}

#[derive(Debug, Serialize)]
struct WebTaskResponse {
    task: TaskStatus,
}

struct SseState {
    events: ManagedEventStream,
    keepalive: tokio::time::Interval,
    closed: bool,
}

#[derive(Debug, Serialize)]
struct WebErrorPayload {
    error: WebErrorEnvelope,
}

#[derive(Debug, Serialize)]
struct WebErrorEnvelope {
    code: &'static str,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<Value>,
}

impl WebError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            details: None,
        }
    }

    fn bad_request(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, code, message)
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, "not_found", message)
    }

    fn upstream(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "control_unavailable",
            message,
        )
    }

    fn with_details(mut self, details: Value) -> Self {
        self.details = Some(details);
        self
    }

    fn into_response(self) -> Response<WebBody> {
        let body = WebErrorPayload {
            error: WebErrorEnvelope {
                code: self.code,
                message: self.message,
                details: self.details,
            },
        };
        json_response(self.status, &body)
    }
}

pub(crate) async fn handle_http(
    req: Request<Incoming>,
    state: Arc<WebState>,
) -> std::result::Result<Response<WebBody>, Infallible> {
    let response = match route_request(req, state).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    };
    Ok(response)
}

async fn route_request(
    req: Request<Incoming>,
    state: Arc<WebState>,
) -> std::result::Result<Response<WebBody>, WebError> {
    let method = req.method().clone();
    let path = normalized_path(req.uri().path());

    if method == Method::GET
        && let Some(app_id) = path.strip_prefix("/api/apps/")
    {
        return handle_app(&state, app_id).await;
    }

    match (method, path.as_str()) {
        (Method::GET, "/") | (Method::GET, "/index.html") => Ok(static_response(
            StatusCode::OK,
            "text/html; charset=utf-8",
            INDEX_HTML,
        )),
        (Method::GET, "/assets/app.css") => Ok(static_response(
            StatusCode::OK,
            "text/css; charset=utf-8",
            APP_CSS,
        )),
        (Method::GET, "/assets/app.js") => Ok(static_response(
            StatusCode::OK,
            "application/javascript; charset=utf-8",
            APP_JS,
        )),
        (Method::GET, "/api/healthz") => Ok(json_response(
            StatusCode::OK,
            &WebHealthz {
                ok: true,
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
        )),
        (Method::GET, "/api/status") => handle_status(&state).await,
        (Method::GET, "/api/session") => handle_session_detail(req, &state).await,
        (Method::GET, "/api/session/graph") => handle_session_graph(req, &state).await,
        (Method::PUT, "/api/session/title") => handle_session_title(req, &state).await,
        (Method::POST, "/api/session/branches") => handle_branch_create(req, &state).await,
        (Method::POST, "/api/session/branches/checkout") => {
            handle_branch_checkout(req, &state).await
        }
        (Method::GET, "/api/data/worklists") => handle_data_worklists(&state).await,
        (Method::GET, "/api/data/worklist-items") => handle_data_worklist_items(req, &state).await,
        (Method::GET, "/api/data/memories") => handle_data_memories(req, &state).await,
        (Method::GET, "/api/apps") => handle_apps(&state).await,
        (Method::POST, "/api/ui/list") => handle_ui_list(req, &state).await,
        (Method::POST, "/api/actions/run") => handle_action_run(req, &state).await,
        (Method::POST, "/api/sessions/open") => handle_session_open(req, &state).await,
        (Method::POST, "/api/sessions/resume") => handle_session_resume(req, &state).await,
        (Method::POST, "/api/tasks/submit") => handle_task_submit(req, &state).await,
        (Method::POST, "/api/tasks/sidestep") => handle_task_sidestep(req, &state).await,
        (Method::POST, "/api/tasks/promote") => handle_task_promote(req, &state).await,
        (Method::GET, "/api/events") => handle_sse_events(req, &state).await,
        _ => Err(WebError::not_found(format!(
            "No turin-web route matches '{}'",
            path
        ))),
    }
}

async fn handle_session_title(
    req: Request<Incoming>,
    state: &WebState,
) -> std::result::Result<Response<WebBody>, WebError> {
    let params: WebSessionTitleRequest = read_json(req).await?;
    validate_session_title(&params)?;
    let session = state
        .client
        .set_session_title(&params.session_id, Some(params.title.trim().to_string()))
        .await
        .map_err(|err| WebError::upstream(format!("Failed to rename session: {err}")))?;
    Ok(json_response(
        StatusCode::OK,
        &WebSessionTitleResponse { session },
    ))
}

fn validate_session_title(params: &WebSessionTitleRequest) -> std::result::Result<(), WebError> {
    if params.session_id.trim().is_empty() {
        return Err(WebError::bad_request(
            "invalid_session_id",
            "Session id must not be empty",
        ));
    }
    let title = params.title.trim();
    if title.is_empty() {
        return Err(WebError::bad_request(
            "invalid_session_title",
            "Session title must not be empty",
        ));
    }
    if title.chars().count() > 120 {
        return Err(WebError::bad_request(
            "invalid_session_title",
            "Session title must not exceed 120 characters",
        ));
    }
    Ok(())
}

async fn handle_data_worklists(
    state: &WebState,
) -> std::result::Result<Response<WebBody>, WebError> {
    let worklists = state
        .client
        .list_worklists(WorklistListParams {
            persistence: None,
            name: None,
            scope: None,
        })
        .await
        .map_err(|err| WebError::upstream(format!("Failed to list worklists: {err}")))?;
    Ok(json_response(
        StatusCode::OK,
        &WebWorklistsResponse { worklists },
    ))
}

async fn handle_data_worklist_items(
    req: Request<Incoming>,
    state: &WebState,
) -> std::result::Result<Response<WebBody>, WebError> {
    let (id, limit) = parse_worklist_items_query(req.uri().query())?;
    let list = state
        .client
        .list_worklist_items(WorklistItemsParams {
            id,
            persistence: None,
            status: None,
            parent_id: None,
            r#where: None,
            claimed_only: false,
            paused_only: false,
            due_only: false,
            limit: Some(limit),
        })
        .await
        .map_err(|err| WebError::upstream(format!("Failed to list worklist items: {err}")))?;
    Ok(json_response(
        StatusCode::OK,
        &WebWorklistItemsResponse { list },
    ))
}

async fn handle_data_memories(
    req: Request<Incoming>,
    state: &WebState,
) -> std::result::Result<Response<WebBody>, WebError> {
    let params = parse_memory_list_query(req.uri().query())?;
    let list = state
        .client
        .list_memories(params)
        .await
        .map_err(|err| WebError::upstream(format!("Failed to list memories: {err}")))?;
    Ok(json_response(StatusCode::OK, &WebMemoriesResponse { list }))
}

async fn handle_session_detail(
    req: Request<Incoming>,
    state: &WebState,
) -> std::result::Result<Response<WebBody>, WebError> {
    let (session_id, message_limit, message_offset) =
        parse_session_detail_query(req.uri().query())?;
    let detail = state
        .client
        .get_session_window_at(&session_id, message_limit, message_offset)
        .await
        .map_err(|err| WebError::upstream(format!("Failed to load session: {}", err)))?;
    Ok(json_response(
        StatusCode::OK,
        &WebSessionDetailResponse { detail },
    ))
}

async fn handle_session_graph(
    req: Request<Incoming>,
    state: &WebState,
) -> std::result::Result<Response<WebBody>, WebError> {
    let session_id = parse_session_id_query(req.uri().query())?;
    let graph = state
        .client
        .get_session_graph(&session_id)
        .await
        .map_err(|err| WebError::upstream(format!("Failed to load session graph: {err}")))?;
    Ok(json_response(
        StatusCode::OK,
        &WebSessionGraphResponse { graph },
    ))
}

async fn handle_branch_create(
    req: Request<Incoming>,
    state: &WebState,
) -> std::result::Result<Response<WebBody>, WebError> {
    let params: WebBranchCreateRequest = read_json(req).await?;
    validate_session_and_name(&params.session_id, &params.name, "branch name")?;
    let branch = state
        .client
        .create_session_branch_from_turn_id(
            params.session_id.trim(),
            params.slot_id,
            params.name.trim(),
            params.from_turn_id,
            params.activate,
        )
        .await
        .map_err(|err| WebError::upstream(format!("Failed to create branch: {err}")))?;
    Ok(json_response(
        StatusCode::CREATED,
        &WebSessionBranchResponse { branch },
    ))
}

async fn handle_branch_checkout(
    req: Request<Incoming>,
    state: &WebState,
) -> std::result::Result<Response<WebBody>, WebError> {
    let params: WebBranchCheckoutRequest = read_json(req).await?;
    validate_session_and_name(&params.session_id, &params.branch, "branch")?;
    let branch = state
        .client
        .checkout_session_branch_in_slot(
            params.session_id.trim(),
            params.slot_id,
            params.branch.trim(),
        )
        .await
        .map_err(|err| WebError::upstream(format!("Failed to check out branch: {err}")))?;
    Ok(json_response(
        StatusCode::OK,
        &WebSessionBranchResponse { branch },
    ))
}

async fn handle_task_sidestep(
    req: Request<Incoming>,
    state: &WebState,
) -> std::result::Result<Response<WebBody>, WebError> {
    let params: WebSidestepRequest = read_json(req).await?;
    if params.session_id.trim().is_empty() || params.prompt.trim().is_empty() {
        return Err(WebError::bad_request(
            "invalid_sidestep",
            "Session id and sidestep prompt must not be empty",
        ));
    }
    let task = state
        .client
        .sidestep_task(
            params.session_id.trim().to_string(),
            params.slot_id,
            params.prompt.trim().to_string(),
            params.mode,
            Some(SidestepContextTargetParams::TurnId {
                turn_id: params.turn_id,
            }),
            params.timeout_ms,
        )
        .await
        .map_err(|err| WebError::upstream(format!("Failed to run sidestep: {err}")))?;
    Ok(json_response(StatusCode::OK, &WebTaskResponse { task }))
}

async fn handle_task_promote(
    req: Request<Incoming>,
    state: &WebState,
) -> std::result::Result<Response<WebBody>, WebError> {
    let params: WebPromoteRequest = read_json(req).await?;
    if params.request_id.trim().is_empty() {
        return Err(WebError::bad_request(
            "invalid_request_id",
            "Task request id must not be empty",
        ));
    }
    let branch_name = params
        .branch_name
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty());
    let branch = state
        .client
        .promote_task(params.request_id.trim(), branch_name)
        .await
        .map_err(|err| WebError::upstream(format!("Failed to promote sidestep: {err}")))?;
    Ok(json_response(
        StatusCode::CREATED,
        &WebSessionBranchResponse { branch },
    ))
}

fn validate_session_and_name(
    session_id: &str,
    name: &str,
    label: &str,
) -> std::result::Result<(), WebError> {
    if session_id.trim().is_empty() || name.trim().is_empty() {
        return Err(WebError::bad_request(
            "invalid_branch_request",
            format!("Session id and {label} must not be empty"),
        ));
    }
    Ok(())
}

async fn handle_session_open(
    req: Request<Incoming>,
    state: &WebState,
) -> std::result::Result<Response<WebBody>, WebError> {
    let params: WebSessionOpenRequest = read_json(req).await?;
    if params.agent_id.trim().is_empty() {
        return Err(WebError::bad_request(
            "invalid_agent_id",
            "Agent id must not be empty",
        ));
    }
    let session = state
        .client
        .open_session(params.agent_id.trim(), params.slot_id)
        .await
        .map_err(|err| WebError::upstream(format!("Failed to open session: {}", err)))?;
    Ok(json_response(
        StatusCode::CREATED,
        &WebSessionResponse { session },
    ))
}

async fn handle_session_resume(
    req: Request<Incoming>,
    state: &WebState,
) -> std::result::Result<Response<WebBody>, WebError> {
    let params: WebSessionResumeRequest = read_json(req).await?;
    if params.session_id.trim().is_empty() {
        return Err(WebError::bad_request(
            "invalid_session_id",
            "Session id must not be empty",
        ));
    }
    let session = state
        .client
        .resume_session(params.session_id.trim(), params.slot_id)
        .await
        .map_err(|err| WebError::upstream(format!("Failed to resume session: {}", err)))?;
    Ok(json_response(
        StatusCode::OK,
        &WebSessionResponse { session },
    ))
}

async fn handle_task_submit(
    req: Request<Incoming>,
    state: &WebState,
) -> std::result::Result<Response<WebBody>, WebError> {
    let params: SubmitTaskParams = read_json(req).await?;
    if params.prompt.trim().is_empty() {
        return Err(WebError::bad_request(
            "invalid_prompt",
            "Prompt must not be empty",
        ));
    }
    let task: TaskStatus = state
        .client
        .request_ok(None, DaemonRequest::TaskSubmit(params))
        .await
        .map_err(|err| WebError::upstream(format!("Failed to submit task: {}", err)))?;
    Ok(json_response(
        StatusCode::ACCEPTED,
        &WebTaskResponse { task },
    ))
}

async fn handle_status(state: &WebState) -> std::result::Result<Response<WebBody>, WebError> {
    let snapshot = DashboardState::snapshot(&state.client)
        .await
        .map_err(|err| WebError::upstream(format!("Failed to load dashboard snapshot: {}", err)))?;
    let ui = ui_registry_from_status(&snapshot.status);
    let response = WebStatusResponse {
        web: WebRuntimeReport {
            ready: true,
            version: env!("CARGO_PKG_VERSION").to_string(),
            bind: state.bind.clone(),
            connection_kind: state.client.kind(),
            connection_target: state.client.target(),
        },
        snapshot,
        ui,
    };
    Ok(json_response(StatusCode::OK, &response))
}

async fn handle_apps(state: &WebState) -> std::result::Result<Response<WebBody>, WebError> {
    let registry = load_ui_registry(&state.client).await?;
    Ok(json_response(
        StatusCode::OK,
        &WebAppsResponse {
            apps: registry.apps().cloned().collect(),
        },
    ))
}

async fn handle_app(
    state: &WebState,
    app_id: &str,
) -> std::result::Result<Response<WebBody>, WebError> {
    if app_id.is_empty() {
        return Err(WebError::bad_request(
            "invalid_app_id",
            "App id must not be empty",
        ));
    }

    let registry = load_ui_registry(&state.client).await?;
    let Some(app) = registry.app(app_id).cloned() else {
        return Err(WebError::not_found(format!(
            "UI app '{}' was not found",
            app_id
        )));
    };
    Ok(json_response(StatusCode::OK, &WebAppResponse { app }))
}

async fn handle_ui_list(
    req: Request<Incoming>,
    state: &WebState,
) -> std::result::Result<Response<WebBody>, WebError> {
    let request: UiListRequest = read_json(req).await?;
    let list = load_ui_list(&state.client, &request).await?;
    Ok(json_response(
        StatusCode::OK,
        &WebListResponse { request, list },
    ))
}

async fn handle_action_run(
    req: Request<Incoming>,
    state: &WebState,
) -> std::result::Result<Response<WebBody>, WebError> {
    let params: HarnessActionRunParams = read_json(req).await?;
    validate_action_run_params(&params)?;
    let result = state
        .client
        .run_harness_action(params)
        .await
        .map_err(|err| WebError::upstream(format!("Failed to run harness action: {}", err)))?;
    Ok(json_response(StatusCode::OK, &WebActionResponse { result }))
}

fn validate_action_run_params(
    params: &HarnessActionRunParams,
) -> std::result::Result<(), WebError> {
    if params.action.trim().is_empty() {
        return Err(WebError::bad_request(
            "invalid_action_request",
            "Action name must not be empty",
        )
        .with_details(json!({
            "field": "action",
            "guidance": "Send the declared harness action name, for example 'release.seed_demo_work'."
        })));
    }
    Ok(())
}

async fn handle_sse_events(
    req: Request<Incoming>,
    state: &WebState,
) -> std::result::Result<Response<WebBody>, WebError> {
    let filter = parse_event_filter(req.uri().query())?;
    let events = state
        .client
        .subscribe_managed(filter)
        .await
        .map_err(|err| WebError::upstream(format!("Failed to subscribe to events: {}", err)))?;
    let mut keepalive = interval(EVENT_KEEPALIVE);
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
            result = state.events.next_event() => {
                match result {
                    Ok(event) => Some((Ok(Frame::data(Bytes::from(format_sse_event(&event)))), state)),
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

async fn load_ui_registry(client: &ControlClient) -> std::result::Result<UiRegistry, WebError> {
    let status = client
        .status()
        .await
        .map_err(|err| WebError::upstream(format!("Failed to load daemon status: {}", err)))?;
    Ok(ui_registry_from_status(&status))
}

fn ui_registry_from_status(status: &DaemonStatus) -> UiRegistry {
    UiRegistry::from_messages(
        status
            .harnesses
            .iter()
            .flat_map(|harness| harness.ui_intents.iter().cloned()),
    )
}

async fn load_ui_list(
    client: &ControlClient,
    request: &UiListRequest,
) -> std::result::Result<WorkItemList, WebError> {
    let worklist_name = worklist_name_from_source(&request.source)?;

    let worklists = client
        .list_worklists(WorklistListParams {
            persistence: None,
            name: Some(worklist_name.to_string()),
            scope: None,
        })
        .await
        .map_err(|err| WebError::upstream(format!("Failed to list worklists: {}", err)))?;
    let Some(worklist) = worklists.first() else {
        return Err(WebError::not_found(format!(
            "Worklist '{}' was not found",
            worklist_name
        )));
    };

    client
        .list_worklist_items(WorklistItemsParams {
            id: worklist.public_id.clone(),
            persistence: worklist.persistence.clone(),
            status: None,
            parent_id: None,
            r#where: (!request.filter.is_empty()).then(|| request.filter.clone()),
            claimed_only: false,
            paused_only: false,
            due_only: false,
            limit: request.limit,
        })
        .await
        .map_err(|err| WebError::upstream(format!("Failed to load worklist items: {}", err)))
}

fn worklist_name_from_source(source: &str) -> std::result::Result<&str, WebError> {
    core_worklist_name_from_source(source).map_err(|err| match err {
        UiWorklistSourceError::Unsupported => WebError::bad_request(
            "unsupported_ui_list_source",
            format!("Unsupported UI list source '{}'", source),
        )
        .with_details(json!({
            "source": source,
            "supported_prefixes": ["worklists.<name>"],
            "guidance": "Model this data as a worklist source or add a deliberate UI list adapter."
        })),
        UiWorklistSourceError::MissingName => WebError::bad_request(
            "invalid_ui_list_source",
            format!("UI list source '{}' is missing a worklist name", source),
        )
        .with_details(json!({
            "source": source,
            "supported_prefixes": ["worklists.<name>"],
            "guidance": "Use a non-empty worklist source, for example 'worklists.release'."
        })),
    })
}

fn parse_event_filter(
    query: Option<&str>,
) -> std::result::Result<RuntimeEventsSubscribeParams, WebError> {
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
                    return Err(WebError::bad_request(
                        "invalid_query",
                        format!("Unsupported query parameter '{}'", other),
                    ));
                }
            }
        }
    }
    Ok(filter)
}

fn parse_session_detail_query(
    query: Option<&str>,
) -> std::result::Result<(String, usize, Option<usize>), WebError> {
    let mut session_id = None;
    let mut message_limit = DEFAULT_MESSAGE_LIMIT;
    let mut message_offset = None;
    if let Some(query) = query {
        for (key, value) in form_urlencoded::parse(query.as_bytes()) {
            match key.as_ref() {
                "session_id" => {
                    if !value.is_empty() {
                        session_id = Some(value.into_owned());
                    }
                }
                "message_limit" => {
                    message_limit = value.parse::<usize>().map_err(|_| {
                        WebError::bad_request(
                            "invalid_message_limit",
                            "message_limit must be a positive integer",
                        )
                    })?;
                }
                "message_offset" => {
                    message_offset = Some(value.parse::<usize>().map_err(|_| {
                        WebError::bad_request(
                            "invalid_message_offset",
                            "message_offset must be a non-negative integer",
                        )
                    })?);
                }
                other => {
                    return Err(WebError::bad_request(
                        "invalid_query",
                        format!("Unsupported query parameter '{}'", other),
                    ));
                }
            }
        }
    }
    if message_limit == 0 || message_limit > MAX_MESSAGE_LIMIT {
        return Err(WebError::bad_request(
            "invalid_message_limit",
            format!("message_limit must be between 1 and {}", MAX_MESSAGE_LIMIT),
        ));
    }
    let session_id = session_id.ok_or_else(|| {
        WebError::bad_request("invalid_session_id", "session_id must not be empty")
    })?;
    Ok((session_id, message_limit, message_offset))
}

fn parse_session_id_query(query: Option<&str>) -> std::result::Result<String, WebError> {
    let mut session_id = None;
    if let Some(query) = query {
        for (key, value) in form_urlencoded::parse(query.as_bytes()) {
            match key.as_ref() {
                "session_id" if !value.is_empty() => session_id = Some(value.into_owned()),
                "session_id" => {}
                other => {
                    return Err(WebError::bad_request(
                        "invalid_query",
                        format!("Unsupported query parameter '{other}'"),
                    ));
                }
            }
        }
    }
    session_id.ok_or_else(|| {
        WebError::bad_request(
            "missing_session_id",
            "session_id query parameter is required",
        )
    })
}

fn parse_worklist_items_query(query: Option<&str>) -> std::result::Result<(String, u32), WebError> {
    let mut id = None;
    let mut limit = DEFAULT_DATA_LIMIT;
    if let Some(query) = query {
        for (key, value) in form_urlencoded::parse(query.as_bytes()) {
            match key.as_ref() {
                "id" if !value.is_empty() => id = Some(value.into_owned()),
                "id" => {}
                "limit" => limit = parse_data_limit(&value)?,
                other => {
                    return Err(WebError::bad_request(
                        "invalid_query",
                        format!("Unsupported query parameter '{other}'"),
                    ));
                }
            }
        }
    }
    let id =
        id.ok_or_else(|| WebError::bad_request("invalid_worklist_id", "id must not be empty"))?;
    Ok((id, limit))
}

fn parse_memory_list_query(query: Option<&str>) -> std::result::Result<MemoryListParams, WebError> {
    let mut params = MemoryListParams {
        persistence: None,
        scope_kind: None,
        scope_key: None,
        include_superseded: false,
        limit: Some(DEFAULT_DATA_LIMIT),
        offset: Some(0),
    };
    if let Some(query) = query {
        for (key, value) in form_urlencoded::parse(query.as_bytes()) {
            match key.as_ref() {
                "scope_kind" if !value.is_empty() => {
                    params.scope_kind = Some(value.into_owned());
                }
                "scope_kind" => {}
                "scope_key" if !value.is_empty() => {
                    params.scope_key = Some(value.into_owned());
                }
                "scope_key" => {}
                "include_superseded" => {
                    params.include_superseded = match value.as_ref() {
                        "true" | "1" => true,
                        "false" | "0" => false,
                        _ => {
                            return Err(WebError::bad_request(
                                "invalid_include_superseded",
                                "include_superseded must be true or false",
                            ));
                        }
                    };
                }
                "limit" => params.limit = Some(parse_data_limit(&value)?),
                "offset" => {
                    params.offset = Some(value.parse::<u32>().map_err(|_| {
                        WebError::bad_request(
                            "invalid_data_offset",
                            "offset must be a non-negative integer",
                        )
                    })?);
                }
                other => {
                    return Err(WebError::bad_request(
                        "invalid_query",
                        format!("Unsupported query parameter '{other}'"),
                    ));
                }
            }
        }
    }
    Ok(params)
}

fn parse_data_limit(value: &str) -> std::result::Result<u32, WebError> {
    let limit = value.parse::<u32>().map_err(|_| {
        WebError::bad_request("invalid_data_limit", "limit must be a positive integer")
    })?;
    if limit == 0 || limit > MAX_DATA_LIMIT {
        return Err(WebError::bad_request(
            "invalid_data_limit",
            format!("limit must be between 1 and {MAX_DATA_LIMIT}"),
        ));
    }
    Ok(limit)
}

fn format_sse_event(event: &EventEnvelope) -> String {
    let payload = serde_json::to_string(&event.data).expect("event payload serializes");
    format!("event: {}\ndata: {}\n\n", event.event, payload)
}

fn format_sse_error(message: &str) -> String {
    format!(
        "event: web.error\ndata: {}\n\n",
        serde_json::to_string(&json!({ "message": message })).expect("web error serializes")
    )
}

async fn read_json<T: DeserializeOwned>(
    req: Request<Incoming>,
) -> std::result::Result<T, WebError> {
    read_json_body(req.into_body()).await
}

async fn read_json_body<T, B>(body: B) -> std::result::Result<T, WebError>
where
    T: DeserializeOwned,
    B: BodyExt,
    B::Error: Into<Box<dyn Error + Send + Sync>>,
{
    let body = Limited::new(body, MAX_JSON_BODY_BYTES)
        .collect()
        .await
        .map_err(json_body_read_error)?
        .to_bytes();
    serde_json::from_slice(&body).map_err(|err| {
        WebError::bad_request(
            "invalid_json",
            format!("Failed to decode request JSON: {}", err),
        )
    })
}

fn json_body_read_error(err: Box<dyn Error + Send + Sync>) -> WebError {
    if err.downcast_ref::<LengthLimitError>().is_some() {
        return WebError::bad_request(
            "request_body_too_large",
            format!("JSON request body exceeds {} bytes", MAX_JSON_BODY_BYTES),
        );
    }
    WebError::bad_request(
        "invalid_request_body",
        format!("Failed to read request body: {}", err),
    )
}

fn normalized_path(raw_path: &str) -> String {
    let path = raw_path.trim_end_matches('/');
    if path.is_empty() {
        "/".to_string()
    } else {
        path.to_string()
    }
}

fn json_response<T: Serialize>(status: StatusCode, value: &T) -> Response<WebBody> {
    let bytes = serde_json::to_vec(value).expect("JSON response serializes");
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "application/json")
        .header(CACHE_CONTROL, "no-store")
        .body(full_body(bytes))
        .expect("JSON response builds")
}

fn static_response(
    status: StatusCode,
    content_type: &'static str,
    body: &'static str,
) -> Response<WebBody> {
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, content_type)
        .header(CACHE_CONTROL, "no-store")
        .body(full_body(body))
        .expect("static response builds")
}

fn full_body(data: impl Into<Bytes>) -> WebBody {
    http_body_util::BodyExt::boxed_unsync(Full::new(data.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_path_trims_trailing_slash() {
        assert_eq!(normalized_path("/api/apps/"), "/api/apps");
        assert_eq!(normalized_path("/api/apps/release/"), "/api/apps/release");
    }

    #[test]
    fn unsupported_list_source_is_bad_request() {
        let err = worklist_name_from_source("tables.release").unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.code, "unsupported_ui_list_source");
        let details = err.details.expect("unsupported source details");
        assert_eq!(details["source"], "tables.release");
        assert_eq!(details["supported_prefixes"][0], "worklists.<name>");
        assert!(
            details["guidance"]
                .as_str()
                .is_some_and(|guidance| guidance.contains("deliberate UI list adapter"))
        );
    }

    #[test]
    fn empty_worklist_source_is_bad_request() {
        let err = worklist_name_from_source("worklists.").unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.code, "invalid_ui_list_source");
        let details = err.details.expect("invalid source details");
        assert_eq!(details["source"], "worklists.");
        assert_eq!(details["supported_prefixes"][0], "worklists.<name>");
        assert!(
            details["guidance"]
                .as_str()
                .is_some_and(|guidance| guidance.contains("non-empty worklist source"))
        );
    }

    #[test]
    fn empty_action_name_is_bad_request() {
        let err = validate_action_run_params(&HarnessActionRunParams {
            action: "  ".to_string(),
            agent_id: None,
            harness_id: Some("default".to_string()),
            params: Value::Null,
        })
        .unwrap_err();

        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.code, "invalid_action_request");
        let details = err.details.expect("invalid action details");
        assert_eq!(details["field"], "action");
        assert!(
            details["guidance"]
                .as_str()
                .is_some_and(|guidance| guidance.contains("declared harness action name"))
        );
    }

    #[test]
    fn error_response_uses_json_envelope() {
        let response = WebError::not_found("missing").into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            response
                .headers()
                .get(CACHE_CONTROL)
                .and_then(|value| value.to_str().ok()),
            Some("no-store")
        );
    }

    #[tokio::test]
    async fn json_body_under_limit_decodes() {
        let decoded = read_json_body::<Value, _>(Full::new(Bytes::from_static(br#"{"ok":true}"#)))
            .await
            .expect("under-limit JSON decodes");

        assert_eq!(decoded["ok"], true);
    }

    #[tokio::test]
    async fn json_body_over_limit_is_rejected() {
        let oversized = Bytes::from(vec![b' '; MAX_JSON_BODY_BYTES + 1]);
        let err = read_json_body::<Value, _>(Full::new(oversized))
            .await
            .unwrap_err();

        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.code, "request_body_too_large");
    }

    #[test]
    fn static_response_uses_declared_content_type() {
        let response = static_response(StatusCode::OK, "text/plain", "hello");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("text/plain")
        );
    }

    #[test]
    fn parse_event_filter_rejects_unknown_query_key() {
        let err = parse_event_filter(Some("bad=value")).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.code, "invalid_query");
    }

    #[test]
    fn parse_event_filter_accepts_known_query_keys() {
        let filter =
            parse_event_filter(Some("agent_id=default&session_id=session-1&slot_id=slot-1"))
                .expect("filter parses");
        assert_eq!(filter.agent_id.as_deref(), Some("default"));
        assert_eq!(filter.session_id.as_deref(), Some("session-1"));
        assert_eq!(filter.slot_id.as_deref(), Some("slot-1"));
    }

    #[test]
    fn sse_event_uses_runtime_event_name() {
        let event = EventEnvelope::new("runtime.snapshot", json!({ "ok": true }));
        let text = format_sse_event(&event);
        assert!(text.contains("event: runtime.snapshot"));
        assert!(text.contains("\"ok\":true"));
    }

    #[test]
    fn session_detail_query_requires_id_and_bounds_window() {
        assert!(parse_session_detail_query(None).is_err());
        assert_eq!(
            parse_session_detail_query(Some("session_id=session-1"))
                .unwrap()
                .1,
            DEFAULT_MESSAGE_LIMIT
        );
        assert_eq!(
            parse_session_detail_query(Some("session_id=session-1&message_limit=96"))
                .unwrap()
                .1,
            96
        );
        assert_eq!(
            parse_session_detail_query(Some(
                "session_id=session-1&message_limit=96&message_offset=240"
            ))
            .unwrap()
            .2,
            Some(240)
        );
        assert!(parse_session_detail_query(Some("session_id=session-1&message_limit=0")).is_err());
        assert!(
            parse_session_detail_query(Some("session_id=session-1&message_limit=257")).is_err()
        );
        assert!(parse_session_detail_query(Some("session_id=session-1&offset=1")).is_err());
        assert!(
            parse_session_detail_query(Some("session_id=session-1&message_offset=old")).is_err()
        );
    }

    #[test]
    fn session_graph_query_requires_only_session_id() {
        assert!(parse_session_id_query(None).is_err());
        assert_eq!(
            parse_session_id_query(Some("session_id=session-1")).unwrap(),
            "session-1"
        );
        assert!(parse_session_id_query(Some("session_id=session-1&offset=1")).is_err());
    }

    #[test]
    fn data_queries_require_identity_and_bound_windows() {
        assert!(parse_worklist_items_query(None).is_err());
        assert_eq!(
            parse_worklist_items_query(Some("id=worklist-1&limit=40")).unwrap(),
            ("worklist-1".to_string(), 40)
        );
        assert!(parse_worklist_items_query(Some("id=worklist-1&limit=0")).is_err());

        let params = parse_memory_list_query(Some(
            "scope_kind=agent&scope_key=researcher&include_superseded=true&limit=50&offset=100",
        ))
        .unwrap();
        assert_eq!(params.scope_kind.as_deref(), Some("agent"));
        assert_eq!(params.scope_key.as_deref(), Some("researcher"));
        assert!(params.include_superseded);
        assert_eq!(params.limit, Some(50));
        assert_eq!(params.offset, Some(100));
        assert!(parse_memory_list_query(Some("limit=251")).is_err());
        assert!(parse_memory_list_query(Some("include_superseded=maybe")).is_err());
    }

    #[test]
    fn session_title_requires_bounded_non_empty_values() {
        assert!(
            validate_session_title(&WebSessionTitleRequest {
                session_id: "session-1".to_string(),
                title: "Project review".to_string(),
            })
            .is_ok()
        );
        assert!(
            validate_session_title(&WebSessionTitleRequest {
                session_id: "".to_string(),
                title: "Project review".to_string(),
            })
            .is_err()
        );
        assert!(
            validate_session_title(&WebSessionTitleRequest {
                session_id: "session-1".to_string(),
                title: " ".to_string(),
            })
            .is_err()
        );
        assert!(
            validate_session_title(&WebSessionTitleRequest {
                session_id: "session-1".to_string(),
                title: "x".repeat(121),
            })
            .is_err()
        );
    }
}
