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
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use tokio::time::{MissedTickBehavior, interval};
use turin_control_client::{ControlClient, DaemonStatus, ManagedEventStream};
use turin_daemon_protocol::{
    EventEnvelope, HarnessActionRunParams, HarnessActionRunResult, RuntimeEventsSubscribeParams,
    WorkItemList, WorklistItemsParams, WorklistListParams,
};
use turin_ui_core::{
    DashboardSnapshot, DashboardState, UiAppRecord, UiListRequest, UiRegistry,
    UiWorklistSourceError, ui_worklist_name_from_source as core_worklist_name_from_source,
};
use url::form_urlencoded;

const MAX_JSON_BODY_BYTES: usize = 1024 * 1024;
const EVENT_KEEPALIVE: Duration = Duration::from_secs(15);
const INDEX_HTML: &str = include_str!("../static/index.html");
const APP_CSS: &str = include_str!("../static/app.css");
const APP_JS: &str = include_str!("../static/app.js");

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
struct WebActionResponse {
    result: HarnessActionRunResult,
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
        (Method::GET, "/api/apps") => handle_apps(&state).await,
        (Method::POST, "/api/ui/list") => handle_ui_list(req, &state).await,
        (Method::POST, "/api/actions/run") => handle_action_run(req, &state).await,
        (Method::GET, "/api/events") => handle_sse_events(req, &state).await,
        _ => Err(WebError::not_found(format!(
            "No turin-web route matches '{}'",
            path
        ))),
    }
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
}
