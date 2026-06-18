use std::convert::Infallible;
use std::sync::Arc;

use bytes::Bytes;
use http::header::CONTENT_TYPE;
use http::{Method, Request, Response, StatusCode};
use http_body_util::{BodyExt, Full, combinators::UnsyncBoxBody};
use hyper::body::Incoming;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;
use turin_control_client::{ControlClient, DaemonStatus};
use turin_daemon_protocol::{
    HarnessActionRunParams, HarnessActionRunResult, WorkItemList, WorklistItemsParams,
    WorklistListParams,
};
use turin_ui_core::{DashboardSnapshot, DashboardState, UiAppRecord, UiListRequest, UiRegistry};

const MAX_JSON_BODY_BYTES: usize = 1024 * 1024;

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

    if method == Method::GET {
        if let Some(app_id) = path.strip_prefix("/api/apps/") {
            return handle_app(&state, app_id).await;
        }
    }

    match (method, path.as_str()) {
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
        (Method::GET, "/api/events") => Err(WebError::new(
            StatusCode::NOT_IMPLEMENTED,
            "not_implemented",
            "The turin-web event stream endpoint is planned but not implemented yet",
        )),
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
    let result = state
        .client
        .run_harness_action(params)
        .await
        .map_err(|err| WebError::upstream(format!("Failed to run harness action: {}", err)))?;
    Ok(json_response(StatusCode::OK, &WebActionResponse { result }))
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
    let worklist_name = source.strip_prefix("worklists.").ok_or_else(|| {
        WebError::bad_request(
            "unsupported_ui_list_source",
            format!("Unsupported UI list source '{}'", source),
        )
    })?;
    if worklist_name.is_empty() {
        return Err(WebError::bad_request(
            "invalid_ui_list_source",
            format!("UI list source '{}' is missing a worklist name", source),
        ));
    }
    Ok(worklist_name)
}

async fn read_json<T: DeserializeOwned>(
    req: Request<Incoming>,
) -> std::result::Result<T, WebError> {
    let body = req
        .into_body()
        .collect()
        .await
        .map_err(|err| {
            WebError::bad_request(
                "invalid_request_body",
                format!("Failed to read request body: {}", err),
            )
        })?
        .to_bytes();
    if body.len() > MAX_JSON_BODY_BYTES {
        return Err(WebError::bad_request(
            "request_body_too_large",
            format!("JSON request body exceeds {} bytes", MAX_JSON_BODY_BYTES),
        ));
    }
    serde_json::from_slice(&body).map_err(|err| {
        WebError::bad_request(
            "invalid_json",
            format!("Failed to decode request JSON: {}", err),
        )
    })
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
        .body(full_body(bytes))
        .expect("JSON response builds")
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
    }

    #[test]
    fn empty_worklist_source_is_bad_request() {
        let err = worklist_name_from_source("worklists.").unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.code, "invalid_ui_list_source");
    }

    #[test]
    fn error_response_uses_json_envelope() {
        let response = WebError::not_found("missing").into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
