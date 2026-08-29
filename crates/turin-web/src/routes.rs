use std::convert::Infallible;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use bytes::Bytes;
use http_body_util::{BodyExt, Full, combinators::UnsyncBoxBody};
use hyper::body::Incoming;
use hyper::header::{CACHE_CONTROL, CONTENT_TYPE};
use hyper::{Method, Request, Response, StatusCode};
use serde::Serialize;
use turin_client::{Client, ConnectionKind, ControlHealth};

mod api;

pub(crate) type WebBody = UnsyncBoxBody<Bytes, Infallible>;

pub(crate) struct WebState {
    pub(crate) assets_dir: PathBuf,
    pub(crate) client: Arc<Client>,
}

#[derive(Serialize)]
struct WebHealth {
    ok: bool,
    version: &'static str,
}

#[derive(Serialize)]
struct WebBootstrap {
    web_version: &'static str,
    runtime: RuntimeHealth,
}

#[derive(Serialize)]
struct RuntimeHealth {
    connection_kind: ConnectionKind,
    ready: bool,
    version: String,
    protocol_version: u32,
    issue_count: usize,
    agent_count: usize,
    harness_count: usize,
    running_agent_count: usize,
    active_task_count: usize,
}

impl From<ControlHealth> for RuntimeHealth {
    fn from(health: ControlHealth) -> Self {
        Self {
            connection_kind: health.connection_kind,
            ready: health.ready,
            version: health.version,
            protocol_version: health.protocol_version,
            issue_count: health.issue_count,
            agent_count: health.agent_count,
            harness_count: health.harness_count,
            running_agent_count: health.running_agent_count,
            active_task_count: health.active_task_count,
        }
    }
}

pub(crate) async fn handle_http(
    request: Request<Incoming>,
    state: Arc<WebState>,
) -> Result<Response<WebBody>, Infallible> {
    let response = match route(request, &state).await {
        Ok(response) => response,
        Err(error) => {
            tracing::warn!(%error, "turin-web request failed");
            json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &serde_json::json!({ "error": "The request could not be completed." }),
            )
        }
    };
    Ok(response)
}

async fn route(request: Request<Incoming>, state: &WebState) -> Result<Response<WebBody>> {
    match (request.method(), request.uri().path()) {
        (&Method::GET, "/api/healthz") => Ok(json_response(
            StatusCode::OK,
            &WebHealth {
                ok: true,
                version: env!("CARGO_PKG_VERSION"),
            },
        )),
        (&Method::GET, "/api/bootstrap") => {
            let runtime = state.client.health().await?;
            Ok(json_response(
                StatusCode::OK,
                &WebBootstrap {
                    web_version: env!("CARGO_PKG_VERSION"),
                    runtime: runtime.into(),
                },
            ))
        }
        (&Method::GET, "/api/agents") => api::list_agents(state).await,
        (&Method::GET, "/api/sessions") => api::list_sessions(&request, state).await,
        (&Method::POST, "/api/sessions") => api::create_session(request, state).await,
        (&Method::GET, "/api/events") => api::stream_events(&request, state).await,
        (_, path) if path.starts_with("/api/sessions/") => api::session_route(request, state).await,
        (_, "/api/healthz" | "/api/bootstrap") => Ok(text_response(
            StatusCode::METHOD_NOT_ALLOWED,
            "Method not allowed",
        )),
        (_, path) if path.starts_with("/api/") => {
            Ok(text_response(StatusCode::NOT_FOUND, "API route not found"))
        }
        (&Method::GET | &Method::HEAD, path) => {
            serve_asset(&state.assets_dir, path, request.method()).await
        }
        _ => Ok(text_response(
            StatusCode::METHOD_NOT_ALLOWED,
            "Method not allowed",
        )),
    }
}

async fn serve_asset(
    root: &Path,
    request_path: &str,
    method: &Method,
) -> Result<Response<WebBody>> {
    let Some(path) = asset_path(root, request_path) else {
        return Ok(text_response(StatusCode::BAD_REQUEST, "Invalid asset path"));
    };

    let selected = if tokio::fs::metadata(&path)
        .await
        .is_ok_and(|metadata| metadata.is_file())
    {
        path
    } else {
        root.join("200.html")
    };

    let bytes = match tokio::fs::read(&selected).await {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(text_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "Turin Web assets are not built. Run `bun run build` in crates/turin-web/frontend.",
            ));
        }
        Err(error) => {
            return Err(error)
                .with_context(|| format!("Failed to read web asset '{}'", selected.display()));
        }
    };

    let body = if method == Method::HEAD {
        Bytes::new()
    } else {
        Bytes::from(bytes)
    };
    let cache_control = if request_path.starts_with("/_app/immutable/") {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    };

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, content_type(&selected))
        .header(CACHE_CONTROL, cache_control)
        .body(Full::new(body).boxed_unsync())
        .expect("static response is valid"))
}

fn asset_path(root: &Path, request_path: &str) -> Option<PathBuf> {
    let relative = request_path.trim_start_matches('/');
    let relative = if relative.is_empty() {
        Path::new("index.html")
    } else {
        Path::new(relative)
    };

    if relative
        .components()
        .any(|component| !matches!(component, Component::Normal(_) | Component::CurDir))
    {
        return None;
    }
    Some(root.join(relative))
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("css") => "text/css; charset=utf-8",
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "application/javascript; charset=utf-8",
        Some("json") => "application/json; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("webp") => "image/webp",
        Some("woff2") => "font/woff2",
        _ => "application/octet-stream",
    }
}

pub(super) fn json_response<T: Serialize>(status: StatusCode, value: &T) -> Response<WebBody> {
    let body = serde_json::to_vec(value).expect("web response serialization is infallible");
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "application/json; charset=utf-8")
        .header(CACHE_CONTROL, "no-store")
        .body(Full::new(Bytes::from(body)).boxed_unsync())
        .expect("JSON response is valid")
}

pub(super) fn text_response(status: StatusCode, value: &'static str) -> Response<WebBody> {
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "text/plain; charset=utf-8")
        .header(CACHE_CONTROL, "no-store")
        .body(Full::new(Bytes::from_static(value.as_bytes())).boxed_unsync())
        .expect("text response is valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_paths_cannot_escape_the_build_root() {
        let root = Path::new("/tmp/turin-web");
        assert_eq!(asset_path(root, "/"), Some(root.join("index.html")));
        assert_eq!(
            asset_path(root, "/_app/immutable/app.js"),
            Some(root.join("_app/immutable/app.js"))
        );
        assert_eq!(asset_path(root, "/../secret"), None);
    }

    #[test]
    fn common_asset_content_types_are_explicit() {
        assert_eq!(
            content_type(Path::new("app.js")),
            "application/javascript; charset=utf-8"
        );
        assert_eq!(
            content_type(Path::new("app.css")),
            "text/css; charset=utf-8"
        );
        assert_eq!(content_type(Path::new("font.woff2")), "font/woff2");
    }
}
