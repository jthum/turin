mod support;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde_json::json;
use tempfile::TempDir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio::time::{Instant, sleep, timeout};
use turin_channel_core::{
    ChannelConversationKey, ChannelKind, ChannelMessageRef, ChannelSessionScope, ChannelUser,
    InboundEvent,
};
use turin_channel_runner::{ChannelDriver, ChannelProgressUpdate};
use turin_channel_runner::{ChannelRunner, RunnerConfig};
use turin_channel_telegram::{TelegramChannelDriver, TelegramChannelDriverConfig};

struct DaemonHarness {
    _tempdir: Arc<TempDir>,
    workspace_root: PathBuf,
    endpoint: PathBuf,
    join: JoinHandle<Result<()>>,
}

struct TelegramMockServer {
    base_url: String,
    sent_messages: Arc<Mutex<Vec<serde_json::Value>>>,
    requests: Arc<Mutex<Vec<TelegramRequestRecord>>>,
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    join: JoinHandle<Result<()>>,
}

#[derive(Debug, Clone)]
struct TelegramRequestRecord {
    method: String,
    body: serde_json::Value,
}

struct TelegramMockState {
    get_updates_responses: VecDeque<serde_json::Value>,
    send_message_responses: VecDeque<serde_json::Value>,
    edit_message_responses: VecDeque<serde_json::Value>,
    sent_messages: Vec<serde_json::Value>,
    requests: Vec<TelegramRequestRecord>,
}

impl DaemonHarness {
    async fn start() -> Result<Self> {
        let tempdir = Arc::new(tempfile::tempdir()?);
        let workspace_root = tempdir.path().join("workspace");
        let config_path = support::write_mock_runtime_config(
            &workspace_root,
            "Telegram channel integration",
            "PONG",
        )?;
        std::fs::create_dir_all(support::channel_runtime_dir(&workspace_root, "telegram"))?;
        let endpoint = support::workspace_daemon_socket(&workspace_root);
        let serve_config_path = config_path.clone();
        let join =
            tokio::spawn(async move { turin::daemon::server::serve(&serve_config_path).await });

        let deadline = Instant::now() + Duration::from_secs(10);
        let client = turin_daemon_client::DaemonClient::new(&endpoint);
        loop {
            if client.handshake().await.is_ok() {
                break;
            }
            if join.is_finished() {
                let result = join
                    .await
                    .context("daemon task join failed before endpoint bind")?;
                return Err(result
                    .err()
                    .unwrap_or_else(|| anyhow!("daemon exited before creating daemon endpoint")));
            }
            if Instant::now() >= deadline {
                join.abort();
                return Err(anyhow!(
                    "Timed out waiting for daemon endpoint '{}'",
                    endpoint.display()
                ));
            }
            sleep(Duration::from_millis(25)).await;
        }

        Ok(Self {
            _tempdir: tempdir,
            workspace_root,
            endpoint,
            join,
        })
    }

    fn runner(&self) -> ChannelRunner {
        ChannelRunner::new(
            turin_daemon_client::DaemonClient::new(&self.endpoint),
            RunnerConfig {
                channel_id: "telegram".to_string(),
                state_path: support::channel_runtime_dir(&self.workspace_root, "telegram")
                    .join("bindings.json"),
                access_state_path: support::channel_runtime_dir(&self.workspace_root, "telegram")
                    .join("access.json"),
                idle_ttl: Some(Duration::from_secs(600)),
                access_policy: Default::default(),
                tools: Default::default(),
            },
        )
    }

    async fn stop(self) -> Result<()> {
        let client = turin_daemon_client::DaemonClient::new(&self.endpoint);
        let _: serde_json::Value = client
            .request_ok(
                None,
                turin_daemon_protocol::DaemonRequest::DaemonStop(Default::default()),
            )
            .await?;
        let _ = timeout(Duration::from_secs(5), self.join)
            .await
            .context("timed out waiting for daemon to exit")??;
        Ok(())
    }
}

impl TelegramMockServer {
    async fn start_with_responses(
        get_updates_responses: Vec<serde_json::Value>,
        send_message_responses: Vec<serde_json::Value>,
        edit_message_responses: Vec<serde_json::Value>,
    ) -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let base_url = format!("http://{}", addr);
        let sent_messages = Arc::new(Mutex::new(Vec::new()));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let state = Arc::new(Mutex::new(TelegramMockState {
            get_updates_responses: get_updates_responses.into(),
            send_message_responses: send_message_responses.into(),
            edit_message_responses: edit_message_responses.into(),
            sent_messages: Vec::new(),
            requests: Vec::new(),
        }));
        let sent_messages_for_task = Arc::clone(&sent_messages);
        let requests_for_task = Arc::clone(&requests);
        let state_for_task = Arc::clone(&state);
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::watch::channel(false);

        let join = tokio::spawn(async move {
            loop {
                tokio::select! {
                    changed = shutdown_rx.changed() => {
                        if changed.is_ok() && *shutdown_rx.borrow() {
                            break;
                        }
                    }
                    accepted = listener.accept() => {
                        let (mut stream, _) = accepted?;
                        let (path, body) = read_http_request(&mut stream).await?;
                        let response = handle_telegram_request(&path, body, &state_for_task, &sent_messages_for_task, &requests_for_task)?;
                        write_http_response(&mut stream, &response).await?;
                    }
                }
            }
            Result::<()>::Ok(())
        });

        Ok(Self {
            base_url,
            sent_messages,
            requests,
            shutdown_tx,
            join,
        })
    }

    async fn stop(self) -> Result<()> {
        let _ = self.shutdown_tx.send(true);
        let _ = timeout(Duration::from_secs(5), self.join)
            .await
            .context("timed out waiting for telegram mock server shutdown")??;
        Ok(())
    }
}

async fn wait_for_telegram_outbound(
    server: &TelegramMockServer,
    run: &mut JoinHandle<Result<()>>,
) -> Result<serde_json::Value> {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if let Some(first) = server
            .sent_messages
            .lock()
            .expect("telegram mock sent_messages lock poisoned")
            .first()
            .cloned()
        {
            return Ok(first);
        }
        if run.is_finished() {
            match timeout(Duration::from_secs(1), run).await {
                Ok(Ok(Ok(()))) => {
                    let methods = server
                        .requests
                        .lock()
                        .expect("telegram mock requests lock poisoned")
                        .iter()
                        .map(|request| request.method.clone())
                        .collect::<Vec<_>>();
                    anyhow::bail!(
                        "telegram channel runner exited before outbound response; requests: {:?}",
                        methods
                    );
                }
                Ok(Ok(Err(err))) => {
                    return Err(err.context("telegram channel runner exited before outbound response"));
                }
                Ok(Err(join_err)) => {
                    return Err(anyhow!("telegram channel runner join failed: {join_err}"));
                }
                Err(_) => anyhow::bail!(
                    "timed out joining telegram channel runner after early exit"
                ),
            }
        }
        if Instant::now() >= deadline {
            let methods = server
                .requests
                .lock()
                .expect("telegram mock requests lock poisoned")
                .iter()
                .map(|request| request.method.clone())
                .collect::<Vec<_>>();
            anyhow::bail!(
                "telegram channel did not produce outbound response; requests: {:?}",
                methods
            );
        }
        sleep(Duration::from_millis(25)).await;
    }
}

fn handle_telegram_request(
    path: &str,
    body: serde_json::Value,
    state: &Arc<Mutex<TelegramMockState>>,
    sent_messages: &Arc<Mutex<Vec<serde_json::Value>>>,
    requests: &Arc<Mutex<Vec<TelegramRequestRecord>>>,
) -> Result<serde_json::Value> {
    let method = path.rsplit('/').next().unwrap_or_default();
    let record = TelegramRequestRecord {
        method: method.to_string(),
        body: body.clone(),
    };
    {
        let mut guard = state.lock().expect("telegram mock state lock poisoned");
        guard.requests.push(record.clone());
    }
    requests
        .lock()
        .expect("telegram mock requests lock poisoned")
        .push(record);
    match method {
        "getUpdates" => {
            let response = state
                .lock()
                .expect("telegram mock state lock poisoned")
                .get_updates_responses
                .pop_front()
                .unwrap_or_else(|| json!({ "ok": true, "result": [] }));
            Ok(response)
        }
        "sendMessage" => {
            let response = {
                let mut guard = state.lock().expect("telegram mock state lock poisoned");
                guard.sent_messages.push(body.clone());
                guard.send_message_responses.pop_front().unwrap_or_else(|| {
                    json!({
                        "ok": true,
                        "result": {
                            "message_id": 1
                        }
                    })
                })
            };
            sent_messages
                .lock()
                .expect("telegram mock sent_messages lock poisoned")
                .push(body.clone());
            Ok(response)
        }
        "sendMessageDraft" => Ok(json!({
            "ok": true,
            "result": true
        })),
        "editMessageText" => {
            let response = state
                .lock()
                .expect("telegram mock state lock poisoned")
                .edit_message_responses
                .pop_front()
                .unwrap_or_else(|| {
                    json!({
                        "ok": true,
                        "result": {
                            "message_id": 1
                        }
                    })
                });
            Ok(response)
        }
        "sendChatAction" => Ok(json!({
            "ok": true,
            "result": true
        })),
        _ => Ok(json!({
            "ok": false,
            "error_code": 404,
            "description": format!("unknown Telegram method for path '{}'", path)
        })),
    }
}

async fn read_http_request(
    stream: &mut tokio::net::TcpStream,
) -> Result<(String, serde_json::Value)> {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 2048];
    let header_end = loop {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            return Err(anyhow!("telegram mock server received empty request"));
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(index) = find_header_end(&buffer) {
            break index;
        }
    };

    let header = String::from_utf8(buffer[..header_end].to_vec())
        .context("telegram mock server request header must be utf-8")?;
    let content_length = header
        .lines()
        .find_map(|line| {
            line.strip_prefix("Content-Length:")
                .or_else(|| line.strip_prefix("content-length:"))
                .map(str::trim)
        })
        .map(str::parse::<usize>)
        .transpose()?
        .unwrap_or(0);

    while buffer.len() < header_end + 4 + content_length {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
    }

    let request_line = header
        .lines()
        .next()
        .context("telegram mock server missing request line")?;
    let path = request_line
        .split_whitespace()
        .nth(1)
        .context("telegram mock server missing request path")?
        .to_string();

    let body_start = header_end + 4;
    let body = if content_length == 0 {
        serde_json::json!({})
    } else {
        serde_json::from_slice(&buffer[body_start..body_start + content_length])
            .context("telegram mock server body must be json")?
    };

    Ok((path, body))
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

async fn write_http_response(
    stream: &mut tokio::net::TcpStream,
    body: &serde_json::Value,
) -> Result<()> {
    let body = serde_json::to_vec(body)?;
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(response.as_bytes()).await?;
    stream.write_all(&body).await?;
    Ok(())
}

fn sample_update(chat_id: i64, message_thread_id: Option<i64>, text: &str) -> serde_json::Value {
    json!({
        "update_id": 1,
        "message": {
            "message_id": 41,
            "message_thread_id": message_thread_id,
            "chat": {
                "id": chat_id,
                "title": "Ops Chat",
                "type": if chat_id < 0 { "supergroup" } else { "private" }
            },
            "from": {
                "id": 7,
                "is_bot": false,
                "first_name": "Nina",
                "username": "nina"
            },
            "text": text
        }
    })
}

fn sample_inbound_event(chat_id: i64, text: &str) -> InboundEvent {
    let conversation = ChannelConversationKey {
        channel: ChannelKind::new("telegram"),
        workspace_id: "telegram".to_string(),
        room_id: Some(chat_id.to_string()),
        thread_id: chat_id.to_string(),
        user_id: Some("7".to_string()),
    };
    InboundEvent {
        message: ChannelMessageRef {
            conversation: conversation.clone(),
            message_id: "41".to_string(),
        },
        conversation,
        user: ChannelUser {
            id: "7".to_string(),
            display_name: Some("Nina".to_string()),
            username: Some("nina".to_string()),
        },
        session_scope: ChannelSessionScope::User,
        text: text.to_string(),
        attachments: Vec::new(),
        metadata: json!({
            "telegram_message_id": 41,
            "telegram_chat_id": chat_id
        })
        .as_object()
        .cloned()
        .unwrap_or_default(),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn telegram_channel_driver_round_trip_with_daemon_runner() -> Result<()> {
    let daemon = DaemonHarness::start().await?;
    let runner = daemon.runner();
    let server = TelegramMockServer::start_with_responses(
        vec![json!({ "ok": true, "result": [sample_update(-100777, Some(555), "Say pong")] })],
        vec![json!({
            "ok": true,
            "result": {
                "message_id": 5,
                "from": {
                    "id": 8702474519_i64,
                    "is_bot": true,
                    "first_name": "Turin",
                    "username": "the_turin_bot"
                },
                "chat": {
                    "id": 498502840_i64,
                    "first_name": "Jayadeep",
                    "last_name": "Thum",
                    "username": "jthum",
                    "type": "private"
                },
                "date": 1774430415_i64,
                "reply_to_message": {
                    "message_id": 41,
                    "from": {
                        "id": 498502840_i64,
                        "is_bot": false,
                        "first_name": "Jayadeep",
                        "last_name": "Thum",
                        "username": "jthum",
                        "language_code": "en"
                    },
                    "chat": {
                        "id": 498502840_i64,
                        "first_name": "Jayadeep",
                        "last_name": "Thum",
                        "username": "jthum",
                        "type": "private"
                    },
                    "date": 1774430411_i64,
                    "text": "Hello."
                },
                "text": "PONG"
            }
        })],
        vec![],
    )
    .await?;

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let mut driver = TelegramChannelDriver::from_config(
        "telegram-test",
        TelegramChannelDriverConfig {
            base_url: server.base_url.clone(),
            workspace_id: "telegram".to_string(),
            chat_ids: vec!["-100777".to_string()],
            accept_all_chats: false,
            token: "test-token".to_string(),
            poll_timeout_secs: 0,
            poll_interval: Duration::from_millis(25),
            max_updates_per_poll: 10,
            start_from_latest: false,
            ignore_bot_messages: true,
            respond_mode: turin_channel_telegram::TelegramRespondMode::All,
            session_scope: ChannelSessionScope::User,
            session_scope_dm: None,
            session_scope_group: None,
            session_scope_channel: None,
            stream_mode: turin_channel_runner::ChannelStreamMode::Off,
            stream_thinking: false,
            persist_thinking: false,
        },
        shutdown_rx,
    )?;

    let mut run =
        tokio::spawn(async move { runner.run_driver("default", &mut driver, Some(5_000)).await });
    let outbound = wait_for_telegram_outbound(&server, &mut run).await?;
    assert_eq!(outbound["chat_id"], "-100777");
    assert_eq!(outbound["message_thread_id"], 555);
    assert_eq!(outbound["reply_to_message_id"], 41);
    assert!(
        outbound["text"]
            .as_str()
            .is_some_and(|text| text.contains("PONG")),
        "outbound payload: {}",
        outbound
    );

    let binding_state = tokio::fs::read_to_string(
        support::channel_runtime_dir(&daemon.workspace_root, "telegram").join("bindings.json"),
    )
    .await
    .context("telegram channel binding state should exist")?;
    let binding_state: serde_json::Value = serde_json::from_str(&binding_state)?;
    let binding_keys = binding_state["bindings"]
        .as_object()
        .context("telegram channel bindings should be an object")?;
    assert!(binding_keys.keys().any(|key| {
        key.contains("\"channel\":\"telegram\"") && key.contains("\"thread_id\":\"555\"")
    }));

    let _ = shutdown_tx.send(true);
    let _ = timeout(Duration::from_secs(5), run)
        .await
        .context("timed out waiting for telegram channel runner shutdown")??;

    server.stop().await?;
    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn telegram_channel_driver_retries_transient_poll_and_send_failures() -> Result<()> {
    let daemon = DaemonHarness::start().await?;
    let runner = daemon.runner();
    let server = TelegramMockServer::start_with_responses(
        vec![
            json!({
                "ok": false,
                "error_code": 429,
                "description": "Too Many Requests: retry later",
                "parameters": { "retry_after": 0 }
            }),
            json!({ "ok": true, "result": [sample_update(-100777, Some(555), "Say pong")] }),
        ],
        vec![
            json!({
                "ok": false,
                "error_code": 429,
                "description": "Too Many Requests: retry later",
                "parameters": { "retry_after": 0 }
            }),
            json!({
                "ok": true,
                "result": { "message_id": 2 }
            }),
        ],
        vec![],
    )
    .await?;

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let mut driver = TelegramChannelDriver::from_config(
        "telegram-test",
        TelegramChannelDriverConfig {
            base_url: server.base_url.clone(),
            workspace_id: "telegram".to_string(),
            chat_ids: vec!["-100777".to_string()],
            accept_all_chats: false,
            token: "test-token".to_string(),
            poll_timeout_secs: 0,
            poll_interval: Duration::from_millis(25),
            max_updates_per_poll: 10,
            start_from_latest: false,
            ignore_bot_messages: true,
            respond_mode: turin_channel_telegram::TelegramRespondMode::All,
            session_scope: ChannelSessionScope::User,
            session_scope_dm: None,
            session_scope_group: None,
            session_scope_channel: None,
            stream_mode: turin_channel_runner::ChannelStreamMode::Off,
            stream_thinking: false,
            persist_thinking: false,
        },
        shutdown_rx,
    )?;

    let mut run =
        tokio::spawn(async move { runner.run_driver("default", &mut driver, Some(5_000)).await });
    let _ = wait_for_telegram_outbound(&server, &mut run).await?;

    let _ = shutdown_tx.send(true);
    let _ = timeout(Duration::from_secs(5), run)
        .await
        .context("timed out waiting for telegram channel runner shutdown")??;

    server.stop().await?;
    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn telegram_channel_driver_streams_progress_before_final_message() -> Result<()> {
    let daemon = DaemonHarness::start().await?;
    let runner = daemon.runner();
    let server = TelegramMockServer::start_with_responses(
        vec![json!({
            "ok": true,
            "result": [sample_update(498502840, None, "Say pong")]
        })],
        vec![json!({
            "ok": true,
            "result": {
                "message_id": 5,
                "chat": {
                    "id": 498502840_i64,
                    "first_name": "Jayadeep",
                    "type": "private"
                },
                "date": 1774430415_i64,
                "text": "PONG"
            }
        })],
        vec![],
    )
    .await?;

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let mut driver = TelegramChannelDriver::from_config(
        "telegram-test",
        TelegramChannelDriverConfig {
            base_url: server.base_url.clone(),
            workspace_id: "telegram".to_string(),
            chat_ids: vec!["498502840".to_string()],
            accept_all_chats: false,
            token: "test-token".to_string(),
            poll_timeout_secs: 0,
            poll_interval: Duration::from_millis(25),
            max_updates_per_poll: 10,
            start_from_latest: false,
            ignore_bot_messages: true,
            respond_mode: turin_channel_telegram::TelegramRespondMode::All,
            session_scope: ChannelSessionScope::User,
            session_scope_dm: None,
            session_scope_group: None,
            session_scope_channel: None,
            stream_mode: turin_channel_runner::ChannelStreamMode::Draft,
            stream_thinking: false,
            persist_thinking: false,
        },
        shutdown_rx,
    )?;

    let run =
        tokio::spawn(async move { runner.run_driver("default", &mut driver, Some(5_000)).await });

    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        let methods = server
            .requests
            .lock()
            .expect("telegram mock requests lock poisoned")
            .iter()
            .map(|request| request.method.clone())
            .collect::<Vec<_>>();
        if methods.iter().any(|method| method == "sendMessage")
            && methods
                .iter()
                .any(|method| method == "sendMessageDraft" || method == "editMessageText")
            && methods.iter().any(|method| method == "sendChatAction")
        {
            break;
        }
        sleep(Duration::from_millis(25)).await;
    }

    let methods = server
        .requests
        .lock()
        .expect("telegram mock requests lock poisoned")
        .iter()
        .map(|request| request.method.clone())
        .collect::<Vec<_>>();
    assert!(
        methods.iter().any(|method| method == "sendChatAction"),
        "expected typing action in request log: {methods:?}"
    );
    assert!(
        methods
            .iter()
            .any(|method| method == "sendMessageDraft" || method == "editMessageText"),
        "expected streaming preview request in request log: {methods:?}"
    );
    assert!(
        methods.iter().any(|method| method == "sendMessage"),
        "expected final sendMessage in request log: {methods:?}"
    );
    let draft_or_edit = server
        .requests
        .lock()
        .expect("telegram mock requests lock poisoned")
        .iter()
        .find(|request| request.method == "sendMessageDraft" || request.method == "editMessageText")
        .cloned()
        .context("expected draft or edit request")?;
    assert!(
        draft_or_edit
            .body
            .get("text")
            .and_then(|value| value.as_str())
            .is_some_and(|text| text.contains("PONG")),
        "expected streamed preview text in request body: {:?}",
        draft_or_edit.body
    );

    let _ = shutdown_tx.send(true);
    let _ = timeout(Duration::from_secs(5), run)
        .await
        .context("timed out waiting for telegram channel runner shutdown")??;

    server.stop().await?;
    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn telegram_progress_preview_can_include_thinking_text() -> Result<()> {
    let server = TelegramMockServer::start_with_responses(vec![], vec![], vec![]).await?;
    let (_shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let mut driver = TelegramChannelDriver::from_config(
        "telegram-test",
        TelegramChannelDriverConfig {
            base_url: server.base_url.clone(),
            workspace_id: "telegram".to_string(),
            chat_ids: vec!["498502840".to_string()],
            accept_all_chats: false,
            token: "test-token".to_string(),
            poll_timeout_secs: 0,
            poll_interval: Duration::from_millis(25),
            max_updates_per_poll: 10,
            start_from_latest: false,
            ignore_bot_messages: true,
            respond_mode: turin_channel_telegram::TelegramRespondMode::All,
            session_scope: ChannelSessionScope::User,
            session_scope_dm: None,
            session_scope_group: None,
            session_scope_channel: None,
            stream_mode: turin_channel_runner::ChannelStreamMode::Draft,
            stream_thinking: true,
            persist_thinking: false,
        },
        shutdown_rx,
    )?;

    let event = sample_inbound_event(498502840, "Say pong");
    driver
        .send_progress(
            &event,
            ChannelProgressUpdate::StreamingPreview {
                text: "Partial answer".to_string(),
                thinking: Some("Reasoning step".to_string()),
            },
        )
        .await?;

    let draft_or_edit = server
        .requests
        .lock()
        .expect("telegram mock requests lock poisoned")
        .iter()
        .find(|request| request.method == "sendMessageDraft" || request.method == "editMessageText")
        .cloned()
        .context("expected draft or edit request")?;
    let preview = draft_or_edit
        .body
        .get("text")
        .and_then(|value| value.as_str())
        .context("expected preview text in request body")?;
    assert!(preview.contains("Thinking"), "preview text: {preview}");
    assert!(
        preview.contains("Reasoning step"),
        "preview text: {preview}"
    );
    assert!(
        preview.contains("Partial answer"),
        "preview text: {preview}"
    );

    server.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn telegram_block_stream_finalization_ignores_not_modified_edit_errors() -> Result<()> {
    let server = TelegramMockServer::start_with_responses(
        vec![],
        vec![json!({
            "ok": true,
            "result": {
                "message_id": 11
            }
        })],
        vec![json!({
            "ok": false,
            "error_code": 400,
            "description": "Bad Request: message is not modified: specified new message content and reply markup are exactly the same as a current content and reply markup of the message"
        })],
    )
    .await?;
    let (_shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let mut driver = TelegramChannelDriver::from_config(
        "telegram-test",
        TelegramChannelDriverConfig {
            base_url: server.base_url.clone(),
            workspace_id: "telegram".to_string(),
            chat_ids: vec!["498502840".to_string()],
            accept_all_chats: false,
            token: "test-token".to_string(),
            poll_timeout_secs: 0,
            poll_interval: Duration::from_millis(25),
            max_updates_per_poll: 10,
            start_from_latest: false,
            ignore_bot_messages: true,
            respond_mode: turin_channel_telegram::TelegramRespondMode::All,
            session_scope: ChannelSessionScope::User,
            session_scope_dm: None,
            session_scope_group: None,
            session_scope_channel: None,
            stream_mode: turin_channel_runner::ChannelStreamMode::Block,
            stream_thinking: false,
            persist_thinking: false,
        },
        shutdown_rx,
    )?;

    let event = sample_inbound_event(498502840, "Say pong");
    driver
        .send_progress(
            &event,
            ChannelProgressUpdate::StreamingPreview {
                text: "PONG".to_string(),
                thinking: None,
            },
        )
        .await?;
    driver
        .send(
            &event.conversation,
            turin_channel_core::OutboundMessage::text("PONG"),
        )
        .await?;

    let requests = server
        .requests
        .lock()
        .expect("telegram mock requests lock poisoned")
        .clone();
    let send_count = requests
        .iter()
        .filter(|request| request.method == "sendMessage")
        .count();
    let edit_count = requests
        .iter()
        .filter(|request| request.method == "editMessageText")
        .count();

    assert_eq!(send_count, 1, "request log: {requests:#?}");
    assert_eq!(edit_count, 1, "request log: {requests:#?}");

    server.stop().await
}
