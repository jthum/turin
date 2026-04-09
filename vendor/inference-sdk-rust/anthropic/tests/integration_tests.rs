use anthropic_sdk::AnthropicRequestExt;
use anthropic_sdk::RequestOptions;
use anthropic_sdk::{
    Client, ClientConfig,
    types::message::{Content, ContentBlock, Message, MessageRequest, Role},
};
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn test_create_message() {
    let mock_server: MockServer = MockServer::start().await;

    let expected_body = json!({
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{
            "type": "text",
            "text": "Hello, world!"
        }],
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5
        }
    });

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(expected_body))
        .mount(&mock_server)
        .await;

    let config = ClientConfig::new("test-key".to_string())
        .unwrap()
        .with_base_url(mock_server.uri());

    let client = Client::from_config(config).unwrap();

    let request = MessageRequest::builder()
        .model("claude-3-opus-20240229")
        .messages(vec![Message {
            role: Role::User,
            content: Content::Text("Hi".to_string()),
        }])
        .build();

    let response = client
        .messages()
        .create(request)
        .await
        .expect("Failed to create message");

    assert_eq!(response.id, "msg_123");
    match &response.content[0] {
        ContentBlock::Text { text } => assert_eq!(text, "Hello, world!"),
        _ => panic!("Unexpected content type"),
    }
}

#[tokio::test]
async fn test_retry_on_500() {
    let mock_server: MockServer = MockServer::start().await;

    let success_body = json!({
        "id": "msg_retry",
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens":0,"output_tokens":0}
    });

    // Fail once with 500, then succeed
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(500))
        .up_to_n_times(1)
        .mount(&mock_server)
        .await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_body))
        .mount(&mock_server)
        .await;

    let config = ClientConfig::new("test-key".to_string())
        .unwrap()
        .with_base_url(mock_server.uri())
        .with_max_retries(2);

    let client = Client::from_config(config).unwrap();
    let request = MessageRequest::builder()
        .model("claude-3-opus-20240229")
        .messages(vec![Message {
            role: Role::User,
            content: Content::Text("Retry me".to_string()),
        }])
        .build();

    let response = client
        .messages()
        .create(request)
        .await
        .expect("Should succeed after retry");
    assert_eq!(response.id, "msg_retry");
}

#[tokio::test]
async fn test_request_options() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("custom-header", "custom-value"))
        .and(header("anthropic-beta", "test-beta"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_options",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude-3-opus-20240229",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens":0,"output_tokens":0}
        })))
        .mount(&mock_server)
        .await;

    let config = ClientConfig::new("test-key".to_string())
        .unwrap()
        .with_base_url(mock_server.uri());

    let client = Client::from_config(config).unwrap();

    let request = MessageRequest::builder()
        .model("claude-3-opus-20240229")
        .messages(vec![Message {
            role: Role::User,
            content: Content::Text("Hi".to_string()),
        }])
        .build();

    let options = RequestOptions::new()
        .with_header("custom-header", "custom-value")
        .unwrap()
        .beta("test-beta")
        .unwrap();

    let _ = client
        .messages()
        .create_with_options(request, options)
        .await
        .expect("Failed to create message with options");
}

#[tokio::test]
async fn test_error_response_400() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(400).set_body_string("Bad request: invalid model"))
        .mount(&mock_server)
        .await;

    let config = ClientConfig::new("test-key".to_string())
        .unwrap()
        .with_base_url(mock_server.uri());
    let client = Client::from_config(config).unwrap();

    let request = MessageRequest::builder()
        .model("invalid-model")
        .messages(vec![Message {
            role: Role::User,
            content: Content::Text("Hi".to_string()),
        }])
        .build();

    let result = client.messages().create(request).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("400"), "Error should contain status code 400");
}

#[tokio::test]
async fn test_retry_exhaustion() {
    let mock_server = MockServer::start().await;

    // Always return 500 — retries should eventually give up
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .mount(&mock_server)
        .await;

    let config = ClientConfig::new("test-key".to_string())
        .unwrap()
        .with_base_url(mock_server.uri())
        .with_max_retries(1);
    let client = Client::from_config(config).unwrap();

    let request = MessageRequest::builder()
        .model("claude-3-opus-20240229")
        .messages(vec![Message {
            role: Role::User,
            content: Content::Text("Hi".to_string()),
        }])
        .build();

    let result = client.messages().create(request).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("500"), "Error should contain status code 500");
}

#[tokio::test]
async fn test_debug_redacts_api_key() {
    let config = ClientConfig::new("sk-ant-super-secret-key-12345".to_string()).unwrap();
    let debug_output = format!("{:?}", config);
    assert!(
        !debug_output.contains("sk-ant-super-secret-key-12345"),
        "Debug output should not contain the raw API key"
    );
    assert!(
        debug_output.contains("[REDACTED]"),
        "Debug output should show [REDACTED]"
    );
}
