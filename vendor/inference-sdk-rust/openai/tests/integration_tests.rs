use openai_sdk::{
    Client,
    types::chat::{ChatCompletionRequest, ChatContent, ChatMessage, ChatRole},
};
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn test_create_chat_completion() {
    let mock_server = MockServer::start().await;

    let response_body = json!({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            },
            "finish_reason": "stop",
            "logprobs": null
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        },
        "system_fingerprint": "fp_abc123"
    });

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_body))
        .mount(&mock_server)
        .await;

    let client = Client::from_config(
        openai_sdk::client::ClientConfig::new("test-key".to_string())
            .unwrap()
            .with_base_url(mock_server.uri()),
    )
    .unwrap();

    let request = ChatCompletionRequest::builder()
        .model("gpt-4o")
        .messages(vec![ChatMessage {
            role: ChatRole::User,
            content: Some(ChatContent::Text("Hello!".to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }])
        .build();

    let response = client
        .chat()
        .create(request)
        .await
        .expect("Failed to create chat completion");

    assert_eq!(response.id, "chatcmpl-123");
    assert_eq!(response.choices.len(), 1);
    assert_eq!(response.choices[0].finish_reason.as_deref(), Some("stop"));

    if let Some(ChatContent::Text(text)) = &response.choices[0].message.content {
        assert_eq!(text, "Hello! How can I help you today?");
    } else {
        panic!("Expected text content");
    }

    assert_eq!(response.usage.as_ref().unwrap().total_tokens, 21);
}

#[tokio::test]
async fn test_retry_on_429() {
    let mock_server = MockServer::start().await;

    let success_body = json!({
        "id": "chatcmpl-retry",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o",
        "choices": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    });

    // Fail once with 429, then succeed
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429))
        .up_to_n_times(1)
        .mount(&mock_server)
        .await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_body))
        .mount(&mock_server)
        .await;

    let client = Client::from_config(
        openai_sdk::client::ClientConfig::new("test-key".to_string())
            .unwrap()
            .with_base_url(mock_server.uri())
            .with_max_retries(2),
    )
    .unwrap();

    let request = ChatCompletionRequest::builder()
        .model("gpt-4o")
        .messages(vec![ChatMessage {
            role: ChatRole::User,
            content: Some(ChatContent::Text("Retry me".to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }])
        .build();

    let response = client
        .chat()
        .create(request)
        .await
        .expect("Should succeed after retry");
    assert_eq!(response.id, "chatcmpl-retry");
}

#[tokio::test]
async fn test_error_response_400() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(400).set_body_string("Bad request: invalid model"))
        .mount(&mock_server)
        .await;

    let client = Client::from_config(
        openai_sdk::client::ClientConfig::new("test-key".to_string())
            .unwrap()
            .with_base_url(mock_server.uri()),
    )
    .unwrap();

    let request = ChatCompletionRequest::builder()
        .model("invalid-model")
        .messages(vec![ChatMessage {
            role: ChatRole::User,
            content: Some(ChatContent::Text("Hi".to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }])
        .build();

    let result = client.chat().create(request).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("400"), "Error should contain status code 400");
}

#[tokio::test]
async fn test_retry_exhaustion() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .mount(&mock_server)
        .await;

    let client = Client::from_config(
        openai_sdk::client::ClientConfig::new("test-key".to_string())
            .unwrap()
            .with_base_url(mock_server.uri())
            .with_max_retries(1),
    )
    .unwrap();

    let request = ChatCompletionRequest::builder()
        .model("gpt-4o")
        .messages(vec![ChatMessage {
            role: ChatRole::User,
            content: Some(ChatContent::Text("Hi".to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }])
        .build();

    let result = client.chat().create(request).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("500"), "Error should contain status code 500");
}

#[tokio::test]
async fn test_debug_redacts_api_key() {
    let config =
        openai_sdk::client::ClientConfig::new("sk-super-secret-key-12345".to_string()).unwrap();
    let debug_output = format!("{:?}", config);
    assert!(
        !debug_output.contains("sk-super-secret-key-12345"),
        "Debug output should not contain the raw API key"
    );
    assert!(
        debug_output.contains("[REDACTED]"),
        "Debug output should show [REDACTED]"
    );
}
