# OpenAI Rust SDK

Unofficial Rust client for the OpenAI Chat Completions API, supporting GPT-4o, o1, and all current models.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
openai-sdk = { path = "openai" } # From workspace root
tokio = { version = "1.0", features = ["full"] }
serde_json = "1.0"
futures = "0.3"
```

## Usage

### Simple Completion

```rust
use openai_sdk::{Client, types::chat::{ChatCompletionRequest, ChatMessage, ChatRole, ChatContent}};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(std::env::var("OPENAI_API_KEY")?)?;

    let request = ChatCompletionRequest::builder()
        .model("gpt-4o-mini")
        .messages(vec![ChatMessage {
            role: ChatRole::User,
            content: Some(ChatContent::Text("Hello!".to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }])
        .max_tokens(1024_u32)
        .build();

    let response = client.chat().create(request).await?;

    for choice in response.choices {
        if let Some(ChatContent::Text(text)) = choice.message.content {
            println!("{}", text);
        }
    }
    Ok(())
}
```

### Streaming

```rust
use futures::StreamExt;
use openai_sdk::{Client, types::chat::{ChatCompletionRequest, ChatMessage, ChatRole, ChatContent}};

let mut stream = client.chat().create_stream(request).await?;

while let Some(chunk) = stream.next().await {
    match chunk {
        Ok(chunk) => {
            for choice in &chunk.choices {
                if let Some(content) = &choice.delta.content {
                    print!("{}", content);
                }
            }
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

### Tool Calls

```rust
use openai_sdk::types::chat::{Tool, FunctionDefinition};
use serde_json::json;

let request = ChatCompletionRequest::builder()
    .model("gpt-4o")
    .messages(messages)
    .tools(vec![Tool {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "get_weather".to_string(),
            description: Some("Get the weather for a city".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }),
            strict: None,
        },
    }])
    .build();
```

### Structured Outputs

```rust
use openai_sdk::types::chat::{ResponseFormat, JsonSchemaConfig};
use serde_json::json;

let request = ChatCompletionRequest::builder()
    .model("gpt-4o")
    .messages(messages)
    .response_format(ResponseFormat::JsonSchema {
        json_schema: JsonSchemaConfig {
            name: "person".to_string(),
            description: Some("A person record".to_string()),
            schema: json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "age": { "type": "integer" }
                },
                "required": ["name", "age"]
            }),
            strict: Some(true),
        },
    })
    .build();
```

## Request Options

Customize individual requests with `RequestOptions`:

```rust
use openai_sdk::RequestOptions;
use std::time::Duration;

let options = RequestOptions::new()
    .with_timeout(Duration::from_secs(120))
    .with_max_retries(5);

let response = client.chat().create_with_options(request, options).await?;
```

## Configuration

```rust
use openai_sdk::client::ClientConfig;
use openai_sdk::Client;
use std::time::Duration;

let config = ClientConfig::new(api_key)?
    .with_timeout(Duration::from_secs(30))
    .with_max_retries(3)
    .with_base_url("https://my-proxy.example.com/v1"); // Custom endpoint

let client = Client::from_config(config)?;
```

## Supported Features

| Feature | Status |
|---------|--------|
| Chat Completions | ✅ |
| Streaming | ✅ |
| Tool Calls | ✅ |
| Structured Outputs | ✅ |
| Multimodal (image URLs) | ✅ |
| Retry with backoff | ✅ |
| API key redaction | ✅ |
