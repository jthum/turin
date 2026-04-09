---
name: OpenAI Rust SDK Skill
description: Instructions for agents on how to use the openai-sdk library to interact with OpenAI models.
---

# OpenAI Rust SDK Skill

This skill provides guidelines for using the `openai-sdk` Rust library.

## Dependencies

Add the following to your `Cargo.toml`:

```toml
[dependencies]
openai-sdk = { path = "openai" } # From workspace root
tokio = { version = "1", features = ["full"] }
serde_json = "1.0"
futures = "0.3"
```

## Core Concepts

### 1. Initialization
Initialize the client with an API key. `Client::new()` returns a `Result`.

```rust
use openai_sdk::Client;
let client = Client::new(api_key)?;
```

### 2. Constructing Requests (Builder Pattern)
Use the Builder pattern to create `ChatCompletionRequest`. String fields like `model` accept `&str` directly via `#[builder(into)]`.

```rust
use openai_sdk::types::chat::{ChatCompletionRequest, ChatMessage, ChatRole, ChatContent};

let request = ChatCompletionRequest::builder()
    .model("gpt-4o")
    .messages(vec![ChatMessage {
        role: ChatRole::User,
        content: Some(ChatContent::Text("Your prompt here".to_string())),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }])
    .max_tokens(1024_u32)
    .build();
```

### 3. Sending Requests
Use `client.chat().create(request)` for standard responses or `create_stream(request)` for streaming.

```rust
let response = client.chat().create(request).await?;
```

### 4. Handling Responses
Responses contain a list of `Choice`s, each with a `message`.

```rust
use openai_sdk::types::chat::ChatContent;

for choice in response.choices {
    if let Some(ChatContent::Text(text)) = choice.message.content {
        println!("{}", text);
    }
}
```

### 5. Streaming
Stream responses and process chunks incrementally.

```rust
use futures::StreamExt;

let mut stream = client.chat().create_stream(request).await?;

while let Some(chunk_result) = stream.next().await {
    match chunk_result {
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

### 6. Tool Calls
The SDK supports OpenAI's function calling. Check for `tool_calls` in the response message.

```rust
if let Some(tool_calls) = &response.choices[0].message.tool_calls {
    for tc in tool_calls {
        println!("Call: {} with args: {}", tc.function.name, tc.function.arguments);
    }
}
```

### 7. System Messages
Use `ChatRole::System` to set the system prompt.

```rust
let messages = vec![
    ChatMessage {
        role: ChatRole::System,
        content: Some(ChatContent::Text("You are a helpful assistant.".to_string())),
        name: None, tool_calls: None, tool_call_id: None,
    },
    ChatMessage {
        role: ChatRole::User,
        content: Some(ChatContent::Text("Hello!".to_string())),
        name: None, tool_calls: None, tool_call_id: None,
    },
];
```

## Best Practices for Agents

1. **Always use the Builder**: Avoid constructing `ChatCompletionRequest` fields manually.
2. **Handle Errors**: The SDK uses `SdkError`. Handle `ApiError` (4xx/5xx) appropriately.
3. **Streaming**: For long-running tasks, prefer streaming to provide real-time feedback.
4. **System Messages**: Use `ChatRole::System` to define agent persona and instructions.
5. **Tool Calls**: For agentic workflows, use the `tools` field to define available functions and handle `tool_calls` in the response loop.
