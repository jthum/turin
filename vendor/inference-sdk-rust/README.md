# Inference SDK (Rust)

A modular Rust workspace for interacting with LLM inference APIs. Each provider gets its own crate, built on a shared core that provides a **Capability Normalizer** layer.

## Crates

| Crate | Description |
|-------|-------------|
| [`inference-sdk-core`](core/) | **Normalization Layer**: Traits (`InferenceProvider`, `EmbeddingProvider`), standardized inference/embedding types, and shared logic. |
| [`anthropic-sdk`](anthropic/) | Anthropic Messages API implementation. |
| [`openai-sdk`](openai/) | OpenAI Chat & Embeddings API implementation. |
| [`inference-sdk-registry`](registry/) | Provider driver registry/factory for building inference and embedding providers from generic driver config. |

## Normalization Layer

The SDK provides unified provider traits so you can write provider-agnostic inference and embedding code.

For inference, use `InferenceProvider`:

```rust
use inference_sdk_core::{
    InferenceContent, InferenceMessage, InferenceProvider, InferenceRequest, InferenceRole,
    SdkError,
};
use std::sync::Arc;

async fn run_inference(provider: Arc<dyn InferenceProvider>, prompt: &str) -> Result<(), SdkError> {
    let request = InferenceRequest::builder()
        .model("model-name")
        .system("Standardized system prompt")
        .messages(vec![InferenceMessage {
            role: InferenceRole::User,
            content: vec![InferenceContent::Text { text: prompt.to_string() }],
            tool_call_id: None,
        }])
        .build();

    // Works for both OpenAI and Anthropic!
    let result = provider.complete(request, None).await?;
    println!("Response: {}", result.text());
    
    if let Some(reason) = result.stop_reason {
        println!("Stop Reason: {:?}", reason);
    }
    
    Ok(())
}
```

## Quick Start

### 1. Simple Completion (Unified)

```rust
use anthropic_sdk::Client; // or openai_sdk::Client
use inference_sdk_core::{InferenceProvider, InferenceRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(std::env::var("API_KEY")?)?;
    
    let request = InferenceRequest::builder()
        .model("claude-3-5-sonnet-20241022")
        .system("You are a helpful assistant")
        .messages(vec![/* ... */])
        .build();

    let response = client.complete(request, None).await?; // Standardized method
    println!("{}", response.text());
    Ok(())
}
```

`InferenceResult::text()` returns only normalized text blocks. If you need tool calls or thinking blocks, inspect `InferenceResult::content` directly.

### 2. Streaming (Unified)

```rust
use openai_sdk::Client;
use inference_sdk_core::{InferenceProvider, InferenceRequest, InferenceEvent};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(std::env::var("OPENAI_API_KEY")?)?;
    let request = InferenceRequest::builder().model("gpt-4o").messages(vec![/*...*/]).build();

    let mut stream = client.stream(request, None).await?;
    while let Some(event_res) = stream.next().await {
        match event_res? {
            InferenceEvent::MessageDelta { content } => print!("{}", content),
            InferenceEvent::MessageEnd { .. } => println!("\nDone."),
            _ => {}
        }
    }
    Ok(())
}
```

## Architecture

```
inference-sdk-rust/
├── core/        → Normalization Layer: inference + embeddings traits and shared types.
├── anthropic/   → Implementation of InferenceProvider for Claude.
├── openai/      → Implementation of InferenceProvider for GPT/Embeddings.
└── registry/    → Driver registry/factory for inference and embedding provider instantiation.
```

## Quality and Contract Docs

- Stream/Event Contract: `docs/STREAM_EVENT_CONTRACT.md`
- Provider Implementation Guide: `docs/PROVIDER_IMPLEMENTATION_GUIDE.md`
- Fuzzing Guide: `docs/FUZZING.md`
- Performance Guards: `docs/PERFORMANCE_GUARDS.md`
- Migration Notes: `docs/MIGRATIONS.md`

## License

MIT
