#![no_main]

use anthropic_sdk::normalization::AnthropicStreamAdapter;
use anthropic_sdk::types::message::StreamEvent;
use futures_util::stream;
use inference_sdk_core::{InferenceEvent, InferenceResult, SdkError, validate_event_sequence};
use libfuzzer_sys::fuzz_target;
use openai_sdk::normalization::OpenAiStreamAdapter;
use openai_sdk::types::chat::ChatCompletionChunk;
use std::sync::OnceLock;
use tokio::runtime::{Builder, Runtime};

const MAX_INPUT_BYTES: usize = 16 * 1024;
const MAX_LINES: usize = 128;

fn runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build tokio runtime for fuzz target")
    })
}

fn process_openai_line(
    adapter: &mut OpenAiStreamAdapter,
    line: &str,
    output: &mut Vec<Result<InferenceEvent, SdkError>>,
) {
    if let Ok(chunk) = serde_json::from_str::<ChatCompletionChunk>(line) {
        output.extend(adapter.process_chunk(chunk));
    }
}

fn process_anthropic_line(
    adapter: &mut AnthropicStreamAdapter,
    line: &str,
    output: &mut Vec<Result<InferenceEvent, SdkError>>,
) {
    if let Ok(event) = serde_json::from_str::<StreamEvent>(line) {
        output.extend(adapter.process_event(event));
    }
}

fn validate_and_assemble(events: Vec<Result<InferenceEvent, SdkError>>) {
    let ok_events: Vec<InferenceEvent> = events
        .iter()
        .filter_map(|e| match e {
            Ok(event) => Some(event.clone()),
            Err(_) => None,
        })
        .collect();
    if !ok_events.is_empty() {
        let _ = validate_event_sequence(&ok_events);
    }

    // Ensure event assembly never panics for any adapter output shape.
    let stream = Box::pin(stream::iter(events));
    let _ = runtime().block_on(InferenceResult::from_stream(stream));
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let Ok(input) = std::str::from_utf8(data) else {
        return;
    };

    let mut openai_adapter = OpenAiStreamAdapter::new();
    let mut anthropic_adapter = AnthropicStreamAdapter::new();

    let mut openai_events = Vec::new();
    let mut anthropic_events = Vec::new();

    for line in input.lines().take(MAX_LINES) {
        process_openai_line(&mut openai_adapter, line, &mut openai_events);
        process_anthropic_line(&mut anthropic_adapter, line, &mut anthropic_events);
    }

    validate_and_assemble(openai_events);
    validate_and_assemble(anthropic_events);
});
