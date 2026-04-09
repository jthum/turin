#![no_main]

use futures_util::stream;
use inference_sdk_core::{InferenceEvent, InferenceResult, SdkError, StopReason};
use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;
use tokio::runtime::{Builder, Runtime};

const MAX_INPUT_BYTES: usize = 8 * 1024;
const MAX_EVENTS: usize = 512;

fn runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build tokio runtime for fuzz target")
    })
}

fn synthesize_events(data: &[u8]) -> Vec<Result<InferenceEvent, SdkError>> {
    let mut events: Vec<Result<InferenceEvent, SdkError>> = Vec::new();

    if let Some(first) = data.first()
        && first % 11 == 0
    {
        events.push(Err(SdkError::ProviderError(
            "synthetic upstream error".to_string(),
        )));
    }

    if data.first().map(|b| b % 5 != 0).unwrap_or(true) {
        events.push(Ok(InferenceEvent::MessageStart {
            role: "assistant".to_string(),
            model: "fuzz-model".to_string(),
            provider_id: "fuzz".to_string(),
        }));
    }

    let mut tool_seq = 0u64;
    for (idx, byte) in data.iter().enumerate() {
        if events.len() >= MAX_EVENTS {
            break;
        }

        match byte % 8 {
            0 => events.push(Ok(InferenceEvent::MessageDelta {
                content: format!("m{:02x}", byte),
            })),
            1 => events.push(Ok(InferenceEvent::ThinkingDelta {
                content: format!("t{:02x}", byte),
            })),
            2 => events.push(Ok(InferenceEvent::ThinkingSignatureDelta {
                signature: format!("s{:02x}", byte),
            })),
            3 => {
                tool_seq += 1;
                events.push(Ok(InferenceEvent::ToolCallStart {
                    id: format!("call_{}_{}", idx, tool_seq),
                    name: "tool".to_string(),
                }));
            }
            4 => events.push(Ok(InferenceEvent::ToolCallDelta {
                delta: format!("{{\"b\":{}}}", byte),
            })),
            5 => {
                events.push(Ok(InferenceEvent::MessageEnd {
                    input_tokens: (idx % 128) as u32,
                    output_tokens: ((idx + 1) % 128) as u32,
                    stop_reason: Some(StopReason::Unknown),
                }));
                break;
            }
            6 => events.push(Ok(InferenceEvent::MessageStart {
                role: "assistant".to_string(),
                model: "dup-model".to_string(),
                provider_id: "fuzz".to_string(),
            })),
            _ => {}
        }
    }

    if let Some(last) = data.last()
        && *last % 3 == 0
    {
        events.push(Ok(InferenceEvent::MessageEnd {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: Some(StopReason::EndTurn),
        }));
    }

    events
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let events = synthesize_events(data);
    let event_stream = Box::pin(stream::iter(events));
    let _ = runtime().block_on(InferenceResult::from_stream(event_stream));
});
