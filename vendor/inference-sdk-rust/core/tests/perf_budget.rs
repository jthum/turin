use futures_util::stream;
use inference_sdk_core::{
    InferenceContent, InferenceEvent, InferenceResult, SdkError, StopReason,
    validate_event_sequence,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const VALIDATE_ITERATIONS: usize = 4_000;
const TEXT_ITERATIONS: usize = 300;
const TOOL_ITERATIONS: usize = 120;

fn build_text_events(delta_count: usize, delta_len: usize) -> Vec<InferenceEvent> {
    let mut events = Vec::with_capacity(delta_count + 2);
    events.push(InferenceEvent::MessageStart {
        role: "assistant".to_string(),
        model: "perf-model".to_string(),
        provider_id: "perf".to_string(),
    });

    let delta = "x".repeat(delta_len);
    for _ in 0..delta_count {
        events.push(InferenceEvent::MessageDelta {
            content: delta.clone(),
        });
    }

    events.push(InferenceEvent::MessageEnd {
        input_tokens: 16,
        output_tokens: 32,
        stop_reason: Some(StopReason::EndTurn),
    });
    events
}

fn build_tool_events(payload_len: usize, chunk_len: usize) -> Vec<InferenceEvent> {
    let mut events = Vec::new();
    events.push(InferenceEvent::MessageStart {
        role: "assistant".to_string(),
        model: "perf-model".to_string(),
        provider_id: "perf".to_string(),
    });
    events.push(InferenceEvent::ToolCallStart {
        id: "call_perf".to_string(),
        name: "store_blob".to_string(),
    });

    let payload = "z".repeat(payload_len);
    let json = serde_json::json!({ "payload": payload }).to_string();
    for bytes in json.as_bytes().chunks(chunk_len) {
        events.push(InferenceEvent::ToolCallDelta {
            delta: String::from_utf8(bytes.to_vec()).expect("delta chunk must be valid UTF-8"),
        });
    }

    events.push(InferenceEvent::MessageEnd {
        input_tokens: 32,
        output_tokens: 64,
        stop_reason: Some(StopReason::ToolUse),
    });
    events
}

fn assert_within_budget(name: &str, elapsed: Duration, budget: Duration) {
    assert!(
        elapsed <= budget,
        "{name} exceeded budget: elapsed={elapsed:?} budget={budget:?}"
    );
}

fn measure_validate_event_sequence_large_message() -> Duration {
    let mut events = Vec::new();
    events.push(InferenceEvent::MessageStart {
        role: "assistant".to_string(),
        model: "perf-model".to_string(),
        provider_id: "perf".to_string(),
    });
    for _ in 0..10_000 {
        events.push(InferenceEvent::MessageDelta {
            content: "0123456789abcdef".to_string(),
        });
    }
    events.push(InferenceEvent::MessageEnd {
        input_tokens: 16,
        output_tokens: 32,
        stop_reason: Some(StopReason::EndTurn),
    });

    let start = Instant::now();
    for _ in 0..VALIDATE_ITERATIONS {
        validate_event_sequence(black_box(&events)).expect("event sequence should be valid");
    }
    start.elapsed()
}

fn measure_from_stream_text_assembly() -> Duration {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let events = build_text_events(4_000, 16);
    let expected_text_len = 4_000 * 16;

    let start = Instant::now();
    for _ in 0..TEXT_ITERATIONS {
        let stream_events: Vec<Result<InferenceEvent, SdkError>> = events
            .iter()
            .cloned()
            .map(Ok::<InferenceEvent, SdkError>)
            .collect();
        let stream = Box::pin(stream::iter(stream_events));
        let result = runtime
            .block_on(InferenceResult::from_stream(stream))
            .expect("stream assembly should succeed");
        let text_len = result.text().len();
        assert_eq!(text_len, expected_text_len);
        black_box(text_len);
    }
    start.elapsed()
}

fn measure_from_stream_tool_delta_assembly() -> Duration {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let events = build_tool_events(64 * 1024, 32);
    let expected_payload_len = 64 * 1024;

    let start = Instant::now();
    for _ in 0..TOOL_ITERATIONS {
        let stream_events: Vec<Result<InferenceEvent, SdkError>> = events
            .iter()
            .cloned()
            .map(Ok::<InferenceEvent, SdkError>)
            .collect();
        let stream = Box::pin(stream::iter(stream_events));
        let result = runtime
            .block_on(InferenceResult::from_stream(stream))
            .expect("stream assembly should succeed");

        let payload_len = result
            .content
            .iter()
            .find_map(|part| match part {
                InferenceContent::ToolUse { input, .. } => input
                    .get("payload")
                    .and_then(|v| v.as_str())
                    .map(|s| s.len()),
                _ => None,
            })
            .expect("tool payload should exist");

        assert_eq!(payload_len, expected_payload_len);
        black_box(payload_len);
    }
    start.elapsed()
}

#[derive(Debug, Deserialize)]
struct PerfBaseline {
    max_regression_pct: f64,
    metrics_ns: HashMap<String, u64>,
}

fn perf_baseline_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("perf_baseline.json")
}

fn load_baseline() -> PerfBaseline {
    let path = perf_baseline_path();
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    serde_json::from_str::<PerfBaseline>(&raw)
        .unwrap_or_else(|e| panic!("invalid baseline JSON in {}: {e}", path.display()))
}

fn assert_not_regressed(metric: &str, measured: Duration, baseline: &PerfBaseline) {
    let measured_ns = measured.as_nanos() as u64;
    let baseline_ns = *baseline
        .metrics_ns
        .get(metric)
        .unwrap_or_else(|| panic!("missing baseline metric '{metric}'"));

    let tolerance = 1.0 + (baseline.max_regression_pct / 100.0);
    let allowed_ns = (baseline_ns as f64 * tolerance) as u64;

    assert!(
        measured_ns <= allowed_ns,
        "performance regression for {metric}: measured={}ns baseline={}ns allowed={}ns (max_regression_pct={}%)",
        measured_ns,
        baseline_ns,
        allowed_ns,
        baseline.max_regression_pct
    );
}

/// Performance guardrail for event-order validation.
///
/// Ignored by default because these are budget checks intended for CI perf gating.
#[test]
#[ignore = "run in CI release mode as a performance budget check"]
fn perf_budget_validate_event_sequence_large_message() {
    let elapsed = measure_validate_event_sequence_large_message();
    eprintln!(
        "metric=validate_event_sequence_large_message_ns value={}",
        elapsed.as_nanos()
    );
    assert_within_budget(
        "validate_event_sequence_large_message",
        elapsed,
        Duration::from_secs(5),
    );
}

/// Performance guardrail for text stream assembly.
#[test]
#[ignore = "run in CI release mode as a performance budget check"]
fn perf_budget_from_stream_text_assembly() {
    let elapsed = measure_from_stream_text_assembly();
    eprintln!(
        "metric=from_stream_text_assembly_ns value={}",
        elapsed.as_nanos()
    );
    assert_within_budget("from_stream_text_assembly", elapsed, Duration::from_secs(6));
}

/// Performance guardrail for long tool-delta JSON assembly and parse.
#[test]
#[ignore = "run in CI release mode as a performance budget check"]
fn perf_budget_from_stream_tool_delta_assembly() {
    let elapsed = measure_from_stream_tool_delta_assembly();
    eprintln!(
        "metric=from_stream_tool_delta_assembly_ns value={}",
        elapsed.as_nanos()
    );
    assert_within_budget(
        "from_stream_tool_delta_assembly",
        elapsed,
        Duration::from_secs(6),
    );
}

/// Historical regression check against committed baseline metrics.
#[test]
#[ignore = "run in CI release mode as a performance regression check"]
fn perf_regression_against_baseline() {
    let baseline = load_baseline();

    // Light warm-up to reduce one-time noise.
    black_box(measure_validate_event_sequence_large_message());
    black_box(measure_from_stream_text_assembly());
    black_box(measure_from_stream_tool_delta_assembly());

    let validate_elapsed = measure_validate_event_sequence_large_message();
    let text_elapsed = measure_from_stream_text_assembly();
    let tool_elapsed = measure_from_stream_tool_delta_assembly();

    eprintln!(
        "baseline-check validate={}ns text={}ns tool={}ns",
        validate_elapsed.as_nanos(),
        text_elapsed.as_nanos(),
        tool_elapsed.as_nanos()
    );

    assert_not_regressed(
        "validate_event_sequence_large_message_ns",
        validate_elapsed,
        &baseline,
    );
    assert_not_regressed("from_stream_text_assembly_ns", text_elapsed, &baseline);
    assert_not_regressed(
        "from_stream_tool_delta_assembly_ns",
        tool_elapsed,
        &baseline,
    );
}
