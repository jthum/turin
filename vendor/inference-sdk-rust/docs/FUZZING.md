# Fuzzing Guide

This repository includes `cargo-fuzz` targets under `fuzz/`.

## Targets

1. `provider_stream_json`
   - Fuzzes provider stream JSON payload parsing and adapter mapping.
   - Tries both OpenAI and Anthropic stream event payloads from arbitrary input.
2. `core_event_stream`
   - Fuzzes normalized event-stream assembly (`InferenceResult::from_stream`) using synthesized event sequences.

## Local run

```bash
cargo install cargo-fuzz
cargo fuzz run provider_stream_json --fuzz-dir fuzz -- -max_total_time=300
cargo fuzz run core_event_stream --fuzz-dir fuzz -- -max_total_time=300
```

## Corpus

Seed corpora are checked in:

- `fuzz/corpus/provider_stream_json`
- `fuzz/corpus/core_event_stream`

These should be expanded whenever new stream payload variants are introduced.
