# Turin Perf Suite

This is a repo-local measurement harness. It is intentionally not a member of the main Cargo workspace, so it does not affect normal builds or shipped binaries.

The first scenario stresses hot session history and large tool outputs without using a real inference provider.

For a static source/binary footprint baseline:

```bash
cargo run --manifest-path tools/perf-suite/Cargo.toml -- \
  footprint \
  --top-files 40
```

`footprint` scans Rust source roots, excludes obvious test/bench/example
directories, reports LOC by area, lists the largest shipped source files, and
records release binary sizes when known artifacts already exist. It does not
build release binaries by itself.

```bash
cargo run --manifest-path tools/perf-suite/Cargo.toml -- \
  hot-history \
  --turns 50 \
  --payload-bytes 262144 \
  --response-bytes 4096 \
  --tool-every 4 \
  --sample-every 5
```

For a small smoke run:

```bash
cargo run --manifest-path tools/perf-suite/Cargo.toml -- \
  hot-history \
  --turns 2 \
  --payload-bytes 1024 \
  --response-bytes 512
```

`hot-history` defaults to one large `read_file` tool call every four prompts.
That keeps the scenario below Turin's built-in 32-tool-call safety window during
fast local runs while still measuring large persisted tool results. Set
`--tool-every 1` only when you intentionally want to exercise rate limiting, or
`--tool-every 0` to measure message payloads without native tool calls.

Hot-history runs also accept:

```bash
--hot-history-profile default|performance|debug
--hot-history-max-messages 64
--hot-history-max-tool-result-bytes 65536
```

Use the default profile to measure the memory-safe baseline, performance to keep
a larger resident window, and debug to compare against effectively unbounded hot
history.

Reports are written to `.workspace/perf-reports/` by default as JSON and Markdown.

The fake channel scenario starts the real daemon, uses the channel runner with an in-process mock driver, and keeps inference mocked:

```bash
cargo run --manifest-path tools/perf-suite/Cargo.toml -- \
  fake-channel \
  --messages 25 \
  --message-bytes 512 \
  --response-bytes 1024
```

The channel scale scenario measures daemon/channel footprint at cumulative message checkpoints across one or more logical sessions:

```bash
cargo run --release --manifest-path tools/perf-suite/Cargo.toml -- \
  channel-scale \
  --sessions 2 \
  --messages-per-session 1000 \
  --checkpoints 10,100,200,1000 \
  --message-bytes 512 \
  --response-bytes 1024
```

Channel scenarios default to 256-byte inbound messages and 1 KiB mocked assistant
responses. To isolate mostly metadata overhead, make both values intentionally
small, for example `--message-bytes 16 --response-bytes 4`.

## What It Measures

- process RSS from `/proc`
- process PSS from `/proc/self/smaps_rollup` when available
- state DB size, split into main DB, WAL, SHM, and total bytes
- session history length
- persisted active-branch message count
- hot-window message offset and whether the session is currently pruned
- approximate resident history payload bytes
- hot-window tool result count and tool-result error count
- turn count
- elapsed wall time
- fake channel outbound count, for channel scenarios
- active logical sessions and messages per session, for scale scenarios

## Compile Cost

This Rust harness links Turin as a library so it can use in-process mock inference and mock channel drivers. That gives good control, but it is not the lightest possible runner.

To avoid a separate `tools/perf-suite/target/` directory, run with a shared target dir:

```bash
CARGO_TARGET_DIR=target cargo run --manifest-path tools/perf-suite/Cargo.toml -- \
  channel-scale --sessions 1 --messages-per-session 100
```

A future black-box runner can use an already-built `target/release/turin` binary plus the filesystem channel to avoid compiling this harness at all.

## Why Mock Inference

Most runtime footprint questions do not require a real model. Mock inference lets the suite stress Turin's own work:

- prompt/history construction
- streaming event normalization
- tool execution
- persistence
- large tool-result retention

Real provider benchmarking can be added as a separate opt-in profile later.

## Next Scenarios

- daemon plus mocked sidecar protocol
- idle-runtime hibernation and memory release
- concurrent sessions across agents
