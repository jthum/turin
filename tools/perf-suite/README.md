# Turin Perf Suite

This is a repo-local measurement harness. It is intentionally not a member of the main Cargo workspace, so it does not affect normal builds or shipped binaries.

The first scenario stresses hot session history and large tool outputs without using a real inference provider.

For a static source/binary footprint baseline, prefer the no-build script:

```bash
tools/footprint-report --top-files 40
```

It uses standard shell tools, writes JSON and Markdown reports to
`.workspace/perf-reports/`, and does not compile or link Turin.

The Rust harness also has a `footprint` subcommand for environments where a
single binary runner is preferable, but it links the perf-suite crate:

```bash
cargo run --manifest-path tools/perf-suite/Cargo.toml -- \
  footprint \
  --top-files 40
```

Both footprint paths scan Rust source roots, exclude obvious test/bench/example
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
JSON reports keep the raw snapshots. Markdown reports add a short summary table
with first, final, delta, and peak values for the main memory, storage, and
session-volume metrics.

Runtime scenarios suppress streamed turn output by default so long runs do not
flood the terminal. Add `--verbose-turn-output` to hot-history, fake-channel,
channel-scale, or idle-runtime when debugging the mocked conversation itself.

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

The black-box channel scale scenario measures an already-built `turin` daemon
binary as a separate child process instead of measuring the perf-suite process:

```bash
CARGO_TARGET_DIR=target cargo run --manifest-path tools/perf-suite/Cargo.toml -- \
  blackbox-channel-scale \
  --turin-binary target/release/turin \
  --sessions 2 \
  --messages-per-session 200 \
  --checkpoints 10,100,200 \
  --message-bytes 512 \
  --response-bytes 1024 \
  --agent-idle-timeout-seconds 1 \
  --post-run-idle-wait-ms 1500 \
  --checkpoint-state-db-after-idle
```

In black-box mode, live DB reads can be unavailable while the daemon owns the
state DB lock. The scenario samples daemon memory while live and applies the
optional DB checkpoint after daemon shutdown.

The black-box task scale scenario measures the daemon task path without the
channel runner layer. It launches the same already-built daemon binary, opens
one live session, submits direct daemon tasks, waits for each task, and samples
the daemon child process at task-count checkpoints:

```bash
CARGO_TARGET_DIR=target cargo run --manifest-path tools/perf-suite/Cargo.toml -- \
  blackbox-task-scale \
  --turin-binary target/release/turin \
  --tasks 1000 \
  --checkpoints 10,100,200,1000 \
  --prompt-bytes 32 \
  --response-bytes 1024 \
  --agent-idle-timeout-seconds 1 \
  --post-run-idle-wait-ms 1500 \
  --checkpoint-state-db-after-idle
```

Use it against `blackbox-channel-scale` to separate channel-runner overhead from
core daemon task/runtime/persistence overhead. When the daemon supports live
history diagnostics, this scenario also records the current hot-history length
and message offset so long-session memory growth can be compared against the
bounded hot window. After-runner/after-idle rows include daemon task-cache
metrics, including total serialized task snapshot bytes.

For heap attribution when aggregate counters are not enough, build a temporary
profiling daemon:

```bash
CARGO_TARGET_DIR=target cargo build --profile profiling -p turin --bin turin --features heap-profile
```

Then run a black-box scenario against `target/profiling/turin`. The
`heap-profile` feature is off by default and uses `dhat` to write
`dhat-heap.json` when the daemon exits. Set
`TURIN_HEAP_PROFILE_PATH=/path/to/dhat-heap.json` to keep the heap file beside
the perf report. Keep this for local perf work only; normal release binaries
should not enable it.

The persistence scale scenario measures the state-store path without daemon,
channel runner, peer runtime, or provider execution:

```bash
CARGO_TARGET_DIR=target cargo run --manifest-path tools/perf-suite/Cargo.toml -- \
  persistence-scale \
  --tasks 5000 \
  --checkpoints 1000,5000 \
  --prompt-bytes 32 \
  --response-bytes 1024
```

By default it writes the same user/assistant message shape without materializing
the active branch. Add `--read-active-branch-at-checkpoints` to read and drop the
full active branch before each checkpoint, which helps separate write-path
retention from read/materialization retention.

Add `--include-daemon-events` to also persist representative task, turn, and
stream event rows for each task. Those reports include `persisted_events` and
`persisted_event_payload_bytes`, which help compare state DB growth from message
content versus lifecycle/stream metadata.

Channel scenarios default to 256-byte inbound messages and 1 KiB mocked assistant
responses. To isolate mostly metadata overhead, make both values intentionally
small, for example `--message-bytes 16 --response-bytes 4`.

Channel scenarios also accept:

```bash
--agent-idle-timeout-seconds 1
--post-run-idle-wait-ms 1500
--checkpoint-state-db-after-idle
--trim-allocator-after-idle
```

Use these together to separate persisted storage growth from live daemon runtime
retention. Channel reports include `live_sessions` for daemon-held runtime
sessions; `active_sessions` remains the planned number of logical channel
conversations in scale scenarios. `--checkpoint-state-db-after-idle` separates
WAL/checkpoint effects from runtime memory. `--trim-allocator-after-idle` is a
diagnostic knob for Linux/glibc builds: it records whether retained PSS looks
like allocator retention after live sessions have been released.

For black-box daemon runs, `--trim-allocator-after-idle` cannot trim the child
daemon process. To test daemon-side peer-runtime idle trimming, run the daemon
profile with:

```bash
TURIN_TRIM_ALLOCATOR_ON_PEER_IDLE=1
```

That opt-in daemon setting calls `malloc_trim(0)` after a peer runtime releases
its session on idle shutdown. It is intended for diagnostics and low-memory
deployments where lower retained RSS/PSS is worth the trim cost.

The idle runtime scenario submits mocked peer-agent requests, samples memory while
the runtime is live, then waits for idle hibernation:

```bash
cargo run --manifest-path tools/perf-suite/Cargo.toml -- \
  idle-runtime \
  --requests 25 \
  --response-bytes 4096 \
  --idle-timeout-seconds 1 \
  --max-wait-ms 5000
```

Use `--idle-timeout-seconds 0` to measure immediate hibernation after each logical
request; the scenario waits for release between requests in that mode. The report
uses the `active_sessions` column as the count of live peer runtime sessions for
this scenario.

## What It Measures

- process RSS from `/proc`
- process PSS from `/proc/self/smaps_rollup` when available
- daemon child RSS/PSS from `/proc/<pid>/smaps_rollup` in black-box scenarios
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
- live daemon runtime session count, for channel and idle-runtime scenarios

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
- concurrent sessions across agents
