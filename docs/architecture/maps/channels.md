# Channels Map

## Purpose

The channel subsystem connects external chat platforms to Turin daemon sessions. It owns sidecar process orchestration, adapter manifests, settings validation, inbound event normalization, access/pairing policy, conversation-to-session bindings, task submission, streaming progress, and outbound rendering.

This subsystem should preserve three guarantees:

- adapters convert platform events into a common `InboundEvent` shape before daemon submission
- access and pairing decisions are enforced by the shared runner, not separately per adapter
- sidecar setup and run-loop behavior stays consistent across adapters

## Files

Shared protocol and runner crates:

- `crates/turin-channel-core/src/lib.rs`
  - Common channel protocol types: manifests, auth-flow requests/responses, conversations, users, attachments, inbound events, outbound messages, routing decisions, and settings helpers.
- `crates/turin-channel-runner/src/lib.rs`
  - Public runner API and `ChannelDriver` trait.
- `crates/turin-channel-runner/src/sidecar.rs`
  - Shared sidecar runtime setup: settings JSON parsing, auth-flow request parsing, tracing setup, shutdown watch, runner construction, presence announcement, heartbeat, and driver run handoff.
- `crates/turin-channel-runner/src/access.rs`
  - Pairing, allowed/banned user policy, approved/pending room state, access snapshots.
- `crates/turin-channel-runner/src/bindings.rs`
  - Durable conversation-to-session binding file storage.
- `crates/turin-channel-runner/src/driver_loop.rs`
  - Inbound event authorization, per-conversation queueing, worker spawning, streaming progress, completion delivery.
- `crates/turin-channel-runner/src/task_payloads.rs`
  - `InboundEvent` to task payload conversion and task snapshot to `OutboundMessage` conversion.
- `crates/turin-channel-runner/src/stream.rs`
  - Channel streaming modes and progress-preview policy.
- `crates/turin-channel-runner/src/presence.rs`
  - Runner hello and heartbeat helpers.
- `crates/turin-channel-host/src/lib.rs`
  - Host-side sidecar discovery and process invocation used by the daemon and `turin-manager`: binary/env names, workspace fallback, manifest description, settings validation, auth-flow commands, and runner-kind discovery.

Daemon-side runtime:

- `src/daemon/channels.rs`
  - Channel runtime supervisor, desired-state sync, sidecar process lifecycle, heartbeat supervision, and runtime event emission.
- `src/daemon/channels/runtime_state.rs`
  - Runtime snapshot structs and named runtime-state transitions.
- `src/daemon/channel_runners.rs`
  - Daemon-facing wrapper around host-side sidecar discovery plus the built-in `fs` manifest.
- `src/daemon/state/channel_validation.rs`
  - Channel config validation against adapter manifests and runtime state.
- `src/daemon/server/dispatch/channel.rs`
  - Daemon protocol dispatch for channel operations.

Adapter crates:

- `crates/turin-channel-telegram/src/*`
  - Telegram settings, polling/API, inbound normalization, media handling, delivery, outbound rendering.
- `crates/turin-channel-rocketchat/src/*`
  - Rocket.Chat settings, REST/realtime transport, inbound normalization, outbound rendering.
- `crates/turin-channel-discord/src/*`
  - Discord settings, REST/gateway transport, inbound normalization, outbound rendering.
- `crates/turin-channel-whatsapp/src/*`
  - WhatsApp settings, auth flow, bot runtime, inbound normalization, outbound rendering.

## Data Flow

Sidecar startup:

1. The daemon launches `turin-channel-<kind> run` with channel id, agent id, daemon endpoint, binding paths, access state path, idle timeout, and settings JSON.
2. The adapter binary parses CLI args and calls `prepare_channel_sidecar_run`.
3. Shared setup parses common settings, builds `ChannelAccessPolicy`, prepares the daemon client, runner, shutdown watch, heartbeat watch, runtime directory, common tools config, and task timeout.
4. The adapter constructs its platform-specific `ChannelDriver`.
5. The sidecar announces presence, starts heartbeat, and calls `run_driver`.

Inbound event:

1. The adapter receives a platform event and normalizes it into `InboundEvent`.
2. `driver_loop.rs` authorizes the event through shared access/pairing policy.
3. The runner serializes the conversation key and either reuses or opens a daemon session.
4. `task_payloads.rs` maps event text and attachments into task prompt/content.
5. The daemon task is submitted, optionally streamed, then waited on.
6. The task result is converted to `OutboundMessage`.
7. The adapter renders and sends the platform-specific outbound message.

Pairing:

1. When pairing is off, user allow/ban policy applies directly.
2. When pairing is pending or auto, access state is keyed by room/conversation.
3. Auto pairing approves eligible rooms immediately.
4. Pending pairing stores a pending room and sends the pending-approval message once per room.
5. Approved rooms still honor `allowed_users` and `banned_users`.

Conversation bindings:

1. A `ChannelConversationKey` is serialized as the binding key.
2. Existing bindings are routed through `decide_routing`.
3. Expired or reset bindings open fresh sessions.
4. Reused bindings resume existing sessions.
5. Bindings are saved after the session is established.

## Invariants

- Platform adapters should own platform normalization and rendering only.
- Shared access policy belongs in `turin-channel-runner`, not individual adapters.
- Common sidecar startup behavior belongs in `sidecar.rs`.
- Host-side sidecar process discovery/invocation belongs in `turin-channel-host`, not separately in the daemon and manager.
- Optional string-enum settings should use shared settings helpers from `turin-channel-core` when the adapter only needs local allowed-value mapping.
- Plain text/code-block rendering and line-aware message splitting should use shared helpers from `turin-channel-core` when the adapter has no platform-specific formatting rule for that step.
- Runtime snapshot state changes should use transition helpers from `src/daemon/channels/runtime_state.rs`; avoid open-coded edits to state, error fields, transition times, counters, and handshake timestamps.
- Each adapter must validate platform-specific settings without requiring live external credentials.
- Auth-flow request parsing should use shared runner helpers so setup commands report consistent parse errors.
- `ChannelDriver::user_matches_selector` remains adapter-specific because platform identity formats differ.
- Conversation binding keys must be stable JSON representations of `ChannelConversationKey`.
- Pending approval notifications should be sent once per unapproved room until the pending room is seen again.
- Banned users override approved rooms and allowed users.
- Sidecar heartbeats should stop when the shared shutdown watch is set.
- Adapter `main.rs` files should stay thin: CLI dispatch, adapter construction, adapter-specific validation, and auth-flow calls.

## Common Changes

Add a new channel adapter:

1. Implement adapter settings parsing and `ChannelDriver`.
2. Keep sidecar `main.rs` thin and use `prepare_channel_sidecar_run`.
3. Implement `adapter_manifest`, `validate_settings`, and auth-flow functions if needed.
4. Add adapter tests for settings, inbound normalization, and outbound rendering.
5. Run the shared runner tests and the new adapter tests.

Change access or pairing behavior:

1. Change `crates/turin-channel-runner/src/access.rs` or `driver_loop.rs`.
2. Update runner tests in `crates/turin-channel-runner/src/tests.rs`.
3. Run `cargo test -p turin-channel-runner`.
4. Run daemon channel tests if runtime snapshots or supervisor behavior change.

Change task payload mapping:

1. Change `crates/turin-channel-runner/src/task_payloads.rs`.
2. Check attachment, structured outbound, and assistant-content tests.
3. Run `cargo test -p turin-channel-runner`.

Change sidecar startup:

1. Change `crates/turin-channel-runner/src/sidecar.rs`.
2. Keep adapter `main.rs` files consistent and thin.
3. Run all adapter package tests, not only runner tests.

Change daemon channel supervision:

1. Change `src/daemon/channels.rs` or `src/daemon/channel_runners.rs`.
2. Keep runtime state changes behind the snapshot transition helpers.
3. Run daemon channel tests.
4. Check runtime snapshot and heartbeat/restart behavior.

## Tests

Focused shared runner tests:

```sh
cargo test -p turin-channel-runner
```

Adapter tests:

```sh
cargo test -p turin-channel-telegram
cargo test -p turin-channel-rocketchat
cargo test -p turin-channel-discord
cargo test -p turin-channel-whatsapp
```

Daemon channel tests:

```sh
cargo test -p turin daemon::channels::tests
cargo test -p turin channel --lib
```

Compile and formatting checks:

```sh
cargo check -p turin-channel-runner
cargo check -p turin-channel-telegram -p turin-channel-rocketchat -p turin-channel-discord -p turin-channel-whatsapp
cargo fmt --all -- --check
git diff --check
```

## Current Shape

The runner crate now owns common sidecar runtime setup. This removed repeated setup code from Telegram, Rocket.Chat, Discord, and WhatsApp sidecar binaries while keeping adapter-specific driver construction in each adapter.

The host crate owns external sidecar discovery and subprocess command helpers. This keeps the daemon and `turin-manager` aligned on env overrides, workspace `cargo run -p` fallback, manifest decoding, settings validation, and auth-flow command behavior.

Shared channel settings helpers cover common optional string-enum parsing. Telegram, Rocket.Chat, and runner access policy use the shared shape while keeping each allowed-value table and error text local.

Shared outbound text helpers cover plain text/code-block rendering and line-aware content splitting. Telegram, Discord, and Rocket.Chat use those helpers where their behavior is identical; Telegram HTML rendering, Rocket.Chat table wrapping/reply quoting, Discord embeds/components/files, and WhatsApp plain rendering stay adapter-owned.

Daemon channel supervision keeps runtime-state mutation behind named transition helpers in `runtime_state.rs`. This is meant to prevent drift between start, restart, stale-heartbeat, clean-stop, shutdown, and failure paths.

The current module split is deliberate:

- `turin-channel-core` answers "what common channel protocol shapes exist?"
- `turin-channel-runner` answers "how does a channel event become a daemon task and response?"
- `turin-channel-runner/src/sidecar.rs` answers "how does a sidecar process start consistently?"
- `turin-channel-host` answers "how does a Turin host process find and invoke sidecar binaries?"
- adapter crates answer "how does this platform map to and from Turin channel shapes?"
- daemon channel modules answer "how are sidecars configured, supervised, state-tracked, and surfaced?"

Likely future cleanup areas:

- compare adapter settings parsing for remaining duplicated range/string handling
- consider shared outbound rendering helpers only where platform formatting rules genuinely overlap
- use perf tests to measure sidecar memory and daemon memory over long channel sessions
