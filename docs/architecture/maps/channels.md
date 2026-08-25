# Channels Map

## Purpose

Channels connect external messaging services to Turin. A channel is an
independently operated client: it translates platform events into ordinary
Turin session and task operations, then renders Turin events and results back
to the platform.

Turin core does not know what a channel is. It sees generic client requests,
an optional opaque `origin_id`, sessions, tasks, events, and governance state.
Channel credentials, configuration, access policy, conversation bindings,
retries, rendering, and process lifecycle remain channel-owned.

## Boundary

- A channel runner is an independently launched Turin client, not a daemon
  plugin or daemon-managed sidecar.
- Channel tooling stores configuration under `.turin/channels` by convention.
  This directory is not part of `TurinConfig`, the daemon registry, or daemon
  filesystem watching.
- Conversation bindings and access state are durable channel-owned files.
- Binding and access mutations use atomic replacement under per-file OS locks,
  so a runner and a separate channel state command cannot lose each other's updates.
- A runner accepts `run --config <channel-config> --turin-config <turin-config>`.
  Shared startup derives the channel id from the config directory, loads the
  adjacent Turin `.env`, resolves the local daemon endpoint, checks daemon
  compatibility, and stores runtime files under `<channel-dir>/runtime`.
- The daemon has no channel ids, channel operations, channel capabilities,
  presence heartbeat, or channel runtime snapshots.
- Channel failure cannot affect daemon readiness or unrelated clients.
- A process manager may launch one channel per process or a future channel host
  may run several; neither topology changes Turin core.

## Files

Shared channel implementation:

- `crates/turin-channel-core/src/*`
  - Normalized messaging types, manifests, settings, auth-flow DTOs, and
    conversation routing decisions.
- `crates/turin-channel-runner/src/*`
  - Access policy, durable bindings, inbound queues, task submission, progress
    streaming, completion delivery, shared runner startup, atomic state I/O,
    and daemon-free state management commands.
- `crates/turin-channel-host/src/*`
  - Optional adapter discovery and invocation used by setup tooling. It is not
    a Turin daemon dependency.
- `crates/turin-manager/src/setup/channels/*`
  - Channel discovery, configuration, and local configuration inventory.

Concrete channels:

- `crates/turin-channel-telegram/src/*`
- `crates/turin-channel-rocketchat/src/*`
- `crates/turin-channel-discord/src/*`
- `crates/turin-channel-whatsapp/src/*`
- `crates/turin-channel-fs/src/*`
  - Standalone filesystem channel and inexpensive integration-test adapter.

## Data Flow

1. An operator or process manager launches a channel runner with its channel
   configuration and Turin connection details.
2. The adapter normalizes a platform event into `InboundEvent`.
3. The shared runner enforces access policy and resolves its durable platform
   conversation binding.
4. The runner opens or resumes a Turin session and submits an ordinary task,
   identifying the channel instance only as opaque `origin_id` provenance.
5. The runner follows task events, renders progress and completion for the
   platform, and updates its own binding state.

Channel state management:

1. Each concrete runner exposes `state --config <path> access ...` and
   `state --config <path> bindings ...`.
2. These commands validate adapter kind but do not connect to the daemon and
   remain usable while the channel is disabled.
3. Access commands list, approve, reject, or revoke room state. Binding commands
   list or clear exact platform-conversation bindings.

## Vocabulary

- **Channel** is the user-facing integration concept, such as a Telegram or
  WhatsApp channel.
- **Channel runner** is the independent process implementing that channel.
- **Platform channel**, **room**, and **thread** identify destinations inside
  the external service; do not overload them with the configured Turin channel
  instance id when code needs both values.
- **Relay** is not a Turin domain term. It is too broad and may describe remote,
  web, daemon, proxy, or transport infrastructure.

## Invariants

- No production crate in Turin core, daemon protocol, client, or shared
  UI may depend on channel DTOs or channel process state.
- Access checks happen before task submission and remain channel-owned.
- Binding keys are stable serialized `ChannelConversationKey` values.
- A binding is reusable only while its configured agent matches the requested
  agent. Rebinding a channel to another agent starts a fresh Turin session.
- Concurrent events in one runner must not create competing sessions for the
  same durable conversation key.
- Adapters normalize inbound events before the runner handles them.
- Shared policy belongs in `turin-channel-runner`; platform formatting and
  identity rules remain adapter-owned.
- Channel presence and health are observed by the channel's process manager,
  not announced to Turin daemon.
- Root-level channel dev-dependencies exist only for cross-system integration
  tests and must not move into normal Turin dependencies.

## Common Changes

Change access or binding behavior:

1. Update `turin-channel-runner/src/access.rs`, `bindings.rs`, or
   `driver_loop.rs`.
2. Update runner tests.
3. Check at least one concrete channel.

Add a channel:

1. Implement `ChannelDriver` and adapter settings.
2. Use the shared channel-runner preparation path.
3. Add manifest, normalization, rendering, and delivery tests.
4. Do not add daemon registry, configuration, or protocol operations.

Change manager setup:

1. Update `turin-manager` and `turin-channel-host`.
2. Preserve channel-owned `.turin/channels` storage.
3. Do not add fields to `TurinConfig` for channel credentials or lifecycle.

## Tests

```sh
cargo test -p turin-channel-runner -p turin-channel-fs
cargo test -p turin-channel-telegram
cargo test -p turin-channel-rocketchat
cargo test -p turin-channel-discord
cargo test -p turin-channel-whatsapp
cargo check --workspace --all-targets
cargo fmt --all -- --check
git diff --check
```
