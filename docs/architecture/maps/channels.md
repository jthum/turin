# Messaging Relays Map

## Purpose

Messaging relays connect external services to Turin through the same daemon
protocol used by any other client. Turin core does not discover, configure,
launch, supervise, or expose messaging channels. It sees an opaque request
origin plus ordinary sessions, tasks, events, and governance context.

The relay subsystem owns platform credentials, normalized inbound and outbound
messages, access and pairing policy, durable conversation bindings, retries,
streaming previews, and adapter process lifecycle.

## Boundary

- A relay is an independently launched Turin client, not a daemon plugin.
- Relay configuration lives under `.turin/relays` by manager convention and is
  not part of `TurinConfig` or daemon registry state.
- Conversation bindings and access state are durable relay-owned files.
- The daemon accepts generic client requests and persists opaque `origin_id`
  provenance; it has no channel ids, channel protocol operations, heartbeat,
  or runtime snapshots.
- Multiple platform adapters may run independently. A future combined relay
  host may supervise several adapters without changing Turin core.

## Files

Shared relay implementation:

- `crates/turin-channel-core/src/*`
  - Existing package name for normalized messaging types, manifests, settings,
    auth-flow DTOs, and conversation routing decisions.
- `crates/turin-channel-runner/src/*`
  - Access policy, durable bindings, inbound queues, task submission, progress
    streaming, completion delivery, and shared adapter startup.
- `crates/turin-channel-host/src/*`
  - Optional adapter discovery and invocation used by setup tooling. It is not
    a Turin daemon dependency.
- `crates/turin-manager/src/setup/channels/*`
  - Messaging relay discovery and configuration tooling.

Adapters:

- `crates/turin-channel-telegram/src/*`
- `crates/turin-channel-rocketchat/src/*`
- `crates/turin-channel-discord/src/*`
- `crates/turin-channel-whatsapp/src/*`
- `crates/turin-channel-fs/src/*`
  - Standalone filesystem relay and inexpensive integration-test adapter.

The `turin-channel-*` package names are retained as messaging-domain names.
They do not imply daemon ownership. A physical rename to `turin-relay-*` would
be a separate repository/package migration with no architectural effect.

## Data Flow

1. An operator or process launches an adapter with its relay configuration and
   Turin connection details.
2. The adapter normalizes a platform event into `InboundEvent`.
3. The shared runner enforces access policy and resolves the durable platform
   conversation binding.
4. The runner opens/resumes a Turin session and submits an ordinary task with
   opaque origin provenance.
5. It follows task events, renders progress/completion for the platform, and
   updates its own binding state.

## Invariants

- No crate in Turin core, daemon protocol, control client, or shared UI may
  depend on messaging channel DTOs or relay process state.
- Access checks happen before task submission and remain relay-owned.
- Binding keys are stable serialized `ChannelConversationKey` values.
- Adapters normalize inbound events before the runner handles them.
- Shared policy belongs in `turin-channel-runner`; platform formatting and
  identity rules remain adapter-owned.
- Adapter failure must not affect daemon readiness or unrelated clients.
- Relay presence and health are observed by the relay's process manager, not
  announced to Turin daemon.

## Common Changes

Change access or binding behavior:

1. Update `turin-channel-runner/src/access.rs`, `bindings.rs`, or
   `driver_loop.rs`.
2. Update runner tests.
3. Check at least one concrete adapter.

Add an adapter:

1. Implement `ChannelDriver` and adapter settings.
2. Use the shared sidecar preparation/run path.
3. Add manifest, normalization, rendering, and delivery tests.
4. Do not add daemon registry or protocol operations.

Change manager setup:

1. Update `turin-manager` and `turin-channel-host`.
2. Preserve relay-owned `.turin/relays` storage.
3. Do not add fields to `TurinConfig` for adapter credentials or lifecycle.

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
