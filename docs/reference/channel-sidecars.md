# Channel Runners

Channel runners are independent Turin clients that adapt external messaging
services into ordinary session, task, and event operations. Turin core does
not load, supervise, or configure them.

The filename is retained as a stable documentation link; **channel runner** is
the current domain term.

## Process Contract

Every runner exposes:

- `run --config <channel-config> [--turin-config <turin-config>]`
- `describe`
- `validate-settings --settings-json <json>`
- `setup-auth-flow-start --request-json <json>` when auth flows are supported
- `setup-auth-flow-poll --request-json <json>` when auth flows are supported

`--turin-config` defaults to `.turin/config.toml`.

The shared run path:

1. Reads and validates the channel-owned TOML file.
2. Rejects disabled channels and configs for a different adapter kind.
3. Derives the channel instance id from the config's parent directory.
4. Loads `.env` beside the Turin config without overriding exported values.
5. Resolves and checks the local Turin daemon endpoint.
6. Stores bindings, access state, and adapter runtime data under
   `<channel-dir>/runtime`.
7. Runs in the foreground until interrupted or terminated.

Example:

```bash
turin-channel-telegram run \
  --config .turin/channels/telegram/config.toml \
  --turin-config .turin/config.toml
```

During workspace development, Turin Manager prints the equivalent
`cargo run -p turin-channel-<kind> -- ...` command.

## Channel Config

Channel configuration is stored by convention at:

```text
.turin/channels/<channel-id>/config.toml
```

Reserved top-level fields are:

```toml
enabled = true
kind = "telegram"
agent_id = "default"
idle_timeout_seconds = 900 # optional
```

All other top-level values and tables are adapter or shared-runner settings.
Common shared settings include access policy, task timeout, and downward-only
tool selection. Credentials should normally be referenced through environment
variables rather than stored directly in this file.

The channel directory and its runtime files are owned by the channel client.
The Turin daemon does not scan `.turin/channels` or expose channel status.

## Manifest Protocol

`describe` returns a `ChannelAdapterManifest`. The current manifest protocol
version is `2`.

Top-level manifest sections are:

- `protocol_version`
- `kind`
- `display_name`
- `runtime`
- `setup`
- `install`

`runtime` describes normalized capabilities and identity behavior. `setup`
describes generic manager prompts, validation, required secrets, and auth
flows. `install` identifies the runner binary.

Turin Manager validates manifests while discovering and configuring runners.
The daemon does not consume this protocol.

## Generic Setup

Manifest setup fields may use:

- `text`
- `secret`
- `boolean`
- `number`
- `select`
- `multi_select`
- `string_list`

Setup results can target channel settings or environment variables. Additional
target kinds remain reserved for broader setup workflows.

Manifest-declared auth flows let a runner request non-text setup without
hard-coding an adapter into Turin Manager. Current generic flow kinds are
`oauth_device_code` and `qr_pairing`.

## Manager Workflow

```bash
turin-manager channels list
turin-manager channels configure telegram
turin-manager channels status
turin-manager doctor
```

`configure` discovers the runner, renders prompts from its manifest, validates
the assembled settings through the runner, stages channel config and optional
`.env` diffs, then prints the exact foreground launch command.

`status` reports configuration readiness only. Runtime health belongs to the
operator's process manager. A future multi-channel host may supervise several
runners without changing Turin core or this client contract.
