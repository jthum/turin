# Channel Sidecars

This document defines Turin's current sidecar protocol for external channel adapters.

## Protocol Version

Current sidecar manifest protocol version:

- `2`

Turin validates sidecar manifests against this version when:

- the daemon probes a runner with `describe`
- `turin-manager` discovers/configures a runner
- a runner sends its live startup handshake to the daemon

## Required Commands

An external channel sidecar is expected to expose these CLI commands:

- `describe`
- `validate-settings --settings-json <json>`
- `run --channel-id <id> --agent-id <id> --daemon-endpoint <path> --bindings-path <path> --access-state-path <path> --settings-json <json> [--idle-ttl-secs <secs>]`
- `setup-auth-flow-start --request-json <json>`
- `setup-auth-flow-poll --request-json <json>`

If a channel does not use manifest-declared auth flows yet, `setup-auth-flow-start` and `setup-auth-flow-poll` may return an explicit unsupported error.

## Manifest Shape

`describe` returns a `ChannelAdapterManifest`.

Top-level sections:

- `protocol_version`
- `kind`
- `display_name`
- `runtime`
- `setup`
- `install`

### Runtime

`runtime` describes properties the daemon/UI can reason about generically:

- `session_scopes`
- `enum_settings`
- `capabilities`
- `identity_selectors`

### Setup

`setup` describes what `turin-manager` can render generically:

- `required_secrets`
- `instructions`
- `setup_url`
- `validation_checks`
- `config_fields`
- `auth_flows`

### Install

`install` currently exposes:

- `binary_name`

## Generic Setup Field Types

Current field types rendered by `turin-manager`:

- `text`
- `secret`
- `boolean`
- `number`
- `select`
- `multi_select`
- `string_list`

Field metadata can include:

- `label`
- `prompt`
- `help`
- `hint`
- `example`
- `required`
- `advanced`
- `default`
- `options`
- `visible_if`
- `target`
- `validate`

## Targets

Setup results resolve to explicit targets:

- `channel_setting`
- `root_config`
- `agent_config`
- `env_var`
- `local_secret_store`

Current `turin-manager` support is strongest for:

- `channel_setting`
- `env_var`

Other target kinds are reserved for broader Turin-wide configuration flows.

## Auth Flows

Manifest-declared auth flows let a sidecar request a non-text setup step without hardcoding the manager to a specific channel.

Current generic flow kinds:

- `oauth_device_code`
- `qr_pairing`

The manager starts a flow with `setup-auth-flow-start` and polls it with `setup-auth-flow-poll`.

Current flow display payloads can include:

- `message`
- `verification_uri`
- `verification_uri_complete`
- `user_code`
- `qr_text`
- `pairing_code`
- `expires_in_secs`
- `poll_interval_secs`

Flow completion returns resolved target/value pairs that the manager applies like any other setup result.

## Live Startup Handshake

`describe` remains the tooling and discovery path.

At runtime, external sidecars now also send a live startup handshake to the daemon via:

- `channel.runner.hello`

The handshake includes:

- `channel_id`
- `manifest`
- `runner_binary`
- `runner_version`
- `pid`

The daemon keeps an external channel in `starting` until the sidecar sends this hello. Once received, the runtime moves to `running` and records handshake metadata in the channel runtime snapshot.

## Manager Integration

`turin-manager` consumes the sidecar protocol generically through:

- `turin-manager channels list`
- `turin-manager channels configure <kind>`
- `turin-manager channels status`
- `turin-manager doctor`

`turin-manager channels configure <kind>` renders setup fields from the manifest, then calls the sidecar's `validate-settings` command on the assembled settings before it writes config files.

That means a new official or community sidecar can integrate with the manager without channel-specific code in the manager, as long as it stays within the shared manifest and auth-flow protocol.
