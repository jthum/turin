# ADR 003: Typed NDJSON Daemon Protocol

- Status: Accepted
- Date: 2026-03-03

## Context

Daemon mode needed:

- a local control-plane API
- transport suitable for CLI, desktop, and web bridge clients
- event subscription support
- a format simple enough to inspect manually during development

The early daemon implementation used string op names plus ad hoc params handling.

## Decision

Use a local Unix-socket NDJSON protocol with typed request/response/event envelopes.

Characteristics:

- one JSON object per line
- request/response correlation by request ID
- typed request enum instead of raw stringly-typed dispatch
- typed error codes instead of free-form error strings
- separate event stream messages for daemon subscriptions

## Consequences

Positive:

- easy local debugging with ordinary text tooling
- good fit for CLI and future local clients
- cleaner protocol evolution than raw string op handling
- better foundation for channels/control-plane work

Negative:

- transport is local-first, not remote-first
- protocol versioning still matters once external clients depend on it

## Rejected alternatives

- opaque binary protocol
  - rejected as unnecessary complexity
- plain ad hoc string-op JSON forever
  - rejected because it does not scale cleanly to multiple clients or richer event flows
