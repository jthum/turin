# Turin Remote

`turin-remote` is the network bridge for Turin’s daemon control plane.

It exposes authenticated HTTP access to the existing daemon request/response API and remote-friendly event streaming over SSE and WebSocket.

`turin-remote` does not replace the daemon. It sits in front of the local daemon endpoint and forwards requests through the same typed control surface used by the CLI and local clients.

## What It Provides

- authenticated HTTP access to daemon operations
- authenticated SSE event streaming
- authenticated WebSocket event streaming
- simple LAN/VPS deployment path without changing the daemon transport itself

## Prerequisites

Make sure the Turin daemon is already running:

```bash
turin daemon ensure --config turin.toml
```

`turin-remote` expects the daemon endpoint configured in the same `turin.toml`.

## Config

Add or review the `[remote]` section in `turin.toml`:

```toml
[remote]
bind = "127.0.0.1:9324"
auth_token_env = "TURIN_REMOTE_TOKEN"
event_keepalive_secs = 15
allow_non_loopback = false
```

Setting notes:

- `bind`: HTTP listen address for `turin-remote`
- `auth_token_env`: env var that contains the bearer token required by remote clients
- `event_keepalive_secs`: keepalive interval for SSE/WebSocket streams
- `allow_non_loopback`: require explicit opt-in before binding to `0.0.0.0`, LAN, or public interfaces

Defaults:

- `bind = "127.0.0.1:9324"`
- `auth_token_env = "TURIN_REMOTE_TOKEN"`
- `event_keepalive_secs = 15`
- `allow_non_loopback = false`

## Start The Remote Bridge

Export a token:

```bash
export TURIN_REMOTE_TOKEN="replace-with-a-long-random-token"
```

Then start the server:

```bash
turin-remote --config turin.toml
```

Useful overrides:

```bash
turin-remote \
  --config turin.toml \
  --bind 0.0.0.0:9324 \
  --allow-non-loopback \
  --auth-token-env TURIN_REMOTE_TOKEN \
  --event-keepalive-secs 10
```

You can also pass the token directly:

```bash
turin-remote --config turin.toml --auth-token "replace-me"
```

## Auth

All remote API routes except `/healthz` require:

```http
Authorization: Bearer <token>
```

The token grants effective control-plane access to the daemon. Treat it like an operator credential.

## Endpoints

### `GET /healthz`

Unauthenticated process liveness probe.

Response shape:

```json
{
  "ok": true,
  "version": "0.24.0"
}
```

### `GET /v1/health`

Authenticated remote + daemon health view.

Use this when you want to know whether the daemon behind `turin-remote` is actually reachable and ready.

### `POST /v1/daemon/request`

Authenticated generic daemon request proxy.

Request body is the same typed daemon wire shape used over local IPC:

```json
{
  "id": "req-1",
  "op": "daemon.ping",
  "params": {}
}
```

The response body is the daemon `ResponseEnvelope` unchanged. That means daemon-level validation or not-found failures still come back as `ok: false` in the JSON payload even when the HTTP status is `200`.

### `GET /v1/events`

Authenticated SSE stream of daemon runtime events.

Optional query parameters:

- `agent_id`
- `session_id`

The first event is typically `runtime.snapshot`, followed by live daemon events.

### `GET /v1/events/ws`

Authenticated WebSocket stream of daemon runtime events.

Optional query parameters:

- `agent_id`
- `session_id`

Each WebSocket text frame contains a serialized daemon `EventEnvelope`.

## Curl Examples

Health:

```bash
curl -s http://127.0.0.1:9324/healthz | jq
```

Authenticated health:

```bash
curl -s http://127.0.0.1:9324/v1/health \
  -H "Authorization: Bearer $TURIN_REMOTE_TOKEN" | jq
```

Daemon ping over HTTP:

```bash
curl -s http://127.0.0.1:9324/v1/daemon/request \
  -H "Authorization: Bearer $TURIN_REMOTE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "op": "daemon.ping",
    "params": {}
  }' | jq
```

SSE event tail:

```bash
curl -N http://127.0.0.1:9324/v1/events \
  -H "Authorization: Bearer $TURIN_REMOTE_TOKEN"
```

Filtered SSE event tail:

```bash
curl -N "http://127.0.0.1:9324/v1/events?agent_id=default" \
  -H "Authorization: Bearer $TURIN_REMOTE_TOKEN"
```

## Error Model

Remote-layer failures use HTTP error status codes with a JSON body like:

```json
{
  "error": {
    "code": "unauthorized",
    "message": "Missing or invalid bearer token"
  }
}
```

Common cases:

- `401 unauthorized`
- `400 invalid_request_body`
- `400 invalid_query`
- `503 daemon_unavailable`

Daemon-level failures are different:

- `POST /v1/daemon/request` still returns the daemon `ResponseEnvelope`
- look at `ok`, `error.code`, and `error.message` inside that JSON body

## Deployment Notes

- `turin-remote` binds to loopback by default
- non-loopback bind is refused unless `[remote].allow_non_loopback = true` or `--allow-non-loopback` is passed
- for LAN/VPS access, explicitly opt in to a non-loopback bind and put it behind TLS or a reverse proxy
- the daemon itself remains local-transport based; `turin-remote` is the network bridge

## Smoke Check

There is also a local operator smoke script:

```bash
scripts/remote_smoke.sh --token "replace-with-a-long-random-token"
```

It creates a temporary mock-backed workspace, starts the daemon and `turin-remote`, checks auth behavior, verifies a daemon ping round-trip, and confirms that SSE produces an initial runtime event.

## Browser Note

The current stream auth model is header-based bearer auth.

That works well for CLI tools, scripts, server-side apps, and custom clients. Browser-native `EventSource` and some WebSocket flows usually want same-origin proxying or a browser-specific auth strategy on top.
