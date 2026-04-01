#!/usr/bin/env bash
set -euo pipefail

BIN="target/release/turin"
REMOTE_BIN="target/release/turin-remote"
TOKEN="${TURIN_REMOTE_TOKEN:-}"
BIND="127.0.0.1:19324"
WORKSPACE_ROOT=""
KEEP_TMP=0

usage() {
  cat <<'USAGE'
Usage: scripts/remote_smoke.sh [options]

Options:
  --bin PATH                 Turin binary path (default: target/release/turin)
  --remote-bin PATH          turin-remote binary path (default: target/release/turin-remote)
  --token VALUE              Bearer token to require (or set TURIN_REMOTE_TOKEN)
  --bind ADDR                Remote bind address (default: 127.0.0.1:19324)
  --workspace-root PATH      Optional existing workspace root
  --keep-tmp                 Keep temporary workspace for inspection
  -h, --help                 Show this help

Notes:
  - This is a local smoke test for daemon + turin-remote wiring.
  - Non-loopback bind requires turin-remote --allow-non-loopback.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin)
      BIN="$2"
      shift 2
      ;;
    --remote-bin)
      REMOTE_BIN="$2"
      shift 2
      ;;
    --token)
      TOKEN="$2"
      shift 2
      ;;
    --bind)
      BIND="$2"
      shift 2
      ;;
    --workspace-root)
      WORKSPACE_ROOT="$2"
      shift 2
      ;;
    --keep-tmp)
      KEEP_TMP=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$TOKEN" ]]; then
  echo "Missing remote token. Use --token or TURIN_REMOTE_TOKEN." >&2
  exit 1
fi

if [[ ! -x "$BIN" ]]; then
  echo "Turin binary not found or not executable: $BIN" >&2
  exit 1
fi

if [[ ! -x "$REMOTE_BIN" ]]; then
  echo "turin-remote binary not found or not executable: $REMOTE_BIN" >&2
  exit 1
fi

if [[ -n "$WORKSPACE_ROOT" ]]; then
  ROOT="$WORKSPACE_ROOT"
  mkdir -p "$ROOT"
else
  ROOT="$(mktemp -d /tmp/turin-remote-smoke.XXXXXX)"
fi

cleanup() {
  set +e
  if [[ -n "${REMOTE_PID:-}" ]]; then
    kill "$REMOTE_PID" >/dev/null 2>&1 || true
    wait "$REMOTE_PID" >/dev/null 2>&1 || true
  fi
  "$BIN" daemon stop --config "$ROOT/turin.toml" >/dev/null 2>&1 || true
  if [[ $KEEP_TMP -eq 0 && -z "$WORKSPACE_ROOT" ]]; then
    rm -rf "$ROOT"
  fi
}
trap cleanup EXIT

mkdir -p "$ROOT/.turin/harnesses"
printf '%s\n' "-- remote smoke harness" "function on_turn_prepare(ctx)" "  return ALLOW" "end" > "$ROOT/.turin/harnesses/main.lua"

printf '%s\n' \
  '[agent]' \
  'id = "default"' \
  'system_prompt = "You are a concise assistant."' \
  'model = "mock-model"' \
  'provider = "mock"' \
  '' \
  '[kernel]' \
  "workspace_root = \"$ROOT\"" \
  '' \
  '[persistence.state]' \
  "path = \"$ROOT/.turin/state.db\"" \
  '' \
  '[harness]' \
  "directory = \"$ROOT/.turin/harnesses\"" \
  'fs_root = "."' \
  '' \
  '[providers.mock]' \
  'type = "mock"' \
  'base_url = "PONG"' \
  > "$ROOT/turin.toml"

"$BIN" daemon start --config "$ROOT/turin.toml" --log-level error >"$ROOT/daemon.log" 2>&1 &
DAEMON_PID=$!

for _ in {1..120}; do
  if "$BIN" daemon ping --config "$ROOT/turin.toml" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done

if ! "$BIN" daemon ping --config "$ROOT/turin.toml" >/dev/null 2>&1; then
  echo "Daemon did not start in time. Log: $ROOT/daemon.log" >&2
  exit 1
fi

REMOTE_ARGS=(--config "$ROOT/turin.toml" --bind "$BIND" --auth-token "$TOKEN" --event-keepalive-secs 2 --log-level error)
if [[ "$BIND" != 127.0.0.1:* && "$BIND" != localhost:* && "$BIND" != "[::1]"* ]]; then
  REMOTE_ARGS+=(--allow-non-loopback)
fi

"$REMOTE_BIN" "${REMOTE_ARGS[@]}" >"$ROOT/remote.log" 2>&1 &
REMOTE_PID=$!

for _ in {1..120}; do
  if curl -fsS "http://$BIND/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done

echo "[1/4] Checking public health endpoint..."
curl -fsS "http://$BIND/healthz" >/dev/null

echo "[2/4] Verifying auth challenge..."
UNAUTH_STATUS=$(curl -s -o /dev/null -w '%{http_code}' "http://$BIND/v1/health")
if [[ "$UNAUTH_STATUS" != "401" ]]; then
  echo "Expected 401 from unauthenticated /v1/health, got $UNAUTH_STATUS" >&2
  exit 1
fi

echo "[3/4] Proxying daemon ping through turin-remote..."
PING_RESPONSE=$(curl -fsS "http://$BIND/v1/daemon/request" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"op":"daemon.ping","params":{}}')
if [[ "$PING_RESPONSE" != *'"ok":true'* ]]; then
  echo "Unexpected daemon ping response: $PING_RESPONSE" >&2
  exit 1
fi

echo "[4/4] Confirming initial SSE snapshot..."
SSE_OUTPUT=$(curl -Ns --max-time 3 "http://$BIND/v1/events" \
  -H "Authorization: Bearer $TOKEN" || true)
if [[ "$SSE_OUTPUT" != *'event: runtime.snapshot'* ]]; then
  echo "Did not observe runtime.snapshot over SSE. Remote log: $ROOT/remote.log" >&2
  exit 1
fi

echo "turin-remote smoke succeeded."
