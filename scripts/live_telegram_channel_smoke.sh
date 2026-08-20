#!/usr/bin/env bash
set -euo pipefail

BIN="target/release/turin"
CHANNEL_BIN="target/release/turin-channel-telegram"
ENV_FILE=""
TOKEN_ENV_NAME="${TELEGRAM_TOKEN_ENV_NAME:-TELEGRAM_BOT_TOKEN}"
CHAT_ID="${TELEGRAM_CHAT_ID:-}"
WORKSPACE_ROOT=""
KEEP_TMP=0

usage() {
  cat <<'USAGE'
Usage: scripts/live_telegram_channel_smoke.sh [options]

Options:
  --bin PATH                 Turin binary path (default: target/release/turin)
  --channel-bin PATH         Telegram runner path (default: target/release/turin-channel-telegram)
  --env-file PATH            Source environment variables before running
  --token-env-name NAME      Env var name holding Telegram bot token (default: TELEGRAM_BOT_TOKEN)
  --chat-id ID               Telegram numeric chat id (or set TELEGRAM_CHAT_ID)
  --workspace-root PATH      Optional existing workspace root
  --keep-tmp                 Keep temporary workspace for inspection
  -h, --help                 Show this help

Required:
  - Telegram bot token must be set in the selected token env var.
  - Telegram numeric chat id must be provided.

Notes:
  - This validates the independent Telegram runner startup path.
  - If the bot still has an active webhook, Telegram polling will fail until the webhook is removed.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin)
      BIN="$2"
      shift 2
      ;;
    --channel-bin)
      CHANNEL_BIN="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --token-env-name)
      TOKEN_ENV_NAME="$2"
      shift 2
      ;;
    --chat-id)
      CHAT_ID="$2"
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

if [[ -n "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

if [[ -z "$CHAT_ID" ]]; then
  echo "Missing Telegram chat id. Use --chat-id or TELEGRAM_CHAT_ID." >&2
  exit 1
fi

if [[ ! "$CHAT_ID" =~ ^-?[0-9]+$ ]]; then
  echo "Telegram chat id must be numeric: $CHAT_ID" >&2
  exit 1
fi

if [[ -z "${!TOKEN_ENV_NAME:-}" ]]; then
  echo "Missing Telegram token env '$TOKEN_ENV_NAME'." >&2
  exit 1
fi

if [[ ! -x "$BIN" ]]; then
  echo "Turin binary not found or not executable: $BIN" >&2
  exit 1
fi

if [[ ! -x "$CHANNEL_BIN" ]]; then
  echo "Telegram channel runner not found or not executable: $CHANNEL_BIN" >&2
  exit 1
fi

if [[ -n "$WORKSPACE_ROOT" ]]; then
  ROOT="$WORKSPACE_ROOT"
  mkdir -p "$ROOT"
else
  ROOT="$(mktemp -d /tmp/turin-telegram-live.XXXXXX)"
fi

cleanup() {
  set +e
  if [[ -n "${CHANNEL_PID:-}" ]]; then
    kill "$CHANNEL_PID" >/dev/null 2>&1 || true
    wait "$CHANNEL_PID" >/dev/null 2>&1 || true
  fi
  "$BIN" daemon stop --config "$ROOT/turin.toml" >/dev/null 2>&1 || true
  if [[ $KEEP_TMP -eq 0 && -z "$WORKSPACE_ROOT" ]]; then
    rm -rf "$ROOT"
  fi
}
trap cleanup EXIT

mkdir -p "$ROOT/.turin/harnesses"
cat > "$ROOT/.turin/harnesses/main.lua" <<'LUA'
-- live telegram channel smoke harness
function on_turn_prepare(ctx)
  return ALLOW
end
LUA

cat > "$ROOT/turin.toml" <<EOF
[agent]
id = "default"
system_prompt = "You are a concise assistant."
model = "mock-model"
provider = "mock"

[kernel]
workspace_root = "$ROOT"

[persistence.state]
path = "$ROOT/.turin/state.db"

[harness]
directory = "$ROOT/.turin/harnesses"
fs_root = "."

[providers.mock]
type = "mock"
base_url = "PONG"
EOF

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
  kill "$DAEMON_PID" >/dev/null 2>&1 || true
  wait "$DAEMON_PID" >/dev/null 2>&1 || true
  exit 1
fi

CHANNEL_DIR="$ROOT/.turin/channels/telegram-live"
mkdir -p "$CHANNEL_DIR"
cat > "$CHANNEL_DIR/config.toml" <<EOF
enabled = true
kind = "telegram"
agent_id = "default"
token_env = "$TOKEN_ENV_NAME"
chat_id = $CHAT_ID
start_from_latest = true
poll_timeout_seconds = 10
EOF

echo "[1/3] Starting independent Telegram channel runner..."
"$CHANNEL_BIN" run \
  --config "$CHANNEL_DIR/config.toml" \
  --turin-config "$ROOT/turin.toml" >"$ROOT/channel.log" 2>&1 &
CHANNEL_PID=$!

echo "[2/3] Checking runner stability..."
sleep 3
if ! kill -0 "$CHANNEL_PID" >/dev/null 2>&1; then
  wait "$CHANNEL_PID" || true
  echo "Telegram runner exited during startup. See $ROOT/channel.log" >&2
  exit 1
fi

echo "[3/3] Runner is active (pid $CHANNEL_PID)."
echo "Telegram channel smoke succeeded."
