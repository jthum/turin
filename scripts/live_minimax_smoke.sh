#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/live_minimax_smoke.sh [options]

Manual (opt-in) live smoke tests for Turin against an Anthropic-compatible endpoint
such as MiniMax. This script does NOT run during normal cargo builds/tests.

Options:
  --env-file PATH        Source env vars from a file (e.g. ~/Documents/minimax.env)
  --binary PATH          Turin binary path (default: target/release/turin)
  --cases LIST           Comma-separated cases (default: basic,tool_read,tool_error,governed_denial)
                         Available: basic,tool_read,tool_error,tool_write_read,governed_denial,peer_agent,queue_steer
  --debug-requests       Enable Anthropic SDK request dumps (ANTHROPIC_SDK_DEBUG_REQUESTS=1)
  --keep-tmp             Keep temp directories after success/failure (for debugging)
  -h, --help             Show this help

Required env vars (via shell or --env-file):
  ANTHROPIC_API_KEY
  ANTHROPIC_BASE_URL
  ANTHROPIC_MODEL
USAGE
}

ENV_FILE=""
BINARY="target/release/turin"
CASES="basic,tool_read,tool_error,governed_denial"
DEBUG_REQUESTS=0
KEEP_TMP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="${2:-}"
      shift 2
      ;;
    --binary)
      BINARY="${2:-}"
      shift 2
      ;;
    --cases)
      CASES="${2:-}"
      shift 2
      ;;
    --debug-requests)
      DEBUG_REQUESTS=1
      shift
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
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is required}"
: "${ANTHROPIC_BASE_URL:?ANTHROPIC_BASE_URL is required}"
: "${ANTHROPIC_MODEL:?ANTHROPIC_MODEL is required}"

if [[ ! -x "$BINARY" ]]; then
  echo "Turin binary not found or not executable: $BINARY" >&2
  echo "Build it first: cargo build --release" >&2
  exit 1
fi

normalize_base_url() {
  local base="$1"
  while [[ "$base" == */ ]]; do
    base="${base%/}"
  done
  if [[ "$base" != */v1 ]]; then
    base="$base/v1"
  fi
  printf '%s\n' "$base"
}

ANTHROPIC_BASE_URL_NORM="$(normalize_base_url "$ANTHROPIC_BASE_URL")"
export ANTHROPIC_API_KEY ANTHROPIC_MODEL
if [[ "$DEBUG_REQUESTS" -eq 1 ]]; then
  export ANTHROPIC_SDK_DEBUG_REQUESTS=1
fi

TMP_DIRS=()
cleanup() {
  local rc=$?
  if [[ "$KEEP_TMP" -eq 0 ]]; then
    for d in "${TMP_DIRS[@]:-}"; do
      [[ -n "$d" && -d "$d" ]] && rm -rf "$d"
    done
  else
    printf '\nKept temp directories:\n'
    for d in "${TMP_DIRS[@]:-}"; do
      printf '  %s\n' "$d"
    done
  fi
  exit "$rc"
}
trap cleanup EXIT

make_temp_env() {
  local prefix="$1"
  local dir
  dir="$(mktemp -d "/tmp/${prefix}.XXXXXX")"
  TMP_DIRS+=("$dir")
  mkdir -p "$dir/harnesses" "$dir/work"
  printf '%s\n' '-- no-op harness' > "$dir/harnesses/main.lua"
  printf '%s\n' "$dir"
}

write_harness_main() {
  local dir="$1"
  local content="$2"
  printf '%s\n' "$content" > "$dir/harnesses/main.lua"
}

write_config() {
  local dir="$1"
  local system_prompt="$2"
  local max_turns="$3"
  cat > "$dir/turin.toml" <<EOF_CFG
[agent]
system_prompt = "$system_prompt"
model = "$ANTHROPIC_MODEL"
provider = "anthropic"

[kernel]
workspace_root = "$dir/work"
max_turns = $max_turns

[persistence]
database_path = "$dir/state.db"

[harness]
directory = "$dir/harnesses"

[providers.anthropic]
type = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
base_url = "$ANTHROPIC_BASE_URL_NORM"
EOF_CFG
}

run_basic() {
  local dir out
  dir="$(make_temp_env turin-live-basic)"
  write_config "$dir" "You are a concise assistant. Reply exactly as instructed." 3
  out="$dir/out.txt"

  printf '\n[CASE] basic\n'
  "$BINARY" run --config "$dir/turin.toml" --prompt "Reply with exactly: PONG" --log-level warn | tee "$out"

  if rg -q '^PONG$' "$out"; then
    printf '[PASS] basic (tmp=%s)\n' "$dir"
  else
    printf '[FAIL] basic (tmp=%s)\n' "$dir" >&2
    return 1
  fi
}

run_tool_read() {
  local dir out nonce prompt
  dir="$(make_temp_env turin-live-toolread)"
  write_config "$dir" "You are a coding assistant. Use tools when needed." 8
  nonce="MINIMAX_TOOL_NONCE_$(date +%s)_$RANDOM"
  printf '%s\n' "$nonce" > "$dir/work/nonce.txt"
  out="$dir/out.txt"
  prompt='Use the read_file tool to read nonce.txt and then reply with exactly the file contents, with no extra text.'

  printf '\n[CASE] tool_read\n'
  "$BINARY" run --config "$dir/turin.toml" --prompt "$prompt" --log-level warn | tee "$out"

  if rg -q "^${nonce}$" "$out"; then
    printf '[PASS] tool_read (tmp=%s)\n' "$dir"
  else
    printf '[FAIL] tool_read (tmp=%s)\n' "$dir" >&2
    return 1
  fi
}

run_tool_error() {
  local dir out prompt
  dir="$(make_temp_env turin-live-toolerror)"
  write_config "$dir" "You are a coding assistant. Use tools when explicitly instructed." 8
  out="$dir/out.txt"
  prompt='First, call the read_file tool on missing_file_that_does_not_exist_12345.txt. Then reply with exactly: TOOL_ERROR_OK'

  printf '\n[CASE] tool_error\n'
  "$BINARY" run --config "$dir/turin.toml" --prompt "$prompt" --log-level warn | tee "$out"

  if rg -q '^TOOL_ERROR_OK$' "$out"; then
    printf '[PASS] tool_error (tmp=%s)\n' "$dir"
  else
    printf '[FAIL] tool_error (tmp=%s)\n' "$dir" >&2
    return 1
  fi
}

run_tool_write_read() {
  local dir out nonce prompt
  dir="$(make_temp_env turin-live-toolwriteread)"
  write_config "$dir" "You are a coding assistant. Use tools when needed and follow exact output instructions." 10
  nonce="MINIMAX_WRITE_READ_NONCE_$(date +%s)_$RANDOM"
  out="$dir/out.txt"
  prompt="Use write_file to create nonce2.txt with exactly this content: ${nonce}. Then use read_file to read nonce2.txt and reply with exactly the file contents, no extra text."

  printf '\n[CASE] tool_write_read\n'
  "$BINARY" run --config "$dir/turin.toml" --prompt "$prompt" --log-level warn | tee "$out"

  if [[ ! -f "$dir/work/nonce2.txt" ]]; then
    printf '[FAIL] tool_write_read (tmp=%s) nonce2.txt missing\n' "$dir" >&2
    return 1
  fi
  if ! rg -q "^${nonce}$" "$dir/work/nonce2.txt"; then
    printf '[FAIL] tool_write_read (tmp=%s) file contents mismatch\n' "$dir" >&2
    return 1
  fi
  if rg -q "^${nonce}$" "$out"; then
    printf '[PASS] tool_write_read (tmp=%s)\n' "$dir"
  else
    printf '[FAIL] tool_write_read (tmp=%s)\n' "$dir" >&2
    return 1
  fi
}

run_governed_denial() {
  local dir out prompt
  dir="$(make_temp_env turin-live-governed)"
  write_config "$dir" "You are a concise assistant. Reply exactly as instructed." 4
  cat >> "$dir/turin.toml" <<'EOF_GOV'

[governance]
profile = "governed"
enforcement_enabled = true
EOF_GOV
  write_harness_main "$dir" '
function on_session_start(event)
  local ok, err = runtime.policy.set("queue.max_depth", 1)
  if ok then
    log("GOVERNED_DENIAL_UNEXPECTED_ALLOW")
  elseif err and tostring(err):find("Governance denial", 1, true) then
    log("GOVERNED_DENIAL_OK")
  else
    log("GOVERNED_DENIAL_UNEXPECTED_ERR:" .. tostring(err))
  end
  return ALLOW
end
'
  out="$dir/out.txt"
  prompt='Reply with exactly: GOVERNED_MAIN_OK'

  printf '\n[CASE] governed_denial\n'
  "$BINARY" run --config "$dir/turin.toml" --prompt "$prompt" --log-level warn 2>&1 | tee "$out"

  if ! rg -q '^\[harness\] GOVERNED_DENIAL_OK$' "$out"; then
    printf '[FAIL] governed_denial (tmp=%s) missing denial sentinel\n' "$dir" >&2
    return 1
  fi
  if rg -q '^GOVERNED_MAIN_OK$' "$out"; then
    printf '[PASS] governed_denial (tmp=%s)\n' "$dir"
  else
    printf '[FAIL] governed_denial (tmp=%s) main response mismatch\n' "$dir" >&2
    return 1
  fi
}

run_peer_agent() {
  local dir out prompt
  dir="$(make_temp_env turin-live-peeragent)"
  write_config "$dir" "You are a concise assistant. Reply exactly as instructed." 6
  cat >> "$dir/turin.toml" <<EOF_PEER

[agents.worker]
id = "worker"
system_prompt = "You are a worker agent. Follow exact output instructions."
model = "$ANTHROPIC_MODEL"
provider = "anthropic"
mode = "stateful"
EOF_PEER
  write_harness_main "$dir" '
function on_session_start(event)
  if not event or not event.identity or event.identity.agent_id ~= "default" then
    return ALLOW
  end

  local out, err = agent.complete(
    "Reply with exactly: PEER_AGENT_WORKER_OK",
    { agent_id = "worker", timeout_ms = 45000 }
  )

  if out == "PEER_AGENT_WORKER_OK" then
    log("PEER_AGENT_SMOKE_OK")
  else
    log("PEER_AGENT_SMOKE_FAIL:" .. tostring(err or out))
  end
  return ALLOW
end
'
  out="$dir/out.txt"
  prompt='Reply with exactly: PEER_AGENT_MAIN_OK'

  printf '\n[CASE] peer_agent\n'
  "$BINARY" run --config "$dir/turin.toml" --prompt "$prompt" --log-level warn 2>&1 | tee "$out"

  if ! rg -q '^\[harness\] PEER_AGENT_SMOKE_OK$' "$out"; then
    printf '[FAIL] peer_agent (tmp=%s) missing peer success sentinel\n' "$dir" >&2
    return 1
  fi
  if rg -q '^PEER_AGENT_MAIN_OK$' "$out"; then
    printf '[PASS] peer_agent (tmp=%s)\n' "$dir"
  else
    printf '[FAIL] peer_agent (tmp=%s) main response mismatch\n' "$dir" >&2
    return 1
  fi
}

run_queue_steer() {
  local dir out prompt
  dir="$(make_temp_env turin-live-queuesteer)"
  write_config "$dir" "You are a concise assistant. Reply exactly as instructed." 8
  write_harness_main "$dir" '
local queued_once = false

function on_all_tasks_complete(event)
  if queued_once then
    return ALLOW
  end
  queued_once = true
  log("QUEUE_STEER_HOOK_OK")
  return MODIFY, {
    "Reply with exactly: QUEUE_STEER_FOLLOWUP_OK"
  }
end
'
  out="$dir/out.txt"
  prompt='Reply with exactly: QUEUE_STEER_MAIN_OK'

  printf '\n[CASE] queue_steer\n'
  "$BINARY" run --config "$dir/turin.toml" --prompt "$prompt" --log-level warn 2>&1 | tee "$out"

  if ! rg -q '^\[harness\] QUEUE_STEER_HOOK_OK$' "$out"; then
    printf '[FAIL] queue_steer (tmp=%s) missing queue steering sentinel\n' "$dir" >&2
    return 1
  fi
  if ! rg -q '^QUEUE_STEER_MAIN_OK$' "$out"; then
    printf '[FAIL] queue_steer (tmp=%s) missing main response\n' "$dir" >&2
    return 1
  fi
  if rg -q '^QUEUE_STEER_FOLLOWUP_OK$' "$out"; then
    printf '[PASS] queue_steer (tmp=%s)\n' "$dir"
  else
    printf '[FAIL] queue_steer (tmp=%s) missing followup response\n' "$dir" >&2
    return 1
  fi
}

IFS=',' read -r -a CASE_LIST <<< "$CASES"
PASS_COUNT=0
FAIL_COUNT=0

printf 'Turin live smoke tests (manual/opt-in)\n'
printf 'Binary: %s\n' "$BINARY"
printf 'Model: %s\n' "$ANTHROPIC_MODEL"
printf 'Base URL: %s\n' "$ANTHROPIC_BASE_URL_NORM"

for case_name in "${CASE_LIST[@]}"; do
  case_name="${case_name// /}"
  if [[ -z "$case_name" ]]; then
    continue
  fi

  if "run_${case_name}"; then
    PASS_COUNT=$((PASS_COUNT + 1))
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    # Stop on first failure to preserve the failing temp dir context if --keep-tmp is set.
    break
  fi
done

printf '\nSummary: %d passed, %d failed\n' "$PASS_COUNT" "$FAIL_COUNT"
[[ "$FAIL_COUNT" -eq 0 ]]
