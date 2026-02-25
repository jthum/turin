#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/live_minimax_smoke.sh [options]

Manual (opt-in) live suite tests for Turin against MiniMax Anthropic-compatible
or OpenAI-compatible endpoints. This script does NOT run during normal cargo
builds/tests.

Options:
  --env-file PATH        Source env vars from a file (e.g. ~/Documents/minimax.env)
  --binary PATH          Turin binary path (default: target/release/turin)
  --api-format NAME      Provider wire format: anthropic|openai (default: anthropic)
  --suite NAME           Suite preset: smoke|core|all (default: smoke)
  --log-level LEVEL      Turin log level for live runs (default: error)
  --report-json PATH     Write a JSON summary report to PATH (use - for stdout)
  --cases LIST           Comma-separated cases (default: basic,tool_read,tool_error,governed_denial)
                         Available: basic,tool_read,tool_error,tool_write_read,governed_denial,peer_agent,queue_steer,runtime_db,grant_flow,token_reject_task,immutable_audit,peer_grant
  --debug-requests       Enable Anthropic SDK request dumps (ANTHROPIC_SDK_DEBUG_REQUESTS=1)
  --keep-tmp             Keep temp directories after success/failure (for debugging)
  -h, --help             Show this help

Required env vars (via shell or --env-file):
  Default (anthropic):
    ANTHROPIC_API_KEY
    ANTHROPIC_BASE_URL
    ANTHROPIC_MODEL
  OpenAI-compatible mode (--api-format openai):
    OPENAI_BASE_URL
    OPENAI_MODEL (optional; falls back to ANTHROPIC_MODEL)
    OPENAI_API_KEY (optional; falls back to ANTHROPIC_API_KEY)
USAGE
}

ENV_FILE=""
BINARY="target/release/turin"
API_FORMAT="anthropic"
CASES="basic,tool_read,tool_error,governed_denial"
SUITE="smoke"
CASES_EXPLICIT=0
DEBUG_REQUESTS=0
KEEP_TMP=0
LOG_LEVEL="error"
REPORT_JSON=""

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
    --api-format)
      API_FORMAT="${2:-}"
      shift 2
      ;;
    --suite)
      SUITE="${2:-}"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="${2:-}"
      shift 2
      ;;
    --report-json)
      REPORT_JSON="${2:-}"
      shift 2
      ;;
    --cases)
      CASES="${2:-}"
      CASES_EXPLICIT=1
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

case "$API_FORMAT" in
  anthropic|openai) ;;
  *)
    echo "Unknown --api-format: $API_FORMAT (expected anthropic|openai)" >&2
    exit 2
    ;;
esac

case "$SUITE" in
  smoke) ;;
  core) ;;
  all) ;;
  "")
    SUITE="smoke"
    ;;
  *)
    echo "Unknown suite: $SUITE (expected smoke|core|all)" >&2
    exit 2
    ;;
esac

SMOKE_CASES="basic,tool_read,tool_error,governed_denial"
CORE_CASES="basic,tool_read,tool_error,tool_write_read,governed_denial,peer_agent,queue_steer,runtime_db,grant_flow,token_reject_task,immutable_audit,peer_grant"
ALL_CASES="$CORE_CASES"

# If --cases was not explicitly set, derive from --suite.
if [[ "$CASES_EXPLICIT" -eq 0 ]]; then
  case "$SUITE" in
    smoke) CASES="$SMOKE_CASES" ;;
    core) CASES="$CORE_CASES" ;;
    all) CASES="$ALL_CASES" ;;
  esac
fi

if [[ -n "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

LIVE_API_KEY=""
LIVE_BASE_URL=""
LIVE_MODEL=""
TURIN_PROVIDER_KIND=""

case "$API_FORMAT" in
  anthropic)
    : "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is required}"
    : "${ANTHROPIC_BASE_URL:?ANTHROPIC_BASE_URL is required}"
    : "${ANTHROPIC_MODEL:?ANTHROPIC_MODEL is required}"
    LIVE_API_KEY="$ANTHROPIC_API_KEY"
    LIVE_BASE_URL="$ANTHROPIC_BASE_URL"
    LIVE_MODEL="$ANTHROPIC_MODEL"
    TURIN_PROVIDER_KIND="anthropic"
    ;;
  openai)
    : "${OPENAI_BASE_URL:?OPENAI_BASE_URL is required for --api-format openai}"
    if [[ -n "${OPENAI_API_KEY:-}" ]]; then
      LIVE_API_KEY="$OPENAI_API_KEY"
    else
      : "${ANTHROPIC_API_KEY:?OPENAI_API_KEY or ANTHROPIC_API_KEY is required for --api-format openai}"
      LIVE_API_KEY="$ANTHROPIC_API_KEY"
    fi
    LIVE_BASE_URL="$OPENAI_BASE_URL"
    if [[ -n "${OPENAI_MODEL:-}" ]]; then
      LIVE_MODEL="$OPENAI_MODEL"
    else
      : "${ANTHROPIC_MODEL:?OPENAI_MODEL or ANTHROPIC_MODEL is required for --api-format openai}"
      LIVE_MODEL="$ANTHROPIC_MODEL"
    fi
    TURIN_PROVIDER_KIND="openai"
    ;;
esac

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

LIVE_BASE_URL_NORM="$(normalize_base_url "$LIVE_BASE_URL")"
export TURIN_LIVE_API_KEY="$LIVE_API_KEY"
if [[ "$DEBUG_REQUESTS" -eq 1 ]]; then
  # SDK request dumps are useful for provider-compat debugging.
  export ANTHROPIC_SDK_DEBUG_REQUESTS=1
  export OPENAI_SDK_DEBUG_REQUESTS=1
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

sqlite_scalar() {
  local db="$1"
  local sql="$2"
  sqlite3 -batch -noheader "$db" "$sql" | tr -d '\r\n'
}

qmatch() {
  local pattern="$1"
  local file="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -q -- "$pattern" "$file"
  else
    grep -Eq -- "$pattern" "$file"
  fi
}

json_escape() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\r'/\\r}"
  s="${s//$'\t'/\\t}"
  printf '%s' "$s"
}

write_report_json() {
  local rc="$1"
  [[ -z "$REPORT_JSON" ]] && return 0

  local generated_at
  generated_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  local json
  json="{"
  json+="\"generated_at\":\"$(json_escape "$generated_at")\","
  json+="\"binary\":\"$(json_escape "$BINARY")\","
  json+="\"api_format\":\"$(json_escape "$API_FORMAT")\","
  json+="\"provider_kind\":\"$(json_escape "$TURIN_PROVIDER_KIND")\","
  json+="\"model\":\"$(json_escape "$LIVE_MODEL")\","
  json+="\"base_url\":\"$(json_escape "$LIVE_BASE_URL_NORM")\","
  json+="\"suite\":\"$(json_escape "$SUITE")\","
  json+="\"cases_explicit\":$([[ "$CASES_EXPLICIT" -eq 1 ]] && printf true || printf false),"
  json+="\"cases_requested\":\"$(json_escape "$CASES")\","
  json+="\"log_level\":\"$(json_escape "$LOG_LEVEL")\","
  json+="\"pass_count\":$PASS_COUNT,"
  json+="\"fail_count\":$FAIL_COUNT,"
  json+="\"exit_code\":$rc,"
  json+="\"cases\":["

  local i
  for ((i = 0; i < ${#CASE_RESULT_NAMES[@]}; i++)); do
    [[ "$i" -gt 0 ]] && json+=","
    json+="{"
    json+="\"name\":\"$(json_escape "${CASE_RESULT_NAMES[$i]}")\","
    json+="\"status\":\"$(json_escape "${CASE_RESULT_STATUS[$i]}")\","
    json+="\"duration_sec\":${CASE_RESULT_DURATIONS[$i]}"
    if [[ -n "${CASE_RESULT_TMPS[$i]}" ]]; then
      json+=",\"tmp_dir\":\"$(json_escape "${CASE_RESULT_TMPS[$i]}")\""
    fi
    json+="}"
  done
  json+="]}"

  if [[ "$REPORT_JSON" == "-" ]]; then
    printf '%s\n' "$json"
  else
    printf '%s\n' "$json" > "$REPORT_JSON"
    printf 'Wrote JSON report: %s\n' "$REPORT_JSON"
  fi
}

run_turin_capture() {
  local out="$1"
  shift

  local max_attempts="${TURIN_LIVE_RETRIES:-3}"
  local attempt=1
  : > "$out"

  while true; do
    if [[ "$attempt" -gt 1 ]]; then
      printf '[retry] rerunning after transient network/provider transport failure (attempt %d/%d)\n' \
        "$attempt" "$max_attempts" | tee -a "$out"
    fi

    set +e
    "$@" 2>&1 | tee -a "$out"
    local rc="${PIPESTATUS[0]}"
    set -e

    if [[ "$rc" -eq 0 ]]; then
      return 0
    fi

    if [[ "$attempt" -ge "$max_attempts" ]]; then
      return "$rc"
    fi

    if qmatch 'Network error:|dns error|Temporary failure in name resolution|error sending request for url|Connection reset by peer|timed out' "$out"; then
      sleep "$attempt"
      attempt=$((attempt + 1))
      continue
    fi

    return "$rc"
  done
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
model = "$LIVE_MODEL"
provider = "$TURIN_PROVIDER_KIND"

[kernel]
workspace_root = "$dir/work"
max_turns = $max_turns

[persistence]
database_path = "$dir/state.db"

[harness]
directory = "$dir/harnesses"

[providers.$TURIN_PROVIDER_KIND]
type = "$TURIN_PROVIDER_KIND"
api_key_env = "TURIN_LIVE_API_KEY"
base_url = "$LIVE_BASE_URL_NORM"
EOF_CFG
}

run_basic() {
  local dir out
  dir="$(make_temp_env turin-live-basic)"
  LAST_CASE_TMP="$dir"
  write_config "$dir" "You are a concise assistant. Reply exactly as instructed." 3
  out="$dir/out.txt"

  printf '\n[CASE] basic\n'
  run_turin_capture "$out" \
    "$BINARY" run --config "$dir/turin.toml" --prompt "Reply with exactly: PONG" --log-level "$LOG_LEVEL"

  if qmatch '^PONG$' "$out"; then
    printf '[PASS] basic (tmp=%s)\n' "$dir"
  else
    printf '[FAIL] basic (tmp=%s)\n' "$dir" >&2
    return 1
  fi
}

run_tool_read() {
  local dir out nonce prompt
  dir="$(make_temp_env turin-live-toolread)"
  LAST_CASE_TMP="$dir"
  write_config "$dir" "You are a coding assistant. Use tools when needed." 8
  nonce="MINIMAX_TOOL_NONCE_$(date +%s)_$RANDOM"
  printf '%s\n' "$nonce" > "$dir/work/nonce.txt"
  out="$dir/out.txt"
  prompt='Use the read_file tool to read nonce.txt and then reply with exactly the file contents, with no extra text.'

  printf '\n[CASE] tool_read\n'
  run_turin_capture "$out" \
    "$BINARY" run --config "$dir/turin.toml" --prompt "$prompt" --log-level "$LOG_LEVEL"

  if qmatch "^${nonce}$" "$out"; then
    printf '[PASS] tool_read (tmp=%s)\n' "$dir"
  else
    printf '[FAIL] tool_read (tmp=%s)\n' "$dir" >&2
    return 1
  fi
}

run_tool_error() {
  local dir out prompt
  dir="$(make_temp_env turin-live-toolerror)"
  LAST_CASE_TMP="$dir"
  write_config "$dir" "You are a coding assistant. Use tools when explicitly instructed." 8
  out="$dir/out.txt"
  prompt='First, call the read_file tool on missing_file_that_does_not_exist_12345.txt. Then reply with exactly: TOOL_ERROR_OK'

  printf '\n[CASE] tool_error\n'
  run_turin_capture "$out" \
    "$BINARY" run --config "$dir/turin.toml" --prompt "$prompt" --log-level "$LOG_LEVEL"

  if qmatch '^TOOL_ERROR_OK$' "$out"; then
    printf '[PASS] tool_error (tmp=%s)\n' "$dir"
  else
    printf '[FAIL] tool_error (tmp=%s)\n' "$dir" >&2
    return 1
  fi
}

run_tool_write_read() {
  local dir out nonce prompt
  dir="$(make_temp_env turin-live-toolwriteread)"
  LAST_CASE_TMP="$dir"
  write_config "$dir" "You are a coding assistant. Use tools when needed and follow exact output instructions." 10
  nonce="MINIMAX_WRITE_READ_NONCE_$(date +%s)_$RANDOM"
  out="$dir/out.txt"
  prompt="Use write_file to create nonce2.txt with exactly this content: ${nonce}. Then use read_file to read nonce2.txt and reply with exactly the file contents, no extra text."

  printf '\n[CASE] tool_write_read\n'
  run_turin_capture "$out" \
    "$BINARY" run --config "$dir/turin.toml" --prompt "$prompt" --log-level "$LOG_LEVEL"

  if [[ ! -f "$dir/work/nonce2.txt" ]]; then
    printf '[FAIL] tool_write_read (tmp=%s) nonce2.txt missing\n' "$dir" >&2
    return 1
  fi
  if ! qmatch "^${nonce}$" "$dir/work/nonce2.txt"; then
    printf '[FAIL] tool_write_read (tmp=%s) file contents mismatch\n' "$dir" >&2
    return 1
  fi
  if qmatch "^${nonce}$" "$out"; then
    printf '[PASS] tool_write_read (tmp=%s)\n' "$dir"
  else
    printf '[FAIL] tool_write_read (tmp=%s)\n' "$dir" >&2
    return 1
  fi
}

run_governed_denial() {
  local dir out prompt
  dir="$(make_temp_env turin-live-governed)"
  LAST_CASE_TMP="$dir"
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
  run_turin_capture "$out" \
    "$BINARY" run --config "$dir/turin.toml" --prompt "$prompt" --log-level "$LOG_LEVEL"

  if ! qmatch '^\[harness\] GOVERNED_DENIAL_OK$' "$out"; then
    printf '[FAIL] governed_denial (tmp=%s) missing denial sentinel\n' "$dir" >&2
    return 1
  fi
  if qmatch '^GOVERNED_MAIN_OK$' "$out"; then
    printf '[PASS] governed_denial (tmp=%s)\n' "$dir"
  else
    printf '[FAIL] governed_denial (tmp=%s) main response mismatch\n' "$dir" >&2
    return 1
  fi
}

run_peer_agent() {
  local dir out prompt
  dir="$(make_temp_env turin-live-peeragent)"
  LAST_CASE_TMP="$dir"
  write_config "$dir" "You are a concise assistant. Reply exactly as instructed." 6
  cat >> "$dir/turin.toml" <<EOF_PEER

[agents.worker]
id = "worker"
system_prompt = "You are a worker agent. Follow exact output instructions."
model = "$LIVE_MODEL"
provider = "$TURIN_PROVIDER_KIND"
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

  if type(out) == "string" and string.find(out, "PEER_AGENT_WORKER_OK", 1, true) then
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
  run_turin_capture "$out" \
    "$BINARY" run --config "$dir/turin.toml" --prompt "$prompt" --log-level "$LOG_LEVEL"

  if ! qmatch '^\[harness\] PEER_AGENT_SMOKE_OK$' "$out"; then
    printf '[FAIL] peer_agent (tmp=%s) missing peer success sentinel\n' "$dir" >&2
    return 1
  fi
  if qmatch '^PEER_AGENT_MAIN_OK$' "$out"; then
    printf '[PASS] peer_agent (tmp=%s)\n' "$dir"
  else
    printf '[FAIL] peer_agent (tmp=%s) main response mismatch\n' "$dir" >&2
    return 1
  fi
}

run_queue_steer() {
  local dir out prompt
  dir="$(make_temp_env turin-live-queuesteer)"
  LAST_CASE_TMP="$dir"
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
  run_turin_capture "$out" \
    "$BINARY" run --config "$dir/turin.toml" --prompt "$prompt" --log-level "$LOG_LEVEL"

  if ! qmatch '^\[harness\] QUEUE_STEER_HOOK_OK$' "$out"; then
    printf '[FAIL] queue_steer (tmp=%s) missing queue steering sentinel\n' "$dir" >&2
    return 1
  fi
  if ! qmatch '^QUEUE_STEER_MAIN_OK$' "$out"; then
    printf '[FAIL] queue_steer (tmp=%s) missing main response\n' "$dir" >&2
    return 1
  fi
  if qmatch '^QUEUE_STEER_FOLLOWUP_OK$' "$out"; then
    printf '[PASS] queue_steer (tmp=%s)\n' "$dir"
  else
    printf '[FAIL] queue_steer (tmp=%s) missing followup response\n' "$dir" >&2
    return 1
  fi
}

run_runtime_db() {
  local dir out nonce db_file row_count file_nonce
  dir="$(make_temp_env turin-live-runtimedb)"
  LAST_CASE_TMP="$dir"
  write_config "$dir" "You are a concise assistant. Reply exactly as instructed." 6
  nonce="RUNTIME_DB_NONCE_$(date +%s)_$RANDOM"
  write_harness_main "$dir" "
function on_session_start(event)
  local h, he = runtime.db.open({ path = \"scratch/live_runtime.db\" })
  if h == nil then
    log(\"RUNTIME_DB_FAIL:open:\" .. tostring(he))
    return ALLOW
  end

  local handles, le = runtime.db.list()
  if handles == nil then
    log(\"RUNTIME_DB_FAIL:list:\" .. tostring(le))
    return ALLOW
  end

  local changed1, e1 = runtime.db.exec(
    \"create table if not exists items (id integer primary key, v text not null)\",
    nil,
    { handle = h.handle }
  )
  if changed1 == nil then
    log(\"RUNTIME_DB_FAIL:create:\" .. tostring(e1))
    return ALLOW
  end

  local changed2, e2 = runtime.db.exec(
    \"insert into items (v) values (?1)\",
    { \"${nonce}\" },
    { handle = h.handle }
  )
  if changed2 == nil then
    log(\"RUNTIME_DB_FAIL:insert:\" .. tostring(e2))
    return ALLOW
  end

  local rows, qe = runtime.db.query(
    \"select v from items order by id desc limit 1\",
    nil,
    { handle = h.handle }
  )
  if rows == nil then
    log(\"RUNTIME_DB_FAIL:query:\" .. tostring(qe))
    return ALLOW
  end
  if rows[1] == nil or rows[1].v ~= \"${nonce}\" then
    log(\"RUNTIME_DB_FAIL:row_mismatch:\" .. tostring(rows[1] and rows[1].v))
    return ALLOW
  end

  local closed, ce = runtime.db.close(h.handle)
  if closed ~= true then
    log(\"RUNTIME_DB_FAIL:close:\" .. tostring(ce))
    return ALLOW
  end

  log(\"RUNTIME_DB_OK:${nonce}\")
  return ALLOW
end
"
  out="$dir/out.txt"
  db_file="$dir/work/scratch/live_runtime.db"

  printf '\n[CASE] runtime_db\n'
  run_turin_capture "$out" \
    "$BINARY" run --config "$dir/turin.toml" --prompt 'Reply with exactly: RUNTIME_DB_MAIN_OK' --log-level "$LOG_LEVEL"

  if ! qmatch "^\[harness\] RUNTIME_DB_OK:${nonce}$" "$out"; then
    printf '[FAIL] runtime_db (tmp=%s) missing runtime.db success sentinel\n' "$dir" >&2
    return 1
  fi
  if ! qmatch '^RUNTIME_DB_MAIN_OK$' "$out"; then
    printf '[FAIL] runtime_db (tmp=%s) main response mismatch\n' "$dir" >&2
    return 1
  fi
  if [[ ! -f "$db_file" ]]; then
    printf '[FAIL] runtime_db (tmp=%s) expected db file missing\n' "$dir" >&2
    return 1
  fi
  row_count="$(sqlite_scalar "$db_file" "select count(*) from items;")"
  file_nonce="$(sqlite_scalar "$db_file" "select v from items order by id desc limit 1;")"
  if [[ "$row_count" -lt 1 ]] || [[ "$file_nonce" != "$nonce" ]]; then
    printf '[FAIL] runtime_db (tmp=%s) sqlite verification failed (count=%s nonce=%s)\n' "$dir" "$row_count" "$file_nonce" >&2
    return 1
  fi
  printf '[PASS] runtime_db (tmp=%s)\n' "$dir"
}

run_grant_flow() {
  local dir out audit_issue audit_use audit_revoke
  dir="$(make_temp_env turin-live-grantflow)"
  LAST_CASE_TMP="$dir"
  write_config "$dir" "You are a concise assistant. Reply exactly as instructed." 6
  cat >> "$dir/turin.toml" <<'EOF_GOV'

[governance]
profile = "balanced"
enforcement_enabled = true

[governance.grants]
enabled = true
max_ttl_ms = 60000
require_audit_reason = true
EOF_GOV
  write_harness_main "$dir" '
local grant_checked = false

function on_turn_prepare(ctx)
  if grant_checked then
    return ALLOW
  end
  grant_checked = true

  local before, be = runtime.governance.check("runtime.policy.set")
  if before == nil then
    log("GRANT_FLOW_FAIL:before_check:" .. tostring(be))
    return ALLOW
  end
  if not before.allowed then
    log("GRANT_FLOW_FAIL:before_denied")
    return ALLOW
  end

  local grant, ge = runtime.governance.grant_issue({
    capabilities = { ["runtime.db.query"] = true },
    ttl_ms = 30000,
    max_uses = 2,
    reason = "live grant flow"
  })
  if grant == nil then
    log("GRANT_FLOW_FAIL:grant_issue:" .. tostring(ge))
    return ALLOW
  end

  local cb = runtime.governance.with_grant(grant.grant_id, function()
    local inside, ie = runtime.governance.check("runtime.policy.set")
    if inside == nil then error("inside check failed: " .. tostring(ie)) end
    if inside.allowed then error("runtime.policy.set should be denied inside grant") end

    local ok, err = runtime.policy.set("grant.live.test", true)
    if ok ~= false then error("runtime.policy.set unexpectedly allowed inside grant") end
    if err == nil then error("runtime.policy.set missing denial error inside grant") end
    return "GRANT_FLOW_CB_OK"
  end)
  if cb ~= "GRANT_FLOW_CB_OK" then
    log("GRANT_FLOW_FAIL:callback:" .. tostring(cb))
    return ALLOW
  end

  local after, ae = runtime.governance.check("runtime.policy.set")
  if after == nil then
    log("GRANT_FLOW_FAIL:after_check:" .. tostring(ae))
    return ALLOW
  end
  if not after.allowed then
    log("GRANT_FLOW_FAIL:after_denied")
    return ALLOW
  end

  local revoked, re = runtime.governance.grant_revoke(grant.grant_id)
  if revoked ~= true then
    log("GRANT_FLOW_FAIL:grant_revoke:" .. tostring(re))
    return ALLOW
  end

  log("GRANT_FLOW_OK")
  return ALLOW
end
'
  out="$dir/out.txt"

  printf '\n[CASE] grant_flow\n'
  run_turin_capture "$out" \
    "$BINARY" run --config "$dir/turin.toml" --prompt 'Reply with exactly: GRANT_FLOW_MAIN_OK' --log-level "$LOG_LEVEL"

  if ! qmatch '^\[harness\] GRANT_FLOW_OK$' "$out"; then
    printf '[FAIL] grant_flow (tmp=%s) missing grant success sentinel\n' "$dir" >&2
    return 1
  fi
  if ! qmatch '^GRANT_FLOW_MAIN_OK$' "$out"; then
    printf '[FAIL] grant_flow (tmp=%s) main response mismatch\n' "$dir" >&2
    return 1
  fi
  audit_issue="$(sqlite_scalar "$dir/state.db" "select count(*) from events where event_type='governance_grant_issue';")"
  audit_use="$(sqlite_scalar "$dir/state.db" "select count(*) from events where event_type='governance_grant_use';")"
  audit_revoke="$(sqlite_scalar "$dir/state.db" "select count(*) from events where event_type='governance_grant_revoke';")"
  if [[ "$audit_issue" -lt 1 ]] || [[ "$audit_use" -lt 1 ]] || [[ "$audit_revoke" -lt 1 ]]; then
    printf '[FAIL] grant_flow (tmp=%s) missing grant audit events (issue=%s use=%s revoke=%s)\n' "$dir" "$audit_issue" "$audit_use" "$audit_revoke" >&2
    return 1
  fi
  printf '[PASS] grant_flow (tmp=%s)\n' "$dir"
}

run_token_reject_task() {
  local dir out
  dir="$(make_temp_env turin-live-tokenreject)"
  LAST_CASE_TMP="$dir"
  write_config "$dir" "You are a concise assistant. Reply exactly as instructed." 6
  write_harness_main "$dir" '
function on_session_start(event)
  local ok, err = runtime.policy.set("hook.token_usage.reject_mode", "enforce_task")
  if ok ~= true then
    log("TOKEN_REJECT_SETUP_FAIL:" .. tostring(err))
  else
    log("TOKEN_REJECT_SETUP_OK")
  end
  return ALLOW
end

function on_token_usage(event)
  return REJECT, "live token budget exceeded"
end

function on_task_complete(event)
  log("TOKEN_REJECT_TASK_STATUS:" .. tostring(event.status))
  return ALLOW
end
'
  out="$dir/out.txt"

  printf '\n[CASE] token_reject_task\n'
  run_turin_capture "$out" \
    "$BINARY" run --config "$dir/turin.toml" --prompt 'Reply with exactly: TOKEN_REJECT_TASK_MAIN_OK' --log-level "$LOG_LEVEL"

  if ! qmatch '^\[harness\] TOKEN_REJECT_SETUP_OK$' "$out"; then
    printf '[FAIL] token_reject_task (tmp=%s) setup failed\n' "$dir" >&2
    return 1
  fi
  if ! qmatch '^\[harness\] TOKEN_REJECT_TASK_STATUS:rejected$' "$out"; then
    printf '[FAIL] token_reject_task (tmp=%s) missing rejected task status sentinel\n' "$dir" >&2
    return 1
  fi
  printf '[PASS] token_reject_task (tmp=%s)\n' "$dir"
}

run_immutable_audit() {
  local dir out audit_count
  dir="$(make_temp_env turin-live-immutableaudit)"
  LAST_CASE_TMP="$dir"
  write_config "$dir" "You are a concise assistant. Reply exactly as instructed." 4
  cat >> "$dir/turin.toml" <<'EOF_GOV'

[governance]
profile = "governed"
enforcement_enabled = false

[governance.audit]
mode = "immutable"
EOF_GOV
  write_harness_main "$dir" '
function on_kernel_event(event)
  if event.type == "governance_snapshot" then
    log("IMMUTABLE_AUDIT_REJECT_ATTEMPT")
    return REJECT, "drop governance snapshot"
  end
  return ALLOW
end
'
  out="$dir/out.txt"

  printf '\n[CASE] immutable_audit\n'
  run_turin_capture "$out" \
    "$BINARY" run --config "$dir/turin.toml" --prompt 'Reply with exactly: IMMUTABLE_AUDIT_MAIN_OK' --log-level "$LOG_LEVEL"

  if ! qmatch '^\[harness\] IMMUTABLE_AUDIT_REJECT_ATTEMPT$' "$out"; then
    printf '[FAIL] immutable_audit (tmp=%s) missing hook reject sentinel\n' "$dir" >&2
    return 1
  fi
  if ! qmatch '^IMMUTABLE_AUDIT_MAIN_OK$' "$out"; then
    printf '[FAIL] immutable_audit (tmp=%s) main response mismatch\n' "$dir" >&2
    return 1
  fi
  audit_count="$(sqlite_scalar "$dir/state.db" "select count(*) from events where event_type='governance_snapshot';")"
  if [[ "$audit_count" -lt 1 ]]; then
    printf '[FAIL] immutable_audit (tmp=%s) governance_snapshot not persisted\n' "$dir" >&2
    return 1
  fi
  printf '[PASS] immutable_audit (tmp=%s)\n' "$dir"
}

run_peer_grant() {
  local dir out
  dir="$(make_temp_env turin-live-peergrant)"
  LAST_CASE_TMP="$dir"
  write_config "$dir" "You are a concise assistant. Reply exactly as instructed." 8
  cat >> "$dir/turin.toml" <<EOF_PEER

[governance]
profile = "balanced"
enforcement_enabled = true

[governance.grants]
enabled = true
require_audit_reason = true

[agents.worker]
id = "worker"
system_prompt = "You are a worker agent. Follow exact output instructions."
model = "$LIVE_MODEL"
provider = "$TURIN_PROVIDER_KIND"
mode = "stateful"
EOF_PEER
  write_harness_main "$dir" '
local orchestrator_once = false

function on_turn_prepare(ctx)
  local ident = agent.session.identity()
  if ident and ident.agent_id == "worker" then
    local dec, de = runtime.governance.check("runtime.policy.set")
    if dec == nil then
      log("PEER_GRANT_WORKER_FAIL:check:" .. tostring(de))
      return ALLOW
    end
    if dec.allowed then
      log("PEER_GRANT_WORKER_FAIL:allowed")
      return ALLOW
    end
    local ok, err = runtime.policy.set("peer.grant.test", true)
    if ok ~= false then
      log("PEER_GRANT_WORKER_FAIL:policy_set_allowed")
      return ALLOW
    end
    log("PEER_GRANT_WORKER_OK")
    return ALLOW
  end

  if orchestrator_once then
    return ALLOW
  end
  orchestrator_once = true

  local grant, ge = runtime.governance.grant_issue({
    capabilities = {
      ["runtime.db.query"] = true,
      ["runtime.agent.submit"] = true,
      ["runtime.agent.await"] = true
    },
    reason = "peer grant propagation live test"
  })
  if grant == nil then
    log("PEER_GRANT_ORCH_FAIL:grant_issue:" .. tostring(ge))
    return ALLOW
  end

  local out = runtime.governance.with_grant(grant.grant_id, function()
    local task_id, se = runtime.agent.submit("worker", {
      prompt = "Reply with exactly: PEER_GRANT_WORKER_REPLY_OK"
    })
    if task_id == nil then error("submit failed: " .. tostring(se)) end
    local res, ae = runtime.agent.await(task_id, { timeout_ms = 90000 })
    if res == nil then error("await failed: " .. tostring(ae)) end
    if res.status ~= "success" then error("worker status " .. tostring(res.status)) end
    if type(res.output) ~= "string" or not string.find(res.output, "PEER_GRANT_WORKER_REPLY_OK", 1, true) then
      error("worker output mismatch: " .. tostring(res.output))
    end
    return "PEER_GRANT_WITH_GRANT_OK"
  end)

  if out ~= "PEER_GRANT_WITH_GRANT_OK" then
    log("PEER_GRANT_ORCH_FAIL:with_grant:" .. tostring(out))
    return ALLOW
  end

  log("PEER_GRANT_ORCH_OK")
  return ALLOW
end
'
  out="$dir/out.txt"

  printf '\n[CASE] peer_grant\n'
  run_turin_capture "$out" \
    "$BINARY" run --config "$dir/turin.toml" --prompt 'Reply with exactly: PEER_GRANT_MAIN_OK' --log-level "$LOG_LEVEL"

  if ! qmatch '^\[harness\] PEER_GRANT_WORKER_OK$' "$out"; then
    printf '[FAIL] peer_grant (tmp=%s) missing worker grant sentinel\n' "$dir" >&2
    return 1
  fi
  if ! qmatch '^\[harness\] PEER_GRANT_ORCH_OK$' "$out"; then
    printf '[FAIL] peer_grant (tmp=%s) missing orchestrator grant sentinel\n' "$dir" >&2
    return 1
  fi
  if ! qmatch '^PEER_GRANT_MAIN_OK$' "$out"; then
    printf '[FAIL] peer_grant (tmp=%s) main response mismatch\n' "$dir" >&2
    return 1
  fi
  printf '[PASS] peer_grant (tmp=%s)\n' "$dir"
}

IFS=',' read -r -a CASE_LIST <<< "$CASES"
PASS_COUNT=0
FAIL_COUNT=0
CASE_RESULT_NAMES=()
CASE_RESULT_STATUS=()
CASE_RESULT_DURATIONS=()
CASE_RESULT_TMPS=()

printf 'Turin live smoke tests (manual/opt-in)\n'
printf 'Binary: %s\n' "$BINARY"
printf 'API format: %s (provider=%s)\n' "$API_FORMAT" "$TURIN_PROVIDER_KIND"
printf 'Model: %s\n' "$LIVE_MODEL"
printf 'Base URL: %s\n' "$LIVE_BASE_URL_NORM"
if [[ "$CASES_EXPLICIT" -eq 1 ]]; then
  printf 'Suite: custom (--cases override)\n'
else
  printf 'Suite: %s\n' "$SUITE"
fi
printf 'Log level: %s\n' "$LOG_LEVEL"

for case_name in "${CASE_LIST[@]}"; do
  case_name="${case_name// /}"
  if [[ -z "$case_name" ]]; then
    continue
  fi

  LAST_CASE_TMP=""
  case_start_s="$(date +%s)"
  if "run_${case_name}"; then
    PASS_COUNT=$((PASS_COUNT + 1))
    CASE_RESULT_STATUS+=("passed")
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    CASE_RESULT_STATUS+=("failed")
    case_end_s="$(date +%s)"
    CASE_RESULT_NAMES+=("$case_name")
    CASE_RESULT_DURATIONS+=("$((case_end_s - case_start_s))")
    CASE_RESULT_TMPS+=("${LAST_CASE_TMP:-}")
    # Stop on first failure to preserve the failing temp dir context if --keep-tmp is set.
    break
  fi

  case_end_s="$(date +%s)"
  CASE_RESULT_NAMES+=("$case_name")
  CASE_RESULT_DURATIONS+=("$((case_end_s - case_start_s))")
  CASE_RESULT_TMPS+=("${LAST_CASE_TMP:-}")
done

printf '\nSummary: %d passed, %d failed\n' "$PASS_COUNT" "$FAIL_COUNT"
if [[ "$FAIL_COUNT" -eq 0 ]]; then
  write_report_json 0
  exit 0
else
  write_report_json 1
  exit 1
fi
