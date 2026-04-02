#!/usr/bin/env bash
set -euo pipefail

CARGO_BIN="${CARGO:-cargo}"
PACKAGE="turin"
REPEAT=20
KEEP_LOGS=0
LOG_DIR=""
TESTS=("daemon-restart" "telegram-roundtrip" "telegram-streaming")

usage() {
  cat <<'USAGE'
Usage: scripts/ci_stress.sh [options]

Run the historically timing-sensitive integration tests in a loop so we can
catch CI-only races locally before pushing.

Options:
  --repeat N                Number of iterations per test (default: 20)
  --tests CSV               Comma-separated test ids to run
                            (daemon-restart,telegram-roundtrip,telegram-streaming)
  --log-dir PATH            Directory for per-iteration logs
  --keep-logs               Keep logs even when all runs pass
  --cargo PATH              Cargo binary to use (default: cargo)
  --list                    List supported test ids and exit
  -h, --help                Show this help

Examples:
  scripts/ci_stress.sh
  scripts/ci_stress.sh --repeat 30
  scripts/ci_stress.sh --tests daemon-restart,telegram-roundtrip
USAGE
}

list_tests() {
  cat <<'TESTS'
daemon-restart       daemon_session_resume_round_trip_over_restart
telegram-roundtrip   telegram_channel_driver_round_trip_with_daemon_runner
telegram-streaming   telegram_channel_driver_streams_progress_before_final_message
TESTS
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repeat)
      REPEAT="$2"
      shift 2
      ;;
    --tests)
      IFS=',' read -r -a TESTS <<<"$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --keep-logs)
      KEEP_LOGS=1
      shift
      ;;
    --cargo)
      CARGO_BIN="$2"
      shift 2
      ;;
    --list)
      list_tests
      exit 0
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

if ! [[ "$REPEAT" =~ ^[0-9]+$ ]] || [[ "$REPEAT" -lt 1 ]]; then
  echo "--repeat must be a positive integer" >&2
  exit 1
fi

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="$(mktemp -d /tmp/turin-ci-stress.XXXXXX)"
else
  mkdir -p "$LOG_DIR"
fi

cleanup() {
  if [[ $KEEP_LOGS -eq 0 ]]; then
    rm -rf "$LOG_DIR"
  fi
}
trap cleanup EXIT

run_case() {
  local id="$1"
  local label
  local cargo_args=()

  case "$id" in
    daemon-restart)
      label="daemon_session_resume_round_trip_over_restart"
      cargo_args=(test -q -p "$PACKAGE" daemon_session_resume_round_trip_over_restart --test daemon_integration_tests -- --nocapture)
      ;;
    telegram-roundtrip)
      label="telegram_channel_driver_round_trip_with_daemon_runner"
      cargo_args=(test -q -p "$PACKAGE" telegram_channel_driver_round_trip_with_daemon_runner --test channel_telegram_integration_tests -- --nocapture)
      ;;
    telegram-streaming)
      label="telegram_channel_driver_streams_progress_before_final_message"
      cargo_args=(test -q -p "$PACKAGE" telegram_channel_driver_streams_progress_before_final_message --test channel_telegram_integration_tests -- --nocapture)
      ;;
    *)
      echo "Unknown test id: $id" >&2
      echo "Supported ids:" >&2
      list_tests >&2
      exit 1
      ;;
  esac

  local slug="${id//[^a-z0-9_-]/_}"
  echo "== $id ($label), repeat $REPEAT =="
  for ((i = 1; i <= REPEAT; i++)); do
    local log_file="$LOG_DIR/${slug}-${i}.log"
    echo "[$id] iteration $i/$REPEAT"
    if ! "$CARGO_BIN" "${cargo_args[@]}" >"$log_file" 2>&1; then
      echo "[$id] failed on iteration $i/$REPEAT" >&2
      echo "[$id] log: $log_file" >&2
      cat "$log_file" >&2
      KEEP_LOGS=1
      exit 1
    fi
  done
}

echo "Logs: $LOG_DIR"
for test_id in "${TESTS[@]}"; do
  run_case "$test_id"
done

echo "ci stress run passed"
