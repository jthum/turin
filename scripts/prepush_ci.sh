#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CARGO_BIN="${CARGO:-cargo}"

usage() {
  cat <<'USAGE'
Usage: scripts/prepush_ci.sh [step...]

Run the same core Rust validation steps we expect before pushing.
If no step is provided, runs: fmt check clippy test

Steps:
  fmt       cargo fmt --all --check
  check     cargo check --workspace --all-targets
  clippy    cargo clippy --workspace --all-targets -- -D warnings
  test      cargo test --workspace --all-targets
  test-ignored
            cargo test --workspace --all-targets -- --ignored
  build     cargo build --release for shipped release binaries
  all       fmt + check + clippy + test
  ci        fmt + check + clippy + test + build
  release   fmt + check + clippy + test + build

Examples:
  scripts/prepush_ci.sh
  scripts/prepush_ci.sh clippy test
  scripts/prepush_ci.sh ci
USAGE
}

run_step() {
  local step="$1"
  case "$step" in
    fmt)
      echo "==> cargo fmt --all --check"
      "$CARGO_BIN" fmt --all --check
      ;;
    check)
      echo "==> cargo check --workspace --all-targets"
      "$CARGO_BIN" check --workspace --all-targets
      ;;
    clippy)
      echo "==> cargo clippy --workspace --all-targets -- -D warnings"
      "$CARGO_BIN" clippy --workspace --all-targets -- -D warnings
      ;;
    test)
      echo "==> cargo test --workspace --all-targets"
      "$CARGO_BIN" test --workspace --all-targets
      ;;
    test-ignored)
      echo "==> cargo test --workspace --all-targets -- --ignored"
      "$CARGO_BIN" test --workspace --all-targets -- --ignored
      ;;
    build)
      echo "==> cargo build --release for shipped binaries"
      "$CARGO_BIN" build --release \
        -p turin \
        -p turin-map \
        -p turin-manager \
        -p turin-channel-discord \
        -p turin-channel-telegram \
        -p turin-channel-rocketchat \
        -p turin-channel-whatsapp
      ;;
    all)
      run_step fmt
      run_step check
      run_step clippy
      run_step test
      ;;
    ci)
      run_step fmt
      run_step check
      run_step clippy
      run_step test
      run_step build
      ;;
    release)
      run_step fmt
      run_step check
      run_step clippy
      run_step test
      run_step build
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown step: $step" >&2
      usage >&2
      exit 1
      ;;
  esac
}

if [[ $# -eq 0 ]]; then
  run_step all
else
  for step in "$@"; do
    run_step "$step"
  done
fi
