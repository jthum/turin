#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CARGO_BIN="${CARGO:-cargo}"
OUTPUT_DIR="${TURIN_COVERAGE_DIR:-.workspace/coverage}"

usage() {
  cat <<'USAGE'
Usage: scripts/coverage_report.sh [summary|html|lcov]

Generate local Rust test coverage with cargo-llvm-cov.

Modes:
  summary   Print a text summary to stdout (default)
  html      Write an HTML report under .workspace/coverage/html
  lcov      Write .workspace/coverage/lcov.info

This is intentionally an opt-in diagnostic, not a normal CI gate. Use it before
large refactors to find modules with no safety net, then add meaningful tests
instead of chasing a percentage.
USAGE
}

mode="${1:-summary}"
case "$mode" in
  summary|html|lcov)
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown coverage mode: $mode" >&2
    usage >&2
    exit 1
    ;;
esac

if ! "$CARGO_BIN" llvm-cov --version >/dev/null 2>&1; then
  cat >&2 <<'MSG'
cargo-llvm-cov is not installed.

Install it when you want local coverage reports:
  cargo install cargo-llvm-cov

This script is optional and is not required for normal development or CI.
MSG
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

case "$mode" in
  summary)
    "$CARGO_BIN" llvm-cov --workspace --all-targets
    ;;
  html)
    "$CARGO_BIN" llvm-cov --workspace --all-targets --html --output-dir "$OUTPUT_DIR/html"
    echo "Coverage HTML: $OUTPUT_DIR/html/index.html"
    ;;
  lcov)
    "$CARGO_BIN" llvm-cov --workspace --all-targets --lcov --output-path "$OUTPUT_DIR/lcov.info"
    echo "Coverage LCOV: $OUTPUT_DIR/lcov.info"
    ;;
esac
