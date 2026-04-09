#!/usr/bin/env bash
set -euo pipefail

event_name="${GITHUB_EVENT_NAME:-}"

if [[ -n "${RELEASE_POLICY_RANGE:-}" ]]; then
  range="${RELEASE_POLICY_RANGE}"
elif [[ "$event_name" == "pull_request" && -n "${GITHUB_BASE_REF:-}" ]]; then
  git fetch --no-tags --depth=1 origin "${GITHUB_BASE_REF}" >/dev/null 2>&1
  range="origin/${GITHUB_BASE_REF}...HEAD"
elif git rev-parse --verify HEAD~1 >/dev/null 2>&1; then
  range="HEAD~1...HEAD"
else
  echo "release-policy: no base revision available, skipping."
  exit 0
fi

mapfile -t changed_files < <(git diff --name-only "$range")
if [[ ${#changed_files[@]} -eq 0 ]]; then
  echo "release-policy: no changed files."
  exit 0
fi

file_changed() {
  local needle="$1"
  for f in "${changed_files[@]}"; do
    if [[ "$f" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

public_api_changed=0
for f in "${changed_files[@]}"; do
  if [[ "$f" =~ ^(core|openai|anthropic|registry)/src/.*\.rs$ ]]; then
    if git diff --unified=0 "$range" -- "$f" | grep -Eq '^[+-][[:space:]]*pub([[:space:](]|$)'; then
      public_api_changed=1
      break
    fi
  fi
done

version_changed=0
for manifest in core/Cargo.toml openai/Cargo.toml anthropic/Cargo.toml registry/Cargo.toml; do
  if file_changed "$manifest"; then
    if git diff --unified=0 "$range" -- "$manifest" | grep -Eq '^[+-]version[[:space:]]*='; then
      version_changed=1
      break
    fi
  fi
done

if [[ "$public_api_changed" -eq 0 && "$version_changed" -eq 0 ]]; then
  echo "release-policy: no public API/version changes detected."
  exit 0
fi

fail=0

if ! file_changed "CHANGELOG.md"; then
  echo "release-policy: CHANGELOG.md must be updated for API/version changes."
  fail=1
fi

if [[ "$public_api_changed" -eq 1 ]]; then
  if ! file_changed "docs/MIGRATIONS.md"; then
    echo "release-policy: docs/MIGRATIONS.md must be updated when public API changes."
    fail=1
  fi
fi

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

echo "release-policy: checks passed."
