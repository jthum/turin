# Governance Map

## Purpose

Governance decides whether harness/runtime code may use a capability, call a child agent, or enter a temporary grant.

This subsystem is security-sensitive. Refactors here should be small, test-backed, and behavior-preserving unless the policy model is being deliberately redesigned.

## Files

- `src/kernel/governance.rs`
  - Governance snapshots.
  - Capability decisions.
  - Agent/root/import/grant capability ceilings.
  - Child-agent allowlist enforcement.
  - Temporary grant issue/get/revoke/use validation.
- `src/kernel/governance/capabilities.rs`
  - Profile preset capability maps.
  - Exact/wildcard capability rule matching.
  - Shared bool-rule ceiling checks used by temporary grants, peer delegation, and import delegation.
  - Tool-name to capability-name mapping.
- `src/kernel/config/governance.rs`
  - Governance configuration types and defaults.
- `src/kernel/config/validation.rs`
  - Governance config validation.
- `src/harness/stdlib/governance_support.rs`
  - Active harness subject construction and capability checks.
- `src/harness/stdlib/runtime_governance.rs`
  - Lua `runtime.governance.*` API.
- `src/harness/dx/governance.rs`
  - DX wrapper helpers for grants.

## Data Flow

Capability checks:

1. Build a `GovernanceSubject` from active agent/session/module/root/grant context.
2. Match the requested capability against profile preset rules.
3. Apply ceilings from agent capability profile, agent max capabilities, root max capabilities, delegated import capabilities, and active grant.
4. If enforcement is off, report the decision but allow execution.
5. If enforcement is on, return the denial reason on failure.

Child-agent checks:

1. Resolve the parent agent from the subject.
2. If the parent has no `allowed_child_agents`, allow by default.
3. Otherwise require the target child id to appear in the allowlist.

Temporary grants:

1. `grant_issue` validates enabled grants, non-empty allowed capabilities, TTL, uses, and audit reason.
2. Child grants cannot exceed parent grant capability ceilings.
3. `with_grant` validates grant ancestry, subject access, expiry, and remaining uses.
4. Revoked, expired, missing, or cyclic ancestor grants invalidate dependent grants.

## Invariants

- Exact capability rules outrank wildcard rules.
- The longest matching wildcard wins.
- Open profile allows unmatched capabilities by baseline; other profiles deny unmatched capabilities.
- Empty bool allowlists deny when used as a hard ceiling.
- Empty JSON max-capability maps mean no ceiling.
- Enforcement-disabled mode must still produce accurate observability decisions.
- Temporary grants are bound to the issuing agent/session context when those fields are present.
- Delegated grants cannot widen parent grant capabilities.

## Tests

Focused tests:

```sh
cargo test -p turin --lib kernel::governance
cargo test -p turin --test harness_tests test_runtime_governance_observability_api
cargo test -p turin --test harness_tests test_governance_profile_enforcement_blocks_high_risk_runtime_apis
cargo test -p turin --test harness_tests test_temporary_grant_ceiling_propagates_to_peer_submit
cargo test -p turin --test session_tests test_governance_grant_audit_events_persisted
```

Basic checks:

```sh
cargo check -p turin --lib
cargo fmt --all -- --check
git diff --check
```

## Current Shape

The current pass keeps `governance.rs` as the manager and grant lifecycle file, while `governance/capabilities.rs` owns capability preset/matching logic.

This centralized the exact/wildcard bool-rule matcher that had been duplicated in:

- `src/kernel/governance.rs`
- `src/harness/stdlib/governance_support.rs`
- `src/harness/stdlib/system_globals/imports.rs`

The shared matcher preserves these security rules with dedicated tests:

- exact capability rules outrank wildcard rules
- the longest wildcard rule wins
- wildcard rules match both the prefix capability itself and dotted children
