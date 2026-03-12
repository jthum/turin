# Turin Docs Index

This is the documentation landing page for Turin.

## Start Here

- `README.md` — quickstart, feature overview, canonical API summary, live smoke commands
- `docs/TURIN.md` — Turin philosophy and design framing (kernel vs harness vs inference)
- `docs/ARCHITECTURE.md` — current implementation architecture and module layout
- `docs/DAEMON.md` — daemon mode, filesystem-backed dynamic state, and control API surface
- `docs/adr/README.md` — architecture decision records for the current runtime and daemon shape

## Harness Authoring

- `docs/HOOKS.md` — stable hook lifecycle, payloads, verdict semantics
- `docs/PRIMITIVES.md` — **canonical harness API surface reference** (`runtime.*`, aliases, and DX helpers)
- `docs/HARNESS_COOKBOOK.md` — quick scaffold, template, and mock-test path for new harnesses
- `docs/HARNESS_GUIDE.md` — practical harness patterns, DX examples, governance-aware design
- `docs/HARNESS_LIBRARY.md` — ready-to-use harness library entries and how they are validated
- `library/README.md` — Harness Library catalog from the repository root

## Governance and Security (Opt-In)

- `docs/GOVERNANCE.md` — profiles, capability model, import scoping, agent ceilings, grants, audit modes

## Testing and Validation

- `docs/TESTING.md` — local validation workflow (`cargo test`, clippy, release builds)
- `docs/LIVE_PROVIDER_TESTING.md` — manual/opt-in real provider smoke testing (e.g. MiniMax Anthropic-compatible)

## Recommended Reading Order

### For new users

1. `README.md`
2. `docs/TURIN.md`
3. `docs/HARNESS_COOKBOOK.md`
4. `docs/HOOKS.md`
5. `docs/PRIMITIVES.md`
6. `docs/HARNESS_GUIDE.md`

### For contributors

1. `docs/ARCHITECTURE.md`
2. `docs/adr/README.md`
3. `docs/DAEMON.md`
4. `docs/HOOKS.md`
5. `docs/PRIMITIVES.md`

### For governance-heavy deployments

1. `docs/GOVERNANCE.md`
2. `docs/HOOKS.md`
3. `docs/PRIMITIVES.md`
4. `docs/HARNESS_GUIDE.md`
5. `docs/LIVE_PROVIDER_TESTING.md`
