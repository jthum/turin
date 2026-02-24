# Turin Docs Index (v0.15.0)

This is the documentation landing page for Turin v0.15.0.

## Start Here

- `README.md` — quickstart, feature overview, canonical API summary, live smoke commands
- `docs/TURIN.md` — Turin philosophy and design framing (kernel vs harness vs inference)
- `docs/ARCHITECTURE.md` — current implementation architecture and module layout

## Harness Authoring

- `docs/HOOKS.md` — stable hook lifecycle, payloads, verdict semantics
- `docs/PRIMITIVES.md` — **canonical harness API surface reference** (`runtime.*` + aliases)
- `docs/HARNESS_GUIDE.md` — practical harness patterns, examples, governance-aware design

## Governance and Security (Opt-In)

- `docs/GOVERNANCE.md` — profiles, capability model, import scoping, agent ceilings, grants, audit modes

## Testing and Validation

- `docs/TESTING.md` — local validation workflow (`cargo test`, clippy, release builds)
- `docs/LIVE_PROVIDER_TESTING.md` — manual/opt-in real provider smoke testing (e.g. MiniMax Anthropic-compatible)

## Recommended Reading Order

### For new users

1. `README.md`
2. `docs/TURIN.md`
3. `docs/HOOKS.md`
4. `docs/PRIMITIVES.md`
5. `docs/HARNESS_GUIDE.md`

### For contributors

1. `docs/ARCHITECTURE.md`
2. `docs/HOOKS.md`
3. `docs/PRIMITIVES.md`
4. `docs/GOVERNANCE.md`
5. `docs/TESTING.md`

### For governance-heavy deployments

1. `docs/GOVERNANCE.md`
2. `docs/HOOKS.md`
3. `docs/PRIMITIVES.md`
4. `docs/HARNESS_GUIDE.md`
5. `docs/LIVE_PROVIDER_TESTING.md`
