# Turin Docs

This is the documentation landing page for Turin.

## Start Here

- `README.md` — quickstart, feature overview, canonical API summary, live smoke commands
- `docs/getting-started/index.md` — first steps, quick paths, and entry points
- `docs/concepts/turin.md` — Turin philosophy and design framing (kernel vs harness vs inference)
- `docs/concepts/memory-vs-kv.md` — convention for choosing between searchable memory and exact KV state
- `docs/guides/inference-routing.md` — named inference profiles, context-window management, and compaction policy
- `docs/guides/multimodal.md` — multimodal Phase 1 input, attachment persistence, and current channel/provider support
- `docs/operations/daemon.md` — daemon mode, filesystem-backed dynamic state, and control API surface
- `docs/operations/remote.md` — authenticated HTTP/SSE/WebSocket bridge for remote daemon access
- `docs/operations/ui-clients.md` — local/remote operator shells for the Turin daemon control plane
- `docs/adr/index.md` — architecture decision records for the current runtime and daemon shape

## Sections

- `docs/getting-started/` — quickstart-oriented docs, examples, and first harness workflows
- `docs/concepts/` — product framing, architecture, and governance model
- `docs/guides/` — practical authoring and operator guides
- `docs/reference/` — stable harness API and hook contracts
- `docs/operations/` — daemon operations, testing, and live validation
- `docs/adr/` — architecture decision records

## Recommended Reading Order

### For new users

1. `README.md`
2. `docs/getting-started/index.md`
3. `docs/concepts/turin.md`
4. `docs/getting-started/harness-cookbook.md`
5. `docs/reference/hooks.md`
6. `docs/reference/primitives.md`
7. `docs/concepts/memory-vs-kv.md`
8. `docs/guides/inference-routing.md`
9. `docs/guides/multimodal.md`

### For contributors

1. `docs/concepts/architecture.md`
2. `docs/adr/index.md`
3. `docs/operations/daemon.md`
4. `docs/reference/hooks.md`
5. `docs/reference/primitives.md`
6. `docs/guides/inference-routing.md`
7. `docs/guides/multimodal.md`

### For governance-heavy deployments

1. `docs/concepts/governance.md`
2. `docs/reference/hooks.md`
3. `docs/reference/primitives.md`
4. `docs/guides/harness-guide.md`
5. `docs/operations/live-provider-testing.md`

## Channel Setup

- `docs/guides/channels/telegram.md` — step-by-step Telegram bot, chat-id, webhook, and Turin channel setup
- `docs/guides/channels/whatsapp.md` — step-by-step WhatsApp linked-device setup, personal vs dedicated account guidance, and Turin channel setup
