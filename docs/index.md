# Turin Docs

This is the documentation landing page for Turin.

## Start Here

- `README.md` — quickstart, feature overview, canonical API summary, live smoke commands
- `docs/concepts/what-is-turin.md` — plain-language answer to what Turin is and why it exists
- `docs/concepts/what-can-you-do.md` — scenario-first overview for people asking what Turin is useful for
- `docs/concepts/scenarios.md` — practical workflow blueprints for common Turin systems
- `docs/getting-started/index.md` — first steps, quick paths, and entry points
- `docs/getting-started/choose-first-workflow.md` — choose a first workflow and decide when custom UI is worth adding
- `docs/concepts/turin.md` — Turin philosophy and design framing (kernel vs harness vs inference)
- `docs/concepts/capability-charter.md` — capability promises and behavior-preserving refactor guardrails
- `docs/concepts/security-model.md` — trust boundaries, high-risk surfaces, and hardening guidance
- `docs/concepts/runtime-profiles.md` — recommended operating profiles and risk posture
- `docs/concepts/memory-vs-kv.md` — convention for choosing between searchable memory and exact KV state
- `docs/concepts/worklists.md` — durable work coordination model, core operations, and workflow examples
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

If you are trying to explain Turin to someone evaluating the project rather than
implementing a harness, start with the first three concept pages before diving
into technical reference material.

### For new users

1. `README.md`
2. `docs/concepts/what-is-turin.md`
3. `docs/concepts/what-can-you-do.md`
4. `docs/concepts/scenarios.md`
5. `docs/getting-started/choose-first-workflow.md`
6. `docs/getting-started/index.md`
7. `docs/concepts/turin.md`
8. `docs/getting-started/harness-cookbook.md`
9. `docs/reference/hooks.md`
10. `docs/reference/primitives.md`
11. `docs/concepts/memory-vs-kv.md`
12. `docs/concepts/worklists.md`
13. `docs/guides/inference-routing.md`
14. `docs/guides/multimodal.md`

### For contributors

1. `docs/concepts/architecture.md`
2. `docs/concepts/capability-charter.md`
3. `docs/concepts/security-model.md`
4. `docs/concepts/project-quality-bar.md`
5. `docs/adr/index.md`
6. `docs/operations/daemon.md`
7. `docs/reference/hooks.md`
8. `docs/reference/primitives.md`
9. `docs/concepts/worklists.md`
10. `docs/guides/inference-routing.md`
11. `docs/guides/multimodal.md`

### For governance-heavy deployments

1. `docs/concepts/governance.md`
2. `docs/concepts/security-model.md`
3. `docs/concepts/runtime-profiles.md`
4. `docs/reference/hooks.md`
5. `docs/reference/primitives.md`
6. `docs/guides/harness-guide.md`
7. `docs/operations/live-provider-testing.md`

## Channel Setup

- `docs/guides/channels/telegram.md` — step-by-step Telegram bot, chat-id, webhook, and Turin channel setup
- `docs/guides/channels/whatsapp.md` — step-by-step WhatsApp linked-device setup, personal vs dedicated account guidance, and Turin channel setup
