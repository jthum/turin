# Turin Web Map

## Purpose

`turin-web` is Turin's browser-client host. It connects to the daemon through
`turin-client`, exposes a web-owned HTTP API, and serves a statically built
SvelteKit application.

The browser is a client of `turin-web`; it is not a direct client of the daemon
protocol. `turin-web` should translate typed client operations into deliberate
browser contracts rather than forwarding the complete control protocol.

## Ownership

- `crates/turin-web/src/main.rs`
  - Command-line connection and bind configuration.
- `crates/turin-web/src/server.rs`
  - HTTP listener lifecycle, daemon connection, and loopback bind policy.
- `crates/turin-web/src/routes.rs`
  - Top-level HTTP dispatch, static asset delivery, and SPA fallback.
- `crates/turin-web/src/routes/api.rs`
  - Browser-owned agent, session, bounded transcript, worklist, memory, task
    submission, and SSE contracts. Daemon event envelopes are translated here
    and are not exposed directly to the browser.
- `crates/turin-web/frontend/`
  - SvelteKit 3 static SPA and locally owned shadcn-svelte components.
- `crates/turin-web/frontend/src/lib/components/product/`
  - Product workflow components. Keep navigation, transcript, and composer
    concerns separate as their behavior grows.
- `crates/turin-web/frontend/src/lib/components/product/work-workspace.svelte`
  - Worklist filtering, work-item inspection, guarded operator interventions,
    and navigation to an item's owning session.
- `crates/turin-web/frontend/src/lib/components/product/memory-workspace.svelte`
  - Persisted memory search, scope filtering, bounded browsing, lineage-aware
    inspection, correction, and explicit forgetting.
- `crates/turin-web/frontend/dev/mock-api/`
  - Development-only Vite API adapter. It implements the same browser
    contracts and generates large transcript windows algorithmically.
- `crates/turin-client/`
  - Typed local/remote Turin operations shared with other clients.

## Data Flow

1. `turin-web` creates a `turin_client::Client` for a local daemon or
   `turin-remote` endpoint.
2. Browser requests arrive at the Rust HTTP boundary.
3. API routes call typed `turin-client` methods and project the result into a
   web-owned response shape.
4. Non-API GET/HEAD requests resolve built assets, with `200.html` as the SPA
   fallback.

During development, Vite serves the SPA and proxies `/api` to the Rust host.
Production uses the static adapter output and requires no Node.js process.
`bun run dev:mock` replaces that proxy with the in-process development mock;
mock code is not included in production assets.

## Invariants

- The Rust host owns API, authentication-boundary, and daemon-transport
  concerns; SvelteKit server routes are not a second backend.
- Browser contracts are explicit projections. Do not expose daemon protocol
  envelopes or operational paths merely because they are available.
- Browser transcript offsets count backward from the latest message. The Rust
  boundary translates them to persistence's oldest-first window offsets and
  returns the resolved boundary because complete turns may widen a page.
- New-conversation state remains browser-local until the first message is sent;
  merely opening and abandoning the composer does not create a durable session.
- Message actions may create an exact-turn branch through the narrow web API.
  The browser sends a durable turn ID and does not reproduce graph semantics.
- Assistant message details project Turin's existing per-turn efficiency data.
  Input, output, and provider cache tokens are shown only when reported; the
  web client does not estimate provider cost or invent unavailable metrics.
- Harness selection is local presentation state. Turin Web exposes harness
  identity and agent bindings without creating a runtime-global active harness.
- The desktop shell has stable global workspace navigation. Conversation
  navigation is contextual and appears only while a conversation is open;
  management views use the wider workspace instead of duplicating the session
  list beside a session table.
- Worklists and memories are lazy browser projections over typed `turin-client`
  operations. Opening those destinations performs the first request; the chat
  startup path does not preload either domain. Worklist rows drill into a
  bounded item view, while memory browsing uses bounded pages and an explicit
  load-more action rather than materializing the complete store.
- Memory search is server-backed and observational: it must not increment agent
  retrieval statistics. Correction creates a replacement with visible lineage
  rather than mutating content in place; forgetting requires explicit
  confirmation and uses exact memory identity.
- The Work surface is operational without becoming a second executor. It may
  pause pending work, resume paused work, and request stale-claim release. It
  does not claim, heartbeat, complete, or fail work on behalf of a harness.
- Conversation discovery uses Turin's ranked persisted-session search rather
  than filtering only the browser's current page. In-conversation search adds
  a session target and returns bounded message snippets with durable turn IDs;
  selecting a result loads an active-path window around that exact turn. It
  does not estimate a location from turn count or load the complete transcript.
- Workspace search is explicitly invoked through the command palette and spans
  persisted sessions, active-path messages, tool executions, and events. It
  performs no startup query. Selecting an anchored result opens its session
  directly at one bounded turn-centered window rather than loading latest first.
- Conversation history is fetched in bounded windows. Live text arrives over
  SSE as task, message-start, delta, completion, and failure events.
- The conversation client keeps a bounded resident transcript and can slide in
  both directions. Scroll boundaries fetch adjacent windows automatically,
  while a variable-height virtualizer limits mounted message views to the
  viewport and overscan. A proportional conversation map may seek directly to
  another bounded window without materializing the intervening transcript.
  Evicted messages remain retrievable from the API.
- Explicit conversation navigation is latest-intent-wins. It aborts any older
  browser window request and ignores obsolete responses, while automatic
  scroll-boundary loading never starts a competing request.
- Provider message content is untrusted. The browser may render Markdown, but
  raw HTML and unsafe link schemes must not become executable markup. Remote
  images require deliberate user navigation rather than automatic loading.
- A selected cold session is resumed before its event subscription is opened;
  this establishes a live event receiver but does not make browser navigation
  state part of the Turin runtime.
- Local bind is the safe default. Non-loopback binding requires explicit opt-in
  and an external authenticated boundary until web authentication is designed.
- Unknown `/api/*` paths return API errors and never fall through to the SPA.
- Static paths cannot escape the configured asset root.
- The frontend is a consumer product surface, not a diagnostic dump. Add a
  navigation destination only when its workflow exists.
- shadcn-svelte components are source-owned building blocks. Product styling
  may evolve without introducing a runtime component-library dependency.
- The checked-in shadcn configuration uses the Maia style with the Mist token
  palette and Outfit variable typography. Extend those generated primitives
  instead of recreating equivalent controls in product components.
- Bun is the pinned frontend package manager. `bun.lock` is authoritative; do
  not add npm, pnpm, or Yarn lockfiles alongside it.
- Harness UI intents remain an experimental runtime capability; this client is
  not required to reproduce the deleted exploratory renderer.

## Common Changes

Add a browser capability:

1. Add or reuse a typed operation in `turin-client`.
2. Define the smallest web-owned request/response contract in `turin-web`.
3. Add the typed frontend API module.
4. Build the workflow and its loading, empty, failure, and narrow-screen states.
5. Test the Rust boundary and run frontend checks.

Add a UI primitive:

1. Add it through the shadcn-svelte CLI from `frontend/`.
2. Keep the generated source local under `src/lib/components/ui`.
3. Adapt tokens centrally instead of scattering one-off utility overrides.

## Tests

```sh
cargo test -p turin-web
cargo check -p turin-web

cd crates/turin-web/frontend
bun run check
bun run build
```

Also run `cargo fmt --all -- --check` and `git diff --check`.
