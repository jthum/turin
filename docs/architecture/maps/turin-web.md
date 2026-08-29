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
  - Browser-owned agent, session, bounded transcript, task submission, and SSE
    contracts. Daemon event envelopes are translated here and are not exposed
    directly to the browser.
- `crates/turin-web/frontend/`
  - SvelteKit 3 static SPA and locally owned shadcn-svelte components.
- `crates/turin-web/frontend/src/lib/components/product/`
  - Product workflow components. Keep navigation, transcript, and composer
    concerns separate as their behavior grows.
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
- Conversation history is fetched in bounded windows. Live text arrives over
  SSE as task, message-start, delta, completion, and failure events.
- The conversation client keeps a bounded resident transcript and can slide in
  both directions. Evicted messages remain retrievable from the API.
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
