# Turin Web Client Direction

`turin-web` is the web-facing client surface for Turin.

The first implementation is server/API-first with a small dependency-free
browser shell. It exists to prove the web boundary against the same semantic UI
model used by `turin-app` and `turin-tui` before a larger frontend structure is
chosen.

## Role

`turin-web` should be a client over the Turin control layer, not a replacement
daemon and not a second runtime.

It should provide:

- a web-friendly API for runtime status, harness UI intent, list data, action
  execution, and events
- a browser client that renders semantic harness UI intent
- a deployment path that works locally or through `turin-remote`

It should not provide:

- a parallel persistence model
- renderer-specific harness APIs
- durable UI session state inside the runtime by default
- a new worklist/query engine before the existing control surface proves
  insufficient

## Suggested Layering

Initial layering should stay thin:

1. Turin daemon remains the source of truth.
2. `turin-control-client` remains the typed Rust facade.
3. `turin-web` exposes web-oriented endpoints and event streams.
4. The browser client renders semantic UI intent with Dashbase and a lightweight
   frontend runtime.

This keeps web behavior aligned with local and remote clients.

## Initial API Shape

Start with a small API that mirrors what the current clients already need.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/status` | Implemented. Current dashboard snapshot plus UI registry derived from harness UI intent. |
| `GET /api/apps` | Implemented. Harness UI app registry derived from semantic UI intent. |
| `GET /api/apps/{app_id}` | Implemented. One app's screens, menus, panes, and declared surfaces. |
| `POST /api/ui/list` | Implemented for semantic worklist sources such as `worklists.release`. |
| `POST /api/actions/run` | Implemented. Runs a harness action with typed daemon params. |
| `GET /api/events` | Implemented. SSE stream of runtime and UI intent events for invalidation. |
| `GET /api/healthz` | Implemented. Web process liveness. |
| `GET /` | Implemented. Minimal same-origin browser shell. |
| `GET /assets/app.css` | Implemented. First-party shell styling. |
| `GET /assets/app.js` | Implemented. First-party shell behavior. |

The current version proxies through `turin-control-client` rather than exposing
new daemon operations.

## Browser Client State

The browser should own ephemeral UI state:

- selected app
- selected screen
- selected list row
- open pane/modal
- form drafts
- filters/search text
- local loading/error state
- event cursors/reconnect state

This mirrors `turin-app` and `turin-tui`: runtime state is shared, UI session
state is local unless a harness explicitly persists something through Turin
primitives.

## Semantic Rendering Rules

The web client should consume the same intent vocabulary as the other clients:

- `app`, `screen`, `menu`, `pane`
- `text`, `section`, `action`, `list`, `form`
- `activity`, `detail`, `report`, `chart`
- dynamic `notice`, `open`, `show`, `badge`, `focus`, `refresh`

For v1 parity:

- worklist-backed lists should render as tables by default
- worklist-backed activity/detail can reuse the same bounded adapters as app/TUI
- forms should support text, number, integer, boolean, options, and textarea
- worklist-backed reports/charts can render lightweight summary and breakdown
  adapters until their shared query shape is clearer
- unsupported sources should show explicit fallback messages rather than fail
  silently

## Remote Deployment

There are two reasonable deployment modes:

- `turin-web` co-located with the daemon and connected over local IPC
- `turin-web` connected through `turin-remote`

The second mode should reuse existing bearer-token expectations rather than
inventing a browser-specific auth model immediately.

For browser SSE/WebSocket auth, prefer a same-origin web server session or
server-side proxy over putting long-lived Turin remote tokens directly into
client-side JavaScript.

The Release Operator web smoke covers both deployment modes: direct local IPC
and `turin-web` connected through `turin-remote`.

## Browser Shell

The current browser shell is intentionally small:

- no frontend build system
- no persistent browser-side storage
- same-origin calls to the `turin-web` API
- local selected app/screen/list row/form draft/action-running state
- table rendering for worklist-backed lists
- inline selected-row detail for worklist-backed lists
- explicit client-side fallback copy for unsupported list sources without
  turning them into failed fetches
- worklist-backed activity and detail rendering, including confirmed work-item
  action buttons when item action payloads exist
- typed form defaults, local drafts, required fields, options, and scalar
  coercion
- action start/completion/failure feedback, latest action result panels, and
  duplicate-run suppression
- lightweight worklist-backed report summaries and chart breakdowns
- SSE invalidation for runtime/UI/action events
- local handling for dynamic `ui.open`, `ui.show`, `ui.focus`, and `ui.notify`
- placeholders for unsupported activity/detail/report/chart sources

This is a validation shell, not the final Dashbase/Svelte decision. It should
stay easy to replace once the web UX shape is better proven.

## Technology Bias

The web client should stay lean.

Preferred direction:

- Dashbase for semantic CSS
- Svelte or a similarly lightweight frontend runtime if a framework is needed
- no Electron-style desktop packaging
- no heavy client state framework unless the UI proves it needs one

The web surface should be fast on older machines and should not become the
dominant memory cost of running Turin.

## Current Implementation Slice

The first useful slice is complete:

1. `crates/turin-web` provides a small Hyper HTTP/1 server and CLI.
2. It connects through `turin-control-client` using local config, explicit local
   endpoint, or `turin-remote`.
3. It exposes status, apps, one app, semantic list loading, action execution,
   and liveness routes.
4. It exposes SSE events for invalidation without adding live query semantics.
5. It serves a minimal browser shell from the same process, including
   client-local dynamic UI navigation handling.
6. It has an integration smoke using the Release Operator harness.

Next work should tighten shared report/chart semantics across app/TUI/web, then
decide whether to keep iterating on the static shell or introduce a light
frontend build step.
