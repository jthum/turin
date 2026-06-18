# Turin Web Client Direction

`turin-web` is the intended web-facing client surface for Turin.

It is not implemented yet. This page defines the first target shape so the web
work starts from the same UI/UX model as `turin-app` and `turin-tui`.

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
| `GET /api/status` | Current dashboard snapshot: health, agents, sessions, tasks, harnesses, and static UI intent. |
| `GET /api/apps` | Harness UI app registry derived from semantic UI intent. |
| `GET /api/apps/{app_id}` | One app's screens, menus, panes, and declared surfaces. |
| `POST /api/ui/list` | Load a semantic list binding such as `worklists.release`. |
| `POST /api/actions/run` | Run a harness action with typed params. |
| `GET /api/events` | SSE stream of runtime and UI intent events. |
| `GET /api/healthz` | Web process liveness. |

The first version can proxy through `turin-control-client` rather than exposing
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
- reports/charts can begin as semantic placeholders until their data/query
  shape is clearer
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

## Technology Bias

The web client should stay lean.

Preferred direction:

- Dashbase for semantic CSS
- Svelte or a similarly lightweight frontend runtime if a framework is needed
- no Electron-style desktop packaging
- no heavy client state framework unless the UI proves it needs one

The web surface should be fast on older machines and should not become the
dominant memory cost of running Turin.

## First Implementation Slice

The first useful slice should be server/API-first:

1. Add a small `turin-web` crate or binary.
2. Connect through `turin-control-client`.
3. Expose `GET /api/status`, `GET /api/apps`, `POST /api/ui/list`, and
   `POST /api/actions/run`.
4. Add an integration smoke using the Release Operator harness.
5. Only then build the browser UI shell.

This avoids designing a web UI in isolation and keeps the API honest against the
same example harness used by the TUI and desktop app.
