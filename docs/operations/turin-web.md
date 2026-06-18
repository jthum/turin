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

API responses use JSON envelopes and are marked `Cache-Control: no-store`.
Errors have the shape:

```json
{
  "error": {
    "code": "unsupported_ui_list_source",
    "message": "Unsupported UI list source 'tables.release'",
    "details": {
      "source": "tables.release",
      "supported_prefixes": ["worklists."],
      "guidance": "Model this data as a worklist source or add a deliberate UI list adapter."
    }
  }
}
```

SSE responses are also `no-store` and use the runtime event name as the SSE
event name. Stream failures are emitted as `web.error` events before the stream
closes.

## Route Contracts

The API is intentionally small. It mirrors the current operator-client needs
rather than exposing the full daemon protocol.

### `GET /api/status`

Returns the current dashboard snapshot, web-process metadata, and the UI
registry derived from harness intent:

```json
{
  "web": {
    "ready": true,
    "version": "<turin-web-version>",
    "bind": "127.0.0.1:8787",
    "connection_kind": "local",
    "connection_target": ".turin/daemon.sock"
  },
  "snapshot": {},
  "ui": {}
}
```

`snapshot` and `ui` use the same shared Rust types consumed by `turin-app` and
`turin-tui`; browser state such as active screen or selected rows is not stored
there.

### `GET /api/apps`

Returns declared harness UI apps:

```json
{ "apps": [] }
```

An empty array is valid. It means the browser should render the default Turin
operator shell rather than a harness-specific app.

### `GET /api/apps/{app_id}`

Returns one app record:

```json
{ "app": {} }
```

Missing apps return a `404` JSON error envelope. The response contains semantic
screens, menus, panes, badges, and nodes, not renderer state.

### `POST /api/ui/list`

Loads a semantic list request:

```json
{
  "source": "worklists.release",
  "where": {},
  "limit": 50
}
```

Returns:

```json
{
  "request": {
    "source": "worklists.release",
    "where": {},
    "limit": 50
  },
  "list": {
    "worklist_id": "release",
    "items": []
  }
}
```

Only `worklists.*` sources are supported today. Other sources return
`unsupported_ui_list_source` with structured `details.source`,
`details.supported_prefixes`, and author guidance; `worklists.` without a name
returns `invalid_ui_list_source` with the same structured source/guidance
details. This keeps list intent semantic while avoiding a raw daemon-query
escape hatch before the UI data model needs one.

### `POST /api/actions/run`

Runs a harness action through the control layer:

```json
{
  "agent_id": null,
  "harness_id": null,
  "action": "release.seed_demo_work",
  "params": {
    "count": 3
  }
}
```

Returns:

```json
{
  "result": {
    "action": "release.seed_demo_work",
    "agent_id": "default",
    "harness_id": "default",
    "result": {
      "status": "seeded"
    },
    "ui_intents": []
  }
}
```

`ui_intents` contains dynamic UI requests emitted during the action, such as
notice, badge, show, focus, and refresh. The browser applies them as local
presentation hints before refreshing visible data. Browser-local badge overlays
are re-applied after status refreshes so action-returned badges do not disappear
when fresh app records are loaded. The browser still owns confirmation and
duplicate-run suppression locally. Durable workflow outcomes should still be
written through harness/runtime primitives such as worklists, events, memory,
KV, or runtime DB tables.

Declared action failures return the standard `control_unavailable` JSON error
envelope from the web boundary. The browser keeps that failure local: it shows an
error notice and an action-result panel scoped to the app that initiated the
action, preserving the structured error envelope as panel detail without
persisting any UI session state in Turin.

An empty or whitespace-only `action` is rejected before control dispatch with
`invalid_action_request` and structured details:

```json
{
  "error": {
    "code": "invalid_action_request",
    "message": "Action name must not be empty",
    "details": {
      "field": "action",
      "guidance": "Send the declared harness action name, for example 'release.seed_demo_work'."
    }
  }
}
```

### `GET /api/events`

Streams runtime/UI events as Server-Sent Events.

Optional filters:

- `agent_id`
- `session_id`
- `slot_id`

Each SSE frame uses the runtime event name as the SSE event name and the event
data as JSON. If the managed event stream fails, `turin-web` emits a `web.error`
event and closes the stream.

### `GET /api/healthz`

Returns liveness for the web process:

```json
{ "ok": true, "version": "<turin-web-version>" }
```

This does not prove a specific harness app is available; use `/api/status` or
`/api/apps` for that.

### Request Limits

JSON request bodies are limited to 1 MiB. Oversized bodies return
`request_body_too_large`, malformed bodies return `invalid_json`, and failed
control-layer calls return `control_unavailable`.

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
- browser-local pane overlay state for `ui.show` pane targets
- table rendering for worklist-backed lists
- inline selected-row detail for worklist-backed lists
- explicit client-side fallback copy for unsupported semantic data sources
  without turning them into failed fetches
- worklist-backed activity and detail rendering, including confirmed work-item
  action buttons when item action payloads exist
- typed form defaults from field definitions or static form params, local
  drafts that tolerate in-progress text/number edits, required fields, options,
  submit-time scalar coercion, and optional blank fields that preserve static
  params instead of overwriting them with null
- browser-local confirmation overlay for explicit-confirm actions and work-item
  actions
- action start/completion/failure feedback, latest action result panels on
  screens and open panes scoped to the originating app, returned action UI
  intent application, and duplicate-run suppression
- compact state panels for unsupported, missing, loading, empty, and failed
  semantic data surfaces
- lightweight worklist-backed report summaries with next pending item
  highlights, plus chart breakdowns
- SSE invalidation for runtime/action events plus binding-level `ui.refresh`
  invalidation for cached list requests
- local handling for dynamic `ui.open`, `ui.show`, `ui.focus`, and `ui.notice`
- dynamic badge rendering for matching navigation targets and titled node ids,
  including action-returned badges that survive browser status refreshes
- a default runtime console backed by `/api/status` when no harness UI apps are
  declared

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
