# Turin Web Client Direction

`turin-web` is the web-facing client surface for Turin.

The current implementation is server/API-first with a compact Svelte client
and authored CSS. It exists to prove the web boundary, streaming behavior,
bounded conversation loading, and semantic harness UI model before Dashbase
and the production visual system are integrated.

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
4. The Svelte client talks through a typed `TurinClient` interface. The HTTP
   implementation uses same-origin JSON and SSE; a desktop host can later
   provide a bridge implementation without inserting `turin-web` between the
   desktop app and daemon.

This keeps web behavior aligned with local and remote clients.

## Initial API Shape

Start with a small API that mirrors what the current clients already need.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/status` | Implemented. Current dashboard snapshot plus UI registry derived from harness UI intent. |
| `GET /api/session` | Implemented. Bounded durable session detail and request-efficiency projection using encoded `session_id`, `message_limit`, and optional absolute `message_offset` query parameters. |
| `POST /api/sessions/open` | Implemented. Opens a live session for an agent. |
| `POST /api/sessions/resume` | Implemented. Resumes a stored session into a live slot. |
| `POST /api/tasks/submit` | Implemented. Submits a prompt or structured task input. |
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

## Frontend Build

Svelte source lives in `crates/turin-web/frontend`. Vite emits deterministic
asset names into `crates/turin-web/static`, and those built assets are checked
in and embedded by `turin-web`. This means a normal Cargo build does not require
Node.

After changing frontend source, run:

```sh
cd crates/turin-web/frontend
npm install
npm run check
npm run build
```

The temporary client uses custom components and CSS rather than Dashbase. Its
transport and state boundaries are intended to survive the later visual-system
replacement; its CSS is not a compatibility contract.

## Conversation Windowing

The assistant initially requests the latest 100 persisted message rows. It
keeps that bounded data window in the browser and mounts about 30
user/assistant messages at a time. Variable-height spacer estimates preserve
the scrollbar while measured message heights refine the virtual layout.

Scrolling near either render boundary slides the mounted window. Reaching a
data-window boundary requests another absolute window with a 30-row overlap and
restores the first visible message as the scroll anchor. Database windows keep
complete turns together, so a response can be slightly larger than the nominal
limit rather than separating a tool call from its result. The browser does not
fetch or retain the entire session tree by default.

The active session uses a session-scoped SSE subscription. `message_delta`
events are buffered and committed at most once per animation frame. A resumed
runtime rebinds that subscription before task submission. On message or turn
completion, the browser reloads the bounded durable session detail and retires
the transient streamed copy; SSE remains a stream and invalidation channel
rather than a second transcript store. Raw persisted tool-result messages stay
out of the chat transcript, while tool executions attach to the corresponding
assistant tool-use message by tool-call ID.

The assistant header exposes the canonical session reference as a copyable
diagnostic value. This identifies the persisted session without making
navigation or other browser state runtime-owned.

## Request Efficiency

The assistant's **Efficiency** inspector separates facts reported by the
provider from estimates Turin can derive before serialization:

- **Measured:** provider-reported input and output tokens per request, per turn,
  and in total.
- **Estimated:** final request tokens, system/message/tool-schema composition,
  normalized payload bytes, per-message context cost, context utilization, and
  structural compaction counts.
- **Reusable prefix:** the stable request prefix that could be eligible for a
  provider prompt cache. It is an opportunity estimate, not a cache hit or
  billing measurement. Turin's current inference boundary does not expose
  provider cache-read or cache-write counters.

The normalized request estimate is intentionally not described as exact HTTP
body size. Provider adapters may add serialization overhead or tokenize content
differently. Measured provider totals remain authoritative where available.
Older sessions retain their measured totals but cannot reconstruct request
composition that was not recorded at inference time.

Session-wide measured totals and provider-call count remain visible while the
per-request records follow the current bounded transcript window. Scrolling to
an older data window therefore exposes the accounting for those turns without
retaining every request record in the browser.

Durable transcript size does not determine request size. Turin retains the full
branch transcript in Turso, keeps a bounded hot runtime history by default, and
builds a separate token-budgeted provider request from the hot window and any
semantic checkpoint. Structural compaction can trim older tool payloads or
drop older messages without mutating durable history. Consequently, a session
with thousands of stored messages does not automatically resend them all, but
the effective request can still grow close to the configured context budget.
The inspector makes that behavior visible for every newly accepted request.

Current tuning levers are:

- configure the provider/model context window accurately rather than relying
  on Turin's fallback assumption
- keep repeated system instructions and tool definitions concise, because they
  consume input on every request even when a provider can cache their prefix
- use hybrid context compaction and semantic checkpoints to retain older intent
  without retaining every old message in the effective request
- tune hot-history bounds for resident memory independently of provider request
  compaction
- reserve only the output and thinking budget the task needs, because those
  reservations reduce the available input budget
- investigate large tool results and low reusable-prefix ratios before applying
  broad transcript truncation

Per-message estimates describe each stored message's approximate contribution
when included. Per-request accounting is the reliable view of what the final
compacted request actually contained.

API responses use JSON envelopes and are marked `Cache-Control: no-store`.
Errors have the shape:

```json
{
  "error": {
    "code": "unsupported_ui_list_source",
    "message": "Unsupported UI list source 'tables.release'",
    "details": {
      "source": "tables.release",
      "supported_prefixes": ["worklists.<name>"],
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

Only named `worklists.<name>` sources are supported today. Other sources return
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
action, preserving the structured error envelope as panel detail and showing
action/harness/agent metadata when known without persisting any UI session state
in Turin.

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
`request_body_too_large` while the body is being read, malformed bodies return
`invalid_json`, and failed control-layer calls return `control_unavailable`.

## Browser Client State

The browser should own ephemeral UI state:

- selected app
- selected screen
- selected list row, retained by item identity where possible
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
- list rows should support click and keyboard selection without persisting
  selected-row state in the runtime
- worklist-backed activity/detail can reuse the same bounded adapters as app/TUI
- forms should support text, integer, number/float/decimal aliases, boolean,
  options, and textarea
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

The UI contract web smoke covers both deployment modes: direct local IPC
and `turin-web` connected through `turin-remote`.

## Browser Shell

The current browser shell is intentionally small:

- no frontend build system
- no persistent browser-side storage
- same-origin calls to the `turin-web` API
- local selected app/screen/list row/form draft/action-running state
- browser-local pane overlay state for `ui.show` pane targets
- table rendering for worklist-backed lists, including human-readable field
  labels, direction-aware sorted-column markers, filter/sort/limit metadata,
  row-count, and selected row feedback
- click, Enter/Space, ArrowUp/ArrowDown, Home, and End selection for
  worklist-backed list rows
- inline selected-row detail for worklist-backed lists, with browser-local item
  identity retained across re-render where possible and pause/claim/failure
  context shown when present
- explicit client-side fallback copy for unsupported semantic data sources
  without turning them into failed fetches
- guarded source detection before requesting worklist-backed list, activity,
  detail, report, or chart data
- worklist-backed activity and detail rendering, including confirmed work-item
  action buttons when item action payloads exist
- typed form defaults from field definitions or static form params, local
  drafts that tolerate in-progress text/number edits, required fields, options,
  number/float/decimal aliases, submit-time scalar coercion, and optional blank
  fields that preserve static params instead of overwriting them with null
- browser-local confirmation overlay for explicit-confirm actions and work-item
  actions
- accessible browser-local modal behavior for confirmation and pane overlays,
  including dialog roles, Escape close handling, contained Tab navigation, and
  focus staying inside a safe control across overlay rerenders
- action start/completion/failure feedback, latest action result panels on
  screens and open panes scoped to the originating app, returned action UI
  intent application, action/harness/agent metadata when known, explicit
  no-payload completion copy, and duplicate-run suppression
- compact state panels for unsupported, missing, loading, empty, and failed
  semantic data surfaces, including filtered empty-list copy when declared
  `where` constraints match no rows
- lightweight worklist-backed report summaries with no-data copy,
  nonstandard-status buckets when present, and next pending item highlights,
  plus chart breakdowns with count and percentage labels
- SSE invalidation for runtime/action events plus binding-level `ui.refresh`
  invalidation for cached list requests
- split browser-local connection feedback for HTTP refresh health and SSE
  event-stream health, so reconnects are visible without turning SSE into a
  live-query cache
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

Next work should turn the current report/chart smoke parity into a fuller shared
data semantics decision across app/TUI/web, then decide whether to keep
iterating on the static shell or introduce a light frontend build step.
