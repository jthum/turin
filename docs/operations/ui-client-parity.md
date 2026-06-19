# UI Client Parity Matrix

This matrix tracks how Turin clients currently interpret semantic harness UI
intent. It is a maintenance aid, not a promise that every client will render the
same visual interface.

Core rule:

- The runtime emits semantic UI intent.
- Each client owns its local presentation state.
- Clients may render, degrade, or ignore intent according to capability.

## Client Scope

| Surface | Current role |
| --- | --- |
| `turin-app` | Rich graphical operator client built on egui/Cast, including a default runtime console summary when no harness UI app is declared. |
| `turin-tui` | Lean keyboard-first terminal client built on Ratatui, including default-console guidance and summary counts when no harness UI app is declared. |
| `turin-web` | API-first web adapter with a minimal same-origin browser shell and a default runtime console when no harness UI app is declared. See `docs/operations/turin-web.md`. |

## Static UI Intent

| Intent | `turin-app` | `turin-tui` | `turin-web` | Notes |
| --- | --- | --- | --- | --- |
| `app` | Selectable harness app cards/list. | Selectable harness app navigation. | App selector in the browser shell and `/api/apps`. | Multiple apps are allowed; client state remains local. |
| `screen` | Screen tabs within selected app. | Screen navigation plus flattened menu entries. | Screen navigation in shell and `/api/apps/{app_id}`. | Default screen comes from `opens_with`. |
| `menu` | Menu groups and nested menu entries. | Flattened terminal navigation with indentation. | Nested menu navigation in the shell. | Menus are navigation intent, not layout. |
| `pane` | Rendered as an app-local modal/sheet when shown. | Rendered as a terminal-local overlay with pane-local item/action selection when shown. | Exposed through API and rendered as a browser-local overlay when shown. | Pane state remains client-local. |

## Node Rendering

| Node | `turin-app` | `turin-tui` | `turin-web` | Notes |
| --- | --- | --- | --- | --- |
| `text` | Markdown/content block. | Text lines. | Text/content block. | TUI keeps rendering simple. |
| `section` | Visual grouping with nested nodes. | Heading plus indented nested nodes. | Recursive section grouping. | Recursive rendering in all clients. |
| `action` | Button; optional confirmation modal. | Inspector action list; optional confirmation modal. | Button with optional browser-local confirmation overlay. | Runs through `OperatorCommand::RunHarnessAction` or web action API. |
| `list` | Worklist-backed data table with human-readable field labels, sorted-column markers, filter/sort/limit metadata, row-count and selected-row feedback, app-local row selection, inline detail with pause/claim/failure context, filtered empty-state copy, and confirmed item-action dispatch when available; unsupported adapters show metadata. | Compact worklist-backed table with human-readable field labels, sorted-column markers, filter/sort/limit metadata, local row selection, inspector detail with pause/claim/failure context, filtered empty-state copy, and confirmed item-action dispatch when available; unsupported adapters show metadata. | Worklist-backed table through `/api/ui/list`, with human-readable field labels, sorted-column markers, filter/sort/limit metadata, row-count and selected-row feedback, browser-local row selection, keyboard row navigation, inline detail with pause/claim/failure context, and filtered empty-state copy; unsupported adapters show metadata. | Only named `worklists.<name>` sources have loaders today; app/TUI share stateless visible-node request discovery. |
| `worklist` sugar | Same as `list` with worklist source/intent. | Same as `list` with worklist source/intent. | Same as `list` with worklist source/intent. | DX sugar only; not a separate protocol primitive. |
| `form` | Editable Cast form controls with typed scalar coercion for text, integer, number/float/decimal, boolean, options, and multiline fields. | Terminal modal with local drafts, typed scalar coercion for text, integer, number/float/decimal, boolean, options, and `Ctrl+J` multiline entry for textarea/markdown fields. | Browser-local drafts with required fields, options, number/float/decimal aliases, integer validation, booleans, textarea, and typed scalar coercion. | Rich control fidelity remains client-specific. |
| `activity` | Worklist-backed recent activity; unsupported adapters show metadata. | Compact worklist-backed recent activity; unsupported adapters show metadata. | Worklist-backed recent activity; unsupported adapters show metadata. | Uses cached/loaded `worklists.<name>` data for now, not a live event query. |
| `detail` | Worklist-backed snapshot or explicit item detail, including confirmed item-action buttons when available; unsupported adapters show metadata. | Compact worklist-backed snapshot or explicit item detail, including confirmed item-action dispatch when available; unsupported adapters show metadata. | Worklist-backed snapshot/detail, including confirmed item-action buttons when available; unsupported adapters show metadata. | Without `item_id`, clients show a bounded worklist snapshot. |
| `report` | Cast report section with lightweight worklist-backed status summary, nonstandard-status bucket when present, explicit no-data copy, and next pending item highlight; unsupported adapters show metadata. | Lightweight worklist-backed status summary, nonstandard-status bucket when present, explicit no-data copy, and next pending item highlight; unsupported adapters show metadata. | Lightweight worklist-backed status summary, nonstandard-status bucket when present, explicit no-data copy, and next pending item highlight; unsupported adapters show metadata. | App/TUI share stateless worklist derivation helpers; richer query semantics remain future work. |
| `chart` | Cast report/chart section with lightweight worklist-backed bar breakdown and grouping hint; unsupported adapters show metadata. | Lightweight worklist-backed bar breakdown and grouping hint; unsupported adapters show metadata. | Lightweight worklist-backed bar breakdown and grouping hint; unsupported adapters show metadata. | App/TUI share stateless grouping semantics; `intent` and `as` remain advisory. |

## Dynamic UI Intent

| Intent | `turin-app` | `turin-tui` | `turin-web` | Notes |
| --- | --- | --- | --- | --- |
| `notice` | Recent UI notices and global info panels. | Overview notices panel. | Shell notices/status panel. | Bounded in `UiRegistry`. |
| `open` | Selects app/screen locally. | Selects app/screen locally. | Selects app/screen locally. | Runtime does not own active screen state. |
| `show` | Opens screen targets or displays pane targets as local modals. | Opens screen targets or displays pane targets as local terminal overlays. | Opens screen targets or displays pane targets as local overlays. | Rust clients share stateless screen/pane target classification; pane behavior and open state remain client-specific. |
| `badge` | Dynamic count/label badges render on matching screen/menu navigation targets and titled node ids. | Dynamic count/label badges render on matching screen/menu navigation targets and titled node ids. | Dynamic count/label badges render on matching navigation targets and titled node ids; action-returned badge overlays survive browser status refreshes. | Rust clients share badge text derivation; placement remains client-local chrome, not a renderer contract. |
| `focus` | Selects screen, action/form, or node targets locally. | Selects screen, action/form, or node targets locally; forms use terminal action focus. | Selects matching screen, action/form, or node targets locally. | Focus remains client-local; target matching includes ids, screen titles, action labels/names, form titles/actions, and nested nodes where supported. |
| `refresh` | Invalidates matching list bindings and reloads. | Invalidates matching list bindings and reloads. | Invalidates matching browser caches and reloads visible data. | `ui.refresh` can arrive through runtime events or returned harness action UI intents; `harness.action_ran` also refreshes visible lists when no explicit refresh was emitted. Rust clients share stateless refresh request selection while keeping caches client-local. |

## Action Feedback

| Feedback | `turin-app` | `turin-tui` | `turin-web` | Notes |
| --- | --- | --- | --- | --- |
| action started | Info notice before command dispatch. | Info notice before command dispatch. | Button/running state and shell notice before API call, with action/harness/agent metadata when known. | Client-local feedback. |
| action confirmation | Modal confirmation. | Modal confirmation. | Browser-local confirmation overlay. | Used for explicit-confirm actions and work-item actions. |
| action completed | Latest action result panel near selected harness app, including explicit no-payload copy; returned UI intents are applied locally. | Latest action result in harness inspector only when it matches the selected app source, including explicit no-payload copy; returned UI intents are applied locally. | Latest action result panel in shell screens and open panes only for the originating app, with action/harness/agent metadata and explicit no-payload copy when known; returned UI intents are applied locally. | Rust clients share app-scoping; backed by `UiUpdate::HarnessActionCompleted` for Rust clients and action API results for web. |
| action failed | App-scoped latest action failure panel plus global dashboard error notice. | App-scoped latest action failure in the harness inspector plus global dashboard error notice. | Latest action failure panel and shell error notice only for the originating app, with action/harness/agent metadata when known. | Rust clients share app-scoping; error stays operator feedback, while durable failure state belongs in runtime primitives. |

## Web API Coverage

`turin-web` exposes semantic intent and data over HTTP and serves a minimal
browser shell that consumes those routes.

| Route | Current coverage |
| --- | --- |
| `GET /api/status` | Dashboard snapshot plus derived UI registry. |
| `GET /api/apps` | Declared app surfaces from harness UI intent. |
| `GET /api/apps/{app_id}` | One app's screens, menus, panes, and badges. |
| `POST /api/ui/list` | Worklist-backed semantic list loading. |
| `POST /api/actions/run` | Harness action execution with JSON result. |
| `GET /api/events` | SSE runtime/UI event stream for client-side invalidation; the browser shell tracks event-stream health separately from HTTP refresh health. |
| `GET /` | Static browser shell for default runtime status plus app/screen/list/form/action rendering, including typed local form drafts. |

## Current Gaps

- App/TUI now share visible-node request discovery plus lightweight report/chart
  derivation helpers, and the web shell mirrors the same status/grouping rules,
  but richer report/chart rendering still needs a fuller shared query/data
  semantics layer.
- List data loading only supports named `worklists.<name>` sources.
- Dynamic badges currently render on navigation targets and titled node ids;
  field-level or arbitrary inline badge placement remains undefined.
- TUI item selection is local to compact table rows with visible row-range,
  row-position, selected-row, and action-available cues; selected item actions
  can be queued for confirmation, page/boundary keys work within the focused
  region, long tables window around the selected row instead of hiding it
  off-screen, and row identity is preserved across refresh/reorder where
  possible.
- TUI pane overlays support pane-local item/action/form selection using
  terminal-local indices separate from screen focus.
- `turin-web` list rows support row-count and selected-row feedback plus click,
  Enter/Space, ArrowUp/ArrowDown, Home, and End selection with focus restored
  after browser-local re-render.
- `turin-web` report/chart rendering is useful for worklist-backed summaries
  but still not a shared final semantics layer.
- Unsupported data sources remain visible in app, TUI, and web clients with
  fallback copy that names the source, names the current client when relevant,
  and points authors toward `worklists.<name>` or a deliberate client adapter.
- Visible worklist-backed surfaces whose data has not loaded yet remain visible
  with explicit not-yet-loaded copy instead of rendering blank panels.
- App has helper-level checks for default no-harness console copy,
  runtime/work/UI metric grouping, visible screen/pane data requests, and
  confirmed work-item action event construction. App/TUI refresh invalidation
  request selection is covered in shared UI-core helper tests.
- TUI has seed normalized terminal golden fixtures for the default no-harness
  console, a harness screen, a loaded worklist table with selected-row/action
  cues, a loaded report/chart screen, a pane overlay with pane-local selection
  cues, and a form modal with typed previews and validation copy. There is still
  no broader screenshot or terminal-golden suite across clients. Current
  coverage is otherwise semantic helper tests plus app/web/TUI smoke and unit
  tests.

## Update Rule

Update this page whenever a client materially changes how it renders, degrades,
or ignores semantic UI intent.
