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
| `turin-app` | Rich graphical operator client built on egui/Cast. |
| `turin-tui` | Lean keyboard-first terminal client built on Ratatui. |
| `turin-web` | API-first web adapter with a minimal same-origin browser shell. See `docs/operations/turin-web.md`. |

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
| `list` | Worklist-backed data table with app-local row selection, inline detail, and confirmed item-action dispatch when available; unsupported adapters show metadata. | Compact worklist-backed table with local row selection, inspector detail, and confirmed item-action dispatch when available; unsupported adapters show metadata. | Worklist-backed table through `/api/ui/list`, with browser-local row selection and inline detail; unsupported adapters show metadata. | Only `worklists.*` sources have loaders today. |
| `worklist` sugar | Same as `list` with worklist source/intent. | Same as `list` with worklist source/intent. | Same as `list` with worklist source/intent. | DX sugar only; not a separate protocol primitive. |
| `form` | Editable Cast form controls. | Terminal modal with local drafts and typed scalar coercion. | Browser-local drafts with required fields, options, and typed scalar coercion. | Rich control fidelity remains client-specific. |
| `activity` | Worklist-backed recent activity; unsupported adapters show metadata. | Compact worklist-backed recent activity; unsupported adapters show metadata. | Worklist-backed recent activity; unsupported adapters show metadata. | Uses cached/loaded `worklists.*` data for now, not a live event query. |
| `detail` | Worklist-backed snapshot or explicit item detail, including confirmed item-action buttons when available; unsupported adapters show metadata. | Compact worklist-backed snapshot or explicit item detail, including confirmed item-action dispatch when available; unsupported adapters show metadata. | Worklist-backed snapshot/detail, including confirmed item-action buttons when available; unsupported adapters show metadata. | Without `item_id`, clients show a bounded worklist snapshot. |
| `report` | Cast report section with lightweight worklist-backed summary and next pending item highlight; unsupported adapters show metadata. | Lightweight worklist-backed summary and next pending item highlight; unsupported adapters show metadata. | Lightweight worklist-backed summary and next pending item highlight; unsupported adapters show metadata. | App/TUI share stateless worklist derivation helpers; richer query semantics remain future work. |
| `chart` | Cast report/chart section with lightweight worklist-backed bar breakdown; unsupported adapters show metadata. | Lightweight worklist-backed bar breakdown; unsupported adapters show metadata. | Lightweight worklist-backed bar breakdown; unsupported adapters show metadata. | App/TUI share stateless grouping semantics; `intent` and `as` remain advisory. |

## Dynamic UI Intent

| Intent | `turin-app` | `turin-tui` | `turin-web` | Notes |
| --- | --- | --- | --- | --- |
| `notice` | Recent UI notices and global info panels. | Overview notices panel. | Shell notices/status panel. | Bounded in `UiRegistry`. |
| `open` | Selects app/screen locally. | Selects app/screen locally. | Selects app/screen locally. | Runtime does not own active screen state. |
| `show` | Opens screen targets or displays pane targets as local modals. | Opens screen targets or displays pane targets as local terminal overlays. | Opens screen targets or displays pane targets as local overlays. | Pane behavior is intentionally client-specific. |
| `badge` | Dynamic count/label badges render on matching screen/menu navigation targets and titled node ids. | Dynamic count/label badges render on matching screen/menu navigation targets and titled node ids. | Dynamic count/label badges render on matching navigation targets and titled node ids. | Placement remains client-local chrome, not a renderer contract. |
| `focus` | Selects screen/action target locally. | Selects screen/action target locally. | Selects matching screen/action target locally. | Focus remains client-local. |
| `refresh` | Invalidates matching list bindings and reloads. | Invalidates matching list bindings and reloads. | Invalidates matching browser caches and reloads visible data. | `harness.action_ran` also refreshes visible lists when no explicit refresh was emitted. |

## Action Feedback

| Feedback | `turin-app` | `turin-tui` | `turin-web` | Notes |
| --- | --- | --- | --- | --- |
| action started | Info notice before command dispatch. | Info notice before command dispatch. | Button/running state and shell notice before API call. | Client-local feedback. |
| action confirmation | Modal confirmation. | Modal confirmation. | Browser-local confirmation overlay. | Used for explicit-confirm actions and work-item actions. |
| action completed | Latest action result panel near selected harness app. | Latest action result in harness inspector. | Latest action result panel in shell screens and open panes. | Backed by `UiUpdate::HarnessActionCompleted` for Rust clients and action API results for web. |
| action failed | Dashboard error notice from command task. | Dashboard error notice from command task. | Shell error notice from failed action API call. | Error stays operator feedback; durable failure state belongs in runtime primitives. |

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
| `GET /api/events` | SSE runtime/UI event stream for client-side invalidation. |
| `GET /` | Static browser shell for app/screen/list/form/action rendering, including typed local form drafts. |

## Current Gaps

- `report` and `chart` now share lightweight app/TUI derivation helpers and the
  web shell mirrors the same status/grouping rules, but they still need a
  fuller shared query/data semantics layer before rich rendering.
- List data loading only supports worklist sources.
- Dynamic badges currently render on navigation targets and titled node ids;
  field-level or arbitrary inline badge placement remains undefined.
- TUI item selection is local to visible compact table rows; selected item
  actions can be queued for confirmation, and page/boundary keys now work
  within the focused region, but richer table widgets can still improve.
- TUI pane overlays support pane-local item/action/form selection using
  terminal-local indices separate from screen focus.
- `turin-web` report/chart rendering is useful for worklist-backed summaries
  but still not a shared final semantics layer.
- Unsupported data sources remain visible in app, TUI, and web clients with
  fallback copy that names the source and points authors toward `worklists.*`
  or a deliberate client adapter.
- There is no automated screenshot/terminal golden test layer. Current coverage
  is semantic helper tests plus app/web/TUI smoke and unit tests.

## Update Rule

Update this page whenever a client materially changes how it renders, degrades,
or ignores semantic UI intent.
