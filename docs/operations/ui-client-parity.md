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

| Intent | `turin-app` | `turin-tui` | Notes |
| --- | --- | --- | --- |
| `app` | Selectable harness app cards/list. | Selectable harness app navigation. | Multiple apps are allowed; client state remains local. |
| `screen` | Screen tabs within selected app. | Screen navigation plus flattened menu entries. | Default screen comes from `opens_with`. |
| `menu` | Menu groups and nested menu entries. | Flattened terminal navigation with indentation. | Menus are navigation intent, not layout. |
| `pane` | Declared but not rendered as a first-class pane yet. | Noted/degraded when shown dynamically. | Needs a client-specific pane model later. |

## Node Rendering

| Node | `turin-app` | `turin-tui` | Notes |
| --- | --- | --- | --- |
| `text` | Markdown/content block. | Text lines. | TUI keeps rendering simple. |
| `section` | Visual grouping with nested nodes. | Heading plus indented nested nodes. | Recursive rendering in both clients. |
| `action` | Button; optional confirmation modal. | Inspector action list; optional confirmation modal. | Runs through `OperatorCommand::RunHarnessAction`. |
| `list` | Worklist-backed data table; unsupported adapters show metadata. | Compact worklist-backed table; unsupported adapters show metadata. | Only `worklists.*` sources have loaders today. |
| `worklist` sugar | Same as `list` with worklist source/intent. | Same as `list` with worklist source/intent. | DX sugar only; not a separate protocol primitive. |
| `form` | Editable Cast form controls. | Terminal modal with local drafts and typed scalar coercion. | TUI rich text areas degrade to line-oriented text. |
| `activity` | Worklist-backed recent activity; unsupported adapters show metadata. | Compact worklist-backed recent activity; unsupported adapters show metadata. | Uses cached `worklists.*` data for now, not a live event query. |
| `detail` | Worklist-backed snapshot or explicit item detail; unsupported adapters show metadata. | Compact worklist-backed snapshot or explicit item detail; unsupported adapters show metadata. | Without `item_id`, clients show a bounded worklist snapshot. |
| `report` | Placeholder with prompt text. | Placeholder. | Needs data/query semantics before rich rendering. |
| `chart` | Placeholder with source/rendering metadata. | Placeholder. | `intent` and `as` remain advisory. |

## Dynamic UI Intent

| Intent | `turin-app` | `turin-tui` | Notes |
| --- | --- | --- | --- |
| `notice` | Recent UI notices and global info panels. | Overview notices panel. | Bounded in `UiRegistry`. |
| `open` | Selects app/screen locally. | Selects app/screen locally. | Runtime does not own active screen state. |
| `show` | Opens screen when target is a screen; panes are recognized. | Opens screen when target is a screen; panes degrade to a notice. | Pane behavior is intentionally client-specific. |
| `badge` | Recorded in registry; limited visual use today. | Recorded in registry; menu badges display when declared statically. | Dynamic badge rendering needs more client work. |
| `focus` | Selects screen/action target locally. | Selects screen/action target locally. | Focus remains client-local. |
| `refresh` | Invalidates matching list bindings and reloads. | Invalidates matching list bindings and reloads. | `harness.action_ran` also refreshes visible lists when no explicit refresh was emitted. |

## Action Feedback

| Feedback | `turin-app` | `turin-tui` | Notes |
| --- | --- | --- | --- |
| action started | Info notice before command dispatch. | Info notice before command dispatch. | Client-local feedback. |
| action confirmation | Modal confirmation. | Modal confirmation. | Used for explicit-confirm actions. |
| action completed | Latest action result panel near selected harness app. | Latest action result in harness inspector. | Backed by `UiUpdate::HarnessActionCompleted`. |
| action failed | Dashboard error notice from command task. | Dashboard error notice from command task. | Error stays operator feedback; durable failure state belongs in runtime primitives. |

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
| `GET /` | Static browser shell for app/screen/list/form/action rendering. |

## Current Gaps

- `report` and `chart` need real adapters or more explicit semantics.
- List data loading only supports worklist sources.
- Dynamic badge rendering is not yet strong enough in either client.
- The TUI has no item-selection-driven detail pane yet.
- `turin-web` activity/detail/report/chart rendering is still placeholder-level.
- There is no automated screenshot/terminal golden test layer; current coverage is unit and smoke tests.

## Update Rule

Update this page whenever a client materially changes how it renders, degrades,
or ignores semantic UI intent.
