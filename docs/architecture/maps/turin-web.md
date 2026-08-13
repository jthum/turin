# Turin Web Map

## Purpose

`turin-web` is Turin's web-facing client adapter. It exposes a small JSON API
over the existing control layer so browser clients can render the same semantic
UI intent as `turin-app` and `turin-tui`.

Keep this crate thin. It should not host a second runtime, own durable UI
session state, invent renderer-specific harness APIs, or bypass
`turin-control-client` for daemon operations that already have typed helpers.

## Files

- `crates/turin-web/src/main.rs`
  - CLI argument parsing and `ConnectionSpec` selection.
- `crates/turin-web/src/server.rs`
  - Hyper HTTP/1 server startup, shutdown, bind policy, and shared state setup.
- `crates/turin-web/src/routes.rs`
  - JSON route handling for status, bounded session detail, session lifecycle,
    task submission, turn topology, branch/sidestep/promotion workflows, app
    registry, UI list loading, action runs, liveness, SSE event streaming, and
    first-party static routes.
- `crates/turin-web/frontend/`
  - Svelte/TypeScript client source, typed `TurinClient` transport boundary,
    custom CSS, assistant surface, built-in Data Explorer, Agent Orchestration,
    Harness Studio, Work Operations, and semantic harness renderer.
- `crates/turin-web/static/`
  - Checked-in Vite build output embedded by the Rust binary. Cargo builds do
    not require Node; regenerate these assets after frontend changes.
- `crates/turin-web/tests/ui_contract.rs`
  - End-to-end smoke against a temporary daemon and test-only semantic UI
    fixture, both through local IPC and through `turin-remote`.

## Data Flow

1. CLI or caller builds a `WebServeOptions`.
2. `turin-web` connects through `turin-control-client`.
3. HTTP routes call typed control-client helpers.
4. UI app responses derive a `UiRegistry` from harness UI intent in daemon
   status.
5. UI list responses resolve semantic sources such as `worklists.release` into
   bounded worklist item queries.
6. SSE event streams proxy managed runtime subscriptions for invalidation and
   refresh hints.
7. Static browser assets load from the same origin and call `/api/*` routes
   directly.
8. The browser remains responsible for selected app, active screen, form drafts,
   panes, modals, filters, loading, and local error state.
9. Browser `ui.refresh` handling invalidates cached list requests whose
   semantic source matches the refresh binding, then reloads visible data.
10. Browser action responses can include returned dynamic UI intents; the shell
    applies them locally before refreshing visible data.
11. Browser-local dynamic badges are re-applied to freshly fetched app records
    after status refreshes, because action-returned badges are presentation
    hints rather than durable daemon state.
12. Browser list-load failures are cached per semantic request and can be
    retried by deleting that local cache entry and calling `/api/ui/list` again.
13. The assistant loads a bounded latest-message window through
    `GET /api/session`. Absolute offsets slide a 100-message data window while
    preserving the browser's visible-message anchor.
14. The active conversation subscribes to a session-scoped SSE feed. Message
    deltas render directly and are batched to animation frames; terminal stream
    events invalidate and reload the bounded durable window.
15. Follow-up task submission idempotently resumes the selected session in its
    previous slot immediately before enqueueing work, so peer-runtime idle eviction
    does not invalidate a browser conversation.
16. Assistant Markdown is parsed in the browser and sanitized before insertion;
    streaming text uses the same renderer after animation-frame batching.
17. Variable-height transcript virtualization keeps about 30 user/assistant
    presentation messages mounted inside the current data window.
18. `HttpTurinClient` is one implementation of the frontend transport contract.
    An embedded desktop host can provide a bridge implementation without
    changing Svelte presentation components.
19. Session detail carries a bounded efficiency projection. The assistant
    presents provider-measured input/output totals separately from estimated
    request composition, per-message cost, context utilization, payload size,
    compaction, and reusable-prefix opportunity.
20. Live `inference_request`, thinking, tool, and message events drive a
    contained response-status bubble with phase and elapsed time. Terminal
    events still reconcile the surface against durable session detail.
21. The built-in Data Explorer uses typed, bounded endpoints for worklists,
    memories, and sessions. Memory browsing is observational and does not count
    as a retrieval.
22. Opening a conversation requests the latest durable message window, mounts
    its newest render slice, then settles scroll position after layout.
23. Manual title edits use the typed daemon session-title operation and remain
    durable. Automatic naming is harness policy; the browser observes the
    resulting session metadata rather than deriving and persisting its own title.
24. The Assistant Run Center renders the bounded execution projection from
    session detail. Task, plan, turn, tool, execution-policy, branch-outcome,
    and error disclosure remains observational; lifecycle SSE events only
    invalidate and reconcile that durable projection.
25. When the daemon advertises feature-gated perf events, the Assistant
    Efficiency panel displays directly measured retrieval timings, exact query
    counters, and daemon-process memory trends from the existing session event
    stream; normal sessions never display this surface.
26. The Session Graph is loaded only when opened. It renders the daemon's
    durable turn-parent topology and branch heads while keeping selected nodes,
    compact/expanded layout, drafts, and action results browser-local.
27. A graph node may fork by exact turn id, run an ephemeral or fork-sibling
    sidestep from that context, and promote an eligible ephemeral result. Branch
    checkout and activation remain daemon operations and therefore retain live
    runtime ambiguity/busy guards.
28. Selecting a graph node loads a bounded exact-turn transcript projection.
    The browser caches that path locally, highlights the inspected ancestry
    separately from the active path, and keeps mutation under an explicit
    exploration mode.
29. Branch-head comparison is browser-local and read-only. The browser derives
    the exact shared ancestor and divergent turn counts from durable topology,
    then composes two bounded exact-turn projections for recent branch-only
    message and tool previews.
30. Completed assistant turns expose browser-local, hover-disclosed entry points
    for exact-turn inspection and forking without inserting persistent action
    rows into the transcript. The selected turn and dialog draft remain local.
31. The assistant uses each projected message's exact turn id to mark only real
    branch points. Their collapsed path switcher can reveal direct checkout and
    graph inspection, while fork naming stays in a focused modal dialog.
32. Agent Orchestration is a runtime-wide read surface over configured agents,
    aggregate runtime status, independent live slots, and bounded peer-task
    snapshots. It groups related work by stable trace id and links attached
    slots back to their conversations without maintaining a browser task ledger.
33. Browser agent inventory comes from effective `agent_runtimes`. Filesystem
    registry entries enrich peer agents with provider, model, and harness metadata
    but do not contain the bootstrap agent and therefore cannot define the roster.
34. Assistant routing controls keep durable session ownership fixed. Selecting
    another agent starts a new conversation, while selecting a named inference
    context changes the route for subsequent tasks in the current conversation.
35. Harness Studio calls typed harness lifecycle helpers through the control
    client. Inventory and detail cover bootstrap, shared, and agent-local
    harnesses; create/delete apply only to managed shared harnesses, while
    validation and atomic reload retain daemon ownership of runtime state.
36. Work Operations joins bounded worklist inspection, durable schedule
    lifecycle, bounded schedule run history, retained task snapshots, and safe
    task cancellation. It does not add a browser-owned work ledger.
37. Harness Studio and Work Operations are deterministic lazy-loaded chunks.
    Rust embeds and serves those explicit chunk paths while ordinary assistant
    startup loads only the main browser bootstrap.
38. Harness source APIs remain thin control-client adapters. They expose recursive Lua source inspection, whole-candidate validation, and hash-guarded batch saves without making `turin-web` a filesystem owner.
39. Harness Studio presents source authoring, runtime overview, and API reference
    as separate views. Its multi-file editor lazily reads source, keeps additions,
    replacements, and deletions in browser-local buffers, and enables batch save
    only when the exact current candidate has passed daemon validation.

## Invariants

- Runtime state remains in the daemon; web session state remains in the browser.
- Dirty harness buffers remain browser state until an explicit save. Validation may include all dirty additions, replacements, and deletions without persisting any of them.
- Editing any source after candidate validation invalidates that validation.
  Harness Studio must not save a candidate different from the one the daemon
  validated, and stale source hashes must remain visible conflicts rather than
  silently overwriting external file edits.
- A source save response means the daemon persisted file changes. Runtime activation remains asynchronous watcher behavior and may retain the previous valid generation when the saved candidate is invalid.
- Agent selectors must include every effective runtime status. They may join
  registry metadata by id, but must not treat registry peers as the full agent set.
- Browser model controls submit configured inference-context ids, never raw
  provider URLs or unvalidated model overrides. Harness turn policy remains able
  to replace that requested route.
- Opening the Assistant starts from a browser-local fresh-conversation draft. Persisted or live
  sessions are selected explicitly rather than adopting an unrelated channel runtime by default.
- Browser matching between persisted summaries and live sessions must compare bare and
  store-qualified session references by session identity.
- A browser-held live-session snapshot is advisory. Follow-up sends must re-establish
  the session because the daemon may have evicted its idle runtime slot.
- Automatic title generation belongs to the active harness. Browser clients
  must not race that policy with their own persisted first-prompt title.
- Parsed assistant Markdown must be sanitized before using Svelte's raw-HTML rendering.
  Remote images and generated inline styles remain disabled in the transcript.
- Raw persisted tool-result messages remain diagnostic data, not chat bubbles.
  Tool cards correlate to their assistant tool-use message by tool-call id.
- The visible session reference must remain copyable for diagnostics without
  making it browser-owned session state.
- Browser form drafts, confirmation modals, and action-running state are
  memory-local and should not be persisted by `turin-web`.
- Browser option-backed form selects should encode and decode option values as
  JSON scalars so non-string options stay typed through submission.
- Browser multiline form aliases such as `textarea`, `markdown`, and
  `multiline` should render as textarea controls while remaining browser-local
  drafts until submit.
- Browser password-like form fields should use password inputs while keeping
  drafts browser-local and submitted params unchanged.
- Dynamic UI intents returned by action responses are local presentation hints,
  not durable browser session state.
- Browser action feedback may display action/harness/agent routing metadata
  from the request or response, but this is operator context only and should not
  become persisted `turin-web` session state.
- Browser action completion feedback should explicitly distinguish a successful
  action with no result payload from a successful action with JSON detail.
- Browser connection feedback should distinguish HTTP refresh health from SSE
  event-stream health so a successful refresh does not hide event invalidation
  reconnects or stream errors.
- Browser modal/pane overlays should expose dialog semantics, close on Escape,
  keep Tab navigation inside the active overlay, and keep keyboard focus inside
  a safe dialog control across initial render and overlay rerender.
- Browser `ui.focus` handling should resolve screen ids/titles, node ids,
  action names/labels, form actions/titles, and nested section targets while
  keeping the selected screen as browser-local state.
- Browser default-screen selection should resolve `opens_with` through the same
  screen id/title helper as open/show requests before falling back to the first
  declared screen.
- Browser-local dynamic badge overlays may survive status refreshes, but they
  still remain client memory state and must not be persisted by `turin-web`.
- Menu `badge` values select dynamic badge targets. The browser must not expose
  those identifiers, or screen presentation hints, as fallback navigation copy.
- Local and remote control connections should stay behaviorally symmetric.
- `GET /api/apps` and `GET /api/apps/{app_id}` should expose semantic UI
  surfaces, not renderer-specific widget state.
- `POST /api/ui/list` should accept semantic UI list requests first. Add raw
  daemon-query escape hatches only after the UI model proves it needs them.
- Built-in operator data routes should expose Turin-owned semantic projections,
  not arbitrary SQLite tables or embedding blobs.
- Unsupported sources and planned endpoints should return explicit JSON errors,
  not silent empty responses.
- JSON request body limits should be enforced while reading the body, not after
  buffering an oversized request.
- `GET /api/events` is an invalidation/event feed. Browser `ui.refresh`
  handling should stay cache invalidation plus visible-data reload, not grow
  into a live-query result cache. High-frequency message deltas may be rendered
  ephemerally, but durable transcript truth still comes from bounded session
  detail queries.
- Session identifiers are compound and may contain store paths. Pass them as an
  encoded `session_id` query/body value; do not place them in URL path segments.
- Session detail requests must remain bounded. The web boundary rejects zero or
  excessively large message windows rather than exposing unbounded history.
- Session Graph reads must remain explicit and on demand. The turn tree is
  durable conversation topology and must not be conflated with harness-authored
  `runtime.graph.*` semantic nodes and edges.
- Selecting a graph node is browser-local inspection. Persisted active-branch
  state changes only through explicit checkout or activating-fork actions.
- Exact-turn path previews must use the bounded daemon projection rather than
  filtering the currently active transcript in the browser. Preview loading
  and errors remain local to the selected graph node.
- Branch comparison must not imply merge or context-transfer semantics. Exact
  divergence comes from durable turn topology; bounded transcript previews must
  identify branch-only rows omitted by their message window.
- Per-turn chat controls may open inspection or a local fork dialog at an
  active-path turn, but must not fork, checkout, or submit work merely by
  revealing those controls or opening the graph. Ordinary turns should not
  reserve transcript space for hidden controls.
- Inline forks and branch-point chips must use exact message `turn_id` and
  branch `source_turn_id` values. Repeated path-relative turn indexes are not
  sufficient identity once branches diverge.
- Graph forks must use exact turn ids. A path-relative turn index is not a
  stable source once sibling branches exist.
- Request-efficiency UI must label provider-reported usage as measured and
  Turin-derived request/message counts as estimated. A reusable prefix is not
  evidence that a provider cache accepted or billed it differently.
- Provider cache-read and cache-creation counters are displayed only when the
  inference provider reports them. Missing counters remain unavailable rather
  than being inferred or presented as zero.
- The Run Center must use the daemon's bounded typed execution projection. It
  must not fetch the raw event history, retain a second task ledger in browser
  memory, or treat SSE delivery as durable execution truth.
- Agent Orchestration must refresh from dashboard snapshots after runtime and
  task events. Event payloads are invalidation hints; configured agents, live
  slots, and task state remain daemon-owned truth.
- Harness Studio must use daemon validation, reload, scaffold, and guarded
  deletion operations. It must not become an arbitrary filesystem editor or
  classify bootstrap/agent-local harnesses as managed shared harnesses.
- Worklist inspection in Work Operations is observational. Generic browser
  mutations must not bypass harness actions, claims, dependency rules, or
  governance policy.
- Scheduled work creation may expose every field supported by the current
  typed create contract. Do not expose recurrence-family editing until the
  update protocol can explicitly clear existing optional recurrence fields.
- Task cancellation is cooperative and daemon-owned. The browser may request
  cancellation and reconcile status, but must not present it as an immediate
  process kill.
- Live perf diagnostics are a development overlay, not durable session data.
  Internal timings must be described as instrumented measurements, while query
  counters may be called exact. Daemon RSS/PSS deltas are process-level
  correlations rather than allocation attribution.
- Live response indicators should participate in normal chat layout. They must
  not use absolute-positioned cursors that can escape the response bubble.
- Frontend source lives under `frontend/`; `static/` is generated and checked in
  so Rust-only builds remain reproducible without installing Node dependencies.
- Lazy browser chunks use deterministic names and must have matching embedded
  Rust routes. The UI contract smoke should fetch them so a successful frontend
  build cannot leave production dynamic imports returning `404`.
- The browser shell should keep semantic list constraints such as named
  filter/sort/limit metadata visible when rendering list nodes and mark sorted
  columns, including advisory direction when declared, in table headers. Those
  constraints are request metadata, not browser session state.
- Browser selected-row detail should surface existing work-item operational
  fields such as created/updated timestamps, pause, claim, parent, completion,
  and failure context without persisting selection in `turin-web`.
- Browser empty-list copy should name declared filters when they may explain an
  empty worklist-backed result.
- Browser list-load error states should offer a local per-request retry path so
  a failed cache entry does not require a full page reload.
- Browser report/chart surfaces should remain visible with explicit no-data
  copy when the backing worklist has no rows, and chart breakdowns should show
  count plus percentage labels without requiring shared query state.
- Non-loopback binds require explicit opt-in.

## Common Changes

Add a new UI data endpoint:

1. Check whether the control client already has a typed helper.
2. Add the route in `routes.rs` and keep request/response structs serializable.
3. Prefer semantic UI requests over raw daemon protocol shapes for harness UI
   client routes.
4. Extend the UI contract smoke when the behavior can be exercised there.

Add browser support:

1. Keep the HTTP API stable and small.
2. Let the browser own ephemeral navigation and selection state.
3. Use `GET /api/events` for event-driven invalidation rather than adding live
   query semantics prematurely.

## Tests

Focused checks:

```sh
cargo test -p turin-web --lib
cargo test -p turin-web --test ui_contract
cargo test -p turin-web
cargo check -p turin-web
cd crates/turin-web/frontend && npm run check && npm run build
```

Basic checks:

```sh
cargo fmt --all -- --check
git diff --check
```
