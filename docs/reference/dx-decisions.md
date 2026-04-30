# DX Decisions

This page records accepted harness-DX decisions.

Use this document for stable decisions that should guide implementation and documentation.

Use `.workspace` review docs for exploration, alternatives, and unresolved discussion.

## Decision Format

Each decision records:

- status
- decision
- rationale
- affected surfaces
- follow-up

## 2026-04-26 — Stable Substrate Vocabulary

Status:

- accepted

Decision:

- keep `branch_head` as the structural named writable handle
- keep `selected_path` as the materialized read-target term
- keep `context_target` as the umbrella execution-selector term
- keep `graph_node` / `graph_edge` as sparse semantic overlay primitives

Rationale:

- these names correctly preserve the distinction between:
  - structural execution substrate
  - materialized execution read targets
  - sparse semantic overlay

Affected surfaces:

- `runtime.graph.*`
- branch/session APIs
- docs and future DX wrappers

Follow-up:

- helper-layer sugar may soften these names at the call site
- substrate semantics should continue to use these exact terms

## 2026-04-26 — Graph Ref Constructors

Status:

- accepted

Decision:

- add tiny helper constructors for graph refs in the helper layer

Initial target shape:

```lua
graph.node(id)
graph.branch(id)
graph.turn(id)
```

Rationale:

- repeated `{ kind = "...", id = "..." }` tables are noisy
- constructors reduce authoring friction without changing graph semantics

Affected surfaces:

- graph/path authoring helpers
- docs/examples that currently repeat low-level ref tables

Follow-up:

- implement as sugar only
- do not rename or weaken the underlying `GraphRef` substrate shape

## 2026-04-26 — Capability Short Forms In The Helper Layer

Status:

- accepted

Decision:

- helper-layer capability checks may accept short forms such as:
  - `agent.submit`
  - `db.exec`
- short forms elide `runtime.` only
- short forms are accepted in helper-layer capability surfaces such as:
  - `needs(...)`
  - `allowed(...)`
  - `access.check(...)`
  - delegated capability tables in helper-style task options
- fully qualified substrate capability names remain canonical:
  - `runtime.agent.submit`
  - `runtime.db.exec`
  - `runtime.fs.write`

Default rule:

- if a helper receives an unqualified capability string, assume `runtime.` by default

Rationale:

- short forms read more like prose in helpers such as:
  - `needs("agent.submit")`
  - `allowed("db.exec")`
- fully qualified names remain available for explicitness and future disambiguation

Affected surfaces:

- `needs(...)`
- `allowed(...)`
- `access.check(...)`
- helper-layer capability tables
- helper docs/examples

Follow-up:

- keep mapping rules simple and explicit
- avoid clever multi-domain inference beyond the default `runtime.` assumption unless clearly needed later

## 2026-04-26 — Promoted Authoring Style vs Canonical Substrate

Status:

- accepted

Decision:

- Turin should preserve multiple authoring altitudes
- the promoted authoring style should be the full prose-like DX/helper layer wherever it can express the task cleanly
- lower layers should remain documented for exact control and contributor understanding
- `runtime.*` remains the canonical substrate and should stay fully documented

Rationale:

- multiple styles are not inherently bad
- the product should still have one clearly preferred voice
- uncurated multiplicity increases:
  - onboarding cost
  - doc complexity
  - example inconsistency
  - contributor uncertainty about which style to extend

Affected surfaces:

- docs hierarchy
- examples
- helper-layer evolution

Follow-up:

- docs and examples should lead with the prose-like DX/helper style
- this does not require removing every lower-level surface
- redundant helper variants may still be removed later if they do not justify their maintenance cost

## 2026-04-27 — Remove Heavyweight File Cache Surface

Status:

- accepted

Decision:

- deprecate and remove:
  - `cache.file(...)`
  - `runtime.cache.*`
- remove the related file-cache schema and persistence layer:
  - `file_cache_reads`
  - `file_cache_versions`

Rationale:

- the current design stores full file snapshots in main session state
- its strongest claimed benefit, skipping unchanged file content for token savings, is only heuristic
- Turin cannot reliably know whether prior file content is still materially present in the effective inference context
- the strongest real use-case is lightweight invalidation of derived state, which does not justify:
  - full content retention
  - diff storage
  - dedicated cache tables
- richer file-history and diff workflows belong with Git or project-local indexing, not the main Turin state DB

Affected surfaces:

- `cache.file(...)`
- `runtime.cache.*`
- harness docs and examples
- persistence schema and state-store cache code

Follow-up:

- remove the surface completely rather than soft-deprecating it
- replace only the useful parts with lighter primitives and helpers

## 2026-04-27 — Add A Generic Hash Primitive, Not `fs.hash`

Status:

- accepted

Decision:

- add a generic hash primitive for harness authors
- do not add `fs.hash(...)` initially

Initial direction:

```lua
hash.sha256(text)
```

Rationale:

- hashing is useful beyond files:
  - URLs
  - endpoint responses
  - command output
  - structured summaries
  - manual KV-based invalidation patterns
- file-oriented flows can use:
  - `fs.stat(path).hash`
  - or `hash.sha256(fs.read(path))`
- adding both `hash.*` and `fs.hash(...)` immediately would create parallel surface area without enough payoff

Affected surfaces:

- system globals
- helper docs/examples
- future file-observation helpers

Follow-up:

- implement a small generic hash namespace first
- consider file-specific sugar later only if `fs.stat(...).hash` proves too clumsy in practice

## 2026-04-27 — Add `fs.stat(...)` As The File Metadata Primitive

Status:

- accepted

Decision:

- add `fs.stat(path)` as the low-level file metadata primitive
- let `fs.stat(path)` also expose session-relative change tracking for the common authoring path

Initial direction:

- current file facts such as:
  - `hash`
  - `bytes`
  - file timestamps where reliable and useful
- common change-tracking fields such as:
  - `changed`
  - `previous_hash`
  - `seen_before`

Rationale:

- file-oriented workflows need an explicit metadata surface
- this is the right place for lightweight file observation
- it is a cleaner primitive than a heavyweight session cache abstraction
- unlike `fs.read(...)`, `fs.stat(...)` already returns a table, so adding change metadata does not create a polymorphic return-type problem
- the common authoring flow should be simple:

```lua
local spec = fs.stat("SPEC.md")
if spec.changed then
  ...
end
```

Affected surfaces:

- `fs.*`
- file-oriented helper design
- docs/examples that currently point authors toward `cache.file(...)`

Follow-up:

- implement change detection by:
  - computing the current content hash
  - comparing it with the prior session-scoped hash
  - updating the stored hash automatically
- keep `fs.stat(...)` table-shaped and file-focused
- avoid turning it into a polymorphic summary/cache API

## 2026-04-27 — Add `fs.summary(...)` As The Promoted File-Summary Helper

Status:

- accepted

Decision:

- add `fs.summary(path, opts?)` as the promoted high-level helper for file summarization
- implement it on top of:
  - normal file reads
  - the generic hash primitive
  - KV-backed invalidation
  - summary generation primitives

Rationale:

- summarizing a few important files is a dominant and high-value harness workflow
- the happy-path DX is materially better as:

```lua
ctx.system_prompt = ctx.system_prompt
  .. "\n\nSpec:\n" .. fs.summary("SPEC.md")
  .. "\n\nConstraints:\n" .. fs.summary("CONSTRAINTS.md")
```

- this keeps file-centric authoring in the `fs.*` namespace without preserving the current heavyweight cache design

Affected surfaces:

- `fs.*`
- harness guide/examples
- future summary/invalidation helpers

Follow-up:

- keep `fs.summary(...)` file-centric and easy to interpolate into prompts
- prefer the noun form `summary` over a procedural `summarize`
- use lower-level primitives underneath so behavior stays consistent if a more generic summarization surface is added later

## 2026-04-28 — Helper vs Substrate Boundary

Status:

- accepted

Decision:

- canonical substrate owns:
  - durable state semantics
  - governance/capability semantics
  - visibility/durability/write-policy semantics
  - branch/path/execution semantics
  - anything another helper or client may need to reason about directly
- helper layer may:
  - choose better defaults
  - collapse boilerplate
  - compose multiple substrate calls
  - maintain small internal bookkeeping in reserved keys
  - present a more prose-like authoring voice
- helper layer must not:
  - invent durable concepts that do not exist in substrate
  - bypass governance or visibility rules
  - hide surprising writes outside clearly documented helper-owned state
  - become the only place where a core runtime behavior exists

Litmus test:

- if removing the helper would delete a real runtime capability, it belongs in substrate
- if removing the helper would only make code uglier, it belongs in helpers

Reserved helper bookkeeping:

- helper-owned internal state should use a reserved prefix such as `_turin:`

Rationale:

- Turin needs both stable primitives and beautiful authoring
- the boundary must be explicit so helper growth does not quietly turn into parallel semantics

Affected surfaces:

- all helper globals
- `runtime.*`
- future helper-owned KV or summary bookkeeping

Follow-up:

- evaluate new DX proposals against this boundary before implementation

## 2026-04-28 — Harness Error And Return Contract

Status:

- accepted

Decision:

- Lua/harness-facing functions should raise on actual failure
- they should not raise for valid absence
- collection queries should return empty collections for valid empty results
- optional singular reads may return `nil`
- recovery should use `try(...)` or `pcall(...)`

Promoted contract:

- success: return the useful value directly
- failure: raise a human-readable error
- valid empty list/search result: return `{}` or equivalent empty collection
- valid missing singular value: return `nil`

Examples:

```lua
local hits = runtime.memory.search("compiler error", ctx, { limit = 5 })
if #hits > 0 then
  ...
end

local value = session.get("draft")
if value then
  ...
end

local spec = fs.summary("SPEC.md")
```

Rationale:

- Turin should avoid Go-like `value, err` ceremony in harness code
- a universal raise-on-failure contract gives the public Lua surface one voice
- valid absence and valid empty results still need distinct semantics

Affected surfaces:

- all harness-facing Lua bindings
- helper globals
- docs/examples

Follow-up:

- add `try(...)` as a thin DX wrapper over `pcall(...)`
- error strings should remain human-readable and source-named

## 2026-04-28 — Scope Vocabulary

Status:

- accepted

Decision:

- keep built-in scopes:
  - `session.*`
  - `user.*`
  - default agent-scoped helpers such as `remember(...)` / `recall(...)`
- add `scope(kind, key, opts?)` as the DX helper for custom scoping
- all scope proxies should share the same interface:
  - `remember`
  - `recall`
  - `set`
  - `get`
  - `del`
  - `incr`
- keep `runtime.context(...)` as the explicit substrate

Example:

```lua
local project = scope("project", "my-app")
project.remember("uses event sourcing")
project.set("version", "2.1")

local private = scope("agent", "coder", {
  namespace = "scratchpad",
  visibility = "private",
})
private.remember("working hypothesis about the bug")
```

Rationale:

- custom scoping is currently one of the biggest readability weak spots
- Turin should make domain-specific harnesses beautiful without inventing new substrate semantics
- `scope(kind, key, opts?)` supports arbitrary domains without forcing a builder grammar

Affected surfaces:

- `session.*`
- `user.*`
- default memory/KV helpers
- `runtime.context(...)`
- future scope proxy helpers

Follow-up:

- docs must be explicit about the default scope for each helper

## 2026-04-29 — Task And Execution Option Shapes

Status:

- accepted

Decision:

- default execution behavior should be sane enough that the common path needs no options
- `agent.sidestep(prompt)` should work cleanly with defaults
- `agent.sidestep(prompt, "mode")` may use a string shorthand for common modes
- full option tables remain available for custom combinations
- do not introduce global constants for execution modes
- helper-layer execution options should use `target`, not `context_target`

Example:

```lua
agent.sidestep("Analyze this")
agent.sidestep("Analyze this", "ephemeral")
agent.sidestep("Try approach B", "hidden")
agent.sidestep("Analyze this", { mode = "ephemeral", timeout_ms = 5000 })
```

Rationale:

- execution APIs need a no-options happy path
- string shorthand improves readability for common modes without removing full control
- `target` is the right helper-layer word; `context_target` remains the substrate term

Affected surfaces:

- `agent.sidestep(...)`
- helper task/execution option tables
- docs/examples

Follow-up:

- keep substrate execution selectors explicit underneath

## 2026-04-29 — Callable Proxies Are First-Class DX

Status:

- accepted
- implementation pending

Decision:

- callable proxies are first-class recommended DX where a handle-oriented pattern fits naturally
- promoted examples should lead with callable proxies such as:
  - `runtime.agent("reviewer")`
  - `runtime.db.with(sel, fn)`
- substrate forms remain documented as explicit/advanced
- proxies must not hide governance or capability requirements
- rename peer-agent helper `complete` to `ask`
- substrate equivalent should become `runtime.agent.ask(...)`

Rationale:

- proxy-style authoring reads much closer to prose
- this is the right style for common orchestration and DB-handle flows
- `ask` reads more naturally than `complete` for prompt/response interaction

Affected surfaces:

- `runtime.agent(...)`
- `runtime.agent.complete(...)`
- `runtime.db.with(...)`
- docs/examples

Follow-up:

- new namespaces may adopt callable proxy forms when they fit the domain naturally

## 2026-04-29 — Graph Namespace Cleanup

Status:

- accepted
- implementation pending

Decision:

- canonical graph substrate should move from flat underscore verbs to grouped sub-namespaces:
  - `runtime.graph.node.*`
  - `runtime.graph.edge.*`
  - `runtime.graph.path.*`
- initial direction:
  - `runtime.graph.node.create(...)`
  - `runtime.graph.node.list(...)`
  - `runtime.graph.edge.create(...)`
  - `runtime.graph.edge.list(...)`
  - `runtime.graph.path.select(...)`
- keep `selected_path` as the returned read-target shape

Rationale:

- graph surface is likely to grow with more node/edge/path operations
- grouped sub-namespaces scale better than `node_create`, `edge_create`, and similar flat verbs
- this matches the general shape already used elsewhere, such as `runtime.code.search.*`
- `selected_path` remains precise as a materialized execution read-target even if the operation becomes `path.select(...)`

Affected surfaces:

- `runtime.graph.*`
- docs/examples
- helper implementations that call graph substrate functions

Follow-up:

- keep this decision separate from graph builder helpers
- helper layer may use shorter prose like `path` in naming, while the substrate keeps `selected_path` precise

## 2026-04-29 — Graph Builder Helpers

Status:

- accepted
- implementation pending

Decision:

- add helper-layer graph builders on top of the graph substrate
- initial direction:
  - `graph.new(kind, label?)`
  - `graph.node(id)`
  - `graph.branch(id)`
  - `graph.turn(id)`
- graph node proxies may expose fluent methods such as:
  - `:add(target, opts?)`
  - `:link(target, relation, opts?)`
  - `:newest(role?)`
  - `:oldest(role?)`
  - `:all(role?)`
  - `:find(opts?)`

Rationale:

- graph/path substrate is correct but too storage-shaped for pleasant authoring
- helper builders should make graph workflows beautiful before public adoption, not after

Affected surfaces:

- helper graph/path authoring APIs
- harness docs/examples

Follow-up:

- evaluate the first helper set by rewriting example/library harnesses with it
- revise aggressively if the resulting authoring language is not good enough

## 2026-04-29 — Namespace Philosophy

Status:

- accepted

Decision:

- verbs for direct actions
- nouns for derived projections and constructors
- predicate helpers should read as adjective/verb phrases

Rule:

- if the function does the thing, use a verb
- if the returned value is the thing, use a noun

Examples:

- direct actions:
  - `fs.read`
  - `fs.write`
  - `fs.stat`
  - `session.set`
  - `session.get`
- derived projections:
  - `fs.summary`
- constructors:
  - `graph.node(id)`
  - `graph.branch(id)`
- predicates:
  - `allowed("cap")`
  - `needs("cap")`

Rationale:

- this naming rule gives the helper layer a more coherent voice
- it explains why `fs.summary(...)` is better than `fs.summarize(...)` for the intended usage

Affected surfaces:

- helper naming
- docs/examples

## 2026-04-29 — Pre-Launch Removal Discipline

Status:

- accepted

Decision:

- before external adoption, replaced surfaces should be removed entirely rather than aliased
- no soft-deprecation periods
- no compatibility shims
- rewrite internal harnesses/examples immediately after API replacement

Rules for the current phase:

- remove, do not alias
- rewrite, do not patch around old names
- record the change in this document
- run tests immediately after removal and fix breakage in the same pass

Rationale:

- Turin currently has no user-migration burden
- carrying temporary compatibility surface during this phase creates needless bloat

Affected surfaces:

- public harness API
- examples
- harness library
- docs

Follow-up:

- replace this rule with a formal deprecation/migration policy once Turin has external users

## 2026-04-29 — Reference Hierarchy

Status:

- accepted
- implementation pending

Decision:

- documentation should lead with the promoted DX/helper style
- substrate reference should remain fully documented in a separate advanced/explicit section
- a harness author using the common path should not need `runtime.*` to get started
- `api-surface.md` should remain the full inventory across both layers

Rationale:

- Turin’s public story is harness authoring quality
- docs should reflect the recommended voice without hiding the substrate from advanced users or contributors

Affected surfaces:

- docs structure
- guide prose
- reference prose

Follow-up:

- consider a dedicated helper-layer reference such as `dx-helpers.md`

## 2026-04-29 — Example Style Policy

Status:

- accepted

Decision:

- guide examples should default to the promoted DX voice
- substrate examples belong primarily in substrate reference docs
- if a guide needs to show substrate for explanation, it must be explicitly labeled
- avoid mixed-style code blocks that interleave helper and substrate calls without explanation
- scaffold templates should use the promoted DX style

Rationale:

- examples teach style whether we intend them to or not
- mixed-style examples make the surface feel accidental rather than designed

Affected surfaces:

- guides
- scaffolds
- examples
- reference docs

Follow-up:

- when both helper and substrate forms need to be shown, use separate labeled blocks
