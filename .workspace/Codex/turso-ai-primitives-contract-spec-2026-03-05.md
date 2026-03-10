# Turin Spec: Memory v2, Content Cache, and Code Search Primitives

Date: 2026-03-05
Status: Proposed (locked enough to implement)

## 1) Product Contract Rules

1. This is a breaking v2 contract.
2. No backward-compatibility guarantees are required for pre-v2 memory/cache/code-search behavior.
3. No migration, shim, or legacy fallback requirements are assumed by this spec.
4. Simple should be easy; difficult should be possible.
5. Canonical `runtime.*` APIs define behavior. DX wrappers must stay thin and unsurprising.
6. Kernel exposes facts and overrideable defaults; harness code chooses policy.
7. Public IDs are opaque UUIDv7 strings in compact/simple form with no type prefix. Internal row IDs are never part of the API.
8. Where practical, public IDs should be stored as 16-byte UUID blobs internally and formatted as strings only at the API boundary.

## 2) Canonical API Conventions

All new primitives follow current Turin tuple style:
1. Success: `(value, nil)` or `(true, nil)`
2. Failure: `(nil, err)` for value-returning calls; `(false, err)` for bool-returning calls

Common option keys used across namespaces:
1. `limit` (number)
2. `strict` (bool, default `false`)
3. `trace` (bool, default `false`, returns debug metadata where relevant)

Defaults policy:
1. Common-path calls should work without verbose option tables.
2. Option tables exist for override power, not because the default path is incomplete.
3. Wrappers may shorten common paths, but must not hide important behavior.

Error strings are human-readable, deterministic, and prefixed by namespace:
1. `runtime.memory.*:`
2. `runtime.cache.*:`
3. `runtime.code.search.*:`

## 3) Governance Capability Names

## 3.1 Memory v2

1. `runtime.memory.search`
2. `runtime.memory.store`
3. `runtime.memory.feedback`
4. `runtime.memory.correct`
5. `runtime.memory.purge`

## 3.2 Content Cache

1. `runtime.cache.invalidate`
2. `runtime.cache.stats`
3. `runtime.cache.reset`
4. `runtime.cache.read` does not introduce a second capability name; it inherits `fs.read` path and capability enforcement.

## 3.3 Code Search (Turin runtime)

1. `runtime.code.search.lexical`
2. `runtime.code.search.semantic`
3. `runtime.code.search.hybrid`
4. `runtime.code.search.status`

## 4) Memory v2 Primitive Contracts

## 4.1 `runtime.memory.search(query, ctx, opts?) -> rows|nil, err?`

`opts`:
```lua
{
  limit = 5,
  mode = "auto" | "lexical" | "semantic" | "hybrid", -- default "auto"
  min_score = 0.0, -- default 0
  include_metadata = false,
  include_superseded = false,
  trace = false,
  strict = false -- strict only affects requested semantic/hybrid capability availability
}
```

Row shape:
```lua
{
  id = "<uuidv7-simple>",
  content = "...",
  score = 0.0,
  lexical_score = 0.0 | nil,
  semantic_score = 0.0 | nil,
  weight = 1.0,
  retrieval_count = 0,
  last_retrieved_at = "2026-03-05T...Z" | nil,
  metadata = { ... } | nil
}
```

Rules:
1. `mode="auto"` chooses the best available mode in this order:
   - hybrid
   - semantic
   - lexical
2. `mode="semantic"` with no embedding provider:
   - `strict=false`: fall back to lexical if possible, else `(nil, err)`
   - `strict=true`: `(nil, "runtime.memory.search: semantic mode requires an embedding provider")`
3. `mode="hybrid"` with no embedding provider:
   - `strict=false`: degrade to lexical
   - `strict=true`: error
4. Superseded memories are excluded unless `include_superseded=true`.
5. Successful retrieval updates `retrieval_count` and `last_retrieved_at` for returned rows.

## 4.2 `runtime.memory.store(content, ctx, metadata?, opts?) -> memory|nil, err?`

`opts`:
```lua
{
  source_task = "<task-id>" | nil,
  tags = { "fact", "policy", "decision" } | nil,
  storage = "auto" | "lexical_only" | "embedded", -- default "auto"
  trace = false
}
```

Return:
```lua
{
  id = "<uuidv7-simple>",
  stored_at = "2026-03-05T...Z",
  storage = "lexical_only" | "embedded"
}
```

Rules:
1. `storage="auto"` embeds when an embedding provider is available, otherwise stores lexical-only.
2. `storage="lexical_only"` stores without embedding even if an embedding provider is available.
3. `storage="embedded"` errors when embeddings are unavailable.
4. If `opts.tags` is provided, tags are stored alongside metadata and surfaced back through normal metadata retrieval.

## 4.3 `runtime.memory.feedback(memory_id, signal, ctx, opts?) -> state|nil, err?`

`signal`:
1. `"up"`
2. `"down"`
3. numeric delta (for example `-0.25`, `0.10`)

`opts`:
```lua
{
  reason = "string" | nil,
  task_id = "<task-id>" | nil,
  step = 0.1, -- used only for "up"/"down"
  clamp = { min = 0.1, max = 5.0 },
  trace = false
}
```

Return:
```lua
{
  id = "<uuidv7-simple>",
  weight = 1.15,
  updated_at = "2026-03-05T...Z"
}
```

Rules:
1. `"up"` applies `+step`.
2. `"down"` applies `-step`.
3. Numeric signals apply the provided delta directly before clamp.

## 4.4 `runtime.memory.correct(memory_id, content, ctx, metadata?, opts?) -> correction|nil, err?`

`opts`:
```lua
{
  source_task = "<task-id>" | nil,
  tags = { "fact", "policy", "decision" } | nil,
  storage = "auto" | "lexical_only" | "embedded", -- default "auto"
  trace = false
}
```

Semantics:
1. The old memory is marked superseded.
2. The corrected memory is stored as a new record.
3. An old-to-new correction link is recorded.
4. The old memory remains queryable only when `include_superseded=true`.

Return:
```lua
{
  superseded_id = "<uuidv7-simple>",
  replacement_id = "<uuidv7-simple>",
  corrected_at = "2026-03-05T...Z"
}
```

## 4.5 `runtime.memory.purge(ctx, opts?) -> report|nil, err?`

`opts`:
```lua
{
  older_than_days = 30 | nil,
  min_weight = 0.2 | nil,
  max_retrieval_count = 0 | nil,
  only_superseded = false,
  all = false,
  dry_run = true,
  trace = false
}
```

Return:
```lua
{
  scanned = 1200,
  matched = 90,
  purged = 0,   -- 0 in dry_run
  dry_run = true
}
```

Rules:
1. Provided filters combine with logical AND.
2. `all=true` means match all memories in the selected context.
3. At least one filter must be provided unless `all=true`.
4. `dry_run` defaults to `true`.
5. Purge is the only API that physically deletes memory records.

## 5) Content Cache Primitive Contracts

## 5.1 `runtime.cache.read(path, opts?) -> result|nil, err?`

`opts`:
```lua
{
  session_id = "<session-public-id>" | nil, -- default active session public_id
  include_content = true | nil, -- true forces content even when unchanged; omitted/false uses default behavior
  include_previous = false,
  max_diff_lines = 200,
  token_estimate = true,
  trace = false
}
```

Return:
```lua
{
  status = "fresh" | "unchanged" | "changed",
  path = "...",
  hash = "...",
  previous_hash = "..." | nil,
  content = "..." | nil,
  previous_content = "..." | nil,
  diff = "..." | nil,
  diff_truncated = false,
  estimated_tokens_saved = 0,
  read_at = "2026-03-05T...Z"
}
```

Rules:
1. `runtime.cache.read` uses the same safe-path resolution, `harness.fs_root`, and capability enforcement as `fs.read`.
2. Session key defaults to the active Turin session public ID.
3. `opts.session_id` must refer to a valid existing session public ID.
4. `fresh` means first read for the session and includes content by default.
5. `unchanged` means the file hash matches the session's previous read and omits full content by default.
6. `changed` means the file hash differs and includes current content by default.
7. `include_content=true` forces full content even when `status="unchanged"`.
8. `include_previous=true` includes `previous_content` when a prior version exists.
9. `diff` is a unified diff string and is truncated to `max_diff_lines`; when truncated, `diff_truncated=true`.
10. `max_diff_lines=0` disables diff generation.

## 5.2 `runtime.cache.invalidate(path, opts?) -> bool, err?`

`opts`:
```lua
{
  scope = "session" | "global", -- default "session"
  session_id = "<session-public-id>" | nil,
  trace = false
}
```

Rules:
1. `scope="session"` clears the target session's read pointer for the path.
2. `scope="global"` clears all session read pointers for the path and removes cached versions for that path.

## 5.3 `runtime.cache.stats(opts?) -> stats|nil, err?`

`opts`:
```lua
{
  scope = "session" | "global" | "both", -- default "both"
  session_id = "<session-public-id>" | nil,
  trace = false
}
```

Return:
```lua
{
  global = {
    cached_files = 123,
    cached_versions = 432,
    tokens_saved = 50123
  },
  session = {
    id = "<session-public-id>",
    files_seen = 27,
    tokens_saved = 920
  }
}
```

## 5.4 `runtime.cache.reset(opts?) -> report|nil, err?`

`opts`:
```lua
{
  scope = "session" | "global",
  session_id = "<session-public-id>" | nil,
  dry_run = true,
  trace = false
}
```

Return:
```lua
{
  scope = "session",
  removed_versions = 0,
  removed_reads = 15,
  reset_stats = true,
  dry_run = true
}
```

Rules:
1. `scope="session"` resets read pointers and counters for the target session only.
2. `scope="global"` resets all read pointers and counters and may remove all cached versions.
3. `dry_run` defaults to `true`.

## 6) Code Search Contracts (Turin Read Path)

Turin does not own code-index writes in this contract.
`turin-map` owns write/index lifecycle and emits index DBs Turin can discover and read.

## 6.1 Codebase Selector

Allowed selector shapes:
1. root path string:
   - `"/abs/path/repo"`
   - `"relative/repo"`
2. selector table:
```lua
{
  root = "/abs/or/relative/path",
  index_path = "/abs/or/relative/path/.turin/codebase.db" | nil
}
```

Resolution rules:
1. Relative roots resolve against the Turin workspace root.
2. The root path is canonicalized before discovery.
3. Default index path is `<root>/.turin/codebase.db`.

## 6.2 Index Discovery and Validation

Turin discovery sequence:
1. Normalize the codebase selector to `root` and `index_path`.
2. Open the discovered index DB.
3. Validate `index_meta`.
4. Validate expected read views.
5. Negotiate features (`lexical`, `semantic`, `hybrid`).

Required `index_meta` fields:
```lua
{
  schema_revision = 20260305,
  root_path = "/abs/path/repo",
  updated_at = "2026-03-05T...Z",
  capabilities = {
    lexical = true,
    semantic = true,
    hybrid = true,
    languages = { "ts", "js", "python", "php", "go", "rust", "lua" }
  }
}
```

Optional `index_meta` metadata:
```lua
{
  codebase_id = "repo_main" | nil
}
```

Rules:
1. `root_path` is the canonical identity field for discovery and validation.
2. `codebase_id`, when present, is descriptive metadata only and must not drive discovery or validation behavior.

Required stable read views:
1. `v_code_lexical`
2. `v_code_semantic` (when `semantic=true`)
3. `v_code_hybrid` (when `hybrid=true`)

Contract validation failures are always errors. Turin does not fall back to subprocess search.

## 6.3 `runtime.code.search.status(codebase, opts?) -> status|nil, err?`

Return:
```lua
{
  root = "/abs/path/repo",
  index_path = "/abs/path/repo/.turin/codebase.db",
  schema_revision = 20260305,
  updated_at = "2026-03-05T...Z",
  index_age_seconds = 12,
  codebase_id = "repo" | nil,
  capabilities = {
    lexical = true,
    semantic = true,
    hybrid = true,
    languages = { "ts", "js", "python", "php", "go", "rust", "lua" }
  },
  semantic = {
    embedded_chunks = 128,
    embedding_key = "openai:https://api.openai.com/v1:text-embedding-3-small:1536" | nil,
    embedding_dimensions = 1536 | nil,
    vector_format = "float8" | "float32" | nil
  }
}
```

`opts`:
```lua
{
  trace = false
}
```

`index_age_seconds` is computed from `updated_at` and returned as a fact for harness code to interpret.

## 6.4 `runtime.code.search.lexical(codebase, query, opts?) -> rows|nil, err?`

## 6.5 `runtime.code.search.semantic(codebase, query, opts?) -> rows|nil, err?`

## 6.6 `runtime.code.search.hybrid(codebase, query, opts?) -> rows|nil, err?`

Shared `opts`:
```lua
{
  limit = 10,
  languages = { "ts", "js", "python", "php", "go", "rust", "lua" } | nil,
  kinds = { "function", "type", "constant" } | nil,
  min_score = 0.0,
  trace = false,
  strict = false -- strict affects capability/provider fallback only; contract validation failures always error
}
```

Row shape:
```lua
{
  chunk_key = "...",
  path = "src/kernel/governance.rs",
  language = "rust",
  kind = "function",
  name = "capability_decision",
  signature = "fn capability_decision(...)",
  snippet = "...",
  start_line = 101,
  end_line = 132,
  score = 0.0,
  lexical_score = 0.0 | nil,
  semantic_score = 0.0 | nil,
  rank = 1,
  trace = {
    requested_mode = "lexical" | "semantic" | "hybrid" | nil,
    effective_mode = "lexical" | "semantic" | "hybrid",
    fallback_reason = "capability_fallback" | "missing_embedding_provider" | "embedding_profile_mismatch" | nil,
    lexical_rank = 1 | nil,
    semantic_rank = 1 | nil,
    lexical_rrf = 0.0 | nil,
    semantic_rrf = 0.0 | nil,
    fusion = "rrf" | nil
  } | nil
}
```

Capability behavior:
1. Semantic requested and semantic capability absent:
   - `strict=false`: fall back to lexical if available
   - `strict=true`: `(nil, "runtime.code.search.semantic: semantic capability not available for root '...'" )`
2. Semantic requested and no embedding provider is available at query time:
   - `strict=false`: fall back to lexical if available
   - `strict=true`: `(nil, "runtime.code.search.semantic: semantic search requires an embedding provider")`
3. Hybrid requested and hybrid capability absent:
   - `strict=false`: fall back to the best available mode
   - `strict=true`: error
4. Hybrid requested and no embedding provider is available at query time:
   - `strict=false`: fall back to lexical if available
   - `strict=true`: `(nil, "runtime.code.search.hybrid: hybrid search requires an embedding provider")`

## 7) `turin-map` External Contract (Write Path)

`turin-map` is a separate binary and may include heavy parser or grammar dependencies.

Expected commands (initial):
1. `turin-map index --root <path>`
2. `turin-map remove --root <path> --path <file>`
3. `turin-map rebuild --root <path>`
4. `turin-map status --root <path> --json`

Optional path override:
1. Commands may accept `--index-path <path>` to override the default DB location.

Status JSON should include:
1. `root`
2. `index_path`
3. `schema_revision`
4. `updated_at`
5. `codebase_id`
6. `capabilities`
7. `semantic`

## 8) Delightful DX Layer (Release Priority)

Goal: canonical APIs stay explicit, while common harness code reads like intent.

Design rules:
1. Thin wrappers ship with the release once canonical contracts are stable.
2. Wrappers must make the common path shorter, not more magical.
3. No hidden side effects.
4. No governance or capability bypass.
5. Wrappers never remove access to canonical low-level options.

Initial DX targets:
1. `memory.search(query, opts?)`
2. `memory.store(content, metadata?, opts?)`
3. `remember(content, metadata?, opts?)`
4. `recall(query, opts?)`
5. `cache.file(path, opts?)`
6. `code.find(query, opts?)`

Wrapper defaults:
1. `memory.search(...)` and `memory.store(...)` use the active agent/session context helper.
2. `remember(...)` and `recall(...)` are thin intent wrappers over the same default context.
3. `cache.file(path, opts?)` delegates to `runtime.cache.read(path, opts?)`.
4. `code.find(query, opts?)` defaults to the Turin workspace root and allows explicit root or index-path override through `opts`.

Common-path examples:
```lua
local mem = remember("User prefers terse reports")
local hits = recall("previous regression in grants")
local file = cache.file("SPEC.md")
local rows = code.find("grant validation path")
```

Advanced override examples:
```lua
local mem, err = runtime.memory.store(
  "Persist this as lexical only",
  ctx,
  { kind = "note" },
  { storage = "lexical_only", tags = { "policy" } }
)

local rows, err = runtime.code.search.hybrid(
  { root = ".", index_path = ".turin/custom.db" },
  "where is grant validation enforced",
  { languages = { "rust" }, strict = true }
)
```

## 9) Backward-Compatibility Stance

1. Pre-v2 contracts may break freely.
2. No compatibility aliases are required for old memory semantics.
3. Existing DBs may be discarded and rebuilt.
4. Old docs/examples should be updated rather than supported in code.

## 10) Locked Decisions

1. `runtime.memory.purge` defaults `dry_run=true`.
2. `runtime.cache.read` returns full content on unchanged reads when `include_content=true`.
3. Turin does not fall back to `turin-map search --json`.
4. `runtime.memory.feedback` allows numeric direct deltas.
5. Memory public IDs use compact UUIDv7 strings with no `mem_` prefix.
