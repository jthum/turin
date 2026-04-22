# State Store Schema

This document summarizes Turin's current state-store schema at a conceptual level.

The authoritative schema lives in `src/persistence/schema.rs`. This reference exists so contributors can reason about the durable model without reading the full SQL string every time.

Current schema version: `13`

Turin currently does not provide an in-place migration path for incompatible state DB versions. Existing DBs with an older `schema_info.version` are rejected and must be recreated.

## Design Shape

The state store is built around a few durable primitives:

- `sessions` are durable containers.
- `turns` are the structural ancestry graph.
- `branch_heads` are named writable path handles into the turn graph.
- `turn_messages` and `turn_tool_executions` hang off concrete turns.
- `graph_nodes` and `graph_edges` are a sparse semantic overlay for opt-in relationships.
- `events` record runtime/session activity.
- `kv`, `memories`, and file-cache tables provide scoped supporting state.

The important split is:

- Structural facts live in turn/branch tables.
- Branch birth context lives in branch provenance columns.
- Optional semantic relationships live in the sparse graph overlay.

Normal serial execution should not create graph overlay rows unless a harness/app/runtime explicitly records a meaningful relationship.

## Core Routing

### `sessions`

Durable session container.

Key fields:

- `id`: internal integer primary key
- `public_id`: public UUID bytes
- `agent_id`: owning/default agent id
- `metadata`: optional JSON-ish text payload
- `active_branch_head_id`: current persisted active branch head
- `created_at`

The active branch is a convenience/default pointer. Execution-scoped semantics should still use explicit context/write targets where possible.

## Turn Graph

### `turns`

Structural ancestry graph.

Key fields:

- `id`
- `public_id`
- `session_id`
- `parent_turn_id`
- `branch_depth`
- `created_at`

Each turn has at most one structural parent. Cross-path meaning should be represented with the sparse graph overlay, not by changing turn ancestry.

### `branch_heads`

Named writable handles into the turn graph.

Key fields:

- `id`
- `public_id`
- `session_id`
- `name`
- `head_turn_id`
- `created_from_turn_id`
- `origin_kind`
- `origin_task_id`
- `origin_execution_id`
- `origin_metadata`
- `created_at`

`created_from_turn_id` records where the branch forked. `origin_*` records why/how the branch was born, for example `main`, `manual`, `sidestep`, `promotion`, or `conflict_fork`.

### `turn_messages`

Messages attached to a concrete turn.

Key fields:

- `turn_id`
- `role`
- `content`
- `token_count`
- `created_at`

### `turn_tool_executions`

Tool execution records attached to a concrete turn.

Key fields:

- `turn_id`
- `tool_call_id`
- `tool_name`
- `args`
- `output`
- `is_error`
- `duration_ms`
- `verdict`
- `created_at`

## Sparse Semantic Graph Overlay

### `graph_nodes`

Opt-in semantic nodes for concepts such as experiments, branch groups, comparison sets, knowledge paths, or external references.

Key fields:

- `id`
- `public_id`
- `session_id`
- `kind`
- `label`
- `origin_task_id`
- `origin_execution_id`
- `metadata`
- `created_at`

Examples:

- `kind = "experiment"`
- `kind = "branch_group"`
- `kind = "knowledge_path"`
- `kind = "external_reference"`

### `graph_edges`

Opt-in semantic relationships between graph nodes, turns, branch heads, external references, or other addressable things.

Key fields:

- `id`
- `public_id`
- `session_id`
- `source_kind`
- `source_id`
- `target_kind`
- `target_id`
- `relation_kind`
- `source_role`
- `target_role`
- `origin_task_id`
- `origin_execution_id`
- `metadata`
- `created_at`

Examples:

- `experiment -> branch_head`, `relation_kind = "contains"`, `target_role = "candidate"`
- `branch_head -> branch_head`, `relation_kind = "alternative_to"`
- `turn -> turn`, `relation_kind = "convergent"`
- `branch_head -> external_path`, `relation_kind = "mounted_from"`

The graph overlay intentionally does not enforce foreign keys for `source_*` and `target_*`. This preserves cross-type and future external references. Validation belongs in the runtime/harness/app layer.

## Event Log

### `events`

Append-style runtime/session event log.

Key fields:

- `session_id`
- `turn_id`
- `event_type`
- `payload`
- `created_at`

Events may be attached to a turn or session-wide.

## Compatibility Mirror Tables

### `messages`

Message mirror keyed by `session_id` and `turn_index`.

Branch-native reads use `turn_messages`. This table is still written by the current persistence path for compatibility with older query surfaces and examples. It is a removal candidate once those surfaces are ported.

### `tool_executions`

Tool-execution mirror keyed by `session_id` and `turn_index`.

Branch-native reads use `turn_tool_executions`. This table is still written by the current persistence path for compatibility with older query surfaces and examples. It is a removal candidate once those surfaces are ported.

## Scoped State

### `kv`

Scoped key-value store.

Primary key:

- `scope_kind`
- `scope_key`
- `key`

Fields:

- `value`
- `expires_at`
- `updated_at`

## Memory

### `memories`

Durable scoped memory rows with optional embedding data.

Key fields:

- `public_id`
- `scope_kind`
- `scope_key`
- `content`
- `embedding`
- `embedding_key`
- `embedding_dimensions`
- `metadata`
- `weight`
- `retrieval_count`
- `last_retrieved_at`
- `superseded_at`
- `superseded_by_memory_id`
- `created_at`

### `memory_feedback_events`

Feedback deltas for memories.

Key fields:

- `memory_id`
- `delta`
- `reason`
- `task_id`
- `created_at`

## File Cache

### `file_cache_versions`

Stores content-addressed file snapshots used by cache helpers.

Key fields:

- `path`
- `content_hash`
- `content`
- `content_bytes`
- `created_at`

### `file_cache_reads`

Tracks per-session file cache reads and token savings.

Primary key:

- `session_id`
- `path`

Fields:

- `content_hash`
- `tokens_saved`
- `last_read_at`

## Schema Metadata

### `schema_info`

Stores schema metadata such as:

- `version`

The runtime compares this value against the compiled `SCHEMA_VERSION`.

## Indexes

The schema includes indexes for common lookups:

- events by session/turn
- messages by session
- turns by session, parent, and depth
- branch heads by session
- turn messages/tool executions by turn
- graph nodes by session/kind
- graph edges by session/relation/source/target
- memories by scope and embedding profile
- file cache by path

The memory table also has a Turso FTS index on `memories.content`.
