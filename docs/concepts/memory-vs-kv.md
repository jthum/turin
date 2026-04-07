# Memory vs KV

This document defines the recommended convention for choosing between Turin's scoped memory store and scoped KV store.

Short version:

- use `kv` when you know exactly what key you want
- use `memory` when you need to find something relevant later

## Mental Model

`kv` is for durable state.

It behaves like variables or records addressed by a known key:

- `"current_task"`
- `"last_seen_message_id"`
- `"deploy:staging/status"`
- `"user:42/timezone"`

`memory` is for durable knowledge.

It behaves like notes or facts that should be found again by lexical or semantic search:

- "User prefers concise answers"
- "Decision: keep self-messages ignored for WhatsApp personal mode"
- "This repository uses channel sidecars and branch-aware persistence"

## Use KV When

Use `kv` for:

- workflow state
- cursors and offsets
- flags and switches
- counters
- exact mappings
- user/session preferences
- dynamically updated configuration
- canonical documents addressed by stable IDs

Examples:

- `kv.get("status")`
- `kv.set("onboarding_complete", "true")`
- `kv.set("projects/turin/failed_builds", 47)`
- `kv.set("timezone", "+0530")`
- `kv.set("review:pr-123/approved", "yes")`
- `kv.set("adrs/0007", "<full ADR text>")`

## Use Memory When

Use `memory` for:

- learned facts from conversation
- observations and lessons
- summaries and notes
- information you may later search by meaning or keywords
- content where multiple related records may coexist naturally

Examples:

- `memory.search("what do we know about the user's preferences")`
- `memory.store("User prefers code review findings before summaries")`
- `memory.store("Decision: use a private WhatsApp group for personal accounts")`

## Practical Test

If the read path is:

- `kv.get("exact_key")`

then it is usually KV.

If the read path is:

- `memory.search("something about X")`

then it is usually memory.

The main failure mode to avoid is using KV as a poor-man's memory store by inventing keys and hoping you remember them later. If retrieval depends on search, store it as memory.

## Current Turin Storage Shape

Today:

- memories already support JSON metadata
- memory store requests also support tags, which are stored in metadata
- KV is a plain exact key/value store with no separate metadata or tags column

That means:

- use memory metadata and tags for memory-side organization
- use key prefixes or namespaces for KV-side organization

Examples:

- memory metadata: `{ kind = "adr", project = "turin" }`
- memory tags: `{ "adr", "release", "architecture" }`
- KV keys: `adrs/0007`, `users/42/timezone`, `deploy/staging/status`

## ADRs, Decisions, and Notes

For ADR-like content, there are three reasonable patterns.

1. Canonical document by ID:
   - use KV
   - example: `adrs/0007`

2. Searchable knowledge or summary:
   - use memory
   - example: a short memory row summarizing the decision and why it matters

3. Hybrid:
   - store the canonical full text in KV
   - store a searchable summary in memory

The hybrid pattern is often the most useful one.

## Recommended Convention

- choose KV for operational state
- choose memory for searchable knowledge
- use prefixes like `adrs/...` for KV grouping
- use metadata and tags for memory grouping
- only add new schema surface when prefixes and metadata stop being enough
