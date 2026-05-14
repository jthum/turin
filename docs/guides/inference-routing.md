# Inference Routing And Compaction

Turin supports named inference profiles under `[inference.contexts.<name>]`.
Harness code can select them per turn with `turn.inference = "<name>"`.

Example:

```toml
[inference.contexts.default]
provider = "anthropic"
model = "claude-sonnet-4-20250514"

[inference.contexts.fast]
provider = "openai"
model = "gpt-4o-mini"
temperature = 0.2

[inference.contexts.reasoning]
provider = "anthropic"
model = "claude-opus-4-6"
thinking_budget = 4096
fallback = "default"
```

```lua
function on_turn_prepare(turn)
    if turn.token_count > 80000 then
        turn.inference = "fast"
    end
    return ALLOW
end
```

For more maintainable routing, wrap that logic in a helper:

```lua
local function select_inference_context(turn)
  local prompt = string.lower(turn.prompt or "")

  if turn.token_limit > 0 and (turn.token_count / turn.token_limit) > 0.85 then
    return "fast"
  end

  if runtime.inference.available("reasoning")
    and (prompt:find("debug", 1, true) or prompt:find("trace", 1, true))
  then
    return "reasoning"
  end

  return "default"
end

function on_turn_prepare(turn)
  local route = select_inference_context(turn)
  if route ~= "default" then
    turn.inference = route
  end
  return ALLOW
end
```

`runtime.inference.available(name)` returns `true` when the named inference context is configured
for the current agent and `false` otherwise.

Resolution order for a turn is:

- harness-selected `turn.inference`
- configured fallback chain for that named profile
- configured default inference profile
- base `[agent]` provider/model

Unknown profile names do not fail the turn. Turin warns and falls back.

## Context Window Management

Providers can declare an estimated context window:

```toml
[providers.anthropic]
type = "anthropic"
context_window_tokens = 200000
```

Turin uses that estimate to:

- expose `turn.token_count` and `turn.token_limit` during `on_turn_prepare`
- reserve output headroom
- compact older history when a turn would exceed the provider budget

Current token estimation is heuristic, not tokenizer-exact.

## Compaction Policy

Compaction behavior is configured under `[inference.compaction]`:

```toml
[inference.compaction]
mode = "hybrid"       # hybrid | trim_only | summary_only
inference = "fast"    # optional named inference profile for summary generation
trigger_ratio = 0.85  # trigger semantic compaction before the window is fully exhausted
```

`mode`:

- `hybrid`: generate semantic checkpoints first, then structurally trim if needed
- `trim_only`: skip semantic summarisation and only trim older history/tool results
- `summary_only`: use semantic checkpoints but do not structurally trim after that

`inference`:

- selects a named profile from `inference.contexts`
- it is not a raw model name
- use it when you want summarisation to run on a cheaper or faster route than the main turn

`trigger_ratio`:

- `1.0` means wait until the request budget is fully hit before semantic compaction
- lower values trigger checkpoint generation earlier

## Checkpoints

Semantic compaction creates a durable checkpoint summary of older history.
Turin then:

- injects that summary into the outbound system prompt
- drops the covered older raw messages from the request
- preserves newer raw messages directly

Checkpoints are persisted and restored on session resume.

They do not rewrite the underlying transcript history.

## Choosing A Mode

Use `hybrid` when:

- you want long-running continuity
- you still want deterministic trimming as a safety net

Use `trim_only` when:

- exactness matters more than semantic continuity
- you want to avoid extra summarisation cost/latency

Use `summary_only` when:

- preserving older meaning matters more than exact transcript retention
- you accept the risk that the checkpointed request may still be too large for some turns
