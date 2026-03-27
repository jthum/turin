# Virtual Tool Patterns

This block documents the core harness-side virtual tool patterns that Turin now supports.

Use it when you want domain-specific tool names without expanding the native Rust tool surface.

## What It Covers

- simple virtual wrappers over native tools
- multi-call virtual tools
- result callbacks
- virtual-to-virtual composition
- callback follow-up plans
- recursion and depth guards

## Pattern 1: Simple Wrapper

```lua
tool.declare("read_note", {
  description = "Read a note from disk",
  params = {
    path = { type = "string", required = true }
  },
  handler = function(args)
    return tool.call("read_file", { path = args.path })
  end
})
```

## Pattern 2: Multi-Call Sequence

```lua
tool.declare("read_pair", {
  description = "Read two files in order",
  params = {
    first = { type = "string", required = true },
    second = { type = "string", required = true }
  },
  handler = function(args)
    return tool.sequence({
      tool.call("read_file", { path = args.first }),
      tool.call("read_file", { path = args.second })
    })
  end
})
```

## Pattern 3: Result Callback

```lua
tool.declare("summarize_pair", {
  description = "Read two files and summarize them",
  params = {
    first = { type = "string", required = true },
    second = { type = "string", required = true }
  },
  handler = function(args)
    return tool.sequence({
      tool.call("read_file", { path = args.first }),
      tool.call("read_file", { path = args.second })
    }, function(results)
      return {
        content = "Combined: " .. results[1].content .. " | " .. results[2].content,
        is_error = results[1].is_error or results[2].is_error
      }
    end)
  end
})
```

## Pattern 4: Virtual-to-Virtual Composition

Declaration order does not matter. Virtual tool names are resolved after harness load completes.

```lua
tool.declare("read_note_wrapped", {
  description = "Read a note through a later-declared virtual tool",
  params = {
    path = { type = "string", required = true }
  },
  handler = function(args)
    return tool.call("read_note", { path = args.path }, function(result)
      return "wrapped later: " .. result.content
    end)
  end
})

tool.declare("read_note", {
  description = "Read a note from disk",
  params = {
    path = { type = "string", required = true }
  },
  handler = function(args)
    return tool.call("read_file", { path = args.path })
  end
})
```

## Pattern 5: Callback Follow-Up Plan

Callbacks can return another tool plan instead of only final content.

```lua
tool.declare("resolve_pointer", {
  description = "Resolve a pointer file and read the final note",
  params = {
    pointer = { type = "string", required = true }
  },
  handler = function(args)
    return tool.call("read_file", { path = args.pointer }, function(result)
      return tool.call("read_note", { path = result.content })
    end)
  end
})
```

## Guardrails

- `tool.declare(...)` is load-time only
- callbacks run after nested execution; handlers do not await inline
- `on_tool_call` / `on_tool_result` still govern both the outer virtual tool and nested calls
- recursive virtual-tool chains are rejected
- max virtual nesting depth is `8`
