# Reference-Aware Objects

This example shows the new reference-aware runtime object DX:

- passing runtime proxies through payload boundaries
- using `ref(...)` for identity-only transport
- attaching methods to matching proxies with `action.define_on(...)`

Use this when you want harness code to read more like domain flow than request /
reload plumbing.

## Files

- `main.lua`

## What It Does

- creates a project scope proxy and a project worklist item
- emits both direct proxy payloads and identity-only refs
- reacts to the signal with hydrated proxies on the receiving side
- attaches `project:review(...)` and `item:label(...)` methods through
  `action.define_on(...)`

## Copy Into a Project

1. Copy `main.lua` into a harness directory.
2. Run the harness with any prompt.
3. Inspect session state or add temporary logging if you want to watch the
   hydrated proxies move through action and signal boundaries.
