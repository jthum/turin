# Built-in Web Tools Map

## Purpose

The built-in web tools provide constrained HTTP retrieval and web search for agents without requiring shell access. This subsystem owns:

- `web_fetch` URL validation, browser-like request headers, content extraction, truncation, and output metadata.
- `web_search` provider selection, provider-specific request construction, result normalization, fallback across configured providers, and output metadata.
- web-tool config validation used at startup/config load time.

## Files

- `src/tools/builtins/web_tools.rs`
  - Public `WebFetchTool` and `WebSearchTool` implementations.
  - Shared HTTP client/header helpers.
  - Tool config validation and URL validation.
- `src/tools/builtins/web_tools/html.rs`
  - HTML title/body extraction, whitespace collapse, character truncation, and DuckDuckGo Lite result parsing.
- `src/tools/builtins/web_tools/search.rs`
  - Search provider enum, provider config expansion, provider response structs, API request functions, JSON response handling, and result normalization.
- `src/tools/builtins/tests/web_tools.rs`
  - Unit tests for provider parsing, HTML extraction, search result normalization, and config validation.
- `src/tools/builtins/mod.rs`
  - Registers `web_fetch` and `web_search` and exposes the `web` builtin group.

## Data Flow

`web_fetch`:

1. Deserialize typed args.
2. Validate the URL is `http` or `https`.
3. Build a bounded redirect/timeout client.
4. Apply configured browser-like headers.
5. Read the response body, extract readable HTML body text when applicable, collapse whitespace, truncate to the requested limit, and return metadata.

`web_search`:

1. Deserialize typed args and reject empty queries.
2. Clamp result limit to the supported range.
3. Expand configured providers, defaulting to DuckDuckGo Lite when none are configured.
4. Try providers in order until one returns successfully.
5. Normalize provider-specific results into `{ title, url, snippet }` values and return attempted-provider metadata.

## Invariants

- Only `http` and `https` URLs are accepted.
- Search providers are tried in configured order; duplicates are collapsed while preserving first occurrence.
- Brave and Tavily require configured API-key environment variable names when selected.
- SearXNG requires a configured HTTP(S) base URL when selected.
- Provider-specific response quirks stay in `search.rs`; user-facing output shape stays in the `Tool` implementation.
- HTML scraping and text cleanup stay in `html.rs` and should remain deterministic/unit-testable without network access.

## Common Changes

Add a search provider:

1. Add the provider variant and parser alias in `search.rs`.
2. Add config validation in `web_tools.rs`.
3. Add provider request/response handling in `search.rs`.
4. Add focused parsing/config tests in `src/tools/builtins/tests/web_tools.rs`.

Change fetch output:

1. Update `WebFetchTool::execute` in `web_tools.rs`.
2. Keep metadata stable unless the change is intentional and documented.
3. Add extraction/truncation tests when the behavior is parseable without network access.

## Tests

Focused tests:

```sh
cargo test -p turin --lib web_tools
```

Basic compile/format checks:

```sh
cargo check -p turin --lib
cargo fmt --all -- --check
git diff --check
```
