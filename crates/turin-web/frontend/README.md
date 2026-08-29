# Turin Web Frontend

This is the browser client for `turin-web`. It is a SvelteKit 3 static SPA; the
Rust `turin-web` process owns the HTTP/API boundary and serves the production
assets. No Node.js server is required at runtime.

The frontend package manager is pinned in `package.json`; use Bun rather than
creating an additional package-manager lockfile.

## Develop

Start the Rust host from the repository root:

```sh
cargo run -p turin-web
```

Then start Vite in this directory:

```sh
bun install
bun run dev
```

Vite proxies `/api` to `http://127.0.0.1:9330`.

For interface work without a daemon, run the development-only mock adapter:

```sh
bun run dev:mock
```

The mock implements the same browser HTTP and SSE contracts as `turin-web`.
Its long-session scenario generates message windows on demand rather than
checking in large data files. Override its default 10,000-message transcript
when testing different scales:

```sh
TURIN_MOCK_MESSAGE_COUNT=100000 bun run dev:mock
```

Exercise streaming states without changing application code:

```sh
TURIN_MOCK_STREAM=slow bun run dev:mock
TURIN_MOCK_STREAM=error bun run dev:mock
TURIN_MOCK_STREAM=interrupt bun run dev:mock
```

## Build

```sh
bun run check
bun run build
```

The static adapter writes `build/`, which `turin-web` serves by default.
