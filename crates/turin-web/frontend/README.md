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

## Build

```sh
bun run check
bun run build
```

The static adapter writes `build/`, which `turin-web` serves by default.
