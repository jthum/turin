# Turin Web

`turin-web` is the HTTP host for Turin's browser client. It serves a SvelteKit
static SPA and talks to Turin through the same typed `turin-client` facade used
by other operator clients.

The current foundation intentionally covers connection and shell behavior only.
It is the runway for consumer-facing session and conversation workflows, not a
preserved version of the earlier exploratory dashboard.

## Prerequisites

Start the daemon from the repository root:

```sh
cargo run --bin turin -- daemon ensure --config .turin/config.toml
```

Install frontend dependencies once:

```sh
cd crates/turin-web/frontend
npm install
```

## Development

Run the Rust API host in one terminal:

```sh
cargo run -p turin-web -- --config .turin/config.toml
```

Run Vite in another terminal:

```sh
cd crates/turin-web/frontend
npm run dev
```

Open the URL printed by Vite. Development `/api` requests are proxied to
`http://127.0.0.1:9330`.

## Static Build

Build the browser assets:

```sh
cd crates/turin-web/frontend
npm run check
npm run build
```

Then run the Rust host from the repository root:

```sh
cargo run -p turin-web -- --config .turin/config.toml
```

By default it serves `crates/turin-web/frontend/build` on
`http://127.0.0.1:9330`. Override the asset location with `--assets-dir`.

## Remote Turin

The browser still connects only to `turin-web`. The Rust host can use
`turin-remote` as its upstream:

```sh
export TURIN_REMOTE_AUTH_TOKEN="replace-with-your-token"
cargo run -p turin-web -- \
  --remote-url http://server.example:9324
```

Use `--auth-token-env NAME` to choose a different environment variable. Passing
the token directly with `--auth-token` is supported but less suitable for
shell history and process listings.

## Network Boundary

`turin-web` binds to loopback by default. It does not yet implement end-user
authentication. A non-loopback bind is rejected unless
`--allow-non-loopback` is supplied; use that option only behind a trusted
authenticated reverse proxy or equivalent boundary.
