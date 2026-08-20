# Security Model

Turin runs model-proposed actions on a real machine. Its security model is built around explicit trust boundaries: inference proposes, harness decides, and the kernel enforces.

This document describes the intended boundaries and the sharp edges operators should understand.

## Trust Boundaries

### Model Output Is Untrusted

The model can propose text, structured output, and tool calls. The model should not be treated as a trusted actor.

Controls:

- tool calls pass through the harness and tool policy
- high-risk built-ins map to governance capabilities
- tool execution is emitted as events and persisted
- child tool selection cannot exceed parent grants

### Harness Scripts Are Trusted Policy Code

Harness scripts define workflow and policy. They are more trusted than model output, but still run inside a sandbox.

Controls:

- Luau safe stdlib
- Luau sandbox mode
- memory limit
- load-time-only imports and watch registration
- filesystem helpers route through path policy
- governance subject context tracks module/root/import delegation

Assumption:

- a project owner controls the harness directory
- malicious harness code can intentionally authorize dangerous behavior within its granted capabilities

### Kernel Is The Enforcement Layer

The Rust kernel owns execution and persistence. It should enforce invariants even if model output or harness behavior is hostile or malformed.

Controls:

- typed execution state
- durable event emission
- tool execution pipeline
- governance enforcement fallback for high-risk built-ins
- persistence validation
- daemon-side config validation

### Local Workspace Is A Trust Boundary

File tools are confined to the configured workspace root by path validation.

Current behavior:

- parent traversal is rejected
- existing ancestors are canonicalized and checked against the root
- reads/writes are rooted in the workspace

Hardening opportunity:

- symlink and time-of-check/time-of-use behavior should be revisited if workspaces are adversarial or multi-user
- a future `WorkspaceFs`/`PathPolicy` abstraction should centralize file safety, symlink policy, file size limits, and error behavior

### Shell Execution Is High-Risk

`shell_exec` can execute arbitrary shell commands. It is practical for local coding agents but is the largest default tool risk.

Guidance:

- treat shell as a high-risk capability
- consider profile-dependent shell exposure
- avoid exposing shell in remote/channel deployments unless explicitly intended
- audit shell execution events

### MCP Is High-Risk Integration

`bridge_mcp` can spawn external MCP servers and register their tools.

Current behavior:

- `bridge_mcp` is not in the code default exposed tool set
- it is available through explicit selection such as `group:integration` or `group:all`

Guidance:

- treat MCP bridge as process/integration risk, similar to shell
- require explicit opt-in
- prefer allowlisted commands for governed deployments

### Channels Are External Input Boundaries

Channel sidecars receive input from external systems and submit work into Turin.

Controls:

- channel settings validation
- access/pairing policy
- session-scope configuration
- inbound text bounding
- sidecar hello/heartbeat supervision
- daemon-managed channel lifecycle

Guidance:

- use restrictive channel admission for shared or public spaces
- keep channel credentials in environment variables or secret stores
- treat attachments as untrusted input

### Remote Bridge Is A Network Boundary

The remote bridge exposes authenticated HTTP/SSE/WebSocket access to daemon control APIs.

Current behavior:

- bearer token required for control/event routes
- non-loopback bind requires explicit opt-in

Guidance:

- keep default loopback-only behavior
- use TLS or a trusted reverse proxy for non-loopback deployments
- consider constant-time token comparison if non-loopback remote becomes a primary deployment mode

### Persistence Is Sensitive

State stores may contain prompts, responses, tool results, memory, events, file contents, and channel metadata.

Guidance:

- place stores intentionally
- avoid committing `.turin` runtime state
- treat state DBs as sensitive artifacts
- use immutable audit mode where audit history matters

## Security Invariants

These invariants should have characterization or negative tests:

- default tool exposure matches documentation
- children cannot expand native tools beyond parent grants
- `apply_patch` and `bridge_mcp` remain opt-in unless intentionally changed
- governance grants cannot exceed issuer ceilings
- denied tools fail safely
- path traversal is rejected
- remote non-loopback bind requires opt-in
- unauthorized channel users are rejected or quarantined
- oversized inbound content is bounded
- invalid sidecar settings do not start a broken channel
- Unix local IPC sockets are owner-only, and endpoint cleanup never deletes a
  regular file or symlink placed at the configured socket path

## Known Hardening Opportunities

- central `WorkspaceFs`/`PathPolicy` with stronger symlink behavior
- profile-dependent default exposure for shell/process tools
- explicit tool risk taxonomy in config/docs
- constant-time bearer token comparison for remote
- owner-only permissions for the broader local runtime directory on Unix
- allowlisted MCP command policy for governed deployments
- clearer minimal/gov/channel runtime profiles
