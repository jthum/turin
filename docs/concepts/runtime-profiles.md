# Runtime Profiles

Runtime profiles are recommended operating shapes for Turin. They describe which capabilities should be enabled together and which risks should be explicit.

This document is conceptual. Turin already has governance profiles and tool selection config; these runtime profiles describe higher-level product modes that can be implemented through configuration, feature flags, or future composition.

## Why Profiles Matter

Turin can run as a local coding assistant, a governed workflow engine, a daemon with channel sidecars, or a remote-controlled runtime. Those modes should not all have the same default risk surface.

Profiles make defaults legible:

- which tools are exposed
- whether shell is allowed
- whether MCP is allowed
- whether memory is enabled
- whether channels are enabled
- whether remote access is allowed
- what audit posture is expected

## Suggested Profiles

### Minimal Local

Purpose:

- local experimentation
- harness development
- small tests
- no external channel ingress

Default shape:

- local provider or mock provider
- no remote bridge
- no channel sidecars
- no MCP
- minimal tool surface
- local state DB

Risk posture:

- lowest operational risk
- useful as the fast smoke path

### Local Coding

Purpose:

- coding assistant in a trusted local workspace

Default shape:

- filesystem tools
- web fetch/search if configured
- shell allowed if explicitly accepted by the operator profile
- `apply_patch` opt-in or included in coding profile
- memory optional
- code search optional

Risk posture:

- high local authority
- should be used only in trusted workspaces
- shell/process activity should be auditable

### Governed Assistant

Purpose:

- constrained assistant behavior with explicit policy
- human/operator review of risky actions

Default shape:

- governance enabled
- high-risk tools restricted
- grants enabled only with ceilings
- immutable or stricter audit mode where needed
- clear harness import scoping

Risk posture:

- safer for sensitive workflows
- slower but more inspectable

### Daemon Runtime

Purpose:

- long-running local control plane
- dynamic filesystem-managed agents/harnesses/channels
- scheduled or externally submitted work

Default shape:

- daemon IPC enabled
- event subscriptions enabled
- scheduler/worklists optional by config
- channel supervision only for configured sidecars
- explicit store placement

Risk posture:

- local system service risk
- daemon socket and state directory should be protected

### Channel Runtime

Purpose:

- Telegram/Rocket.Chat/Discord/WhatsApp style ingress and egress

Default shape:

- daemon runtime
- selected channel sidecars
- admission/pairing policy
- bounded inbound text/media handling
- no shell/MCP by default unless explicitly configured

Risk posture:

- external input boundary
- use conservative tool exposure and governance

### Remote Controlled

Purpose:

- HTTP/SSE/WebSocket access to daemon control plane

Default shape:

- remote bridge enabled
- bearer token from environment
- loopback-only unless explicitly deployed behind TLS/proxy
- event stream enabled

Risk posture:

- network boundary
- should be treated as production control plane exposure

## Profile Principles

- Process and integration tools should be profile-dependent.
- Channel ingress should not imply shell access.
- Remote access should not imply broader tool access.
- Memory and code search should be optional capabilities.
- High-risk profiles should emit strong audit events.
- Profile defaults must be tested.

## Mapping To Current Config

Current Turin configuration can approximate these profiles through:

- `[tools].allow` / `[tools].exclude`
- `[agent.tools]`
- channel `settings.tools`
- governance profile/capability settings
- daemon/channel config
- remote bind/auth settings
- provider/inference routing settings

Future work could make these first-class profile presets.

