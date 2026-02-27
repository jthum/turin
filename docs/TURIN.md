# Turin: A Substrate for Programmatic Autonomy

Turin is an execution substrate for AI agents, not an agent framework with a baked-in personality or workflow.

Turin provides:

- inference transport and streaming
- deterministic tool execution
- persistence and event logging
- a programmable harness runtime (Luau)
- opt-in governance and capability enforcement

Your harness scripts define behavior. Turin enforces execution.

## The Core Separation

Turin is built around a strict boundary:

- **Inference proposes** — the LLM suggests actions, plans, and tool calls
- **Harness decides** — Lua hooks can allow, reject, escalate, or modify behavior
- **Kernel enforces** — Rust runtime executes only what survives policy and harness checks

This is why Turin can be both:

- a highly autonomous, minimally restricted runtime in isolated environments
- a tightly governed, auditable runtime with capability ceilings and import scoping

## Why Turin Exists

Many agent systems conflate three concerns:

1. inference (the model)
2. execution (tools / state / side effects)
3. policy (what is allowed)

Turin separates them on purpose.

That gives you a few practical advantages:

- **Replaceable behavior**: change harness scripts, not the kernel
- **Deterministic enforcement**: a `REJECT` verdict is code, not a prompt suggestion
- **Composability**: workflows, memory policies, and routing strategies become harness modules
- **Portability**: one binary, many harnesses, many providers

## What Is Stable in the Current Baseline

This release establishes the forward-looking public harness surface:

- stable hook lifecycle semantics (`on_turn_prepare`, `on_plan_submit`, etc.)
- canonical stdlib API under `runtime.*`
- top-level ergonomic aliases (`memory`, `kv`, `agent`, `session`, `user`)
- first-party DX helpers layered on top of the canonical API (`verdict`, `allowed`, `needs`, callable `runtime.db(...)`, callable `runtime.agent(...)`, grant/time/json helpers)
- governance profiles/capabilities/import scoping/grants (opt-in)
- multi-db and multi-agent orchestration primitives

Turin is still pre-1.0, so change is expected. But the direction is now much clearer and more coherent than earlier versions.

## Turin’s Governance Philosophy (Important)

Turin is **not** “secure by force.” It is **flexible by default with opt-in strong governance**.

That means:

- dangerous behavior can be allowed intentionally
- strict enforcement is available when you want it
- governance is explicit and configurable
- user sovereignty is preserved

Examples:

- You can run an open harness that rewrites itself in a sandboxed environment.
- You can also lock imports to read-only roots and give writable submodules only limited capabilities.
- You can apply stricter per-agent ceilings (e.g. reviewer vs coder vs deployer).
- You can use temporary grants to elevate a specific operation for a limited time.

## Mental Model: Kernel as Physics

Turin is easiest to understand if you treat it like an operating substrate:

- The **kernel** is the physics engine.
- The **harness** is the law and workflow.
- The **model** is a planner operating inside that environment.

The kernel should remain boring, deterministic, and provider-agnostic.
Provider quirks belong in the normalized SDK layer (`inference-sdk-rust`), not in Turin core.

## Where to Go Next

- `docs/ARCHITECTURE.md` — implementation architecture and module layout
- `docs/HOOKS.md` — exact hook lifecycle and payloads
- `docs/PRIMITIVES.md` — current canonical stdlib API
- `docs/GOVERNANCE.md` — profiles, capabilities, import scoping, grants
