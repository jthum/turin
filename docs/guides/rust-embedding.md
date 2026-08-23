# Rust Embedding

Turin can be embedded as a Rust library with a compiled harness instead of loading Lua
scripts. This is intended for fixed-purpose applications that want Turin's inference,
session graph, persistence, governance, scheduler, memory, and native-tool machinery
without shipping a scripting VM.

## Dependency

Depend on the kernel package directly. It does not include `mlua` or a scripting adapter:

```toml
[dependencies]
turin = { path = "../turin" }
```

Applications that want Lua add `turin-harness-lua` separately and inject its adapter
through `RuntimeBuilder::with_harness_adapter`. Turin's CLI product performs that
composition by default.

## Default Harness

Implement `Harness` for synchronous lifecycle and request policy. A factory
creates isolated mutable harness state for each active session:

```rust
use anyhow::Result;
use turin::kernel::Kernel;
use turin::kernel::harness_contract::HarnessTurnRequest;
use turin::kernel::harness::{Harness, Verdict};

struct AppHarness;

impl Harness for AppHarness {
    fn on_turn_prepare(&mut self, request: &mut HarnessTurnRequest) -> Result<Verdict> {
        request.system_prompt.push_str("\nPrefer concise answers.");
        Ok(Verdict::Allow)
    }
}

let kernel = Kernel::builder(config)
    .with_default_harness(|| Ok(Box::new(AppHarness) as Box<dyn Harness>))
    .build()?;
```

`with_default_harness` is shorthand for registering the `default` harness ID.

## Multiple Harnesses

Agent configuration remains the authority for selecting a harness. Declare the named
harness and bind the agent to its ID, then register its compiled factory:

```rust
config.harnesses.insert("review".into(), Default::default());
config.agents.get_mut("reviewer").unwrap().harness = Some("review".into());

let kernel = Kernel::builder(config)
    .with_default_harness(default_factory)
    .with_harness("review", review_factory)
    .build()?;
```

The harness declaration preserves the same stable ID and agent-binding model used by
Lua harnesses. In a build without Lua, every initialized harness definition must have a
Rust factory. A missing factory fails explicitly instead of running an empty harness.

Factory registration is construction-time work. The selected factory is stored on the
harness runtime definition, and each session creates its harness directly from that
factory; provider turns do not perform registry lookups.

## Operations And Tools

Use Rust harness callbacks for policy, request preparation, runtime signals, and named
actions. Implement agent-triggered I/O and asynchronous operations as Rust `Tool`
values and install them with `RuntimeBuilder::with_tool_registry`. This keeps governance,
tool exposure, persistence, and effect application inside the kernel rather than giving
harness callbacks unrestricted access to process-wide managers.

Registered application tools participate in the same root, agent, and per-request tool
selection as Turin's built-ins. A custom tool is exposed by default unless an explicit
selection narrows it. Declare a governance capability when execution should pass through
Turin's capability policy:

```rust
#[async_trait::async_trait]
impl turin::tools::Tool for SaveRecord {
    fn name(&self) -> &str { "save_record" }
    fn description(&self) -> &str { "Save an application record" }
    fn parameters_schema(&self) -> serde_json::Value { /* JSON Schema */ }

    fn capability(&self) -> Option<&str> {
        Some("records.write")
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        context: &turin::tools::ToolContext,
    ) -> Result<turin::tools::ToolEffect, turin::tools::ToolError> {
        todo!()
    }
}
```

The capability method defaults to `None`, so existing tools remain source-compatible.
Built-in tools retain their existing static capability mapping. Exact custom tool names
in configuration are checked against the installed registry when the kernel is built,
which catches configuration mistakes before a session starts.

## Runnable Example

The repository includes `examples/rust_harness.rs`:

```sh
cargo run --no-default-features --example rust_harness -- \
  .turin/config.toml "Summarize this runtime"
```

The example loads normal Turin provider and persistence configuration while replacing
the default Lua harness with compiled Rust policy.
