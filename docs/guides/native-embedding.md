# Native Embedding

Turin can be embedded as a Rust library with a compiled harness instead of loading Lua
scripts. This is intended for fixed-purpose applications that want Turin's inference,
session graph, persistence, governance, scheduler, memory, and native-tool machinery
without shipping a scripting VM.

## Dependency

Disable default features to omit `mlua` and the Lua adapter:

```toml
[dependencies]
turin = { path = "../turin", default-features = false }
```

Keeping default features enabled allows compiled and Lua harnesses to coexist.

## Default Harness

Implement `NativeHarness` for synchronous lifecycle and request policy. A factory
creates isolated mutable harness state for each active session:

```rust
use std::sync::Arc;

use anyhow::Result;
use turin::kernel::Kernel;
use turin::kernel::harness_contract::HarnessTurnRequest;
use turin::kernel::native_harness::{NativeHarness, NativeHarnessFactory, Verdict};

struct AppHarness;

impl NativeHarness for AppHarness {
    fn on_turn_prepare(&mut self, request: &mut HarnessTurnRequest) -> Result<Verdict> {
        request.system_prompt.push_str("\nPrefer concise answers.");
        Ok(Verdict::Allow)
    }
}

let factory: Arc<dyn NativeHarnessFactory> =
    Arc::new(|| Ok(Box::new(AppHarness) as Box<dyn NativeHarness>));
let kernel = Kernel::builder(config)
    .with_native_harness_factory(factory)
    .build()?;
```

`with_native_harness_factory` is shorthand for registering the `default` harness ID.

## Multiple Harnesses

Agent configuration remains the authority for selecting a harness. Declare the named
harness and bind the agent to its ID, then register its compiled factory:

```rust
config.harnesses.insert("review".into(), Default::default());
config.agents.get_mut("reviewer").unwrap().harness = Some("review".into());

let kernel = Kernel::builder(config)
    .with_native_harness_factory(default_factory)
    .with_native_harness("review", review_factory)
    .build()?;
```

The harness declaration preserves the same stable ID and agent-binding model used by
Lua harnesses. In a build without Lua, every initialized harness definition must have a
native factory. A missing factory fails explicitly instead of running an empty harness.

Factory registration is construction-time work. The selected factory is stored on the
harness runtime definition, and each session creates its harness directly from that
factory; provider turns do not perform registry lookups.

## Operations And Tools

Use native harness callbacks for policy, request preparation, runtime signals, and named
actions. Implement agent-triggered I/O and asynchronous operations as native `Tool`
values and install them with `RuntimeBuilder::with_tool_registry`. This keeps governance,
tool exposure, persistence, and effect application inside the kernel rather than giving
harness callbacks unrestricted access to process-wide managers.

## Runnable Example

The repository includes `examples/native_harness.rs`:

```sh
cargo run --no-default-features --example native_harness -- \
  .turin/config.toml "Summarize this runtime"
```

The example loads normal Turin provider and persistence configuration while replacing
the default Lua harness with compiled Rust policy.
