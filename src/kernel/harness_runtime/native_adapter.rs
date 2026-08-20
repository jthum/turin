use std::cell::RefCell;

use anyhow::Result;

use super::{HarnessInstance, HarnessTurnServices};
use crate::kernel::harness_contract::{
    HarnessActionRequest, HarnessHook, HarnessSignal, HarnessTurnRequest,
};
use crate::kernel::native_harness::{NativeHarness, NativeHarnessFactory, Verdict};

struct NativeHarnessInstance {
    harness: RefCell<Box<dyn NativeHarness>>,
}

impl HarnessInstance for NativeHarnessInstance {
    fn load_script_str(&mut self, _script: &str) -> Result<()> {
        anyhow::bail!("native harnesses do not load Lua source")
    }

    fn runtime_signal_topics(&self) -> Vec<String> {
        self.harness.borrow().runtime_signal_topics()
    }

    fn evaluate_hook(&self, hook: HarnessHook<'_>) -> Result<Verdict> {
        self.harness.borrow_mut().on_hook(hook)
    }

    fn has_hook(&self, hook_name: &str) -> bool {
        hook_name == "on_turn_prepare"
    }

    fn prepare_turn(
        &self,
        request: &mut HarnessTurnRequest,
        _services: HarnessTurnServices<'_>,
    ) -> Result<Verdict> {
        self.harness.borrow_mut().on_turn_prepare(request)
    }

    fn invoke_action(
        &self,
        request: HarnessActionRequest<'_>,
    ) -> Result<Option<serde_json::Value>> {
        self.harness.borrow_mut().on_action(request)
    }

    fn dispatch_runtime_signal(&self, signal: HarnessSignal<'_>) -> Result<usize> {
        self.harness.borrow_mut().on_signal(signal)?;
        Ok(1)
    }
}

pub(super) fn build_instance(
    factory: &std::sync::Arc<dyn NativeHarnessFactory>,
) -> Result<Box<dyn HarnessInstance>> {
    Ok(Box::new(NativeHarnessInstance {
        harness: RefCell::new(factory.create()?),
    }))
}
