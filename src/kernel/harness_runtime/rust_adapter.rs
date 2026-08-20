use std::cell::RefCell;
use std::sync::Arc;

use anyhow::Result;

use super::{
    HarnessAdapterFactory, HarnessDefinition, HarnessInstance, HarnessRuntimeInitContext,
    HarnessTurnServices,
};
use crate::kernel::harness::{Harness, HarnessFactory, Verdict};
use crate::kernel::harness_contract::{
    HarnessActionRequest, HarnessHook, HarnessSignal, HarnessTurnRequest,
};

struct RustHarnessAdapterFactory {
    factory: Arc<dyn HarnessFactory>,
}

impl HarnessAdapterFactory for RustHarnessAdapterFactory {
    fn name(&self) -> &'static str {
        "rust"
    }

    fn create(
        &self,
        _definition: &HarnessDefinition,
        _ctx: HarnessRuntimeInitContext,
    ) -> Result<Box<dyn HarnessInstance>> {
        Ok(Box::new(RustHarnessInstance {
            harness: RefCell::new(self.factory.create()?),
        }))
    }
}

struct RustHarnessInstance {
    harness: RefCell<Box<dyn Harness>>,
}

impl HarnessInstance for RustHarnessInstance {
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

pub(super) fn factory(factory: Arc<dyn HarnessFactory>) -> Arc<dyn HarnessAdapterFactory> {
    Arc::new(RustHarnessAdapterFactory { factory })
}
