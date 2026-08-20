use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;

use super::harness_contract::{
    HarnessActionRequest, HarnessHook, HarnessSignal, HarnessTurnRequest,
};
pub use crate::harness::verdict::Verdict;

/// A compiled, session-local harness implementation.
///
/// Methods default to `ALLOW`, so fixed-purpose applications only implement the policy
/// surfaces they need. Each active session receives its own instance from the factory.
pub trait NativeHarness: Send {
    /// Signal topic patterns subscribed by this harness definition.
    fn runtime_signal_topics(&self) -> Vec<String> {
        Vec::new()
    }

    fn on_hook(&mut self, _hook: HarnessHook<'_>) -> Result<Verdict> {
        Ok(Verdict::Allow)
    }

    fn on_turn_prepare(&mut self, _request: &mut HarnessTurnRequest) -> Result<Verdict> {
        Ok(Verdict::Allow)
    }

    fn on_signal(&mut self, _signal: HarnessSignal<'_>) -> Result<()> {
        Ok(())
    }

    /// Handles a named action, returning `None` when this harness does not define it.
    fn on_action(
        &mut self,
        _request: HarnessActionRequest<'_>,
    ) -> Result<Option<serde_json::Value>> {
        Ok(None)
    }
}

/// Creates isolated native harness state for a runtime session.
pub trait NativeHarnessFactory: Send + Sync {
    fn create(&self) -> Result<Box<dyn NativeHarness>>;
}

pub(crate) type NativeHarnessFactories = HashMap<String, Arc<dyn NativeHarnessFactory>>;

impl<F> NativeHarnessFactory for F
where
    F: Fn() -> Result<Box<dyn NativeHarness>> + Send + Sync,
{
    fn create(&self) -> Result<Box<dyn NativeHarness>> {
        self()
    }
}
