use anyhow::Result;

use super::harness_contract::{HarnessHook, HarnessTurnRequest};
use crate::harness::verdict::Verdict;

/// A compiled, session-local harness implementation.
///
/// Methods default to `ALLOW`, so fixed-purpose applications only implement the policy
/// surfaces they need. Each active session receives its own instance from the factory.
pub trait NativeHarness: Send {
    fn on_hook(&mut self, _hook: HarnessHook<'_>) -> Result<Verdict> {
        Ok(Verdict::Allow)
    }

    fn on_turn_prepare(&mut self, _request: &mut HarnessTurnRequest) -> Result<Verdict> {
        Ok(Verdict::Allow)
    }
}

/// Creates isolated native harness state for a runtime session.
pub trait NativeHarnessFactory: Send + Sync {
    fn create(&self) -> Result<Box<dyn NativeHarness>>;
}

impl<F> NativeHarnessFactory for F
where
    F: Fn() -> Result<Box<dyn NativeHarness>> + Send + Sync,
{
    fn create(&self) -> Result<Box<dyn NativeHarness>> {
        self()
    }
}
