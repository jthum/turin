use std::sync::Arc;

use crate::harness::engine::HarnessEngine;

use super::harness_runtime::HarnessRuntime;

pub(crate) struct HarnessManager {
    default_runtime: Arc<HarnessRuntime>,
}

impl HarnessManager {
    pub(crate) fn new(default_runtime: HarnessRuntime) -> Self {
        let default_runtime = Arc::new(default_runtime);
        Self { default_runtime }
    }

    pub(crate) fn default_runtime(&self) -> &Arc<HarnessRuntime> {
        &self.default_runtime
    }

    pub(crate) fn lock_default_engine(&self) -> std::sync::MutexGuard<'_, Option<HarnessEngine>> {
        self.resolve_harness(None).lock_engine()
    }

    pub(crate) fn resolve_harness(&self, _agent_id: Option<&str>) -> &Arc<HarnessRuntime> {
        self.default_runtime()
    }
}
