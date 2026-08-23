use std::sync::Arc;

use anyhow::Result;

use super::{HarnessAdapterFactory, rust_adapter_factory};
use crate::kernel::config::TurinConfig;
use crate::kernel::harness::RustHarnessFactories;

/// Resolves harness implementations once while constructing the harness catalog.
pub(crate) struct HarnessAdapterResolver<'a> {
    rust_harness_factories: &'a RustHarnessFactories,
    script_harness_adapter: Option<&'a Arc<dyn HarnessAdapterFactory>>,
}

impl<'a> HarnessAdapterResolver<'a> {
    pub(crate) fn new(
        config: &TurinConfig,
        rust_harness_factories: &'a RustHarnessFactories,
        script_harness_adapter: Option<&'a Arc<dyn HarnessAdapterFactory>>,
    ) -> Result<Self> {
        if let Some(unknown_id) = rust_harness_factories
            .keys()
            .find(|id| id.as_str() != "default" && !config.harnesses.contains_key(*id))
        {
            anyhow::bail!(
                "Rust harness '{}' is not declared in config.harnesses",
                unknown_id
            );
        }

        Ok(Self {
            rust_harness_factories,
            script_harness_adapter,
        })
    }

    pub(crate) fn resolve(&self, harness_id: &str) -> Result<Arc<dyn HarnessAdapterFactory>> {
        if let Some(factory) = self.rust_harness_factories.get(harness_id) {
            return Ok(rust_adapter_factory(Arc::clone(factory)));
        }

        self.script_harness_adapter.cloned().ok_or_else(|| {
            anyhow::anyhow!("Harness '{}' has no registered implementation", harness_id)
        })
    }
}
