use std::sync::Arc;

use anyhow::Result;

use super::{HarnessAdapterFactory, default_script_adapter_factory, rust_adapter_factory};
use crate::kernel::config::TurinConfig;
use crate::kernel::harness::RustHarnessFactories;

/// Resolves harness implementations once while constructing the harness catalog.
pub(crate) struct HarnessAdapterResolver<'a> {
    rust_harness_factories: &'a RustHarnessFactories,
}

impl<'a> HarnessAdapterResolver<'a> {
    pub(crate) fn new(
        config: &TurinConfig,
        rust_harness_factories: &'a RustHarnessFactories,
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
        })
    }

    pub(crate) fn resolve(&self, harness_id: &str) -> Result<Arc<dyn HarnessAdapterFactory>> {
        if let Some(factory) = self.rust_harness_factories.get(harness_id) {
            return Ok(rust_adapter_factory(Arc::clone(factory)));
        }

        default_script_adapter_factory().map_err(|_| {
            anyhow::anyhow!(
                "Harness '{}' has no Rust factory and no script adapter is enabled",
                harness_id
            )
        })
    }
}
