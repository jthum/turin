use std::sync::Arc;

pub use crate::harness::source::HarnessSourceOverlay;
use crate::kernel::harness::HarnessFactory;

mod contract;
mod definition;
mod resolver;
mod rust_adapter;
#[cfg(test)]
mod test_adapter;

pub use contract::{
    HarnessAdapterFactory, HarnessGeneration, HarnessInstance, HarnessLoadMetadata,
    HarnessRuntimeInitContext,
};
#[doc(hidden)]
pub use definition::HarnessDefinition;
pub(crate) use resolver::HarnessAdapterResolver;

fn rust_adapter_factory(factory: Arc<dyn HarnessFactory>) -> Arc<dyn HarnessAdapterFactory> {
    rust_adapter::factory(factory)
}

#[cfg(test)]
pub(crate) fn test_script_adapter_factory() -> Arc<dyn HarnessAdapterFactory> {
    Arc::new(test_adapter::TestHarnessAdapterFactory)
}

#[cfg(test)]
pub(crate) fn test_runtime_builder(
    config: crate::kernel::config::TurinConfig,
) -> crate::kernel::builder::RuntimeBuilder {
    crate::kernel::builder::RuntimeBuilder::new(config)
        .with_harness_adapter(test_script_adapter_factory())
}
