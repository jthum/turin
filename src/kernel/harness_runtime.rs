use std::sync::Arc;

use anyhow::Result;

use crate::kernel::harness::HarnessFactory;

mod contract;
mod definition;
#[cfg(feature = "lua")]
mod lua_adapter;
mod resolver;
mod rust_adapter;

pub(crate) use contract::{HarnessAdapterFactory, HarnessInstance, HarnessRuntimeInitContext};
pub(crate) use definition::HarnessDefinition;
pub(crate) use resolver::HarnessAdapterResolver;

fn rust_adapter_factory(factory: Arc<dyn HarnessFactory>) -> Arc<dyn HarnessAdapterFactory> {
    rust_adapter::factory(factory)
}

#[cfg(feature = "lua")]
fn default_script_adapter_factory() -> Result<Arc<dyn HarnessAdapterFactory>> {
    Ok(lua_adapter::factory())
}

#[cfg(not(feature = "lua"))]
fn default_script_adapter_factory() -> Result<Arc<dyn HarnessAdapterFactory>> {
    anyhow::bail!("No script harness adapter is enabled in this Turin build")
}
