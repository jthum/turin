pub(crate) use turin_core::{
    code_index_reader, display, inference, kernel, persistence, signal_topics, tools, work_items,
};

mod harness;
mod runtime;

pub use runtime::LuaHarnessAdapterFactory;

/// Creates the Lua adapter used by a Turin runtime composition.
pub fn factory() -> std::sync::Arc<dyn turin_core::kernel::harness_runtime::HarnessAdapterFactory> {
    std::sync::Arc::new(LuaHarnessAdapterFactory)
}

/// Creates a runtime builder configured with the Lua harness adapter.
pub fn runtime_builder(
    config: turin_core::kernel::config::TurinConfig,
) -> turin_core::kernel::builder::RuntimeBuilder {
    turin_core::kernel::builder::RuntimeBuilder::new(config).with_harness_adapter(factory())
}

/// Runs the Turin daemon with Lua harness support enabled.
pub async fn serve_daemon(config_path: &std::path::Path) -> anyhow::Result<()> {
    turin_core::daemon::server::serve_with_harness_adapter(config_path, factory()).await
}
