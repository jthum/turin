pub use turin_core::{
    code_index_reader, display, inference, kernel, persistence, schedule_support, signal_topics,
    tools, work_items,
};

pub mod harness;
mod runtime;

pub use runtime::LuaHarnessAdapterFactory;

pub fn factory() -> std::sync::Arc<dyn kernel::harness_runtime::HarnessAdapterFactory> {
    std::sync::Arc::new(LuaHarnessAdapterFactory)
}

pub fn runtime_builder(config: kernel::config::TurinConfig) -> kernel::builder::RuntimeBuilder {
    kernel::builder::RuntimeBuilder::new(config).with_harness_adapter(factory())
}

pub async fn serve_daemon(config_path: &std::path::Path) -> anyhow::Result<()> {
    turin_core::daemon::server::serve_with_harness_adapter(config_path, factory()).await
}
