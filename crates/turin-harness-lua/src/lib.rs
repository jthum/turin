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
