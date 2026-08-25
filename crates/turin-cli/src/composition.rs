use std::path::Path;

use anyhow::Result;
use turin::kernel::builder::RuntimeBuilder;
use turin::kernel::config::TurinConfig;

pub(crate) fn kernel_builder(config: TurinConfig) -> RuntimeBuilder {
    RuntimeBuilder::new(config)
        .paint_stdout(true)
        .with_harness_adapter(turin_harness_lua::factory())
}

pub(crate) async fn serve_daemon(config_path: &Path) -> Result<()> {
    turin::daemon::server::serve_with_harness_adapter(config_path, turin_harness_lua::factory())
        .await
}
