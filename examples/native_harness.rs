use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use turin::kernel::Kernel;
use turin::kernel::config::TurinConfig;
use turin::kernel::harness_contract::HarnessTurnRequest;
use turin::kernel::native_harness::{NativeHarness, NativeHarnessFactory, Verdict};

struct ConciseHarness;

impl NativeHarness for ConciseHarness {
    fn on_turn_prepare(&mut self, request: &mut HarnessTurnRequest) -> Result<Verdict> {
        request
            .system_prompt
            .push_str("\nAnswer directly and keep operational output concise.");
        Ok(Verdict::Allow)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| ".turin/config.toml".to_string());
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "Describe the active runtime configuration.".to_string());
    let config = TurinConfig::from_file(Path::new(&config_path))?;
    let factory: Arc<dyn NativeHarnessFactory> =
        Arc::new(|| Ok(Box::new(ConciseHarness) as Box<dyn NativeHarness>));
    let mut kernel = Kernel::builder(config)
        .with_native_harness_factory(factory)
        .build()?;

    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel.run(&mut session, Some(prompt)).await?;
    kernel.end_session(&mut session).await?;
    Ok(())
}
