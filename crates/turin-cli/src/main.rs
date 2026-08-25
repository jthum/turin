use anyhow::Result;
use clap::Parser;
#[cfg(feature = "heap-profile")]
use std::path::PathBuf;

mod cli;
mod commands;
mod composition;
mod dispatch;

use cli::Cli;
use turin::tracing_support::init_tracing;

#[cfg(feature = "heap-profile")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[tokio::main]
async fn main() -> Result<()> {
    #[cfg(feature = "heap-profile")]
    let _heap_profiler = dhat::Profiler::builder()
        .file_name(
            std::env::var_os("TURIN_HEAP_PROFILE_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("dhat-heap.json")),
        )
        .trim_backtraces(None)
        .build();

    let cli = Cli::parse();
    init_tracing(&cli.log_level, cli.log_file.clone())?;
    dispatch::run(cli).await
}
