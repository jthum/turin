use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

use turin_code_index::code_index_reader::{CodebaseSelector, status as read_status};
use turin_code_index::code_index_writer::{
    build_index_with_options, rebuild_index_with_options, remove_file,
};

mod config;
mod embedding;
mod output;

use config::load_turin_map_config;
use embedding::{EmbeddingArgs, build_options};
use output::{print_build_report, print_remove_report, print_status};

#[derive(Parser, Debug)]
#[command(
    name = "turin-map",
    version,
    about = "Build and inspect Turin code-search indexes"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Build or refresh the code index for a root
    Index(IndexArgs),
    /// Rebuild the code index from scratch
    Rebuild(IndexArgs),
    /// Remove one file from an existing index
    Remove(RemoveArgs),
    /// Show index status
    Status(StatusArgs),
}

#[derive(Args, Debug, Clone)]
struct RootArgs {
    #[arg(long, default_value = ".")]
    root: PathBuf,

    #[arg(long)]
    index_path: Option<PathBuf>,

    #[arg(long)]
    json: bool,
}

#[derive(Args, Debug, Clone)]
struct IndexArgs {
    #[command(flatten)]
    root: RootArgs,

    #[arg(long)]
    config: Option<PathBuf>,

    #[command(flatten)]
    embedding: EmbeddingArgs,
}

#[derive(Args, Debug, Clone)]
struct RemoveArgs {
    #[command(flatten)]
    root: RootArgs,

    #[arg(long)]
    path: PathBuf,
}

#[derive(Args, Debug, Clone)]
struct StatusArgs {
    #[command(flatten)]
    root: RootArgs,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let cwd = std::env::current_dir()?;
    match cli.command {
        Command::Index(args) => {
            let config = load_turin_map_config(&cwd, args.config.as_deref())?;
            let report = build_index_with_options(
                &args.root.root,
                args.root.index_path.as_deref(),
                build_options(&args.embedding, config.as_ref())?,
            )
            .await?;
            print_build_report(args.root.json, &report)?;
        }
        Command::Rebuild(args) => {
            let config = load_turin_map_config(&cwd, args.config.as_deref())?;
            let report = rebuild_index_with_options(
                &args.root.root,
                args.root.index_path.as_deref(),
                build_options(&args.embedding, config.as_ref())?,
            )
            .await?;
            print_build_report(args.root.json, &report)?;
        }
        Command::Remove(args) => {
            let report =
                remove_file(&args.root.root, args.root.index_path.as_deref(), &args.path).await?;
            print_remove_report(args.root.json, &report)?;
        }
        Command::Status(args) => {
            let selector = CodebaseSelector {
                root: absolute_or_self(&cwd, &args.root.root)
                    .to_string_lossy()
                    .to_string(),
                index_path: args
                    .root
                    .index_path
                    .as_ref()
                    .map(|path| absolute_or_self(&cwd, path).to_string_lossy().to_string()),
            };
            let status = read_status(&cwd, selector).await?;
            print_status(args.root.json, &status)?;
        }
    }
    Ok(())
}

fn absolute_or_self(base: &std::path::Path, path: &std::path::Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}
