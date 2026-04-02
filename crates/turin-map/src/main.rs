use anyhow::{Context, Result};
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

const CLI_EXAMPLES: &str = "Examples:
  turin-map index
  turin-map status
  turin-map index --config path/to/.turin/config.toml
  turin-map index --embedding-provider openai --embedding-base-url http://127.0.0.1:11434/v1 --embedding-model your-small-embedding-model --embedding-dimensions 384";

#[derive(Parser, Debug)]
#[command(
    name = "turin-map",
    version,
    about = "Build and inspect Turin code-search indexes",
    after_help = CLI_EXAMPLES
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
    #[arg(long, default_value = ".", help = "Project root to index or inspect")]
    root: PathBuf,

    #[arg(
        long,
        help = "Override the index database path (defaults to <root>/.turin/codebase.db)"
    )]
    index_path: Option<PathBuf>,

    #[arg(
        long,
        help = "Print machine-readable JSON instead of human-readable output"
    )]
    json: bool,
}

#[derive(Args, Debug, Clone)]
struct IndexArgs {
    #[command(flatten)]
    root: RootArgs,

    #[arg(
        long,
        help = "Load provider and embedding defaults from a Turin config file; defaults to ./.turin/config.toml if present"
    )]
    config: Option<PathBuf>,

    #[command(flatten)]
    embedding: EmbeddingArgs,
}

#[derive(Args, Debug, Clone)]
struct RemoveArgs {
    #[command(flatten)]
    root: RootArgs,

    #[arg(long, help = "File path to remove from the existing index")]
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
            let config = load_turin_map_config(&cwd, args.config.as_deref()).with_context(
                || "Hint: run from the Turin project root or pass --config path/to/.turin/config.toml",
            )?;
            let report = build_index_with_options(
                &args.root.root,
                args.root.index_path.as_deref(),
                build_options(&args.embedding, config.as_ref())?,
            )
            .await
            .map_err(add_embedding_hint)?;
            print_build_report(args.root.json, &report)?;
        }
        Command::Rebuild(args) => {
            let config = load_turin_map_config(&cwd, args.config.as_deref()).with_context(
                || "Hint: run from the Turin project root or pass --config path/to/.turin/config.toml",
            )?;
            let report = rebuild_index_with_options(
                &args.root.root,
                args.root.index_path.as_deref(),
                build_options(&args.embedding, config.as_ref())?,
            )
            .await
            .map_err(add_embedding_hint)?;
            print_build_report(args.root.json, &report)?;
        }
        Command::Remove(args) => {
            let report = remove_file(&args.root.root, args.root.index_path.as_deref(), &args.path)
                .await
                .map_err(add_index_hint)?;
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
            let status = read_status(&cwd, selector).await.map_err(add_index_hint)?;
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

fn add_index_hint(err: anyhow::Error) -> anyhow::Error {
    let message = err.to_string();
    if message.contains("index db not found")
        || message.contains("missing required index_meta contract")
        || message.contains("index_meta is empty")
    {
        err.context("Hint: run `turin-map index` from the project root first")
    } else {
        err
    }
}

fn add_embedding_hint(err: anyhow::Error) -> anyhow::Error {
    let message = err.to_string();
    if message.contains("embedding provider")
        || message.contains("Environment variable")
        || message.contains("embedding model")
        || message.contains("embedding configuration")
    {
        err.context(
            "Hint: omit [embeddings] / --embedding-* for lexical-only indexing, or verify the local OpenAI-compatible endpoint, model, and dimensions",
        )
    } else {
        err
    }
}
