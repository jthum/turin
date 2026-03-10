use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

use turin_code_index::code_index_reader::{CodebaseSelector, status as read_status};
use turin_code_index::code_index_writer::{
    build_index_with_options, rebuild_index_with_options, remove_file,
};

mod embedding;

use embedding::{EmbeddingArgs, build_options};

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
    #[arg(long)]
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
    match cli.command {
        Command::Index(args) => {
            let report = build_index_with_options(
                &args.root.root,
                args.root.index_path.as_deref(),
                build_options(&args.embedding)?,
            )
            .await?;
            print_value(args.root.json, &report)?;
        }
        Command::Rebuild(args) => {
            let report = rebuild_index_with_options(
                &args.root.root,
                args.root.index_path.as_deref(),
                build_options(&args.embedding)?,
            )
            .await?;
            print_value(args.root.json, &report)?;
        }
        Command::Remove(args) => {
            let report =
                remove_file(&args.root.root, args.root.index_path.as_deref(), &args.path).await?;
            print_value(args.root.json, &report)?;
        }
        Command::Status(args) => {
            let cwd = std::env::current_dir()?;
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
            print_value(args.root.json, &status)?;
        }
    }
    Ok(())
}

fn print_value(value_as_json: bool, value: &impl serde::Serialize) -> Result<()> {
    if value_as_json {
        println!("{}", serde_json::to_string_pretty(value)?);
    } else {
        println!("{}", serde_json::to_string_pretty(value)?);
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
