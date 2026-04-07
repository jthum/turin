use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use turin_code_index::metadata::CodeIndexSemanticStatus;
use turin_code_index::support::{CODE_INDEX_SCHEMA_REVISION, open_index_connection};
use turin_types::layout::default_code_index_db_for_workspace;

pub mod embeddings;

mod chunking;
mod fs;
mod store;

use chunking::build_chunks;
use embeddings::CodeEmbeddingProvider;
use fs::{collect_indexable_files, normalize_relative_path, read_indexable_file};
use store::{
    current_timestamp, delete_indexed_file, init_schema, insert_chunks, load_index_summary,
    load_indexed_files, should_recreate_index, upsert_indexed_file, write_index_meta,
};

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CodeIndexWriteCapabilities {
    pub lexical: bool,
    pub semantic: bool,
    pub hybrid: bool,
    pub languages: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CodeIndexBuildReport {
    pub root: String,
    pub index_path: String,
    pub schema_revision: i64,
    pub updated_at: String,
    pub codebase_id: Option<String>,
    pub capabilities: CodeIndexWriteCapabilities,
    pub semantic: CodeIndexSemanticStatus,
    pub files_indexed: u64,
    pub chunks_indexed: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CodeIndexRemoveReport {
    pub root: String,
    pub index_path: String,
    pub path: String,
    pub removed_chunks: u64,
    pub updated_at: String,
}

#[derive(Debug, Clone)]
struct CodeChunkRecord {
    chunk_key: String,
    path: String,
    language: String,
    kind: String,
    name: String,
    signature: Option<String>,
    snippet: String,
    search_text: String,
    embedding: Option<Vec<f32>>,
    start_line: i64,
    end_line: i64,
}

#[derive(Debug, Clone)]
struct IndexedFileState {
    content_hash: String,
    language: String,
    embedding_key: String,
    embedding_dimensions: usize,
    chunk_count: u64,
}

#[derive(Debug, Clone)]
struct IndexableFileContent {
    path: String,
    language: String,
    content_hash: String,
    content: String,
}

#[derive(Debug, Clone)]
struct CodeIndexSummary {
    capabilities: CodeIndexWriteCapabilities,
    semantic: CodeIndexSemanticStatus,
    files_indexed: u64,
    chunks_indexed: u64,
}

#[derive(Clone, Default)]
pub struct CodeIndexBuildOptions {
    pub embedding_provider: Option<Arc<dyn CodeEmbeddingProvider>>,
}

pub async fn build_index(root: &Path, index_path: Option<&Path>) -> Result<CodeIndexBuildReport> {
    build_index_with_options(root, index_path, CodeIndexBuildOptions::default()).await
}

pub async fn build_index_with_options(
    root: &Path,
    index_path: Option<&Path>,
    options: CodeIndexBuildOptions,
) -> Result<CodeIndexBuildReport> {
    build_index_internal(root, index_path, false, &options).await
}

pub async fn rebuild_index(root: &Path, index_path: Option<&Path>) -> Result<CodeIndexBuildReport> {
    rebuild_index_with_options(root, index_path, CodeIndexBuildOptions::default()).await
}

pub async fn rebuild_index_with_options(
    root: &Path,
    index_path: Option<&Path>,
    options: CodeIndexBuildOptions,
) -> Result<CodeIndexBuildReport> {
    build_index_internal(root, index_path, true, &options).await
}

async fn build_index_internal(
    root: &Path,
    index_path: Option<&Path>,
    force_rebuild: bool,
    options: &CodeIndexBuildOptions,
) -> Result<CodeIndexBuildReport> {
    let root = std::fs::canonicalize(root)
        .with_context(|| format!("root '{}' does not exist", root.display()))?;
    let index_path = resolve_index_path(&root, index_path)?;
    if let Some(parent) = index_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create '{}'", parent.display()))?;
    }

    if force_rebuild || should_recreate_index(&index_path).await? {
        remove_index_db_if_present(&index_path)?;
    }

    let (_db, conn) = open_index_connection(&index_path).await?;
    init_schema(&conn).await?;

    let run_updated_at = current_timestamp(&conn).await?;
    let existing_files = load_indexed_files(&conn).await?;
    let embedding_key = options
        .embedding_provider
        .as_ref()
        .map(|provider| provider.config_key())
        .unwrap_or_else(|| "none".to_string());
    let embedding_dimensions = options
        .embedding_provider
        .as_ref()
        .map(|provider| provider.dimensions())
        .unwrap_or(0);
    let mut seen_paths = HashSet::new();

    for file in collect_indexable_files(&root)? {
        let relative_path = normalize_relative_path(&root, &file)?;
        seen_paths.insert(relative_path.clone());

        let Some(source) = read_indexable_file(relative_path.clone(), &file)? else {
            delete_indexed_file(&conn, &relative_path).await?;
            continue;
        };

        let unchanged = existing_files.get(&source.path).is_some_and(|existing| {
            existing.content_hash == source.content_hash
                && existing.language == source.language
                && existing.embedding_key == embedding_key
                && existing.embedding_dimensions == embedding_dimensions
                && existing.chunk_count > 0
        });
        if unchanged {
            continue;
        }

        delete_indexed_file(&conn, &source.path).await?;
        let mut chunks = build_chunks(&source.path, &source.language, &source.content);
        if chunks.is_empty() {
            continue;
        }

        if let Some(provider) = &options.embedding_provider {
            let embedding_inputs = chunks
                .iter()
                .map(|chunk| chunk.search_text.clone())
                .collect::<Vec<_>>();
            let embeddings = provider
                .embed_batch(&embedding_inputs)
                .await
                .with_context(|| format!("failed to generate embeddings for '{}'", source.path))?;
            if embeddings.len() != chunks.len() {
                bail!(
                    "embedding provider returned {} vectors for {} chunks in '{}'",
                    embeddings.len(),
                    chunks.len(),
                    source.path
                );
            }
            for (chunk, embedding) in chunks.iter_mut().zip(embeddings) {
                chunk.embedding = Some(embedding);
            }
        }

        insert_chunks(&conn, &chunks).await?;
        upsert_indexed_file(
            &conn,
            &source,
            chunks.len() as u64,
            &embedding_key,
            embedding_dimensions,
            &run_updated_at,
        )
        .await?;
    }

    for stale_path in existing_files.keys() {
        if !seen_paths.contains(stale_path) {
            delete_indexed_file(&conn, stale_path).await?;
        }
    }

    let summary = load_index_summary(&conn).await?;
    let codebase_id = codebase_id_for(&root);
    write_index_meta(&conn, &root, &run_updated_at, &codebase_id, &summary).await?;

    Ok(CodeIndexBuildReport {
        root: root.to_string_lossy().to_string(),
        index_path: index_path.to_string_lossy().to_string(),
        schema_revision: CODE_INDEX_SCHEMA_REVISION,
        updated_at: run_updated_at,
        codebase_id,
        capabilities: summary.capabilities,
        semantic: summary.semantic,
        files_indexed: summary.files_indexed,
        chunks_indexed: summary.chunks_indexed,
    })
}

pub async fn remove_file(
    root: &Path,
    index_path: Option<&Path>,
    file_path: &Path,
) -> Result<CodeIndexRemoveReport> {
    let root = std::fs::canonicalize(root)
        .with_context(|| format!("root '{}' does not exist", root.display()))?;
    let index_path = resolve_index_path(&root, index_path)?;
    let relative_path = normalize_relative_path(&root, file_path)?;
    if !index_path.exists() {
        bail!("index db not found at '{}'", index_path.display());
    }

    let (_db, conn) = open_index_connection(&index_path).await?;
    let removed_chunks = delete_indexed_file(&conn, &relative_path).await?;
    let updated_at = current_timestamp(&conn).await?;
    let summary = load_index_summary(&conn).await?;
    let codebase_id = codebase_id_for(&root);
    write_index_meta(&conn, &root, &updated_at, &codebase_id, &summary).await?;

    Ok(CodeIndexRemoveReport {
        root: root.to_string_lossy().to_string(),
        index_path: index_path.to_string_lossy().to_string(),
        path: relative_path,
        removed_chunks,
        updated_at,
    })
}

fn resolve_index_path(root: &Path, index_path: Option<&Path>) -> Result<PathBuf> {
    match index_path {
        Some(path) if path.is_absolute() => Ok(path.to_path_buf()),
        Some(path) => Ok(root.join(path)),
        None => Ok(default_code_index_db_for_workspace(root)),
    }
}

fn codebase_id_for(root: &Path) -> Option<String> {
    root.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .map(|name| format!("{name}-main"))
}

fn remove_index_db_if_present(index_path: &Path) -> Result<()> {
    if index_path.exists() {
        std::fs::remove_file(index_path)
            .with_context(|| format!("failed to remove '{}'", index_path.display()))?;
    }
    if let Some(wal_path) = wal_path_for(index_path)
        && wal_path.exists()
    {
        std::fs::remove_file(&wal_path)
            .with_context(|| format!("failed to remove '{}'", wal_path.display()))?;
    }
    if let Some(shm_path) = shm_path_for(index_path)
        && shm_path.exists()
    {
        std::fs::remove_file(&shm_path)
            .with_context(|| format!("failed to remove '{}'", shm_path.display()))?;
    }
    Ok(())
}

fn wal_path_for(index_path: &Path) -> Option<PathBuf> {
    let file_name = index_path.file_name()?.to_string_lossy().to_string();
    Some(index_path.with_file_name(format!("{file_name}-wal")))
}

fn shm_path_for(index_path: &Path) -> Option<PathBuf> {
    let file_name = index_path.file_name()?.to_string_lossy().to_string();
    Some(index_path.with_file_name(format!("{file_name}-shm")))
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use tempfile::tempdir;
    use turin_code_index::code_index_reader::{
        CodeSearchMode, CodeSearchRequest, CodebaseSelector,
    };

    use super::chunking::{CHUNK_LINES, build_chunks};
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn builds_indexes_searches_and_removes_files() -> Result<()> {
        let tmp = tempdir()?;
        let root = tmp.path().join("repo");
        let src = root.join("src");
        std::fs::create_dir_all(&src)?;
        std::fs::write(
            src.join("governance.rs"),
            "pub fn capability_decision(capability: &str) -> bool {\n  capability == \"ok\"\n}\n",
        )?;
        std::fs::write(src.join("helpers.rs"), "pub fn helper() -> bool { true }\n")?;

        let report = build_index(&root, None).await?;
        assert_eq!(report.files_indexed, 2);
        assert!(report.chunks_indexed >= 2);
        assert_eq!(report.codebase_id.as_deref(), Some("repo-main"));
        assert!(report.capabilities.lexical);
        assert!(!report.capabilities.semantic);
        assert_eq!(report.semantic.embedded_chunks, 0);

        let rows = lexical_search(tmp.path(), "capability_decision").await?;
        assert_eq!(rows[0].name, "capability_decision");

        let removed = remove_file(&root, None, Path::new("src/governance.rs")).await?;
        assert!(removed.removed_chunks > 0);
        let rows_after_remove = lexical_search(tmp.path(), "capability_decision").await?;
        assert!(rows_after_remove.is_empty());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn incremental_rebuild_drops_removed_files_and_updates_changed_ones() -> Result<()> {
        let tmp = tempdir()?;
        let root = tmp.path().join("repo");
        let src = root.join("src");
        std::fs::create_dir_all(&src)?;
        std::fs::write(
            src.join("alpha.rs"),
            "pub fn alpha_rule() -> bool { true }\n",
        )?;
        std::fs::write(src.join("beta.rs"), "pub fn beta_rule() -> bool { true }\n")?;
        std::fs::write(
            src.join("gamma.rs"),
            "pub fn gamma_rule() -> bool { true }\n",
        )?;

        let first = build_index(&root, None).await?;
        assert_eq!(first.files_indexed, 3);

        std::fs::write(
            src.join("alpha.rs"),
            "pub fn alpha_review() -> bool { true }\n",
        )?;
        std::fs::remove_file(src.join("gamma.rs"))?;

        let second = build_index(&root, None).await?;
        assert_eq!(second.files_indexed, 2);
        assert!(lexical_search(tmp.path(), "alpha_rule").await?.is_empty());
        assert!(!lexical_search(tmp.path(), "alpha_review").await?.is_empty());
        assert!(!lexical_search(tmp.path(), "beta_rule").await?.is_empty());
        assert!(lexical_search(tmp.path(), "gamma_rule").await?.is_empty());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn indexable_file_filters_skip_ignored_targets() -> Result<()> {
        let tmp = tempdir()?;
        let root = tmp.path().join("repo");
        let src = root.join("src");
        let node_modules = root.join("node_modules");
        let vendor = root.join("vendor");
        let target = root.join("target");
        std::fs::create_dir_all(&src)?;
        std::fs::create_dir_all(&node_modules)?;
        std::fs::create_dir_all(&vendor)?;
        std::fs::create_dir_all(&target)?;

        std::fs::write(
            src.join("main.rs"),
            "pub fn indexable_symbol() -> bool { true }\n",
        )?;
        std::fs::write(
            node_modules.join("ignored.ts"),
            "export function ignored_symbol() { return true; }\n",
        )?;
        std::fs::write(
            vendor.join("vendor.rs"),
            "pub fn vendor_symbol() -> bool { true }\n",
        )?;
        std::fs::write(
            target.join("artifact.lua"),
            "function moduleSymbol() return true end\n",
        )?;

        let report = build_index(&root, None).await?;
        assert_eq!(report.files_indexed, 1);
        assert!(report.capabilities.languages.contains(&"rust".to_string()));
        assert!(
            !lexical_search(tmp.path(), "indexable_symbol")
                .await?
                .is_empty()
        );
        assert!(
            lexical_search(tmp.path(), "ignored_symbol")
                .await?
                .is_empty()
        );
        assert!(
            lexical_search(tmp.path(), "vendor_symbol")
                .await?
                .is_empty()
        );
        assert!(lexical_search(tmp.path(), "moduleSymbol").await?.is_empty());
        Ok(())
    }

    #[test]
    fn chunk_builder_discovers_symbols_and_respects_windowing() {
        let content = "use std::fmt;\n\npub fn alpha() {}\n\npub fn beta() {}\n";
        let chunks = build_chunks("src/lib.rs", "rust", content);
        assert!(chunks.iter().any(|chunk| {
            chunk.name == "alpha" && chunk.signature.as_deref() == Some("pub fn alpha() {}")
        }));
        assert!(chunks.iter().any(|chunk| {
            chunk.name == "beta" && chunk.signature.as_deref() == Some("pub fn beta() {}")
        }));
    }

    #[test]
    fn oversized_symbol_chunks_keep_signature_metadata() {
        let mut content = String::from("pub fn oversized() {\n");
        for _ in 0..(CHUNK_LINES * 2) {
            content.push_str("    let value = 1;\n");
        }
        content.push_str("}\n");

        let chunks = build_chunks("src/lib.rs", "rust", &content);
        assert!(chunks.len() >= 2);
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.signature.as_deref() == Some("pub fn oversized() {"))
        );
    }

    async fn lexical_search(
        workspace_root: &Path,
        query: &str,
    ) -> Result<Vec<turin_code_index::code_index_reader::CodeSearchRow>> {
        turin_code_index::code_index_reader::search(
            workspace_root,
            CodebaseSelector {
                root: "repo".to_string(),
                index_path: None,
            },
            CodeSearchMode::Lexical,
            query,
            &CodeSearchRequest {
                limit: 10,
                languages: Vec::new(),
                kinds: Vec::new(),
                min_score: 0.0,
                strict: false,
                trace: false,
            },
            None,
        )
        .await
    }
}
