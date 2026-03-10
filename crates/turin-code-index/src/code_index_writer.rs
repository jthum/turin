use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::shared::{CODE_INDEX_SCHEMA_REVISION, open_index_connection};

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
    pub capabilities: CodeIndexWriteCapabilities,
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
    start_line: i64,
    end_line: i64,
}

#[derive(Debug, Clone)]
struct IndexedFileState {
    content_hash: String,
    language: String,
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
    files_indexed: u64,
    chunks_indexed: u64,
}

mod chunking;
mod fs;
mod store;

use chunking::build_chunks;
use fs::{collect_indexable_files, normalize_relative_path, read_indexable_file};
use store::{
    current_timestamp, delete_indexed_file, init_schema, insert_chunks, load_index_summary,
    load_indexed_files, should_recreate_index, upsert_indexed_file, write_index_meta,
};

pub async fn build_index(root: &Path, index_path: Option<&Path>) -> Result<CodeIndexBuildReport> {
    build_index_internal(root, index_path, false).await
}

pub async fn rebuild_index(root: &Path, index_path: Option<&Path>) -> Result<CodeIndexBuildReport> {
    build_index_internal(root, index_path, true).await
}

async fn build_index_internal(
    root: &Path,
    index_path: Option<&Path>,
    force_rebuild: bool,
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
                && existing.chunk_count > 0
        });
        if unchanged {
            continue;
        }

        delete_indexed_file(&conn, &source.path).await?;
        let chunks = build_chunks(&source.path, &source.language, &source.content);
        if chunks.is_empty() {
            continue;
        }

        insert_chunks(&conn, &chunks).await?;
        upsert_indexed_file(&conn, &source, chunks.len() as u64, &run_updated_at).await?;
    }

    for stale_path in existing_files.keys() {
        if !seen_paths.contains(stale_path) {
            delete_indexed_file(&conn, stale_path).await?;
        }
    }

    let summary = load_index_summary(&conn).await?;
    write_index_meta(&conn, &root, &run_updated_at, &summary.capabilities).await?;

    Ok(CodeIndexBuildReport {
        root: root.to_string_lossy().to_string(),
        index_path: index_path.to_string_lossy().to_string(),
        schema_revision: CODE_INDEX_SCHEMA_REVISION,
        updated_at: run_updated_at,
        capabilities: summary.capabilities,
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
    write_index_meta(&conn, &root, &updated_at, &summary.capabilities).await?;

    Ok(CodeIndexRemoveReport {
        root: root.to_string_lossy().to_string(),
        index_path: index_path.to_string_lossy().to_string(),
        path: relative_path,
        removed_chunks,
        updated_at,
    })
}

fn resolve_index_path(root: &Path, index_path: Option<&Path>) -> Result<PathBuf> {
    Ok(match index_path {
        Some(path) if path.is_absolute() => path.to_path_buf(),
        Some(path) => root.join(path),
        None => root.join(".turin").join("codebase.db"),
    })
}

fn remove_index_db_if_present(index_path: &Path) -> Result<()> {
    if index_path.exists() {
        std::fs::remove_file(index_path)
            .with_context(|| format!("failed to replace '{}'", index_path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_index_reader::{CodeSearchMode, CodeSearchRequest, CodebaseSelector};
    use crate::code_index_writer::chunking::{CHUNK_LINES, build_chunks};
    use std::time::Duration;
    use tempfile::tempdir;

    #[tokio::test]
    async fn build_index_generates_runtime_searchable_db() -> Result<()> {
        let tmp = tempdir()?;
        let root = tmp.path().join("repo");
        let src = root.join("src");
        std::fs::create_dir_all(&src)?;
        std::fs::write(
            src.join("governance.rs"),
            r#"
pub fn capability_decision(capability: &str) -> bool {
    capability == "runtime.code.search.lexical"
}
"#,
        )?;
        std::fs::write(
            root.join("main.lua"),
            r#"
function on_turn_prepare(ctx)
  return ALLOW
end
"#,
        )?;

        let report = build_index(&root, None).await?;
        assert_eq!(report.files_indexed, 2);
        assert!(report.chunks_indexed >= 2);
        assert!(report.capabilities.lexical);
        assert!(!report.capabilities.semantic);

        let rows = lexical_search(tmp.path(), "capability_decision").await?;
        assert!(!rows.is_empty());
        assert_eq!(rows[0].name, "capability_decision");
        assert!(rows[0].score > 0.0);

        let removed = remove_file(&root, None, Path::new("src/governance.rs")).await?;
        assert!(removed.removed_chunks >= 1);

        let rows_after_remove = lexical_search(tmp.path(), "capability_decision").await?;
        assert!(rows_after_remove.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn build_index_refreshes_incrementally() -> Result<()> {
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

        let alpha_before = indexed_file_updated_at(&root, "src/alpha.rs").await?;
        let beta_before = indexed_file_updated_at(&root, "src/beta.rs").await?;
        assert!(alpha_before.is_some());
        assert!(beta_before.is_some());

        std::thread::sleep(Duration::from_millis(20));
        std::fs::write(
            src.join("alpha.rs"),
            "pub fn alpha_review() -> bool { true }\n",
        )?;
        std::fs::remove_file(src.join("gamma.rs"))?;

        let second = build_index(&root, None).await?;
        assert_eq!(second.files_indexed, 2);

        let alpha_after = indexed_file_updated_at(&root, "src/alpha.rs").await?;
        let beta_after = indexed_file_updated_at(&root, "src/beta.rs").await?;
        let gamma_after = indexed_file_updated_at(&root, "src/gamma.rs").await?;
        assert_ne!(alpha_before, alpha_after);
        assert_eq!(beta_before, beta_after);
        assert!(gamma_after.is_none());

        assert!(lexical_search(tmp.path(), "alpha_rule").await?.is_empty());
        assert!(!lexical_search(tmp.path(), "alpha_review").await?.is_empty());
        assert!(!lexical_search(tmp.path(), "beta_rule").await?.is_empty());
        assert!(lexical_search(tmp.path(), "gamma_rule").await?.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn build_index_respects_gitignore_and_default_skip_dirs() -> Result<()> {
        let tmp = tempdir()?;
        let root = tmp.path().join("repo");
        let src = root.join("src");
        let vendor = root.join("vendor");
        let node_modules = root.join("node_modules").join("pkg");
        std::fs::create_dir_all(&src)?;
        std::fs::create_dir_all(&vendor)?;
        std::fs::create_dir_all(&node_modules)?;
        std::fs::write(root.join(".gitignore"), "ignored.rs\n")?;
        std::fs::write(
            src.join("search.rs"),
            "pub fn indexable_symbol() -> bool { true }\n",
        )?;
        std::fs::write(
            root.join("ignored.rs"),
            "pub fn ignored_symbol() -> bool { true }\n",
        )?;
        std::fs::write(
            vendor.join("vendor.rs"),
            "pub fn vendor_symbol() -> bool { true }\n",
        )?;
        std::fs::write(
            node_modules.join("module.js"),
            "export function moduleSymbol() { return true; }\n",
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
        assert!(
            indexed_file_updated_at(&root, "ignored.rs")
                .await?
                .is_none()
        );
        assert!(
            indexed_file_updated_at(&root, "vendor/vendor.rs")
                .await?
                .is_none()
        );

        Ok(())
    }

    #[test]
    fn build_chunks_anchor_chunks_to_symbol_boundaries() {
        let chunks = build_chunks(
            "src/lib.rs",
            "rust",
            "use std::fmt;\n\npub fn alpha() {}\n\npub fn beta() {}\n",
        );

        assert!(chunks.iter().any(|chunk| {
            chunk.name == "alpha"
                && chunk.signature.as_deref() == Some("pub fn alpha() {}")
                && chunk.start_line == 3
        }));
        assert!(chunks.iter().any(|chunk| {
            chunk.name == "beta"
                && chunk.signature.as_deref() == Some("pub fn beta() {}")
                && chunk.start_line == 5
        }));
    }

    #[test]
    fn build_chunks_split_large_symbols_without_losing_symbol_identity() {
        let mut content = String::from("pub fn oversized() {\n");
        for _ in 0..96 {
            content.push_str("    println!(\"x\");\n");
        }
        content.push_str("}\n");

        let symbol_chunks = build_chunks("src/lib.rs", "rust", &content)
            .into_iter()
            .filter(|chunk| chunk.name == "oversized")
            .collect::<Vec<_>>();

        assert!(symbol_chunks.len() >= 2);
        assert!(
            symbol_chunks
                .iter()
                .all(|chunk| chunk.signature.as_deref() == Some("pub fn oversized() {"))
        );
        assert_eq!(symbol_chunks[0].start_line, 1);
        assert!(
            symbol_chunks
                .last()
                .is_some_and(|chunk| chunk.end_line > CHUNK_LINES as i64)
        );
    }

    async fn lexical_search(
        workspace_root: &Path,
        query: &str,
    ) -> Result<Vec<crate::code_index_reader::CodeSearchRow>> {
        crate::code_index_reader::search(
            workspace_root,
            CodebaseSelector {
                root: "repo".to_string(),
                index_path: None,
            },
            CodeSearchMode::Lexical,
            query,
            &CodeSearchRequest {
                limit: 5,
                ..CodeSearchRequest::default()
            },
        )
        .await
    }

    async fn indexed_file_updated_at(root: &Path, relative_path: &str) -> Result<Option<String>> {
        let index_path = resolve_index_path(root, None)?;
        let (_db, conn) = open_index_connection(&index_path).await?;
        let mut rows = conn
            .query(
                "SELECT updated_at FROM indexed_files WHERE path = ?1",
                turso::params![relative_path.to_string()],
            )
            .await?;
        match rows.next().await? {
            Some(row) => Ok(Some(row.get::<String>(0)?)),
            None => Ok(None),
        }
    }
}
