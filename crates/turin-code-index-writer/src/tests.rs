use anyhow::Result;
use tempfile::tempdir;
use turin_code_index::code_index_reader::{CodeSearchMode, CodeSearchRequest, CodebaseSelector};

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

#[test]
fn normalize_relative_path_rejects_absolute_parent_escape() -> Result<()> {
    let tmp = tempdir()?;
    let root = tmp.path().join("repo");
    std::fs::create_dir_all(&root)?;
    let root = std::fs::canonicalize(root)?;

    let err = super::fs::normalize_relative_path(&root, &root.join("../outside.rs"))
        .expect_err("absolute path with parent escape should fail");
    assert!(err.to_string().contains("escapes root"));
    Ok(())
}

#[test]
fn normalize_relative_path_normalizes_absolute_components() -> Result<()> {
    let tmp = tempdir()?;
    let root = tmp.path().join("repo");
    std::fs::create_dir_all(&root)?;
    let root = std::fs::canonicalize(root)?;

    let path = root.join("src/../lib.rs");
    let normalized = super::fs::normalize_relative_path(&root, &path)?;
    assert_eq!(normalized, "lib.rs");
    Ok(())
}

#[cfg(unix)]
#[test]
fn normalize_relative_path_rejects_non_utf8_paths() {
    use std::os::unix::ffi::OsStringExt;

    let path = PathBuf::from(std::ffi::OsString::from_vec(vec![
        b's', b'r', b'c', b'/', 0xff, b'.', b'r', b's',
    ]));
    let err = super::fs::normalize_relative_path(Path::new("/repo"), &path)
        .expect_err("index identity paths should be valid UTF-8");
    assert!(err.to_string().contains("not valid UTF-8"));
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
