use anyhow::Result;
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::tempdir;
use turin_code_index::code_index_reader::{
    CodeSearchMode, CodeSearchRequest, CodebaseSelector, search, status,
};
use turin_code_index_writer::embeddings::{CODE_INDEX_VECTOR_DIM, CodeEmbeddingProvider};
use turin_code_index_writer::{CodeIndexBuildOptions, build_index_with_options};

struct KeywordEmbeddingProvider;

#[async_trait]
impl CodeEmbeddingProvider for KeywordEmbeddingProvider {
    fn config_key(&self) -> String {
        "test:keyword".to_string()
    }

    fn dimensions(&self) -> usize {
        CODE_INDEX_VECTOR_DIM
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let text = text.to_ascii_lowercase();
        let mut vector = vec![0.0; CODE_INDEX_VECTOR_DIM];
        let keywords = [
            "runtime",
            "code",
            "search",
            "namespace",
            "cache",
            "memory",
            "session",
            "dx",
        ];
        for (index, keyword) in keywords.iter().enumerate() {
            if text.contains(keyword) {
                vector[index] = 1.0;
            }
        }
        if vector.iter().all(|value| *value == 0.0) {
            vector[CODE_INDEX_VECTOR_DIM - 1] = 0.01;
        }
        Ok(vector)
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "indexes a real repo tree to smoke-test retrieval quality"]
async fn real_repo_smoke_runtime_harness_index() -> Result<()> {
    let repo_root = repo_root().canonicalize()?;
    let harness_root = repo_root.join("src").join("harness");
    let index_dir = tempdir()?;
    let index_path = index_dir.path().join("harness-smoke.db");
    let provider = Arc::new(KeywordEmbeddingProvider);

    let report = build_index_with_options(
        &harness_root,
        Some(&index_path),
        CodeIndexBuildOptions {
            embedding_provider: Some(provider.clone()),
        },
    )
    .await?;

    assert!(report.capabilities.lexical);
    assert!(report.capabilities.semantic);
    assert!(report.capabilities.hybrid);
    assert!(report.files_indexed > 10);
    assert!(report.chunks_indexed > 10);
    assert_eq!(report.codebase_id.as_deref(), Some("harness-main"));
    assert!(report.semantic.embedded_chunks > 0);
    assert_eq!(
        report.semantic.embedding_dimensions,
        Some(CODE_INDEX_VECTOR_DIM)
    );
    assert_eq!(
        report.semantic.embedding_key.as_deref(),
        Some("test:keyword")
    );
    assert_eq!(
        report.semantic.vector_format,
        Some(turin_code_index::metadata::CodeIndexVectorFormat::Float8)
    );

    let selector = CodebaseSelector {
        root: "src/harness".to_string(),
        index_path: Some(index_path.to_string_lossy().to_string()),
    };

    let index_status = status(&repo_root, selector.clone()).await?;
    assert_eq!(index_status.codebase_id.as_deref(), Some("harness-main"));
    assert!(index_status.capabilities.semantic);
    assert!(index_status.capabilities.hybrid);
    assert!(index_status.semantic.embedded_chunks > 0);
    assert_eq!(
        index_status.semantic.embedding_dimensions,
        Some(CODE_INDEX_VECTOR_DIM)
    );
    assert_eq!(
        index_status.semantic.embedding_key.as_deref(),
        Some("test:keyword")
    );
    assert_eq!(
        index_status.semantic.vector_format,
        Some(turin_code_index::metadata::CodeIndexVectorFormat::Float8)
    );
    assert_eq!(
        Path::new(&index_status.root).canonicalize()?,
        harness_root.canonicalize()?
    );

    let lexical_rows = search(
        &repo_root,
        selector.clone(),
        CodeSearchMode::Lexical,
        "register_runtime_code_namespace",
        &CodeSearchRequest {
            limit: 5,
            languages: vec!["rust".to_string()],
            ..CodeSearchRequest::default()
        },
        None,
    )
    .await?;
    assert!(!lexical_rows.is_empty());
    assert_eq!(lexical_rows[0].name, "register_runtime_code_namespace");
    assert!(
        lexical_rows[0].path.ends_with("stdlib/runtime_code.rs"),
        "unexpected lexical top hit path: {}",
        lexical_rows[0].path
    );

    let lexical_phrase_rows = search(
        &repo_root,
        selector.clone(),
        CodeSearchMode::Lexical,
        "runtime code search namespace",
        &CodeSearchRequest {
            limit: 5,
            languages: vec!["rust".to_string()],
            ..CodeSearchRequest::default()
        },
        None,
    )
    .await?;
    assert!(
        lexical_phrase_rows
            .iter()
            .take(3)
            .any(|row| row.name == "register_runtime_code_namespace"),
        "expected runtime code namespace in top lexical phrase hits, got {:?}",
        lexical_phrase_rows
            .iter()
            .take(3)
            .map(|row| row.name.as_str())
            .collect::<Vec<_>>()
    );

    let path_rows = search(
        &repo_root,
        selector.clone(),
        CodeSearchMode::Lexical,
        "stdlib/runtime_code.rs",
        &CodeSearchRequest {
            limit: 5,
            languages: vec!["rust".to_string()],
            trace: true,
            ..CodeSearchRequest::default()
        },
        None,
    )
    .await?;
    assert!(!path_rows.is_empty());
    assert!(
        path_rows[0].path.ends_with("stdlib/runtime_code.rs"),
        "expected path-oriented query to prioritize runtime_code.rs, got {}",
        path_rows[0].path
    );
    assert_eq!(
        path_rows[0]
            .trace
            .as_ref()
            .and_then(|trace| trace.requested_mode.as_deref()),
        Some("lexical")
    );

    let dx_path_rows = search(
        &repo_root,
        selector.clone(),
        CodeSearchMode::Lexical,
        "dx/code_cache.rs",
        &CodeSearchRequest {
            limit: 5,
            languages: vec!["rust".to_string()],
            ..CodeSearchRequest::default()
        },
        None,
    )
    .await?;
    assert!(!dx_path_rows.is_empty());
    assert!(
        dx_path_rows[0].path.ends_with("dx/code_cache.rs"),
        "expected path-oriented query to prioritize dx/code_cache.rs, got {}",
        dx_path_rows[0].path
    );

    let dx_phrase_rows = search(
        &repo_root,
        selector.clone(),
        CodeSearchMode::Lexical,
        "remember recall helpers",
        &CodeSearchRequest {
            limit: 5,
            languages: vec!["rust".to_string()],
            ..CodeSearchRequest::default()
        },
        None,
    )
    .await?;
    assert!(
        dx_phrase_rows
            .iter()
            .take(3)
            .any(|row| row.name == "register_data_globals" || row.name == "register_scope_helpers"),
        "expected remember/recall helpers in top lexical hits, got {:?}",
        dx_phrase_rows
            .iter()
            .take(3)
            .map(|row| row.name.as_str())
            .collect::<Vec<_>>()
    );

    let hybrid_query = "runtime code search namespace";
    let hybrid_vector = provider.embed(hybrid_query).await?;
    let hybrid_rows = search(
        &repo_root,
        selector,
        CodeSearchMode::Hybrid,
        hybrid_query,
        &CodeSearchRequest {
            limit: 5,
            languages: vec!["rust".to_string()],
            ..CodeSearchRequest::default()
        },
        Some(&hybrid_vector),
    )
    .await?;
    assert!(!hybrid_rows.is_empty());
    assert!(
        hybrid_rows
            .iter()
            .take(3)
            .any(|row| row.name == "register_runtime_code_namespace"),
        "expected runtime code namespace in top hybrid hits, got {:?}",
        hybrid_rows
            .iter()
            .take(3)
            .map(|row| row.name.as_str())
            .collect::<Vec<_>>()
    );
    assert!(
        hybrid_rows
            .iter()
            .take(3)
            .any(|row| row.semantic_score.is_some()),
        "expected semantic contribution in top hybrid hits"
    );

    Ok(())
}
