use super::*;
use tempfile::tempdir;
use turin_code_index_writer::embeddings::CODE_INDEX_VECTOR_DIM;

use crate::support::CODE_INDEX_SCHEMA_REVISION;

fn vector_blob(fill: f32) -> Vec<u8> {
    let mut blob = Vec::with_capacity(CODE_INDEX_VECTOR_DIM * std::mem::size_of::<f32>());
    for _ in 0..CODE_INDEX_VECTOR_DIM {
        blob.extend_from_slice(&fill.to_le_bytes());
    }
    blob
}

fn sparse_vector_blob() -> Vec<u8> {
    let mut blob = Vec::with_capacity(CODE_INDEX_VECTOR_DIM * std::mem::size_of::<f32>());
    blob.extend_from_slice(&1.0_f32.to_le_bytes());
    for _ in 1..CODE_INDEX_VECTOR_DIM {
        blob.extend_from_slice(&0.0_f32.to_le_bytes());
    }
    blob
}

async fn create_synthetic_code_index(
    root: &Path,
    semantic: bool,
    hybrid: bool,
) -> Result<std::path::PathBuf> {
    let index_dir = turin_types::layout::default_layout_root_for_workspace(root);
    std::fs::create_dir_all(&index_dir)?;
    let index_path = turin_types::layout::default_code_index_db_for_workspace(root);
    let db = turso::Builder::new_local(index_path.to_str().unwrap())
        .experimental_index_method(true)
        .build()
        .await?;
    let conn = db.connect()?;
    let root_path = std::fs::canonicalize(root)?;
    let capabilities = serde_json::json!({
        "lexical": true,
        "semantic": semantic,
        "hybrid": hybrid,
        "languages": ["rust", "lua"],
    })
    .to_string();
    conn.execute_batch(
        r#"
CREATE TABLE index_meta (
schema_revision INTEGER NOT NULL,
root_path TEXT NOT NULL,
updated_at TEXT NOT NULL,
capabilities TEXT NOT NULL,
codebase_id TEXT,
embedding_key TEXT,
embedding_dimensions INTEGER,
embedding_vector_format TEXT,
embedded_chunks INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE code_chunks (
chunk_key TEXT PRIMARY KEY,
path TEXT NOT NULL,
language TEXT NOT NULL,
kind TEXT NOT NULL,
name TEXT NOT NULL,
signature TEXT,
snippet TEXT NOT NULL,
search_text TEXT NOT NULL,
embedding BLOB,
start_line INTEGER NOT NULL,
end_line INTEGER NOT NULL,
lexical_score REAL NOT NULL,
semantic_score REAL
);
CREATE INDEX idx_code_chunks_search_fts ON code_chunks USING fts(search_text);
"#,
    )
    .await?;
    conn.execute(
        "INSERT INTO index_meta (schema_revision, root_path, updated_at, capabilities, codebase_id, embedding_key, embedding_dimensions, embedding_vector_format, embedded_chunks) VALUES (?1, ?2, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?3, ?4, ?5, ?6, ?7, ?8)",
        turso::params![
            CODE_INDEX_SCHEMA_REVISION,
            root_path.to_string_lossy().to_string(),
            capabilities,
            "repo-main",
            if semantic { Some("test:synthetic".to_string()) } else { None },
            if semantic { Some(CODE_INDEX_VECTOR_DIM as i64) } else { None },
            if semantic { Some("float8".to_string()) } else { None },
            if semantic { 2_i64 } else { 0_i64 }
        ],
    )
    .await?;
    conn.execute(
        "INSERT INTO code_chunks (chunk_key, path, language, kind, name, signature, snippet, search_text, embedding, start_line, end_line, lexical_score, semantic_score)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, vector8(?9), ?10, ?11, ?12, ?13)",
        turso::params![
            "chunk_rust",
            "src/kernel/governance.rs",
            "rust",
            "function",
            "capability_decision",
            "fn capability_decision(...)",
            "pub fn capability_decision(capability: &str) -> CapabilityDecision",
            "src/kernel/governance.rs\ncapability_decision\nfn capability_decision(...)\npub fn capability_decision(capability: &str) -> CapabilityDecision",
            vector_blob(0.001_f32),
            101_i64,
            132_i64,
            0.91_f64,
            0.82_f64
        ],
    )
    .await?;
    conn.execute(
        "INSERT INTO code_chunks (chunk_key, path, language, kind, name, signature, snippet, search_text, embedding, start_line, end_line, lexical_score, semantic_score)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, vector8(?9), ?10, ?11, ?12, ?13)",
        turso::params![
            "chunk_lua",
            "harnesses/runtime_graph.lua",
            "lua",
            "function",
            "on_turn_prepare",
            "function on_turn_prepare(ctx)",
            "function on_turn_prepare(ctx) local rows = runtime.graph.edges() end",
            "harnesses/runtime_graph.lua\non_turn_prepare\nfunction on_turn_prepare(ctx)\nfunction on_turn_prepare(ctx) local rows = runtime.graph.edges() end",
            sparse_vector_blob(),
            1_i64,
            18_i64,
            0.67_f64,
            0.48_f64
        ],
    )
    .await?;
    conn.execute_batch(
        r#"
CREATE VIEW v_code_lexical AS
SELECT
chunk_key,
path,
language,
kind,
name,
signature,
snippet,
start_line,
end_line,
lexical_score AS score,
lexical_score,
NULL AS semantic_score,
search_text
FROM code_chunks;
"#,
    )
    .await?;
    if semantic {
        conn.execute_batch(
            r#"
CREATE VIEW v_code_semantic AS
SELECT
chunk_key,
path,
language,
kind,
name,
signature,
snippet,
start_line,
end_line,
semantic_score AS score,
NULL AS lexical_score,
semantic_score,
embedding
FROM code_chunks
WHERE semantic_score IS NOT NULL;
"#,
        )
        .await?;
    }
    if hybrid {
        conn.execute_batch(
            r#"
CREATE VIEW v_code_hybrid AS
SELECT
chunk_key,
path,
language,
kind,
name,
signature,
snippet,
start_line,
end_line,
((lexical_score * 0.5) + (COALESCE(semantic_score, 0.0) * 0.5)) AS score,
lexical_score,
semantic_score,
search_text,
embedding
FROM code_chunks;
"#,
        )
        .await?;
    }
    Ok(index_path)
}

#[tokio::test]
async fn status_and_search_follow_contract_and_fallbacks() -> Result<()> {
    let tmp = tempdir()?;
    let root = tmp.path().join("repo");
    std::fs::create_dir_all(&root)?;
    create_synthetic_code_index(&root, true, true).await?;

    let status = status(
        tmp.path(),
        CodebaseSelector {
            root: "repo".to_string(),
            index_path: None,
        },
    )
    .await?;
    assert_eq!(status.schema_revision, CODE_INDEX_SCHEMA_REVISION);
    assert_eq!(status.codebase_id.as_deref(), Some("repo-main"));
    assert!(status.capabilities.semantic);
    assert!(status.capabilities.hybrid);
    assert_eq!(status.semantic.embedded_chunks, 2);
    assert_eq!(
        status.semantic.embedding_dimensions,
        Some(CODE_INDEX_VECTOR_DIM)
    );
    assert_eq!(
        status.semantic.embedding_key.as_deref(),
        Some("test:synthetic")
    );
    assert_eq!(
        status.semantic.vector_format,
        Some(crate::metadata::CodeIndexVectorFormat::Float8)
    );

    let rows = search(
        tmp.path(),
        CodebaseSelector {
            root: "repo".to_string(),
            index_path: None,
        },
        CodeSearchMode::Hybrid,
        "capability",
        &CodeSearchRequest {
            limit: 5,
            languages: vec!["rust".to_string()],
            kinds: vec!["function".to_string()],
            min_score: 0.1,
            strict: false,
            trace: true,
        },
        Some(&vec![0.001_f32; CODE_INDEX_VECTOR_DIM]),
    )
    .await?;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].name, "capability_decision");
    assert_eq!(rows[0].rank, 1);
    assert!(rows[0].lexical_score.is_some());
    assert!(rows[0].semantic_score.is_some());
    assert_eq!(
        rows[0]
            .trace
            .as_ref()
            .and_then(|trace| trace.fusion.as_deref()),
        Some("rrf")
    );

    let lexical_only_root = tmp.path().join("repo_lexical_only");
    std::fs::create_dir_all(&lexical_only_root)?;
    create_synthetic_code_index(&lexical_only_root, false, false).await?;

    let fallback_rows = search(
        tmp.path(),
        CodebaseSelector {
            root: "repo_lexical_only".to_string(),
            index_path: None,
        },
        CodeSearchMode::Semantic,
        "graph",
        &CodeSearchRequest {
            strict: false,
            trace: true,
            ..CodeSearchRequest::default()
        },
        None,
    )
    .await?;
    assert_eq!(fallback_rows.len(), 1);
    assert_eq!(fallback_rows[0].name, "on_turn_prepare");
    assert!(fallback_rows[0].lexical_score.is_some());
    assert!(fallback_rows[0].semantic_score.is_none());
    assert_eq!(
        fallback_rows[0]
            .trace
            .as_ref()
            .and_then(|trace| trace.requested_mode.as_deref()),
        Some("semantic")
    );
    assert_eq!(
        fallback_rows[0]
            .trace
            .as_ref()
            .map(|trace| trace.effective_mode.as_str()),
        Some("lexical")
    );
    assert_eq!(
        fallback_rows[0]
            .trace
            .as_ref()
            .and_then(|trace| trace.fallback_reason.as_deref()),
        Some("capability_fallback")
    );

    let lexical_phrase_rows = search(
        tmp.path(),
        CodebaseSelector {
            root: "repo_lexical_only".to_string(),
            index_path: None,
        },
        CodeSearchMode::Lexical,
        "runtime graph edges",
        &CodeSearchRequest::default(),
        None,
    )
    .await?;
    assert_eq!(lexical_phrase_rows.len(), 1);
    assert_eq!(lexical_phrase_rows[0].name, "on_turn_prepare");

    let lexical_path_rows = search(
        tmp.path(),
        CodebaseSelector {
            root: "repo".to_string(),
            index_path: None,
        },
        CodeSearchMode::Lexical,
        "src/kernel/governance.rs",
        &CodeSearchRequest::default(),
        None,
    )
    .await?;
    assert_eq!(lexical_path_rows.len(), 1);
    assert_eq!(lexical_path_rows[0].name, "capability_decision");

    let lexical_file_rows = search(
        tmp.path(),
        CodebaseSelector {
            root: "repo_lexical_only".to_string(),
            index_path: None,
        },
        CodeSearchMode::Lexical,
        "runtime_graph.lua",
        &CodeSearchRequest::default(),
        None,
    )
    .await?;
    assert_eq!(lexical_file_rows.len(), 1);
    assert_eq!(lexical_file_rows[0].path, "harnesses/runtime_graph.lua");

    let strict_err = search(
        tmp.path(),
        CodebaseSelector {
            root: "repo_lexical_only".to_string(),
            index_path: None,
        },
        CodeSearchMode::Semantic,
        "graph",
        &CodeSearchRequest {
            strict: true,
            ..CodeSearchRequest::default()
        },
        None,
    )
    .await
    .unwrap_err();
    assert!(
        strict_err
            .to_string()
            .contains("semantic capability not available")
    );

    Ok(())
}
