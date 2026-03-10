use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::metadata::CodeIndexSemanticStatus;
use crate::shared::{encode_vector_blob, open_index_connection};

mod query;
mod resolve;

use query::{
    build_lexical_search_sql, build_semantic_search_sql, hybrid_candidate_limit,
    negotiated_search_mode, reciprocal_rank,
};
use resolve::{has_optional_column, validate_index};

const DEFAULT_LIMIT: usize = 10;

#[derive(Debug, Clone)]
pub struct CodebaseSelector {
    pub root: String,
    pub index_path: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeSearchMode {
    Lexical,
    Semantic,
    Hybrid,
}

#[derive(Debug, Clone)]
pub struct CodeSearchRequest {
    pub limit: usize,
    pub languages: Vec<String>,
    pub kinds: Vec<String>,
    pub min_score: f64,
    pub strict: bool,
    pub trace: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CodeIndexCapabilities {
    pub lexical: bool,
    #[serde(default)]
    pub semantic: bool,
    #[serde(default)]
    pub hybrid: bool,
    #[serde(default)]
    pub languages: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CodeIndexStatus {
    pub root: String,
    pub index_path: String,
    pub schema_revision: i64,
    pub updated_at: String,
    pub index_age_seconds: u64,
    pub codebase_id: Option<String>,
    pub capabilities: CodeIndexCapabilities,
    pub semantic: CodeIndexSemanticStatus,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CodeSearchTrace {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_mode: Option<String>,
    pub effective_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexical_rank: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_rank: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexical_rrf: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_rrf: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fusion: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CodeSearchRow {
    pub chunk_key: String,
    pub path: String,
    pub language: String,
    pub kind: String,
    pub name: String,
    pub signature: Option<String>,
    pub snippet: String,
    pub start_line: i64,
    pub end_line: i64,
    pub score: f64,
    pub lexical_score: Option<f64>,
    pub semantic_score: Option<f64>,
    pub rank: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace: Option<CodeSearchTrace>,
}

pub async fn status(workspace_root: &Path, selector: CodebaseSelector) -> Result<CodeIndexStatus> {
    let validated = validate_index(workspace_root, selector).await?;
    Ok(CodeIndexStatus {
        root: validated.root.to_string_lossy().to_string(),
        index_path: validated.index_path.to_string_lossy().to_string(),
        schema_revision: validated.schema_revision,
        updated_at: validated.updated_at,
        index_age_seconds: validated.index_age_seconds,
        codebase_id: validated.codebase_id,
        capabilities: validated.capabilities,
        semantic: validated.semantic,
    })
}

pub async fn search(
    workspace_root: &Path,
    selector: CodebaseSelector,
    requested_mode: CodeSearchMode,
    query: &str,
    request: &CodeSearchRequest,
    query_vector: Option<&[f32]>,
) -> Result<Vec<CodeSearchRow>> {
    let validated = validate_index(workspace_root, selector).await?;
    let query = query.trim();
    if query.is_empty() {
        bail!("query must not be empty");
    }

    let negotiated_mode = negotiated_search_mode(
        requested_mode,
        &validated.capabilities,
        &validated.root,
        request.strict,
    )?;
    let (_db, conn) = open_index_connection(&validated.index_path).await?;
    let mut rows = match negotiated_mode {
        CodeSearchMode::Lexical => {
            lexical_search_rows(
                &conn,
                negotiated_mode.view_name(),
                query,
                request,
                request.limit,
            )
            .await
        }
        CodeSearchMode::Semantic => {
            let query_vector =
                query_vector.context("semantic search requires an embedding query vector")?;
            semantic_search_rows(
                &conn,
                negotiated_mode.view_name(),
                request,
                query_vector,
                request.limit,
            )
            .await
        }
        CodeSearchMode::Hybrid => {
            let query_vector =
                query_vector.context("hybrid search requires an embedding query vector")?;
            hybrid_search_rows(&conn, query, request, query_vector).await
        }
    }?;
    if request.trace {
        annotate_request_trace(
            &mut rows,
            requested_mode,
            negotiated_mode,
            (requested_mode != negotiated_mode).then_some("capability_fallback"),
        );
    }
    Ok(rows)
}

async fn lexical_search_rows(
    conn: &turso::Connection,
    view_name: &str,
    query: &str,
    request: &CodeSearchRequest,
    limit: usize,
) -> Result<Vec<CodeSearchRow>> {
    let has_search_text = has_optional_column(conn, "code_chunks", "search_text").await;
    let source_name = if has_search_text {
        "code_chunks"
    } else {
        view_name
    };
    let (sql, params) =
        build_lexical_search_sql(source_name, true, query, request, has_search_text, limit);
    let mut rows = query_rows(conn, &sql, params).await?;
    if request.trace {
        for row in &mut rows {
            row.trace = Some(CodeSearchTrace {
                requested_mode: None,
                effective_mode: CodeSearchMode::Lexical.as_str().to_string(),
                fallback_reason: None,
                lexical_rank: Some(row.rank),
                semantic_rank: None,
                lexical_rrf: None,
                semantic_rrf: None,
                fusion: None,
            });
        }
    }
    Ok(rows)
}

async fn semantic_search_rows(
    conn: &turso::Connection,
    view_name: &str,
    request: &CodeSearchRequest,
    query_vector: &[f32],
    limit: usize,
) -> Result<Vec<CodeSearchRow>> {
    let (sql, mut params) = build_semantic_search_sql(view_name, request, limit);
    params[0] = turso::Value::Blob(encode_vector_blob(query_vector, "semantic query vector")?);
    let mut rows = query_rows(conn, &sql, params).await?;
    if request.trace {
        for row in &mut rows {
            row.trace = Some(CodeSearchTrace {
                requested_mode: None,
                effective_mode: CodeSearchMode::Semantic.as_str().to_string(),
                fallback_reason: None,
                lexical_rank: None,
                semantic_rank: Some(row.rank),
                lexical_rrf: None,
                semantic_rrf: None,
                fusion: None,
            });
        }
    }
    Ok(rows)
}

async fn query_rows(
    conn: &turso::Connection,
    sql: &str,
    params: Vec<turso::Value>,
) -> Result<Vec<CodeSearchRow>> {
    let mut stmt = conn.prepare(sql).await?;
    let mut rows = stmt.query(params).await?;

    let mut out = Vec::new();
    let mut rank = 1_i64;
    while let Some(row) = rows.next().await? {
        out.push(CodeSearchRow {
            chunk_key: row.get::<String>(0)?,
            path: row.get::<String>(1)?,
            language: row.get::<String>(2)?,
            kind: row.get::<String>(3)?,
            name: row.get::<String>(4)?,
            signature: row.get::<Option<String>>(5)?,
            snippet: row.get::<String>(6)?,
            start_line: row.get::<i64>(7)?,
            end_line: row.get::<i64>(8)?,
            score: row.get::<f64>(9)?,
            lexical_score: row.get::<Option<f64>>(10)?,
            semantic_score: row.get::<Option<f64>>(11)?,
            rank,
            trace: None,
        });
        rank += 1;
    }
    Ok(out)
}

async fn hybrid_search_rows(
    conn: &turso::Connection,
    query: &str,
    request: &CodeSearchRequest,
    query_vector: &[f32],
) -> Result<Vec<CodeSearchRow>> {
    let candidate_limit = hybrid_candidate_limit(request.limit);
    let candidate_request = CodeSearchRequest {
        limit: candidate_limit,
        min_score: 0.0,
        ..request.clone()
    };
    let lexical_rows = lexical_search_rows(
        conn,
        CodeSearchMode::Lexical.view_name(),
        query,
        &candidate_request,
        candidate_limit,
    )
    .await?;
    let semantic_rows = semantic_search_rows(
        conn,
        CodeSearchMode::Semantic.view_name(),
        &candidate_request,
        query_vector,
        candidate_limit,
    )
    .await?;

    let lexical_trace = lexical_rows
        .iter()
        .enumerate()
        .map(|(index, row)| {
            (
                row.chunk_key.clone(),
                ((index + 1) as i64, reciprocal_rank(index + 1)),
            )
        })
        .collect::<HashMap<_, _>>();
    let semantic_trace = semantic_rows
        .iter()
        .enumerate()
        .map(|(index, row)| {
            (
                row.chunk_key.clone(),
                ((index + 1) as i64, reciprocal_rank(index + 1)),
            )
        })
        .collect::<HashMap<_, _>>();

    let mut fused = HashMap::<String, CodeSearchRow>::new();
    for (index, row) in lexical_rows.iter().enumerate() {
        let key = row.chunk_key.clone();
        let entry = fused.entry(key).or_insert_with(|| CodeSearchRow {
            chunk_key: row.chunk_key.clone(),
            path: row.path.clone(),
            language: row.language.clone(),
            kind: row.kind.clone(),
            name: row.name.clone(),
            signature: row.signature.clone(),
            snippet: row.snippet.clone(),
            start_line: row.start_line,
            end_line: row.end_line,
            score: 0.0,
            lexical_score: row.lexical_score.or(Some(row.score)),
            semantic_score: None,
            rank: 0,
            trace: None,
        });
        entry.score += reciprocal_rank(index + 1);
        entry.lexical_score = row.lexical_score.or(Some(row.score));
    }

    for (index, row) in semantic_rows.iter().enumerate() {
        let key = row.chunk_key.clone();
        let entry = fused.entry(key).or_insert_with(|| CodeSearchRow {
            chunk_key: row.chunk_key.clone(),
            path: row.path.clone(),
            language: row.language.clone(),
            kind: row.kind.clone(),
            name: row.name.clone(),
            signature: row.signature.clone(),
            snippet: row.snippet.clone(),
            start_line: row.start_line,
            end_line: row.end_line,
            score: 0.0,
            lexical_score: None,
            semantic_score: row.semantic_score.or(Some(row.score)),
            rank: 0,
            trace: None,
        });
        entry.score += reciprocal_rank(index + 1);
        entry.semantic_score = row.semantic_score.or(Some(row.score));
    }

    let mut rows = fused.into_values().collect::<Vec<_>>();
    rows.retain(|row| row.score >= request.min_score);
    rows.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.start_line.cmp(&b.start_line))
    });
    rows.truncate(request.limit.max(1));
    for (index, row) in rows.iter_mut().enumerate() {
        row.rank = (index + 1) as i64;
        if request.trace {
            let lexical = lexical_trace.get(&row.chunk_key).copied();
            let semantic = semantic_trace.get(&row.chunk_key).copied();
            row.trace = Some(CodeSearchTrace {
                requested_mode: None,
                effective_mode: CodeSearchMode::Hybrid.as_str().to_string(),
                fallback_reason: None,
                lexical_rank: lexical.map(|value| value.0),
                semantic_rank: semantic.map(|value| value.0),
                lexical_rrf: lexical.map(|value| value.1),
                semantic_rrf: semantic.map(|value| value.1),
                fusion: Some("rrf".to_string()),
            });
        }
    }
    Ok(rows)
}

impl CodeSearchMode {
    pub(crate) fn view_name(self) -> &'static str {
        match self {
            Self::Lexical => "v_code_lexical",
            Self::Semantic => "v_code_semantic",
            Self::Hybrid => "v_code_hybrid",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lexical => "lexical",
            Self::Semantic => "semantic",
            Self::Hybrid => "hybrid",
        }
    }
}

impl Default for CodeSearchRequest {
    fn default() -> Self {
        Self {
            limit: DEFAULT_LIMIT,
            languages: Vec::new(),
            kinds: Vec::new(),
            min_score: 0.0,
            strict: false,
            trace: false,
        }
    }
}

fn annotate_request_trace(
    rows: &mut [CodeSearchRow],
    requested_mode: CodeSearchMode,
    effective_mode: CodeSearchMode,
    fallback_reason: Option<&str>,
) {
    for row in rows {
        let trace = row.trace.get_or_insert_with(|| CodeSearchTrace {
            requested_mode: None,
            effective_mode: effective_mode.as_str().to_string(),
            fallback_reason: None,
            lexical_rank: None,
            semantic_rank: None,
            lexical_rrf: None,
            semantic_rrf: None,
            fusion: None,
        });
        trace.requested_mode = Some(requested_mode.as_str().to_string());
        trace.effective_mode = effective_mode.as_str().to_string();
        trace.fallback_reason = fallback_reason.map(str::to_string);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::CODE_INDEX_VECTOR_DIM;
    use crate::shared::CODE_INDEX_SCHEMA_REVISION;
    use tempfile::tempdir;

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
        let index_dir = root.join(".turin");
        std::fs::create_dir_all(&index_dir)?;
        let index_path = index_dir.join("codebase.db");
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
                "harnesses/runtime_cache.lua",
                "lua",
                "function",
                "on_turn_prepare",
                "function on_turn_prepare(ctx)",
                "function on_turn_prepare(ctx) local rows = runtime.cache.stats() end",
                "harnesses/runtime_cache.lua\non_turn_prepare\nfunction on_turn_prepare(ctx)\nfunction on_turn_prepare(ctx) local rows = runtime.cache.stats() end",
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
            "cache",
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
            "runtime cache stats",
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

        let strict_err = search(
            tmp.path(),
            CodebaseSelector {
                root: "repo_lexical_only".to_string(),
                index_path: None,
            },
            CodeSearchMode::Semantic,
            "cache",
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
}
