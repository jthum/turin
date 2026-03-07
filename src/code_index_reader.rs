use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use turso::{Connection, Database, Value};

const CODE_INDEX_SCHEMA_REVISION: i64 = 20260305;
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
    pub capabilities: CodeIndexCapabilities,
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
}

#[derive(Debug, Clone)]
struct ResolvedCodebase {
    root: PathBuf,
    index_path: PathBuf,
}

#[derive(Debug, Clone)]
struct ValidatedIndex {
    root: PathBuf,
    index_path: PathBuf,
    schema_revision: i64,
    updated_at: String,
    index_age_seconds: u64,
    capabilities: CodeIndexCapabilities,
}

pub async fn status(workspace_root: &Path, selector: CodebaseSelector) -> Result<CodeIndexStatus> {
    let validated = validate_index(workspace_root, selector).await?;
    Ok(CodeIndexStatus {
        root: validated.root.to_string_lossy().to_string(),
        index_path: validated.index_path.to_string_lossy().to_string(),
        schema_revision: validated.schema_revision,
        updated_at: validated.updated_at,
        index_age_seconds: validated.index_age_seconds,
        capabilities: validated.capabilities,
    })
}

pub async fn search(
    workspace_root: &Path,
    selector: CodebaseSelector,
    requested_mode: CodeSearchMode,
    query: &str,
    request: &CodeSearchRequest,
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
    let view_name = negotiated_mode.view_name();
    let (_db, conn) = open_index_connection(&validated.index_path).await?;
    let has_search_text = has_optional_column(&conn, view_name, "search_text").await;
    let (sql, params) = build_search_sql(view_name, query, request, has_search_text);
    let mut stmt = conn
        .prepare(&sql)
        .await
        .with_context(|| format!("failed to prepare query for '{view_name}'"))?;
    let mut rows = stmt
        .query(params)
        .await
        .with_context(|| format!("failed to execute query against '{view_name}'"))?;

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
        });
        rank += 1;
    }

    Ok(out)
}

async fn validate_index(
    workspace_root: &Path,
    selector: CodebaseSelector,
) -> Result<ValidatedIndex> {
    let resolved = resolve_codebase(workspace_root, selector)?;
    let (_db, conn) = open_index_connection(&resolved.index_path).await?;
    let (schema_revision, root_path, updated_at, capabilities) = load_index_meta(&conn).await?;

    if schema_revision != CODE_INDEX_SCHEMA_REVISION {
        bail!(
            "unsupported schema_revision {} at '{}'; expected {}",
            schema_revision,
            resolved.index_path.display(),
            CODE_INDEX_SCHEMA_REVISION
        );
    }

    let declared_root = std::fs::canonicalize(&root_path).with_context(|| {
        format!(
            "index_meta.root_path '{}' is not a valid canonical path",
            root_path
        )
    })?;
    if declared_root != resolved.root {
        bail!(
            "index_meta.root_path '{}' does not match resolved root '{}'",
            declared_root.display(),
            resolved.root.display()
        );
    }

    if !capabilities.lexical {
        bail!("index capabilities.lexical must be true");
    }

    validate_view_contract(&conn, "v_code_lexical").await?;
    if capabilities.semantic {
        validate_view_contract(&conn, "v_code_semantic").await?;
    }
    if capabilities.hybrid {
        validate_view_contract(&conn, "v_code_hybrid").await?;
    }

    Ok(ValidatedIndex {
        root: resolved.root,
        index_path: resolved.index_path,
        schema_revision,
        updated_at,
        index_age_seconds: index_age_seconds(&conn).await?,
        capabilities,
    })
}

fn resolve_codebase(workspace_root: &Path, selector: CodebaseSelector) -> Result<ResolvedCodebase> {
    let workspace_root = std::fs::canonicalize(workspace_root).with_context(|| {
        format!(
            "workspace root '{}' does not exist",
            workspace_root.display()
        )
    })?;

    let root_value = selector.root.trim();
    if root_value.is_empty() {
        bail!("codebase.root must not be empty");
    }

    let root = canonicalize_selector_path(&workspace_root, Path::new(root_value))
        .with_context(|| format!("codebase root '{}' not found", root_value))?;

    let index_path = match selector.index_path {
        Some(index_path) => {
            let candidate = PathBuf::from(index_path);
            if candidate.is_absolute() {
                candidate
            } else {
                root.join(candidate)
            }
        }
        None => root.join(".turin").join("codebase.db"),
    };

    let index_path = std::fs::canonicalize(&index_path)
        .with_context(|| format!("index db not found at '{}'", index_path.display()))?;

    Ok(ResolvedCodebase { root, index_path })
}

fn canonicalize_selector_path(base: &Path, candidate: &Path) -> Result<PathBuf> {
    let path = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        base.join(candidate)
    };
    Ok(std::fs::canonicalize(&path)?)
}

async fn open_index_connection(index_path: &Path) -> Result<(Database, Connection)> {
    let index_path = index_path.to_string_lossy().to_string();
    let db = turso::Builder::new_local(&index_path)
        .experimental_index_method(true)
        .build()
        .await
        .with_context(|| format!("failed to open index db '{}'", index_path))?;
    let conn = db.connect()?;
    conn.execute("PRAGMA busy_timeout = 5000;", ()).await.ok();
    Ok((db, conn))
}

async fn load_index_meta(
    conn: &Connection,
) -> Result<(i64, String, String, CodeIndexCapabilities)> {
    let mut rows = conn
        .query(
            "SELECT schema_revision, root_path, updated_at, capabilities FROM index_meta LIMIT 1",
            (),
        )
        .await
        .context("missing required index_meta contract; run `turin-map index --root <path>`")?;
    let row = rows
        .next()
        .await?
        .ok_or_else(|| anyhow!("index_meta is empty; run `turin-map index --root <path>`"))?;

    let schema_revision = row.get::<i64>(0)?;
    let root_path = row.get::<String>(1)?;
    let updated_at = row.get::<String>(2)?;
    let capabilities_json = row.get::<String>(3)?;
    let capabilities = serde_json::from_str::<CodeIndexCapabilities>(&capabilities_json)
        .with_context(|| "index_meta.capabilities must be valid JSON")?;

    Ok((schema_revision, root_path, updated_at, capabilities))
}

async fn index_age_seconds(conn: &Connection) -> Result<u64> {
    let mut rows = conn
        .query(
            "SELECT CAST(strftime('%s', 'now') - strftime('%s', updated_at) AS INTEGER) FROM index_meta LIMIT 1",
            (),
        )
        .await?;
    let row = rows
        .next()
        .await?
        .ok_or_else(|| anyhow!("index_meta is empty"))?;
    let age = row.get::<Option<i64>>(0)?.unwrap_or(0).max(0) as u64;
    Ok(age)
}

async fn validate_view_contract(conn: &Connection, view_name: &str) -> Result<()> {
    let sql = format!(
        "SELECT chunk_key, path, language, kind, name, signature, snippet, start_line, end_line, score, lexical_score, semantic_score FROM {view_name} LIMIT 0"
    );
    conn.query(&sql, ())
        .await
        .with_context(|| format!("missing required read view contract '{view_name}'"))?;
    Ok(())
}

async fn has_optional_column(conn: &Connection, view_name: &str, column_name: &str) -> bool {
    let sql = format!("SELECT {column_name} FROM {view_name} LIMIT 0");
    conn.query(&sql, ()).await.is_ok()
}

fn negotiated_search_mode(
    requested_mode: CodeSearchMode,
    capabilities: &CodeIndexCapabilities,
    root: &Path,
    strict: bool,
) -> Result<CodeSearchMode> {
    match requested_mode {
        CodeSearchMode::Lexical => Ok(CodeSearchMode::Lexical),
        CodeSearchMode::Semantic if capabilities.semantic => Ok(CodeSearchMode::Semantic),
        CodeSearchMode::Semantic if !strict => Ok(CodeSearchMode::Lexical),
        CodeSearchMode::Semantic => bail!(
            "semantic capability not available for root '{}'",
            root.display()
        ),
        CodeSearchMode::Hybrid if capabilities.hybrid => Ok(CodeSearchMode::Hybrid),
        CodeSearchMode::Hybrid if capabilities.semantic && !strict => Ok(CodeSearchMode::Semantic),
        CodeSearchMode::Hybrid if !strict => Ok(CodeSearchMode::Lexical),
        CodeSearchMode::Hybrid => bail!(
            "hybrid capability not available for root '{}'",
            root.display()
        ),
    }
}

fn build_search_sql(
    view_name: &str,
    query: &str,
    request: &CodeSearchRequest,
    has_search_text: bool,
) -> (String, Vec<Value>) {
    let like_value = escape_like_pattern(query);
    let mut params = vec![
        Value::Text(like_value.clone()),
        Value::Text(query.to_string()),
    ];
    let pattern_slot = "?1";
    let exact_slot = "?2";

    let lexical_score_expr = if view_name == CodeSearchMode::Lexical.view_name() {
        Some(format!(
            "CASE \
                WHEN LOWER(name) = LOWER({exact_slot}) THEN 120.0 \
                WHEN LOWER(COALESCE(signature, '')) = LOWER({exact_slot}) THEN 90.0 \
                WHEN LOWER(name) LIKE LOWER({pattern_slot}) ESCAPE '\\' THEN 70.0 \
                WHEN LOWER(COALESCE(signature, '')) LIKE LOWER({pattern_slot}) ESCAPE '\\' THEN 45.0 \
                WHEN LOWER(snippet) LIKE LOWER({pattern_slot}) ESCAPE '\\' THEN 20.0 \
                WHEN LOWER(path) LIKE LOWER({pattern_slot}) ESCAPE '\\' THEN 10.0 \
                ELSE 0.0 \
            END"
        ))
    } else {
        None
    };

    let (mut sql, mut clauses) = if let Some(lexical_score) = lexical_score_expr.as_deref() {
        (
            format!(
                "SELECT chunk_key, path, language, kind, name, signature, snippet, start_line, end_line, {lexical_score} AS score, {lexical_score} AS lexical_score, NULL AS semantic_score FROM {view_name}"
            ),
            vec![lexical_match_clause(pattern_slot, has_search_text)],
        )
    } else {
        (
            format!(
                "SELECT chunk_key, path, language, kind, name, signature, snippet, start_line, end_line, score, lexical_score, semantic_score FROM {view_name}"
            ),
            vec![lexical_match_clause(pattern_slot, has_search_text)],
        )
    };

    if request.min_score > 0.0 {
        params.push(Value::Real(request.min_score));
        if let Some(lexical_score) = lexical_score_expr.as_deref() {
            clauses.push(format!("({lexical_score}) >= ?{}", params.len()));
        } else {
            clauses.push(format!("score >= ?{}", params.len()));
        }
    }

    if !request.languages.is_empty() {
        let slots = push_in_params(&mut params, &request.languages);
        clauses.push(format!("language IN ({})", slots.join(", ")));
    }

    if !request.kinds.is_empty() {
        let slots = push_in_params(&mut params, &request.kinds);
        clauses.push(format!("kind IN ({})", slots.join(", ")));
    }

    if !clauses.is_empty() {
        sql.push_str(" WHERE ");
        sql.push_str(&clauses.join(" AND "));
    }

    sql.push_str(" ORDER BY score DESC, path ASC, start_line ASC");
    let limit = request.limit.max(1);
    params.push(Value::Integer(limit as i64));
    sql.push_str(&format!(" LIMIT ?{}", params.len()));
    (sql, params)
}

fn lexical_match_clause(pattern_slot: &str, has_search_text: bool) -> String {
    if has_search_text {
        format!("search_text LIKE {pattern_slot} ESCAPE '\\'")
    } else {
        format!(
            "(path LIKE {pattern_slot} ESCAPE '\\' OR name LIKE {pattern_slot} ESCAPE '\\' OR COALESCE(signature, '') LIKE {pattern_slot} ESCAPE '\\' OR snippet LIKE {pattern_slot} ESCAPE '\\')"
        )
    }
}

fn push_in_params(params: &mut Vec<Value>, values: &[String]) -> Vec<String> {
    let mut slots = Vec::with_capacity(values.len());
    for value in values {
        params.push(Value::Text(value.clone()));
        slots.push(format!("?{}", params.len()));
    }
    slots
}

fn escape_like_pattern(query: &str) -> String {
    let mut out = String::with_capacity(query.len() + 2);
    out.push('%');
    for ch in query.chars() {
        match ch {
            '\\' | '%' | '_' => {
                out.push('\\');
                out.push(ch);
            }
            _ => out.push(ch),
        }
    }
    out.push('%');
    out
}

impl CodeSearchMode {
    fn view_name(self) -> &'static str {
        match self {
            Self::Lexical => "v_code_lexical",
            Self::Semantic => "v_code_semantic",
            Self::Hybrid => "v_code_hybrid",
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    async fn create_synthetic_code_index(
        root: &Path,
        semantic: bool,
        hybrid: bool,
    ) -> Result<PathBuf> {
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
    codebase_id TEXT
);
CREATE TABLE code_chunks (
    chunk_key TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    language TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    signature TEXT,
    snippet TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    lexical_score REAL NOT NULL,
    semantic_score REAL
);
"#,
        )
        .await?;
        conn.execute(
            "INSERT INTO index_meta (schema_revision, root_path, updated_at, capabilities, codebase_id) VALUES (?1, ?2, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?3, ?4)",
            turso::params![CODE_INDEX_SCHEMA_REVISION, root_path.to_string_lossy().to_string(), capabilities, "repo-main"],
        )
        .await?;
        conn.execute(
            "INSERT INTO code_chunks (chunk_key, path, language, kind, name, signature, snippet, start_line, end_line, lexical_score, semantic_score) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            turso::params![
                "chunk_rust",
                "src/kernel/governance.rs",
                "rust",
                "function",
                "capability_decision",
                "fn capability_decision(...)",
                "pub fn capability_decision(capability: &str) -> CapabilityDecision",
                101_i64,
                132_i64,
                0.91_f64,
                0.82_f64
            ],
        )
        .await?;
        conn.execute(
            "INSERT INTO code_chunks (chunk_key, path, language, kind, name, signature, snippet, start_line, end_line, lexical_score, semantic_score) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            turso::params![
                "chunk_lua",
                "harnesses/runtime_cache.lua",
                "lua",
                "function",
                "on_turn_prepare",
                "function on_turn_prepare(ctx)",
                "function on_turn_prepare(ctx) local rows = runtime.cache.stats() end",
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
    NULL AS semantic_score
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
    semantic_score
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
    semantic_score
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
        assert!(status.capabilities.semantic);
        assert!(status.capabilities.hybrid);

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
            },
        )
        .await?;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].name, "capability_decision");
        assert_eq!(rows[0].rank, 1);
        assert!(rows[0].lexical_score.is_some());
        assert!(rows[0].semantic_score.is_some());

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
                ..CodeSearchRequest::default()
            },
        )
        .await?;
        assert_eq!(fallback_rows.len(), 1);
        assert_eq!(fallback_rows[0].name, "on_turn_prepare");
        assert!(fallback_rows[0].lexical_score.is_some());
        assert!(fallback_rows[0].semantic_score.is_none());

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
