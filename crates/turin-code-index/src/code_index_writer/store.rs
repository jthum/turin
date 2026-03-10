use anyhow::{Context, Result};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use crate::embeddings::CODE_INDEX_VECTOR_DIM;
use crate::metadata::CodeIndexSemanticStatus;
use crate::shared::{encode_vector_blob, open_index_connection};

use super::{
    CODE_INDEX_SCHEMA_REVISION, CodeChunkRecord, CodeIndexSummary, CodeIndexWriteCapabilities,
    IndexableFileContent, IndexedFileState,
};

pub(super) async fn init_schema(conn: &turso::Connection) -> Result<()> {
    conn.execute_batch(INIT_SCHEMA)
        .await
        .context("failed to initialize code index schema")?;
    Ok(())
}

pub(super) async fn should_recreate_index(index_path: &Path) -> Result<bool> {
    if !index_path.exists() {
        return Ok(false);
    }

    let (_db, conn) = open_index_connection(index_path).await?;
    if !table_exists(&conn, "index_meta").await? || !table_exists(&conn, "indexed_files").await? {
        return Ok(true);
    }

    Ok(load_existing_schema_revision(&conn).await? != Some(CODE_INDEX_SCHEMA_REVISION))
}

pub(super) async fn current_timestamp(conn: &turso::Connection) -> Result<String> {
    let mut rows = conn
        .query("SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')", ())
        .await?;
    let row = rows
        .next()
        .await?
        .context("timestamp query returned no row")?;
    Ok(row.get::<String>(0)?)
}

pub(super) async fn load_indexed_files(
    conn: &turso::Connection,
) -> Result<HashMap<String, IndexedFileState>> {
    let mut rows = conn
        .query(
            "SELECT path, content_hash, language, embedding_key, chunk_count FROM indexed_files",
            (),
        )
        .await?;

    let mut out = HashMap::new();
    while let Some(row) = rows.next().await? {
        out.insert(
            row.get::<String>(0)?,
            IndexedFileState {
                content_hash: row.get::<String>(1)?,
                language: row.get::<String>(2)?,
                embedding_key: row.get::<String>(3)?,
                chunk_count: row.get::<i64>(4)? as u64,
            },
        );
    }
    Ok(out)
}

pub(super) async fn insert_chunks(
    conn: &turso::Connection,
    chunks: &[CodeChunkRecord],
) -> Result<()> {
    for chunk in chunks {
        let embedding = chunk
            .embedding
            .as_deref()
            .map(|vector| encode_vector_blob(vector, "code chunk embedding"))
            .transpose()?;
        conn.execute(
            "INSERT INTO code_chunks (chunk_key, path, language, kind, name, signature, snippet, search_text, embedding, start_line, end_line) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            turso::params![
                chunk.chunk_key.clone(),
                chunk.path.clone(),
                chunk.language.clone(),
                chunk.kind.clone(),
                chunk.name.clone(),
                chunk.signature.clone(),
                chunk.snippet.clone(),
                chunk.search_text.clone(),
                embedding,
                chunk.start_line,
                chunk.end_line
            ],
        )
        .await?;
    }
    Ok(())
}

pub(super) async fn upsert_indexed_file(
    conn: &turso::Connection,
    source: &IndexableFileContent,
    chunk_count: u64,
    embedding_key: &str,
    updated_at: &str,
) -> Result<()> {
    conn.execute(
        "INSERT INTO indexed_files (path, content_hash, language, embedding_key, chunk_count, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(path) DO UPDATE SET
           content_hash = excluded.content_hash,
           language = excluded.language,
           embedding_key = excluded.embedding_key,
           chunk_count = excluded.chunk_count,
           updated_at = excluded.updated_at",
        turso::params![
            source.path.clone(),
            source.content_hash.clone(),
            source.language.clone(),
            embedding_key.to_string(),
            chunk_count as i64,
            updated_at.to_string()
        ],
    )
    .await?;
    Ok(())
}

pub(super) async fn delete_indexed_file(
    conn: &turso::Connection,
    relative_path: &str,
) -> Result<u64> {
    let removed_chunks = conn
        .execute(
            "DELETE FROM code_chunks WHERE path = ?1",
            turso::params![relative_path.to_string()],
        )
        .await?;
    conn.execute(
        "DELETE FROM indexed_files WHERE path = ?1",
        turso::params![relative_path.to_string()],
    )
    .await?;
    Ok(removed_chunks as u64)
}

pub(super) async fn load_index_summary(conn: &turso::Connection) -> Result<CodeIndexSummary> {
    let mut rows = conn
        .query(
            "SELECT COUNT(*), COALESCE(SUM(chunk_count), 0) FROM indexed_files",
            (),
        )
        .await?;
    let row = rows
        .next()
        .await?
        .context("indexed_files summary query returned no row")?;
    let files_indexed = row.get::<i64>(0)? as u64;
    let chunks_indexed = row.get::<i64>(1)? as u64;

    let mut semantic_rows = conn
        .query(
            "SELECT COUNT(*) FROM code_chunks WHERE embedding IS NOT NULL",
            (),
        )
        .await?;
    let semantic_row = semantic_rows
        .next()
        .await?
        .context("semantic summary query returned no row")?;
    let embedded_chunks = semantic_row.get::<i64>(0)? as u64;
    let semantic = load_semantic_status(conn, embedded_chunks).await?;

    let mut language_rows = conn
        .query(
            "SELECT DISTINCT language FROM indexed_files ORDER BY language ASC",
            (),
        )
        .await?;
    let mut languages = BTreeSet::new();
    while let Some(row) = language_rows.next().await? {
        languages.insert(row.get::<String>(0)?);
    }

    Ok(CodeIndexSummary {
        capabilities: CodeIndexWriteCapabilities {
            lexical: true,
            semantic: semantic.embedded_chunks > 0,
            hybrid: semantic.embedded_chunks > 0,
            languages: languages.into_iter().collect(),
        },
        semantic,
        files_indexed,
        chunks_indexed,
    })
}

pub(super) async fn write_index_meta(
    conn: &turso::Connection,
    root: &Path,
    updated_at: &str,
    codebase_id: &Option<String>,
    summary: &CodeIndexSummary,
) -> Result<()> {
    conn.execute("DELETE FROM index_meta", ()).await?;
    conn.execute(
        "INSERT INTO index_meta (schema_revision, root_path, updated_at, capabilities, codebase_id, embedding_key, embedding_dimensions, embedded_chunks) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        turso::params![
            CODE_INDEX_SCHEMA_REVISION,
            root.to_string_lossy().to_string(),
            updated_at.to_string(),
            serde_json::to_string(&summary.capabilities)?,
            codebase_id.clone(),
            summary.semantic.embedding_key.clone(),
            summary.semantic.embedding_dimensions.map(|value| value as i64),
            summary.semantic.embedded_chunks as i64
        ],
    )
    .await?;
    Ok(())
}

async fn load_semantic_status(
    conn: &turso::Connection,
    embedded_chunks: u64,
) -> Result<CodeIndexSemanticStatus> {
    if embedded_chunks == 0 {
        return Ok(CodeIndexSemanticStatus::disabled());
    }

    let mut rows = conn
        .query(
            "SELECT embedding_key FROM indexed_files WHERE embedding_key <> 'none' LIMIT 1",
            (),
        )
        .await?;
    let embedding_key = rows
        .next()
        .await?
        .map(|row| row.get::<String>(0))
        .transpose()?
        .unwrap_or_else(|| "unknown".to_string());

    Ok(CodeIndexSemanticStatus::enabled(
        embedded_chunks,
        embedding_key,
        CODE_INDEX_VECTOR_DIM,
    ))
}

async fn table_exists(conn: &turso::Connection, table: &str) -> Result<bool> {
    let mut rows = conn
        .query(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1 LIMIT 1",
            turso::params![table],
        )
        .await?;
    Ok(rows.next().await?.is_some())
}

async fn load_existing_schema_revision(conn: &turso::Connection) -> Result<Option<i64>> {
    let mut rows = conn
        .query("SELECT schema_revision FROM index_meta LIMIT 1", ())
        .await?;
    match rows.next().await? {
        Some(row) => Ok(Some(row.get::<i64>(0)?)),
        None => Ok(None),
    }
}

const INIT_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS index_meta (
    schema_revision INTEGER NOT NULL,
    root_path TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    capabilities TEXT NOT NULL,
    codebase_id TEXT,
    embedding_key TEXT,
    embedding_dimensions INTEGER,
    embedded_chunks INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS indexed_files (
    path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    language TEXT NOT NULL,
    embedding_key TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS code_chunks (
    chunk_key TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    language TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    signature TEXT,
    snippet TEXT NOT NULL,
    search_text TEXT NOT NULL,
    embedding F32_BLOB(1536),
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_indexed_files_language ON indexed_files(language);
CREATE INDEX IF NOT EXISTS idx_code_chunks_path ON code_chunks(path);
CREATE INDEX IF NOT EXISTS idx_code_chunks_search_fts ON code_chunks USING fts(search_text);

DROP VIEW IF EXISTS v_code_lexical;
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
    0.0 AS score,
    0.0 AS lexical_score,
    NULL AS semantic_score,
    search_text
FROM code_chunks;

DROP VIEW IF EXISTS v_code_semantic;
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
    0.0 AS score,
    NULL AS lexical_score,
    0.0 AS semantic_score,
    embedding
FROM code_chunks
WHERE embedding IS NOT NULL;

DROP VIEW IF EXISTS v_code_hybrid;
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
    0.0 AS score,
    0.0 AS lexical_score,
    0.0 AS semantic_score,
    search_text,
    embedding
FROM code_chunks
WHERE embedding IS NOT NULL;
"#;
