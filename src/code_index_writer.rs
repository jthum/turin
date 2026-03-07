use anyhow::{Context, Result, bail};
use ignore::{DirEntry, WalkBuilder};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Component, Path, PathBuf};

const CODE_INDEX_SCHEMA_REVISION: i64 = 20260307;
const CHUNK_LINES: usize = 40;
const CHUNK_OVERLAP: usize = 8;
const MAX_FILE_BYTES: usize = 512 * 1024;
const MAX_SNIPPET_CHARS: usize = 1800;

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
struct DiscoveredSymbol {
    start_line: usize,
    kind: String,
    name: String,
    signature: String,
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
    conn.execute_batch(INIT_SCHEMA)
        .await
        .context("failed to initialize code index schema")?;

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

async fn should_recreate_index(index_path: &Path) -> Result<bool> {
    if !index_path.exists() {
        return Ok(false);
    }

    let (_db, conn) = open_index_connection(index_path).await?;
    if !table_exists(&conn, "index_meta").await? || !table_exists(&conn, "indexed_files").await? {
        return Ok(true);
    }

    Ok(load_existing_schema_revision(&conn).await? != Some(CODE_INDEX_SCHEMA_REVISION))
}

fn remove_index_db_if_present(index_path: &Path) -> Result<()> {
    if index_path.exists() {
        std::fs::remove_file(index_path)
            .with_context(|| format!("failed to replace '{}'", index_path.display()))?;
    }
    Ok(())
}

async fn open_index_connection(index_path: &Path) -> Result<(turso::Database, turso::Connection)> {
    let db = turso::Builder::new_local(index_path.to_str().unwrap())
        .experimental_index_method(true)
        .build()
        .await
        .with_context(|| format!("failed to open '{}'", index_path.display()))?;
    let conn = db.connect()?;
    conn.execute("PRAGMA busy_timeout = 5000;", ()).await.ok();
    Ok((db, conn))
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

async fn load_indexed_files(conn: &turso::Connection) -> Result<HashMap<String, IndexedFileState>> {
    let mut rows = conn
        .query(
            "SELECT path, content_hash, language, chunk_count FROM indexed_files",
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
                chunk_count: row.get::<i64>(3)? as u64,
            },
        );
    }
    Ok(out)
}

async fn insert_chunks(conn: &turso::Connection, chunks: &[CodeChunkRecord]) -> Result<()> {
    for chunk in chunks {
        conn.execute(
            "INSERT INTO code_chunks (chunk_key, path, language, kind, name, signature, snippet, search_text, start_line, end_line) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            turso::params![
                chunk.chunk_key.clone(),
                chunk.path.clone(),
                chunk.language.clone(),
                chunk.kind.clone(),
                chunk.name.clone(),
                chunk.signature.clone(),
                chunk.snippet.clone(),
                chunk.search_text.clone(),
                chunk.start_line,
                chunk.end_line
            ],
        )
        .await?;
    }
    Ok(())
}

async fn upsert_indexed_file(
    conn: &turso::Connection,
    source: &IndexableFileContent,
    chunk_count: u64,
    updated_at: &str,
) -> Result<()> {
    conn.execute(
        "INSERT INTO indexed_files (path, content_hash, language, chunk_count, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(path) DO UPDATE SET
           content_hash = excluded.content_hash,
           language = excluded.language,
           chunk_count = excluded.chunk_count,
           updated_at = excluded.updated_at",
        turso::params![
            source.path.clone(),
            source.content_hash.clone(),
            source.language.clone(),
            chunk_count as i64,
            updated_at.to_string()
        ],
    )
    .await?;
    Ok(())
}

async fn delete_indexed_file(conn: &turso::Connection, relative_path: &str) -> Result<u64> {
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

async fn load_index_summary(conn: &turso::Connection) -> Result<CodeIndexSummary> {
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
            semantic: false,
            hybrid: false,
            languages: languages.into_iter().collect(),
        },
        files_indexed,
        chunks_indexed,
    })
}

async fn write_index_meta(
    conn: &turso::Connection,
    root: &Path,
    updated_at: &str,
    capabilities: &CodeIndexWriteCapabilities,
) -> Result<()> {
    let codebase_id = root
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string);
    conn.execute("DELETE FROM index_meta", ()).await?;
    conn.execute(
        "INSERT INTO index_meta (schema_revision, root_path, updated_at, capabilities, codebase_id) VALUES (?1, ?2, ?3, ?4, ?5)",
        turso::params![
            CODE_INDEX_SCHEMA_REVISION,
            root.to_string_lossy().to_string(),
            updated_at.to_string(),
            serde_json::to_string(capabilities)?,
            codebase_id
        ],
    )
    .await?;
    Ok(())
}

fn collect_indexable_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut builder = WalkBuilder::new(root);
    builder
        .hidden(false)
        .require_git(false)
        .sort_by_file_path(|left, right| left.cmp(right))
        .filter_entry(should_walk_entry);

    let mut out = Vec::new();
    for entry in builder.build() {
        let entry = entry?;
        if !entry
            .file_type()
            .is_some_and(|file_type| file_type.is_file())
        {
            continue;
        }
        let path = entry.into_path();
        if detect_language(&path).is_some() {
            out.push(path);
        }
    }
    Ok(out)
}

fn should_walk_entry(entry: &DirEntry) -> bool {
    if entry.depth() == 0 {
        return true;
    }
    match entry.path().file_name().and_then(|name| name.to_str()) {
        Some(name) => !should_skip_dir(name),
        None => true,
    }
}

fn should_skip_dir(name: &str) -> bool {
    matches!(
        name,
        ".git"
            | ".turin"
            | ".next"
            | ".nuxt"
            | ".svelte-kit"
            | "build"
            | "dist"
            | "node_modules"
            | "target"
            | "vendor"
    )
}

fn read_indexable_file(relative_path: String, file: &Path) -> Result<Option<IndexableFileContent>> {
    let language = match detect_language(file) {
        Some(language) => language,
        None => return Ok(None),
    };
    let bytes = std::fs::read(file)?;
    if bytes.len() > MAX_FILE_BYTES {
        return Ok(None);
    }
    let content_hash = file_content_hash(&bytes);
    let content = match String::from_utf8(bytes) {
        Ok(content) => content,
        Err(_) => return Ok(None),
    };
    if content.trim().is_empty() {
        return Ok(None);
    }

    Ok(Some(IndexableFileContent {
        path: relative_path,
        language: language.to_string(),
        content_hash,
        content,
    }))
}

fn build_chunks(relative_path: &str, language: &str, content: &str) -> Vec<CodeChunkRecord> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let symbols = discover_symbols(language, &lines);
    if symbols.is_empty() {
        return build_chunk_segment(
            relative_path,
            language,
            &lines,
            0,
            fallback_symbol(relative_path, 1),
        );
    }

    let mut out = Vec::new();
    if symbols[0].start_line > 0 {
        out.extend(build_chunk_segment(
            relative_path,
            language,
            &lines[..symbols[0].start_line],
            0,
            fallback_symbol(relative_path, 1),
        ));
    }

    for (index, symbol) in symbols.iter().enumerate() {
        let segment_end = symbols
            .get(index + 1)
            .map(|next| next.start_line)
            .unwrap_or(lines.len());
        out.extend(build_chunk_segment(
            relative_path,
            language,
            &lines[symbol.start_line..segment_end],
            symbol.start_line,
            (
                symbol.kind.clone(),
                symbol.name.clone(),
                Some(symbol.signature.clone()),
            ),
        ));
    }
    out
}

fn fallback_symbol(relative_path: &str, start_line: usize) -> (String, String, Option<String>) {
    (
        "chunk".to_string(),
        format!("{relative_path}:{start_line}"),
        None,
    )
}

fn discover_symbols(language: &str, lines: &[&str]) -> Vec<DiscoveredSymbol> {
    let mut out = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || is_comment_line(language, trimmed) {
            continue;
        }
        if let Some((kind, name)) = parse_signature(language, trimmed) {
            out.push(DiscoveredSymbol {
                start_line: index,
                kind: kind.to_string(),
                name,
                signature: trimmed.to_string(),
            });
        }
    }
    out
}

fn build_chunk_segment(
    relative_path: &str,
    language: &str,
    lines: &[&str],
    base_start: usize,
    metadata: (String, String, Option<String>),
) -> Vec<CodeChunkRecord> {
    if lines.is_empty() {
        return Vec::new();
    }

    let step = CHUNK_LINES.saturating_sub(CHUNK_OVERLAP).max(1);
    let mut start = 0_usize;
    let mut out = Vec::new();
    while start < lines.len() {
        let end = (start + CHUNK_LINES).min(lines.len());
        let chunk_lines = &lines[start..end];
        let snippet = truncate_chars(&chunk_lines.join("\n"), MAX_SNIPPET_CHARS);
        let (kind, name, signature) = metadata.clone();
        let search_text = match &signature {
            Some(signature) => format!("{relative_path}\n{name}\n{signature}\n{snippet}"),
            None => format!("{relative_path}\n{name}\n{snippet}"),
        };
        out.push(CodeChunkRecord {
            chunk_key: format!("{relative_path}:{}", base_start + start + 1),
            path: relative_path.to_string(),
            language: language.to_string(),
            kind,
            name,
            signature,
            snippet,
            search_text,
            start_line: (base_start + start + 1) as i64,
            end_line: (base_start + end) as i64,
        });
        if end == lines.len() {
            break;
        }
        start += step;
    }
    out
}

fn is_comment_line(language: &str, line: &str) -> bool {
    match language {
        "python" => line.starts_with('#'),
        _ => line.starts_with("//") || line.starts_with("--") || line.starts_with("/*"),
    }
}

fn parse_signature(language: &str, line: &str) -> Option<(&'static str, String)> {
    match language {
        "rust" => parse_from_prefixes(
            line,
            &[
                ("function", "pub async fn "),
                ("function", "pub fn "),
                ("function", "async fn "),
                ("function", "fn "),
                ("type", "pub struct "),
                ("type", "struct "),
                ("type", "pub enum "),
                ("type", "enum "),
                ("type", "pub trait "),
                ("type", "trait "),
                ("impl", "impl "),
            ],
        ),
        "lua" => parse_from_prefixes(
            line,
            &[("function", "local function "), ("function", "function ")],
        ),
        "python" => parse_from_prefixes(line, &[("function", "def "), ("type", "class ")]),
        "go" => parse_from_prefixes(
            line,
            &[
                ("function", "func "),
                ("type", "type "),
                ("value", "const "),
                ("value", "var "),
            ],
        ),
        "javascript" | "typescript" => parse_from_prefixes(
            line,
            &[
                ("function", "export async function "),
                ("function", "export function "),
                ("function", "async function "),
                ("function", "function "),
                ("type", "export class "),
                ("type", "class "),
                ("value", "export const "),
                ("value", "const "),
            ],
        ),
        "php" => parse_from_prefixes(line, &[("function", "function "), ("type", "class ")]),
        _ => None,
    }
}

fn parse_from_prefixes(
    line: &str,
    prefixes: &[(&'static str, &'static str)],
) -> Option<(&'static str, String)> {
    for (kind, prefix) in prefixes {
        if let Some(rest) = line.strip_prefix(prefix)
            && let Some(identifier) = extract_identifier(rest)
        {
            return Some((*kind, identifier));
        }
    }
    None
}

fn extract_identifier(input: &str) -> Option<String> {
    let mut identifier = String::new();
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | ':') {
            identifier.push(ch);
        } else if !identifier.is_empty() {
            break;
        } else if ch == '(' {
            break;
        } else if ch.is_whitespace() {
            if identifier.is_empty() {
                continue;
            }
            break;
        } else {
            break;
        }
    }
    if identifier.is_empty() {
        None
    } else {
        Some(identifier)
    }
}

fn normalize_relative_path(root: &Path, path: &Path) -> Result<String> {
    let relative = if path.is_absolute() {
        path.strip_prefix(root)
            .with_context(|| format!("'{}' is outside '{}'", path.display(), root.display()))?
            .to_path_buf()
    } else {
        let mut normalized = PathBuf::new();
        for component in path.components() {
            match component {
                Component::CurDir => {}
                Component::Normal(part) => normalized.push(part),
                Component::ParentDir => {
                    if !normalized.pop() {
                        bail!(
                            "path '{}' escapes root '{}'",
                            path.display(),
                            root.display()
                        );
                    }
                }
                Component::Prefix(_) | Component::RootDir => {
                    bail!(
                        "path '{}' is outside root '{}'",
                        path.display(),
                        root.display()
                    );
                }
            }
        }
        normalized
    };
    let normalized = relative.to_string_lossy().replace('\\', "/");
    if normalized.is_empty() {
        bail!("path must not be empty");
    }
    Ok(normalized)
}

fn detect_language(path: &Path) -> Option<&'static str> {
    match path.extension().and_then(|ext| ext.to_str())? {
        "rs" => Some("rust"),
        "lua" => Some("lua"),
        "py" => Some("python"),
        "go" => Some("go"),
        "php" => Some("php"),
        "js" | "mjs" | "cjs" => Some("javascript"),
        "ts" | "tsx" => Some("typescript"),
        _ => None,
    }
}

async fn current_timestamp(conn: &turso::Connection) -> Result<String> {
    let mut rows = conn
        .query("SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')", ())
        .await?;
    let row = rows
        .next()
        .await?
        .context("timestamp query returned no row")?;
    Ok(row.get::<String>(0)?)
}

fn file_content_hash(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn truncate_chars(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    input.chars().take(max_chars).collect::<String>()
}

const INIT_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS index_meta (
    schema_revision INTEGER NOT NULL,
    root_path TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    capabilities TEXT NOT NULL,
    codebase_id TEXT
);

CREATE TABLE IF NOT EXISTS indexed_files (
    path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    language TEXT NOT NULL,
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
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_index_reader::{CodeSearchMode, CodeSearchRequest, CodebaseSelector};
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
