use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};

const CODE_INDEX_SCHEMA_REVISION: i64 = 20260305;
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

pub async fn build_index(root: &Path, index_path: Option<&Path>) -> Result<CodeIndexBuildReport> {
    let root = std::fs::canonicalize(root)
        .with_context(|| format!("root '{}' does not exist", root.display()))?;
    let index_path = resolve_index_path(&root, index_path)?;
    if let Some(parent) = index_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create '{}'", parent.display()))?;
    }
    if index_path.exists() {
        std::fs::remove_file(&index_path)
            .with_context(|| format!("failed to replace '{}'", index_path.display()))?;
    }

    let db = turso::Builder::new_local(index_path.to_str().unwrap())
        .experimental_index_method(true)
        .build()
        .await
        .with_context(|| format!("failed to open '{}'", index_path.display()))?;
    let conn = db.connect()?;
    conn.execute("PRAGMA busy_timeout = 5000;", ()).await.ok();
    conn.execute_batch(INIT_SCHEMA)
        .await
        .context("failed to initialize code index schema")?;

    let files = collect_indexable_files(&root)?;
    let mut files_indexed = 0_u64;
    let mut chunks_indexed = 0_u64;
    let mut languages = BTreeSet::new();

    for file in files {
        let chunks = build_chunks(&root, &file)?;
        if chunks.is_empty() {
            continue;
        }
        files_indexed += 1;
        for chunk in chunks {
            languages.insert(chunk.language.clone());
            conn.execute(
                "INSERT INTO code_chunks (chunk_key, path, language, kind, name, signature, snippet, search_text, start_line, end_line) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                turso::params![
                    chunk.chunk_key,
                    chunk.path,
                    chunk.language,
                    chunk.kind,
                    chunk.name,
                    chunk.signature,
                    chunk.snippet,
                    chunk.search_text,
                    chunk.start_line,
                    chunk.end_line
                ],
            )
            .await?;
            chunks_indexed += 1;
        }
    }

    let capabilities = CodeIndexWriteCapabilities {
        lexical: true,
        semantic: false,
        hybrid: false,
        languages: languages.into_iter().collect(),
    };
    let updated_at = current_timestamp(&conn).await?;
    let codebase_id = root
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string);
    conn.execute(
        "INSERT INTO index_meta (schema_revision, root_path, updated_at, capabilities, codebase_id) VALUES (?1, ?2, ?3, ?4, ?5)",
        turso::params![
            CODE_INDEX_SCHEMA_REVISION,
            root.to_string_lossy().to_string(),
            updated_at.clone(),
            serde_json::to_string(&capabilities)?,
            codebase_id
        ],
    )
    .await?;

    Ok(CodeIndexBuildReport {
        root: root.to_string_lossy().to_string(),
        index_path: index_path.to_string_lossy().to_string(),
        schema_revision: CODE_INDEX_SCHEMA_REVISION,
        updated_at,
        capabilities,
        files_indexed,
        chunks_indexed,
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

    let db = turso::Builder::new_local(index_path.to_str().unwrap())
        .experimental_index_method(true)
        .build()
        .await
        .with_context(|| format!("failed to open '{}'", index_path.display()))?;
    let conn = db.connect()?;
    let changed = conn
        .execute(
            "DELETE FROM code_chunks WHERE path = ?1",
            turso::params![relative_path.clone()],
        )
        .await?;
    let updated_at = current_timestamp(&conn).await?;
    conn.execute(
        "UPDATE index_meta SET updated_at = ?1",
        turso::params![updated_at.clone()],
    )
    .await?;

    Ok(CodeIndexRemoveReport {
        root: root.to_string_lossy().to_string(),
        index_path: index_path.to_string_lossy().to_string(),
        path: relative_path,
        removed_chunks: changed as u64,
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

fn collect_indexable_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    collect_indexable_files_recursive(root, root, &mut out)?;
    out.sort();
    Ok(out)
}

fn collect_indexable_files_recursive(
    root: &Path,
    current: &Path,
    out: &mut Vec<PathBuf>,
) -> Result<()> {
    for entry in std::fs::read_dir(current)
        .with_context(|| format!("failed to read '{}'", current.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if file_type.is_dir() {
            if should_skip_dir(&name) {
                continue;
            }
            collect_indexable_files_recursive(root, &path, out)?;
            continue;
        }
        if !file_type.is_file() || detect_language(&path).is_none() {
            continue;
        }
        if path.strip_prefix(root).ok().is_some_and(|relative| {
            relative
                .components()
                .any(|component| matches!(component, Component::Normal(part) if part == ".turin"))
        }) {
            continue;
        }
        out.push(path);
    }
    Ok(())
}

fn should_skip_dir(name: &str) -> bool {
    matches!(name, ".git" | ".turin" | "target" | "node_modules")
}

fn build_chunks(root: &Path, file: &Path) -> Result<Vec<CodeChunkRecord>> {
    let language = match detect_language(file) {
        Some(language) => language,
        None => return Ok(Vec::new()),
    };
    let bytes = std::fs::read(file)?;
    if bytes.len() > MAX_FILE_BYTES {
        return Ok(Vec::new());
    }
    let content = match String::from_utf8(bytes) {
        Ok(content) => content,
        Err(_) => return Ok(Vec::new()),
    };
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    let relative_path = normalize_relative_path(root, file)?;
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Ok(Vec::new());
    }

    let step = CHUNK_LINES.saturating_sub(CHUNK_OVERLAP).max(1);
    let mut start = 0_usize;
    let mut out = Vec::new();
    while start < lines.len() {
        let end = (start + CHUNK_LINES).min(lines.len());
        let chunk_lines = &lines[start..end];
        let snippet = truncate_chars(&chunk_lines.join("\n"), MAX_SNIPPET_CHARS);
        let (kind, name, signature) = infer_symbol(language, chunk_lines)
            .unwrap_or_else(|| fallback_symbol(&relative_path, start + 1));
        let search_text = match &signature {
            Some(signature) => format!("{relative_path}\n{name}\n{signature}\n{snippet}"),
            None => format!("{relative_path}\n{name}\n{snippet}"),
        };
        out.push(CodeChunkRecord {
            chunk_key: format!("{relative_path}:{}", start + 1),
            path: relative_path.clone(),
            language: language.to_string(),
            kind,
            name,
            signature,
            snippet,
            search_text,
            start_line: (start + 1) as i64,
            end_line: end as i64,
        });
        if end == lines.len() {
            break;
        }
        start += step;
    }
    Ok(out)
}

fn fallback_symbol(relative_path: &str, start_line: usize) -> (String, String, Option<String>) {
    (
        "chunk".to_string(),
        format!("{relative_path}:{start_line}"),
        None,
    )
}

fn infer_symbol(language: &str, lines: &[&str]) -> Option<(String, String, Option<String>)> {
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || is_comment_line(language, trimmed) {
            continue;
        }
        if let Some((kind, name)) = parse_signature(language, trimmed) {
            return Some((kind.to_string(), name, Some(trimmed.to_string())));
        }
    }
    None
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

fn truncate_chars(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    input.chars().take(max_chars).collect::<String>()
}

const INIT_SCHEMA: &str = r#"
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
    search_text TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL
);

CREATE INDEX idx_code_chunks_path ON code_chunks(path);
CREATE INDEX idx_code_chunks_search_fts ON code_chunks USING fts(search_text);

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

        let rows = crate::code_index_reader::search(
            tmp.path(),
            CodebaseSelector {
                root: "repo".to_string(),
                index_path: None,
            },
            CodeSearchMode::Lexical,
            "capability_decision",
            &CodeSearchRequest {
                limit: 5,
                ..CodeSearchRequest::default()
            },
        )
        .await?;
        assert!(!rows.is_empty());
        assert_eq!(rows[0].name, "capability_decision");
        assert!(rows[0].score > 0.0);

        let removed = remove_file(&root, None, Path::new("src/governance.rs")).await?;
        assert!(removed.removed_chunks >= 1);

        let rows_after_remove = crate::code_index_reader::search(
            tmp.path(),
            CodebaseSelector {
                root: "repo".to_string(),
                index_path: None,
            },
            CodeSearchMode::Lexical,
            "capability_decision",
            &CodeSearchRequest::default(),
        )
        .await?;
        assert!(rows_after_remove.is_empty());

        Ok(())
    }
}
