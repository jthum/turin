use anyhow::{Context, Result, anyhow, bail};
use std::path::{Path, PathBuf};
use turso::Connection;

use crate::shared::{CODE_INDEX_SCHEMA_REVISION, open_index_connection};

use super::{CodeIndexCapabilities, CodebaseSelector};

#[derive(Debug, Clone)]
pub(super) struct ValidatedIndex {
    pub(super) root: PathBuf,
    pub(super) index_path: PathBuf,
    pub(super) schema_revision: i64,
    pub(super) updated_at: String,
    pub(super) index_age_seconds: u64,
    pub(super) capabilities: CodeIndexCapabilities,
}

#[derive(Debug, Clone)]
struct ResolvedCodebase {
    root: PathBuf,
    index_path: PathBuf,
}

pub(super) async fn validate_index(
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

pub(super) async fn has_optional_column(
    conn: &Connection,
    view_name: &str,
    column_name: &str,
) -> bool {
    let sql = format!("SELECT {column_name} FROM {view_name} LIMIT 0");
    conn.query(&sql, ()).await.is_ok()
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
    Ok(row.get::<Option<i64>>(0)?.unwrap_or(0).max(0) as u64)
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
