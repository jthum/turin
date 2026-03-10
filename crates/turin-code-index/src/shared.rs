use anyhow::{Context, Result};
use std::path::Path;
use turso::{Connection, Database};

pub(crate) const CODE_INDEX_SCHEMA_REVISION: i64 = 2026031003;

pub(crate) async fn open_index_connection(index_path: &Path) -> Result<(Database, Connection)> {
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

pub(crate) fn encode_vector_blob(vector: &[f32], context: &str) -> Result<Vec<u8>> {
    if vector.is_empty() {
        anyhow::bail!("{context} must not be empty");
    }

    let mut blob = Vec::with_capacity(vector.len() * std::mem::size_of::<f32>());
    for value in vector {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    Ok(blob)
}
