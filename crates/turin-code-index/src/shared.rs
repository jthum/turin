use anyhow::{Context, Result};
use std::path::Path;
use turso::{Connection, Database};

pub(crate) const CODE_INDEX_SCHEMA_REVISION: i64 = 20260307;

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
