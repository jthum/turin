//! Turso-backed state store for Turin.
//!
//! Provides persistent storage for:
//! - Event log (append-only)
//! - Message history (per session)
//! - Tool execution log
//! - Harness key-value store
//! - Cognitive memories (vector store)
//!
//! Schema definitions live in [`super::schema`], memory search in [`super::search`].

use anyhow::{Context, Result};
use std::sync::Arc;
use turin_daemon_protocol::SessionSearchHitKind;
use turso::{Connection, Database};

use super::schema::*;

mod events;
mod kv;
mod messages;
mod sessions;
#[cfg(test)]
mod tests;
mod tools;

/// The state store manages all Turin persistence.
///
/// It holds a reference to the database engine and spawns connections on demand.
/// This allows it to be efficiently Cloned and shared across threads.
#[derive(Clone)]
pub struct StateStore {
    pub(crate) db: Arc<Database>,
}

#[derive(Debug, Clone)]
pub struct SessionSearchRow {
    pub kind: SessionSearchHitKind,
    pub score: i64,
    pub public_id: Vec<u8>,
    pub agent_id: String,
    pub metadata: Option<String>,
    pub created_at: String,
    pub turn_index: Option<u32>,
    pub role: Option<String>,
    pub tool_name: Option<String>,
    pub event_type: Option<String>,
    pub match_text: String,
}

impl StateStore {
    /// Obtain a database connection with pragmas applied.
    ///
    /// Every connection gets `busy_timeout = 5000` so concurrent writers
    /// (e.g. background persistence + sub-agent KV calls) don't hit
    /// SQLITE_BUSY immediately.
    pub(crate) async fn connect(&self) -> Result<Connection> {
        let conn = self.db.connect()?;
        // busy_timeout is connection-local in SQLite; must be set per connection.
        conn.execute("PRAGMA busy_timeout = 5000;", ()).await.ok();
        Ok(conn)
    }

    /// Open or create a state store at the given path.
    ///
    /// Creates parent directories and initializes the schema if the database is new.
    pub async fn open(db_path: &str) -> Result<Self> {
        // Create parent directories
        let path = std::path::Path::new(db_path);
        if let Some(parent) = path.parent()
            && !parent.exists()
        {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create database directory: {}", parent.display())
            })?;
        }

        let db = turso::Builder::new_local(db_path)
            .experimental_index_method(true)
            .build()
            .await
            .with_context(|| format!("Failed to open database: {}", db_path))?;

        let store = Self { db: Arc::new(db) };
        store.init_schema().await?;

        Ok(store)
    }

    /// Open an in-memory state store (useful for testing).
    pub async fn open_memory() -> Result<Self> {
        let db = turso::Builder::new_local(":memory:")
            .experimental_index_method(true)
            .build()
            .await
            .with_context(|| "Failed to open in-memory database")?;

        let store = Self { db: Arc::new(db) };
        store.init_schema().await?;

        Ok(store)
    }

    /// Initialize the database schema.
    async fn init_schema(&self) -> Result<()> {
        let conn = self.connect().await?;

        match self.schema_version(&conn).await? {
            Some(version) if version != SCHEMA_VERSION.to_string() => {
                anyhow::bail!(
                    "State DB schema version {} is incompatible with runtime schema version {}. Delete and recreate the DB; no migration path is provided.",
                    version,
                    SCHEMA_VERSION
                );
            }
            None if self.has_user_schema(&conn).await? => {
                anyhow::bail!(
                    "State DB has an unversioned or legacy schema. Delete and recreate the DB; no migration path is provided."
                );
            }
            _ => {}
        }

        // 1. Init Core Schema
        conn.execute("PRAGMA journal_mode = WAL;", ()).await.ok();

        conn.execute_batch(INIT_SCHEMA_CORE)
            .await
            .with_context(|| "Failed to initialize database core schema")?;

        // 2. Init native Turso FTS schema. This is part of the required baseline now.
        conn.execute_batch(INIT_SCHEMA_FTS)
            .await
            .with_context(|| "Failed to initialize database FTS schema")?;

        // Record schema version
        conn.execute(
            "INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', ?1)",
            [SCHEMA_VERSION.to_string()],
        )
        .await?;

        Ok(())
    }

    async fn schema_version(&self, conn: &Connection) -> Result<Option<String>> {
        let mut rows = match conn
            .query("SELECT value FROM schema_info WHERE key = 'version'", ())
            .await
        {
            Ok(rows) => rows,
            Err(_) => return Ok(None),
        };

        if let Some(row) = rows.next().await? {
            Ok(Some(row.get::<String>(0)?))
        } else {
            Ok(None)
        }
    }

    async fn has_user_schema(&self, conn: &Connection) -> Result<bool> {
        let mut rows = conn
            .query(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'index', 'trigger') AND name NOT LIKE 'sqlite_%' LIMIT 1",
                (),
            )
            .await?;
        Ok(rows.next().await?.is_some())
    }
    /// Get a new database connection (for advanced operations).
    pub async fn get_connection(&self) -> Result<Connection> {
        self.connect().await
    }

    /// Get the underlying database (for advanced ops, e.g. shutdown).
    #[allow(dead_code)]
    pub fn database(&self) -> &Database {
        &self.db
    }
}

pub(super) fn update_session_title_metadata(
    metadata: Option<&str>,
    title: Option<&str>,
) -> Result<Option<String>> {
    let mut object = match metadata {
        Some(raw) => match serde_json::from_str::<serde_json::Value>(raw).ok() {
            Some(serde_json::Value::Object(map)) => map,
            _ => serde_json::Map::new(),
        },
        None => serde_json::Map::new(),
    };

    match title.map(str::trim).filter(|value| !value.is_empty()) {
        Some(title) => {
            object.insert(
                "title".to_string(),
                serde_json::Value::String(title.to_string()),
            );
        }
        None => {
            object.remove("title");
        }
    }

    if object.is_empty() {
        Ok(None)
    } else {
        Ok(Some(serde_json::to_string(&serde_json::Value::Object(
            object,
        ))?))
    }
}
