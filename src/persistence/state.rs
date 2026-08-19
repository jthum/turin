//! Turso-backed state store for Turin.
//!
//! Provides persistent storage for:
//! - Event log (append-only)
//! - Turn-scoped message history
//! - Turn-scoped tool execution history
//! - Harness key-value store
//! - Cognitive memories (vector store)
//!
//! Schema definitions live in `super::schema`, and memory search logic lives alongside it.

use anyhow::{Context, Result};
use std::sync::Arc;
use turin_daemon_protocol::SessionSearchHitKind;
use turso::{Connection, Database};

use super::schema::*;

mod events;
mod graph;
mod kv;
mod messages;
mod scheduler;
mod sessions;
mod signals;
#[cfg(test)]
mod tests;
mod tools;
mod turns;
mod worklists;

pub use events::SessionCounters;
pub use messages::TokenBoundedMessages;
pub use scheduler::{ScheduledJobInsert, ScheduledJobUpdate};
pub use signals::SignalInsert;
pub use worklists::{WorkItemInsert, WorkItemUpdate};

/// The state store manages all Turin persistence.
///
/// It holds a reference to the database engine and spawns connections on demand.
/// This allows it to be efficiently Cloned and shared across threads.
#[derive(Clone)]
pub struct StateStore {
    pub(crate) db: Arc<Database>,
}

#[derive(Debug, thiserror::Error)]
pub enum TurnWriteError {
    /// The selected branch head moved after the execution chose its write target.
    #[error(
        "Branch head changed while preparing turn write target: expected {expected_head_turn_id:?}, found {found_head_turn_id:?}"
    )]
    BranchHeadChanged {
        expected_head_turn_id: i64,
        found_head_turn_id: Option<i64>,
    },
}

pub fn is_turn_write_conflict(error: &anyhow::Error) -> bool {
    error.downcast_ref::<TurnWriteError>().is_some()
}

/// Selects which persisted turn path should be read for a session-scoped query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionReadTarget {
    /// Read from the session row's current active branch head.
    ActiveBranch,
    /// Read from an explicit branch head regardless of the session row's active head.
    BranchHead(i64),
    /// Read the materialized path up to a specific turn.
    TurnId(i64),
    /// Read an explicit ordered turn path.
    SelectedPath(Vec<i64>),
}

impl SessionReadTarget {
    pub fn branch_head(branch_head_id: Option<i64>) -> Self {
        match branch_head_id {
            Some(branch_head_id) => Self::BranchHead(branch_head_id),
            None => Self::ActiveBranch,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnWriteTarget {
    /// Allocate or validate the next turn on a branch head using optimistic expectations.
    BranchAdvance {
        branch_head_id: Option<i64>,
        expected_head_turn_id: Option<i64>,
        turn_index: u32,
    },
    /// Reuse a previously allocated persisted turn row.
    ExistingTurn { turn_id: i64, turn_index: u32 },
}

impl TurnWriteTarget {
    pub const fn active_branch(turn_index: u32) -> Self {
        Self::BranchAdvance {
            branch_head_id: None,
            expected_head_turn_id: None,
            turn_index,
        }
    }

    pub const fn branch_head(branch_head_id: Option<i64>, turn_index: u32) -> Self {
        Self::BranchAdvance {
            branch_head_id,
            expected_head_turn_id: None,
            turn_index,
        }
    }

    pub const fn branch_head_with_expectation(
        branch_head_id: Option<i64>,
        expected_head_turn_id: Option<i64>,
        turn_index: u32,
    ) -> Self {
        Self::BranchAdvance {
            branch_head_id,
            expected_head_turn_id,
            turn_index,
        }
    }

    pub const fn existing_turn(turn_id: i64, turn_index: u32) -> Self {
        Self::ExistingTurn {
            turn_id,
            turn_index,
        }
    }

    pub const fn turn_index(self) -> u32 {
        match self {
            Self::BranchAdvance { turn_index, .. } | Self::ExistingTurn { turn_index, .. } => {
                turn_index
            }
        }
    }
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
