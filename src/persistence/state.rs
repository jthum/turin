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
use turso::{Connection, Database};

use super::schema::*;

/// The state store manages all Turin persistence.
///
/// It holds a reference to the database engine and spawns connections on demand.
/// This allows it to be efficiently Cloned and shared across threads.
#[derive(Clone)]
pub struct StateStore {
    pub(crate) db: Arc<Database>,
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
    // ─── Sessions ────────────────────────────────────────────────

    /// Create a new session in the database, returning its internal ID.
    pub async fn create_session(
        &self,
        public_id: uuid::Uuid,
        agent_id: &str,
        metadata: Option<&str>,
    ) -> Result<i64> {
        let conn = self.connect().await?;
        let public_id_bytes = public_id.into_bytes().to_vec();

        conn.execute(
            "INSERT INTO sessions (public_id, agent_id, metadata) VALUES (?1, ?2, ?3)",
            turso::params![public_id_bytes, agent_id, metadata],
        )
        .await
        .context("Failed to insert into sessions table")?;

        let id = conn.last_insert_rowid();
        Ok(id)
    }

    /// Retrieve a session's internal ID given its public Uuid.
    pub async fn get_session_by_public_id(&self, public_id: uuid::Uuid) -> Result<Option<i64>> {
        let conn = self.connect().await?;
        let public_id_bytes = public_id.into_bytes().to_vec();

        let mut rows = conn
            .query(
                "SELECT id FROM sessions WHERE public_id = ?1",
                turso::params![public_id_bytes],
            )
            .await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(row.get(0)?))
        } else {
            Ok(None)
        }
    }

    /// Retrieve a full session row by internal ID.
    pub async fn get_session_row(&self, session_id: i64) -> Result<Option<SessionRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT id, public_id, agent_id, metadata, created_at FROM sessions WHERE id = ?1",
                [session_id],
            )
            .await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(SessionRow {
                id: row.get::<i64>(0)?,
                public_id: row.get::<Vec<u8>>(1)?,
                agent_id: row.get::<String>(2)?,
                metadata: row.get::<Option<String>>(3)?,
                created_at: row.get::<String>(4)?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Retrieve a full session row by public UUID.
    pub async fn get_session_row_by_public_id(
        &self,
        public_id: uuid::Uuid,
    ) -> Result<Option<SessionRow>> {
        let conn = self.connect().await?;
        let public_id_bytes = public_id.into_bytes().to_vec();
        let mut rows = conn
            .query(
                "SELECT id, public_id, agent_id, metadata, created_at FROM sessions WHERE public_id = ?1",
                turso::params![public_id_bytes],
            )
            .await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(SessionRow {
                id: row.get::<i64>(0)?,
                public_id: row.get::<Vec<u8>>(1)?,
                agent_id: row.get::<String>(2)?,
                metadata: row.get::<Option<String>>(3)?,
                created_at: row.get::<String>(4)?,
            }))
        } else {
            Ok(None)
        }
    }

    // ─── Event Log ───────────────────────────────────────────────

    /// Persist a KernelEvent to the event log.
    pub async fn insert_event(
        &self,
        session_id: i64,
        event_type: &str,
        payload: &serde_json::Value,
    ) -> Result<()> {
        let conn = self.connect().await?;
        let payload_str = serde_json::to_string(payload)?;
        conn.execute(
            "INSERT INTO events (session_id, event_type, payload) VALUES (?1, ?2, ?3)",
            turso::params![session_id, event_type, payload_str],
        )
        .await
        .with_context(|| format!("Failed to insert event for session: {}", session_id))?;
        Ok(())
    }

    /// Get all events for a session, ordered by creation time.
    pub async fn get_events(&self, session_id: i64) -> Result<Vec<EventRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT id, session_id, event_type, payload, created_at FROM events WHERE session_id = ?1 ORDER BY id",
                [session_id],
            )
            .await?;

        let mut events = Vec::new();
        while let Some(row) = rows.next().await? {
            events.push(EventRow {
                id: row.get::<i64>(0)?,
                session_id: row.get::<i64>(1)?,
                event_type: row.get::<String>(2)?,
                payload: row.get::<String>(3)?,
                created_at: row.get::<String>(4)?,
            });
        }
        Ok(events)
    }

    /// List recent sessions, ordered by last activity.
    pub async fn list_sessions(&self, limit: usize, offset: usize) -> Result<Vec<i64>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT session_id FROM events GROUP BY session_id ORDER BY MAX(id) DESC LIMIT ?1 OFFSET ?2",
                turso::params![limit as i64, offset as i64],
            )
            .await?;

        let mut sessions = Vec::new();
        while let Some(row) = rows.next().await? {
            sessions.push(row.get(0)?);
        }
        Ok(sessions)
    }

    /// List recent session rows ordered by last activity.
    pub async fn list_session_rows(&self, limit: usize, offset: usize) -> Result<Vec<SessionRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT s.id, s.public_id, s.agent_id, s.metadata, s.created_at
                FROM sessions s
                LEFT JOIN events e ON e.session_id = s.id
                GROUP BY s.id
                ORDER BY COALESCE(MAX(e.id), s.id) DESC
                LIMIT ?1 OFFSET ?2
                "#,
                turso::params![limit as i64, offset as i64],
            )
            .await?;

        let mut sessions = Vec::new();
        while let Some(row) = rows.next().await? {
            sessions.push(SessionRow {
                id: row.get::<i64>(0)?,
                public_id: row.get::<Vec<u8>>(1)?,
                agent_id: row.get::<String>(2)?,
                metadata: row.get::<Option<String>>(3)?,
                created_at: row.get::<String>(4)?,
            });
        }
        Ok(sessions)
    }

    // ─── Message History ─────────────────────────────────────────

    /// Insert a message into the history.
    pub async fn insert_message(
        &self,
        session_id: i64,
        turn_index: u32,
        role: &str,
        content: &serde_json::Value,
        token_count: Option<u64>,
    ) -> Result<()> {
        let conn = self.connect().await?;
        let content_str = serde_json::to_string(content)?;
        conn
            .execute(
                "INSERT INTO messages (session_id, turn_index, role, content, token_count) VALUES (?1, ?2, ?3, ?4, ?5)",
                turso::params![
                    session_id,
                    turn_index as i64,
                    role,
                    content_str,
                    token_count.map(|t| t as i64),
                ],
            )
            .await
            .with_context(|| format!("Failed to insert message for session: {}", session_id))?;
        Ok(())
    }

    /// Get all messages for a session.
    pub async fn get_messages(&self, session_id: i64) -> Result<Vec<MessageRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT id, session_id, turn_index, role, content, token_count, created_at FROM messages WHERE session_id = ?1 ORDER BY id",
                [session_id],
            )
            .await?;

        let mut messages = Vec::new();
        while let Some(row) = rows.next().await? {
            messages.push(MessageRow {
                id: row.get::<i64>(0)?,
                session_id: row.get::<i64>(1)?,
                turn_index: row.get::<i64>(2)? as u32,
                role: row.get::<String>(3)?,
                content: row.get::<String>(4)?,
                token_count: row.get::<Option<i64>>(5)?.map(|t| t as u64),
                created_at: row.get::<String>(6)?,
            });
        }
        Ok(messages)
    }

    // ─── Tool Executions ─────────────────────────────────────────

    /// Log a tool execution.
    #[allow(clippy::too_many_arguments)]
    pub async fn insert_tool_execution(
        &self,
        session_id: i64,
        turn_index: u32,
        tool_call_id: &str,
        tool_name: &str,
        args: &serde_json::Value,
        output: Option<&str>,
        is_error: bool,
        duration_ms: Option<u64>,
        verdict: &str,
    ) -> Result<()> {
        let conn = self.connect().await?;
        let args_str = serde_json::to_string(args)?;
        conn
            .execute(
                "INSERT INTO tool_executions (session_id, turn_index, tool_call_id, tool_name, args, output, is_error, duration_ms, verdict) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                turso::params![
                    session_id,
                    turn_index as i64,
                    tool_call_id,
                    tool_name,
                    args_str,
                    output,
                    is_error as i64,
                    duration_ms.map(|d| d as i64),
                    verdict,
                ],
            )
            .await
            .with_context(|| format!("Failed to insert tool execution for session: {}", session_id))?;
        Ok(())
    }

    /// Get all tool executions for a session.
    pub async fn get_tool_executions(&self, session_id: i64) -> Result<Vec<ToolExecutionRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT id, session_id, turn_index, tool_call_id, tool_name, args, output, is_error, duration_ms, verdict, created_at FROM tool_executions WHERE session_id = ?1 ORDER BY id",
                [session_id],
            )
            .await?;

        let mut execs = Vec::new();
        while let Some(row) = rows.next().await? {
            execs.push(ToolExecutionRow {
                id: row.get::<i64>(0)?,
                session_id: row.get::<i64>(1)?,
                turn_index: row.get::<i64>(2)? as u32,
                tool_call_id: row.get::<String>(3)?,
                tool_name: row.get::<String>(4)?,
                args: row.get::<String>(5)?,
                output: row.get::<Option<String>>(6)?,
                is_error: row.get::<i64>(7)? != 0,
                duration_ms: row.get::<Option<i64>>(8)?.map(|d| d as u64),
                verdict: row.get::<String>(9)?,
                created_at: row.get::<String>(10)?,
            });
        }
        Ok(execs)
    }

    // ─── Harness KV Store ────────────────────────────────────────

    /// Set a key-value pair in the harness store.
    pub async fn kv_set(&self, key: &str, value: &str) -> Result<()> {
        const MAX_KV_VALUE_SIZE: usize = 1_048_576; // 1MB

        if value.len() > MAX_KV_VALUE_SIZE {
            anyhow::bail!(
                "KV value exceeds maximum size of {} bytes (got {})",
                MAX_KV_VALUE_SIZE,
                value.len()
            );
        }

        let conn = self.connect().await?;
        conn
            .execute(
                "INSERT OR REPLACE INTO harness_kv (key, value, updated_at) VALUES (?1, ?2, datetime('now'))",
                turso::params![key, value],
            )
            .await
            .with_context(|| format!("Failed to set KV pair for key: {}", key))?;
        Ok(())
    }

    /// Get a value from the harness store.
    pub async fn kv_get(&self, key: &str) -> Result<Option<String>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT value FROM harness_kv WHERE key = ?1 AND (expires_at IS NULL OR expires_at > datetime('now'))",
                [key],
            )
            .await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(row.get::<String>(0)?))
        } else {
            Ok(None)
        }
    }

    /// Delete a key from the harness store.
    pub async fn kv_delete(&self, key: &str) -> Result<()> {
        let conn = self.connect().await?;
        conn.execute("DELETE FROM harness_kv WHERE key = ?1", [key])
            .await?;
        Ok(())
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

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_schema_initialization() {
        let store = StateStore::open_memory().await.unwrap();

        // Check schema version
        let conn = store.get_connection().await.unwrap();
        let mut rows = conn
            .query("SELECT value FROM schema_info WHERE key = 'version'", ())
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let version: String = row.get(0).unwrap();
        assert_eq!(version, SCHEMA_VERSION.to_string());
    }

    #[tokio::test]
    async fn test_insert_and_get_events() {
        let store = StateStore::open_memory().await.unwrap();
        let session = 1;

        store
            .insert_event(session, "session_start", &json!({"session_id": session}))
            .await
            .unwrap();
        store
            .insert_event(session, "turn_start", &json!({"turn_index": 0}))
            .await
            .unwrap();

        let events = store.get_events(session).await.unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, "session_start");
        assert_eq!(events[1].event_type, "turn_start");
    }

    #[tokio::test]
    async fn test_events_isolated_by_session() {
        let store = StateStore::open_memory().await.unwrap();

        store
            .insert_event(1, "session_start", &json!({}))
            .await
            .unwrap();
        store
            .insert_event(2, "session_start", &json!({}))
            .await
            .unwrap();

        let events_a = store.get_events(1).await.unwrap();
        let events_b = store.get_events(2).await.unwrap();
        assert_eq!(events_a.len(), 1);
        assert_eq!(events_b.len(), 1);
    }

    #[tokio::test]
    async fn test_insert_and_get_messages() {
        let store = StateStore::open_memory().await.unwrap();
        let session = 1;

        store
            .insert_message(
                session,
                0,
                "user",
                &json!([{"type": "text", "text": "hello"}]),
                None,
            )
            .await
            .unwrap();
        store
            .insert_message(
                session,
                0,
                "assistant",
                &json!([{"type": "text", "text": "hi!"}]),
                Some(10),
            )
            .await
            .unwrap();

        let msgs = store.get_messages(session).await.unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[1].role, "assistant");
        assert_eq!(msgs[1].token_count, Some(10));
    }

    #[tokio::test]
    async fn test_insert_and_get_tool_executions() {
        let store = StateStore::open_memory().await.unwrap();
        let session = 1;

        store
            .insert_tool_execution(
                session,
                0,
                "call_1",
                "read_file",
                &json!({"path": "main.rs"}),
                Some("fn main() {}"),
                false,
                Some(15),
                "allow",
            )
            .await
            .unwrap();

        let execs = store.get_tool_executions(session).await.unwrap();
        assert_eq!(execs.len(), 1);
        assert_eq!(execs[0].tool_name, "read_file");
        assert_eq!(execs[0].output, Some("fn main() {}".to_string()));
        assert!(!execs[0].is_error);
        assert_eq!(execs[0].duration_ms, Some(15));
        assert_eq!(execs[0].verdict, "allow");
    }

    #[tokio::test]
    async fn test_tool_execution_with_error() {
        let store = StateStore::open_memory().await.unwrap();
        let session = 1;

        store
            .insert_tool_execution(
                session,
                1,
                "call_2",
                "shell_exec",
                &json!({"command": "rm -rf /"}),
                Some("Permission denied"),
                true,
                Some(5),
                "reject",
            )
            .await
            .unwrap();

        let execs = store.get_tool_executions(session).await.unwrap();
        assert_eq!(execs.len(), 1);
        assert!(execs[0].is_error);
        assert_eq!(execs[0].verdict, "reject");
    }

    #[tokio::test]
    async fn test_kv_set_get_delete() {
        let store = StateStore::open_memory().await.unwrap();

        // Set
        store.kv_set("budget_remaining", "1000").await.unwrap();

        // Get
        let val = store.kv_get("budget_remaining").await.unwrap();
        assert_eq!(val, Some("1000".to_string()));

        // Update
        store.kv_set("budget_remaining", "500").await.unwrap();
        let val = store.kv_get("budget_remaining").await.unwrap();
        assert_eq!(val, Some("500".to_string()));

        // Delete
        store.kv_delete("budget_remaining").await.unwrap();
        let val = store.kv_get("budget_remaining").await.unwrap();
        assert_eq!(val, None);
    }

    #[tokio::test]
    async fn test_kv_get_nonexistent() {
        let store = StateStore::open_memory().await.unwrap();
        let val = store.kv_get("nonexistent").await.unwrap();
        assert_eq!(val, None);
    }

    #[tokio::test]
    async fn test_file_based_store() {
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("test.db");
        let db_path_str = db_path.to_str().unwrap();

        // Create and populate
        {
            let store = StateStore::open(db_path_str).await.unwrap();
            store
                .insert_event(1, "session_start", &json!({}))
                .await
                .unwrap();
            store.kv_set("key1", "value1").await.unwrap();
        }

        // Reopen and verify persistence
        {
            let store = StateStore::open(db_path_str).await.unwrap();
            let events = store.get_events(1).await.unwrap();
            assert_eq!(events.len(), 1);

            let val = store.kv_get("key1").await.unwrap();
            assert_eq!(val, Some("value1".to_string()));
        }
    }

    #[tokio::test]
    async fn test_hybrid_search() {
        let store = StateStore::open_memory()
            .await
            .expect("Failed to open state store");

        let session = 1;

        // Insert memories
        store
            .insert_memory(session, "The secret code is 12345", Some(&[1.0, 0.0]), &json!({}))
            .await
            .unwrap();

        store
            .insert_memory(session, "Apples are red", Some(&[0.0, 1.0]), &json!({}))
            .await
            .unwrap();

        // Test 1: Vector Search
        let results = store
            .search_memories(session, Some(&[1.0, 0.0]), None, 10, 0.0, false, false)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].content.contains("secret code"));
        assert!(results[0].semantic_score.is_some());

        // Test 2: Lexical Search
        let results = store
            .search_memories(session, None, Some("12345"), 10, 0.0, false, false)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("secret code"));
        assert!(results[0].lexical_score.is_some());

        // Test 3: Hybrid Search
        let results = store
            .search_memories(session, Some(&[0.0, 1.0]), Some("12345"), 10, 0.0, false, false)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        let found_secret = results.iter().any(|r| r.content.contains("secret code"));
        let found_apples = results.iter().any(|r| r.content.contains("Apples"));
        assert!(found_secret, "Hybrid search missing lexical result");
        assert!(found_apples, "Hybrid search missing vector result");
    }

    #[tokio::test]
    async fn test_lexical_search_without_embeddings() {
        let store = StateStore::open_memory()
            .await
            .expect("Failed to open state store");
        let session = 1;

        store
            .insert_memory(session, "The secret code is 12345", None, &json!({}))
            .await
            .unwrap();

        let results = store
            .search_memories(session, None, Some("12345"), 10, 0.0, false, false)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("secret code"));

        let hybrid_results = store
            .search_memories(session, Some(&[1.0, 0.0]), Some("12345"), 10, 0.0, false, false)
            .await
            .unwrap();
        assert_eq!(hybrid_results.len(), 1);
        assert!(hybrid_results[0].content.contains("secret code"));
    }

    #[tokio::test]
    async fn test_memory_store_returns_public_id_and_updates_retrieval_metadata() {
        let store = StateStore::open_memory()
            .await
            .expect("Failed to open state store");
        let session = 1;

        let stored = store
            .insert_memory(
                session,
                "alpha memory",
                None,
                &json!({ "kind": "note", "source": "test" }),
            )
            .await
            .expect("memory insert should succeed");

        assert_eq!(stored.public_id.len(), 16);
        assert_eq!(stored.storage.as_str(), "lexical_only");
        assert!(stored.stored_at.contains('T'));

        let rows = store
            .search_memories(session, None, Some("alpha"), 5, 0.0, true, false)
            .await
            .expect("memory search should succeed");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].public_id, stored.public_id);
        assert_eq!(rows[0].retrieval_count, 1);
        assert!(rows[0].last_retrieved_at.is_some());
        let metadata = rows[0].metadata.as_deref().expect("metadata should be present");
        assert!(metadata.contains("\"kind\":\"note\""));
        assert!(metadata.contains("\"source\":\"test\""));
    }
}
