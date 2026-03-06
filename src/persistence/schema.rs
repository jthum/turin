//! Database schema definitions and row types for Turin state store.

// ─── Schema Constants ───────────────────────────────────────────

/// Schema version — bump when changing table structure.
pub(crate) const SCHEMA_VERSION: u32 = 4;

/// SQL statements to initialize the core database schema.
pub(crate) const INIT_SCHEMA_CORE: &str = r#"
-- Core routing and identity envelope
CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    public_id  BLOB(16) UNIQUE NOT NULL,
    agent_id   TEXT NOT NULL,
    metadata   TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Core event log (append-only)
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    event_type  TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Message history (per session)
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    turn_index  INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    token_count INTEGER,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Harness key-value store
CREATE TABLE IF NOT EXISTS harness_kv (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    expires_at  TEXT,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Tool execution log
CREATE TABLE IF NOT EXISTS tool_executions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES sessions(id),
    turn_index    INTEGER NOT NULL,
    tool_call_id  TEXT NOT NULL,
    tool_name     TEXT NOT NULL,
    args          TEXT NOT NULL,
    output        TEXT,
    is_error      INTEGER NOT NULL DEFAULT 0,
    duration_ms   INTEGER,
    verdict       TEXT NOT NULL DEFAULT 'allow',
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_info (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_executions_session ON tool_executions(session_id);

-- Cognitive Memory
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    content     TEXT NOT NULL,
    embedding   F32_BLOB(1536),
    metadata    TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"#;

/// Native Turso FTS schema
pub(crate) const INIT_SCHEMA_FTS: &str = r#"
CREATE INDEX IF NOT EXISTS idx_memories_fts ON memories USING fts (content);
"#;

// ─── Row Types ───────────────────────────────────────────────

/// A row from the `sessions` table.
#[derive(Debug, Clone)]
pub struct SessionRow {
    pub id: i64,
    pub public_id: Vec<u8>,
    pub agent_id: String,
    pub metadata: Option<String>,
    pub created_at: String,
}

/// A row from the `events` table.
#[derive(Debug, Clone)]
pub struct EventRow {
    pub id: i64,
    pub session_id: i64,
    pub event_type: String,
    pub payload: String,
    pub created_at: String,
}

/// A row from the `messages` table.
#[derive(Debug, Clone)]
pub struct MessageRow {
    pub id: i64,
    pub session_id: i64,
    pub turn_index: u32,
    pub role: String,
    pub content: String,
    pub token_count: Option<u64>,
    pub created_at: String,
}

/// A row from the `tool_executions` table.
#[derive(Debug, Clone)]
pub struct ToolExecutionRow {
    pub id: i64,
    pub session_id: i64,
    pub turn_index: u32,
    pub tool_call_id: String,
    pub tool_name: String,
    pub args: String,
    pub output: Option<String>,
    pub is_error: bool,
    pub duration_ms: Option<u64>,
    pub verdict: String,
    pub created_at: String,
}

/// A row from the `memories` table.
#[derive(Debug, Clone)]
pub struct MemoryRow {
    pub id: i64,
    pub session_id: i64,
    pub content: String,
    pub metadata: String,
    pub created_at: String,
    pub score: f64,
}
