//! Database schema definitions and row types for Turin state store.

// ─── Schema Constants ───────────────────────────────────────────

/// Schema version — bump when changing table structure.
pub(crate) const SCHEMA_VERSION: u32 = 9;

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

-- Scoped key-value store
CREATE TABLE IF NOT EXISTS kv (
    scope_kind TEXT NOT NULL,
    scope_key  TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    expires_at TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (scope_kind, scope_key, key)
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
CREATE INDEX IF NOT EXISTS idx_kv_scope ON kv(scope_kind, scope_key);

-- Cognitive Memory
CREATE TABLE IF NOT EXISTS memories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    public_id     BLOB(16) UNIQUE NOT NULL,
    scope_kind    TEXT NOT NULL,
    scope_key     TEXT NOT NULL,
    content       TEXT NOT NULL,
    embedding     BLOB,
    embedding_key TEXT,
    embedding_dimensions INTEGER,
    metadata      TEXT,
    weight        REAL NOT NULL DEFAULT 1.0,
    retrieval_count INTEGER NOT NULL DEFAULT 0,
    last_retrieved_at TEXT,
    superseded_at TEXT,
    superseded_by_memory_id INTEGER REFERENCES memories(id),
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (metadata IS NULL OR json_valid(metadata))
);

CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope_kind, scope_key);
CREATE INDEX IF NOT EXISTS idx_memories_embedding_profile ON memories(scope_kind, scope_key, embedding_key, embedding_dimensions);

CREATE TABLE IF NOT EXISTS memory_feedback_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id   INTEGER NOT NULL REFERENCES memories(id),
    delta       REAL NOT NULL,
    reason      TEXT,
    task_id     TEXT,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_memory_feedback_events_memory ON memory_feedback_events(memory_id);

CREATE TABLE IF NOT EXISTS file_cache_versions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    path         TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    content      TEXT NOT NULL,
    content_bytes INTEGER NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(path, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_file_cache_versions_path ON file_cache_versions(path);

CREATE TABLE IF NOT EXISTS file_cache_reads (
    session_id   INTEGER NOT NULL REFERENCES sessions(id),
    path         TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    tokens_saved INTEGER NOT NULL DEFAULT 0,
    last_read_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (session_id, path)
);

CREATE INDEX IF NOT EXISTS idx_file_cache_reads_path ON file_cache_reads(path);
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
    pub public_id: Vec<u8>,
    pub scope_kind: String,
    pub scope_key: String,
    pub content: String,
    pub metadata: Option<String>,
    pub created_at: String,
    pub score: f64,
    pub lexical_score: Option<f64>,
    pub semantic_score: Option<f64>,
    pub weight: f64,
    pub retrieval_count: u64,
    pub last_retrieved_at: Option<String>,
    pub superseded_at: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryStorageKind {
    LexicalOnly,
    Embedded,
}

impl MemoryStorageKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LexicalOnly => "lexical_only",
            Self::Embedded => "embedded",
        }
    }
}

#[derive(Debug, Clone)]
pub struct StoredMemoryRow {
    pub public_id: Vec<u8>,
    pub stored_at: String,
    pub storage: MemoryStorageKind,
}

#[derive(Debug, Clone)]
pub struct MemoryFeedbackState {
    pub public_id: Vec<u8>,
    pub weight: f64,
    pub updated_at: String,
}

#[derive(Debug, Clone)]
pub struct MemoryCorrectionRow {
    pub superseded_public_id: Vec<u8>,
    pub replacement_public_id: Vec<u8>,
    pub corrected_at: String,
}

#[derive(Debug, Clone)]
pub struct MemoryPurgeReport {
    pub matched: usize,
    pub deleted: usize,
    pub dry_run: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheReadStatus {
    Fresh,
    Unchanged,
    Changed,
}

impl CacheReadStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fresh => "fresh",
            Self::Unchanged => "unchanged",
            Self::Changed => "changed",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheReadResult {
    pub status: CacheReadStatus,
    pub path: String,
    pub hash: String,
    pub previous_hash: Option<String>,
    pub content: Option<String>,
    pub previous_content: Option<String>,
    pub diff: Option<String>,
    pub diff_truncated: bool,
    pub estimated_tokens_saved: u64,
    pub read_at: String,
}

#[derive(Debug, Clone)]
pub struct CacheGlobalStats {
    pub cached_files: u64,
    pub cached_versions: u64,
    pub tokens_saved: u64,
}

#[derive(Debug, Clone)]
pub struct CacheSessionStats {
    pub public_id: Vec<u8>,
    pub files_seen: u64,
    pub tokens_saved: u64,
}

#[derive(Debug, Clone, Default)]
pub struct CacheStatsReport {
    pub global: Option<CacheGlobalStats>,
    pub session: Option<CacheSessionStats>,
}

#[derive(Debug, Clone)]
pub struct CacheResetReport {
    pub scope: String,
    pub removed_versions: u64,
    pub removed_reads: u64,
    pub reset_stats: bool,
    pub dry_run: bool,
}
