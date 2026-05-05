//! Database schema definitions and row types for Turin state store.

// ─── Schema Constants ───────────────────────────────────────────

/// Schema version — bump when changing table structure.
pub(crate) const SCHEMA_VERSION: u32 = 18;

/// SQL statements to initialize the core database schema.
pub(crate) const INIT_SCHEMA_CORE: &str = r#"
-- Core routing and identity envelope
CREATE TABLE IF NOT EXISTS sessions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    public_id             BLOB(16) UNIQUE NOT NULL,
    agent_id              TEXT NOT NULL,
    metadata              TEXT,
    active_branch_head_id INTEGER REFERENCES branch_heads(id),
    created_at            TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Core event log (append-only)
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    turn_id     INTEGER REFERENCES turns(id),
    event_type  TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Branch-native turn graph
CREATE TABLE IF NOT EXISTS turns (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    public_id     BLOB(16) UNIQUE NOT NULL,
    session_id    INTEGER NOT NULL REFERENCES sessions(id),
    parent_turn_id INTEGER REFERENCES turns(id),
    branch_depth  INTEGER NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS branch_heads (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    public_id            BLOB(16) UNIQUE NOT NULL,
    session_id           INTEGER NOT NULL REFERENCES sessions(id),
    name                 TEXT NOT NULL,
    head_turn_id         INTEGER REFERENCES turns(id),
    created_from_turn_id INTEGER REFERENCES turns(id),
    origin_kind          TEXT NOT NULL DEFAULT 'manual',
    origin_task_id       TEXT,
    origin_execution_id  TEXT,
    origin_metadata      TEXT,
    created_at           TEXT NOT NULL DEFAULT (datetime('now')),
    CHECK (origin_metadata IS NULL OR json_valid(origin_metadata)),
    UNIQUE(session_id, name)
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id      INTEGER NOT NULL REFERENCES turns(id),
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    token_count INTEGER,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS tool_executions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id      INTEGER NOT NULL REFERENCES turns(id),
    tool_call_id TEXT NOT NULL,
    tool_name    TEXT NOT NULL,
    args         TEXT NOT NULL,
    output       TEXT,
    is_error     INTEGER NOT NULL DEFAULT 0,
    duration_ms  INTEGER,
    verdict      TEXT NOT NULL DEFAULT 'allow',
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Sparse semantic graph overlay. This stores opt-in meaning across existing
-- entities without replacing the turn tree or branch heads.
CREATE TABLE IF NOT EXISTS graph_nodes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    public_id           BLOB(16) UNIQUE NOT NULL,
    session_id          INTEGER REFERENCES sessions(id),
    kind                TEXT NOT NULL,
    label               TEXT,
    origin_task_id      TEXT,
    origin_execution_id TEXT,
    metadata            TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    CHECK (metadata IS NULL OR json_valid(metadata))
);

CREATE TABLE IF NOT EXISTS graph_edges (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    public_id           BLOB(16) UNIQUE NOT NULL,
    session_id          INTEGER REFERENCES sessions(id),
    source_kind         TEXT NOT NULL,
    source_id           TEXT NOT NULL,
    target_kind         TEXT NOT NULL,
    target_id           TEXT NOT NULL,
    relation_kind       TEXT NOT NULL,
    source_role         TEXT,
    target_role         TEXT,
    origin_task_id      TEXT,
    origin_execution_id TEXT,
    metadata            TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    CHECK (metadata IS NULL OR json_valid(metadata))
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

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_info (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_turn ON events(turn_id);
CREATE INDEX IF NOT EXISTS idx_kv_scope ON kv(scope_kind, scope_key);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_parent ON turns(parent_turn_id);
CREATE INDEX IF NOT EXISTS idx_turns_session_depth ON turns(session_id, branch_depth);
CREATE INDEX IF NOT EXISTS idx_branch_heads_session ON branch_heads(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_turn ON messages(turn_id);
CREATE INDEX IF NOT EXISTS idx_tool_executions_turn ON tool_executions(turn_id);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_session_kind ON graph_nodes(session_id, kind);
CREATE INDEX IF NOT EXISTS idx_graph_edges_session_relation ON graph_edges(session_id, relation_kind);
CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_kind, source_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_kind, target_id);

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

-- Durable scheduler jobs
CREATE TABLE IF NOT EXISTS scheduled_jobs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    public_id        BLOB(16) UNIQUE NOT NULL,
    agent_id         TEXT NOT NULL,
    prompt           TEXT NOT NULL,
    state_target     TEXT,
    store_target     TEXT,
    next_run_unix_ms INTEGER NOT NULL,
    interval_seconds INTEGER,
    overlap_policy   TEXT NOT NULL DEFAULT 'skip',
    enabled          INTEGER NOT NULL DEFAULT 1,
    running_task_id  TEXT,
    pending_rerun    INTEGER NOT NULL DEFAULT 0,
    last_run_unix_ms INTEGER,
    last_status      TEXT,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_due ON scheduled_jobs(enabled, next_run_unix_ms);
CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_running ON scheduled_jobs(running_task_id);

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
    pub active_branch_head_id: Option<i64>,
    pub created_at: String,
}

/// A row from the `events` table.
#[derive(Debug, Clone)]
pub struct EventRow {
    pub id: i64,
    pub session_id: i64,
    pub turn_id: Option<i64>,
    pub event_type: String,
    pub payload: String,
    pub turn_index: Option<u32>,
    pub created_at: String,
}

/// A turn-scoped message row materialized for a selected session path.
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

/// A row from the `turns` table.
#[derive(Debug, Clone)]
pub struct TurnRow {
    pub id: i64,
    pub public_id: Vec<u8>,
    pub session_id: i64,
    pub parent_turn_id: Option<i64>,
    pub branch_depth: u32,
    pub created_at: String,
}

/// A row from the `branch_heads` table.
#[derive(Debug, Clone)]
pub struct BranchHeadRow {
    pub id: i64,
    pub public_id: Vec<u8>,
    pub session_id: i64,
    pub name: String,
    pub head_turn_id: Option<i64>,
    pub head_turn_depth: Option<u32>,
    pub created_from_turn_id: Option<i64>,
    pub origin_kind: String,
    pub origin_task_id: Option<String>,
    pub origin_execution_id: Option<String>,
    pub origin_metadata: Option<String>,
    pub created_at: String,
    pub is_active: bool,
}

/// Compact durable provenance for why a branch head exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchProvenance {
    pub origin_kind: String,
    pub origin_task_id: Option<String>,
    pub origin_execution_id: Option<String>,
    pub origin_metadata: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ScheduledJobRow {
    pub id: i64,
    pub public_id: Vec<u8>,
    pub agent_id: String,
    pub prompt: String,
    pub state_target: Option<String>,
    pub store_target: Option<String>,
    pub next_run_unix_ms: i64,
    pub interval_seconds: Option<u64>,
    pub overlap_policy: String,
    pub enabled: bool,
    pub running_task_id: Option<String>,
    pub pending_rerun: bool,
    pub last_run_unix_ms: Option<i64>,
    pub last_status: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

impl BranchProvenance {
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            origin_kind: kind.into(),
            origin_task_id: None,
            origin_execution_id: None,
            origin_metadata: None,
        }
    }

    pub fn manual() -> Self {
        Self::new("manual")
    }

    pub fn main() -> Self {
        Self::new("main")
    }

    pub fn sidestep() -> Self {
        Self::new("sidestep")
    }

    pub fn conflict_fork(task_id: Option<String>, execution_id: Option<String>) -> Self {
        Self {
            origin_kind: "conflict_fork".to_string(),
            origin_task_id: task_id,
            origin_execution_id: execution_id,
            origin_metadata: None,
        }
    }

    pub fn promotion(task_id: Option<String>) -> Self {
        Self {
            origin_kind: "promotion".to_string(),
            origin_task_id: task_id,
            origin_execution_id: None,
            origin_metadata: None,
        }
    }
}

/// Generic reference used by the sparse semantic graph overlay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphRef {
    pub kind: String,
    pub id: String,
}

impl GraphRef {
    pub fn new(kind: impl Into<String>, id: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            id: id.into(),
        }
    }
}

/// A row from the `graph_nodes` table.
#[derive(Debug, Clone)]
pub struct GraphNodeRow {
    pub id: i64,
    pub public_id: Vec<u8>,
    pub session_id: Option<i64>,
    pub kind: String,
    pub label: Option<String>,
    pub origin_task_id: Option<String>,
    pub origin_execution_id: Option<String>,
    pub metadata: Option<String>,
    pub created_at: String,
}

/// A row from the `graph_edges` table.
#[derive(Debug, Clone)]
pub struct GraphEdgeRow {
    pub id: i64,
    pub public_id: Vec<u8>,
    pub session_id: Option<i64>,
    pub source: GraphRef,
    pub target: GraphRef,
    pub relation_kind: String,
    pub source_role: Option<String>,
    pub target_role: Option<String>,
    pub origin_task_id: Option<String>,
    pub origin_execution_id: Option<String>,
    pub metadata: Option<String>,
    pub created_at: String,
}

/// Input for creating a sparse semantic graph edge.
#[derive(Debug, Clone)]
pub struct GraphEdgeCreate {
    pub session_id: Option<i64>,
    pub source: GraphRef,
    pub target: GraphRef,
    pub relation_kind: String,
    pub source_role: Option<String>,
    pub target_role: Option<String>,
    pub provenance: GraphProvenance,
    pub metadata: Option<serde_json::Value>,
}

impl GraphEdgeCreate {
    pub fn new(source: GraphRef, target: GraphRef, relation_kind: impl Into<String>) -> Self {
        Self {
            session_id: None,
            source,
            target,
            relation_kind: relation_kind.into(),
            source_role: None,
            target_role: None,
            provenance: GraphProvenance::default(),
            metadata: None,
        }
    }
}

/// Optional provenance attached to semantic graph nodes and edges.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GraphProvenance {
    pub origin_task_id: Option<String>,
    pub origin_execution_id: Option<String>,
}

impl GraphProvenance {
    pub fn new(origin_task_id: Option<String>, origin_execution_id: Option<String>) -> Self {
        Self {
            origin_task_id,
            origin_execution_id,
        }
    }
}

/// A turn-scoped tool execution row materialized for a selected session path.
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
