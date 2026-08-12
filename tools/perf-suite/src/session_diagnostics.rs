use std::collections::{HashMap, HashSet};
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use serde::Serialize;
use tempfile::TempDir;
use turin::inference::content::{decode_content_json, encode_content_json};
use turin::inference::provider::InferenceContent;
use turin::kernel::session_refs::parse_session_reference;
use turin::persistence::state::{SessionReadTarget, StateStore, TurnWriteTarget};
use turso::{Connection, Database, Value as SqlValue};
use uuid::Uuid;

#[derive(Parser)]
pub(crate) struct SessionLabArgs {
    /// Number of turns on the active path in both datasets.
    #[arg(long, default_value_t = 1_000)]
    turns: usize,

    /// Number of off-path branches to add to the graph dataset.
    #[arg(long, default_value_t = 8)]
    branches: usize,

    /// Number of additional turns to append to every off-path branch.
    #[arg(long, default_value_t = 25)]
    branch_turns: usize,

    /// Target bytes in each synthetic user message.
    #[arg(long, default_value_t = 128)]
    prompt_bytes: usize,

    /// Target bytes in each synthetic assistant message.
    #[arg(long, default_value_t = 1_024)]
    response_bytes: usize,

    /// Target bytes in each synthetic event payload field.
    #[arg(long, default_value_t = 256)]
    event_bytes: usize,

    /// Warmup materializations excluded from timing summaries.
    #[arg(long, default_value_t = 1)]
    warmups: usize,

    /// Measured materializations per implementation.
    #[arg(long, default_value_t = 7)]
    samples: usize,

    /// Optional persistent workspace. If omitted, an ephemeral temp dir is used.
    #[arg(long)]
    workspace_root: Option<PathBuf>,

    /// Report output directory.
    #[arg(long, default_value = ".workspace/perf-reports")]
    report_dir: PathBuf,
}

#[derive(Parser)]
pub(crate) struct SessionInspectArgs {
    /// Offline Turin state database to copy and inspect.
    #[arg(long)]
    state_db: PathBuf,

    /// Bare or store-qualified Turin session id.
    #[arg(long)]
    session_id: String,

    /// Warmup materializations excluded from timing summaries.
    #[arg(long, default_value_t = 1)]
    warmups: usize,

    /// Measured materializations per implementation.
    #[arg(long, default_value_t = 7)]
    samples: usize,

    /// Report output directory.
    #[arg(long, default_value = ".workspace/perf-reports")]
    report_dir: PathBuf,
}

#[derive(Debug, Serialize)]
struct SessionDiagnosticsReport {
    scenario: String,
    source: String,
    config: serde_json::Value,
    session: SessionIdentity,
    stores: Vec<StoreStorage>,
    retrieval: Vec<RetrievalBenchmark>,
    notes: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SessionIdentity {
    public_id: String,
    internal_id: i64,
    agent_id: String,
}

#[derive(Debug, Clone, Default, Serialize)]
struct FileStorage {
    main_bytes: u64,
    wal_bytes: u64,
    shm_bytes: u64,
    total_bytes: u64,
}

impl FileStorage {
    fn growth_from(&self, baseline: &Self) -> Self {
        let main_bytes = self.main_bytes.saturating_sub(baseline.main_bytes);
        let wal_bytes = self.wal_bytes.saturating_sub(baseline.wal_bytes);
        let shm_bytes = self.shm_bytes.saturating_sub(baseline.shm_bytes);
        Self {
            main_bytes,
            wal_bytes,
            shm_bytes,
            total_bytes: main_bytes
                .saturating_add(wal_bytes)
                .saturating_add(shm_bytes),
        }
    }
}

#[derive(Debug, Serialize)]
struct StoreStorage {
    label: String,
    physical: FileStorage,
    baseline_physical: Option<FileStorage>,
    data_growth: Option<FileStorage>,
    logical_owned_bytes: u64,
    logical_associated_bytes: u64,
    structure: SessionStructure,
    categories: Vec<StorageCategory>,
}

#[derive(Debug, Default, Serialize)]
struct SessionStructure {
    total_turns: usize,
    active_path_turns: usize,
    off_path_turns: usize,
    branches: usize,
    messages: usize,
    events: usize,
}

#[derive(Debug, Serialize)]
struct StorageCategory {
    name: String,
    ownership: String,
    rows: usize,
    logical_bytes: u64,
    description: String,
}

#[derive(Debug, Serialize)]
struct RetrievalBenchmark {
    implementation: String,
    description: String,
    warmups: usize,
    samples: Vec<RetrievalSample>,
    message_load: TimingSummary,
    event_load: TimingSummary,
    decode: TimingSummary,
    total: TimingSummary,
    message_rows: usize,
    event_rows: usize,
    payload_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
struct RetrievalSample {
    message_load_us: u128,
    event_load_us: u128,
    decode_us: u128,
    total_us: u128,
}

#[derive(Debug, Default, Serialize)]
struct TimingSummary {
    min_us: u128,
    median_us: u128,
    p95_us: u128,
    max_us: u128,
}

struct FlatStore {
    _database: Database,
    connection: Connection,
    path: PathBuf,
}

struct SyntheticDataset {
    _temp_guard: Option<TempDir>,
    workspace_root: PathBuf,
    graph_path: PathBuf,
    graph_store: StateStore,
    flat_store: FlatStore,
    session: SessionIdentity,
    graph_baseline: FileStorage,
    flat_baseline: FileStorage,
}

struct LoadedPayloads {
    messages: Vec<String>,
    events: Vec<String>,
}

pub(crate) async fn run_session_lab(args: SessionLabArgs) -> Result<()> {
    validate_run_args(args.turns, args.samples)?;
    anyhow::ensure!(
        args.branches == 0 || args.branch_turns > 0,
        "--branch-turns must be greater than zero when branches are requested"
    );

    let dataset = build_synthetic_dataset(&args).await?;
    let graph_storage = graph_storage(
        "turin-graph",
        &dataset.graph_path,
        &dataset.graph_store,
        dataset.session.internal_id,
        &dataset.session.public_id,
        Some(dataset.graph_baseline.clone()),
    )
    .await?;
    let flat_storage = flat_storage(
        &dataset.flat_store,
        dataset.session.internal_id,
        Some(dataset.flat_baseline.clone()),
    )
    .await?;

    let retrieval = vec![
        benchmark_current_graph(
            &dataset.graph_store,
            dataset.session.internal_id,
            args.warmups,
            args.samples,
        )
        .await?,
        benchmark_set_graph(
            &dataset.graph_store,
            dataset.session.internal_id,
            args.warmups,
            args.samples,
        )
        .await?,
        benchmark_flat(
            &dataset.flat_store.connection,
            dataset.session.internal_id,
            args.warmups,
            args.samples,
        )
        .await?,
    ];
    ensure_equivalent_payloads(&retrieval)?;

    let report = SessionDiagnosticsReport {
        scenario: "session-lab".to_string(),
        source: dataset.workspace_root.display().to_string(),
        config: serde_json::json!({
            "turns": args.turns,
            "branches": args.branches,
            "branch_turns": args.branch_turns,
            "prompt_bytes": args.prompt_bytes,
            "response_bytes": args.response_bytes,
            "event_bytes": args.event_bytes,
            "warmups": args.warmups,
            "samples": args.samples,
        }),
        session: dataset.session,
        stores: vec![graph_storage, flat_storage],
        retrieval,
        notes: standard_notes(true),
    };

    write_reports(&args.report_dir, &report)?;
    print_summary(&report);
    Ok(())
}

pub(crate) async fn run_session_inspect(args: SessionInspectArgs) -> Result<()> {
    validate_run_args(1, args.samples)?;
    anyhow::ensure!(
        args.state_db.is_file(),
        "state database '{}' does not exist",
        args.state_db.display()
    );

    let original_physical = file_storage(&args.state_db);
    let temp = tempfile::tempdir().context("failed to create inspection workspace")?;
    let copied_path = temp.path().join("state.db");
    copy_store_files(&args.state_db, &copied_path)?;
    let store = StateStore::open(path_text(&copied_path)?).await?;

    let session_ref = parse_session_reference(&args.session_id)?;
    let public_id = Uuid::parse_str(&session_ref.public_id)
        .with_context(|| format!("invalid session id '{}'", session_ref.public_id))?;
    let row = store
        .get_session_row_by_public_id(public_id)
        .await?
        .ok_or_else(|| anyhow!("session '{}' was not found", session_ref.public_id))?;
    let identity = SessionIdentity {
        public_id: public_id.simple().to_string(),
        internal_id: row.id,
        agent_id: row.agent_id,
    };

    let mut storage = graph_storage(
        "turin-graph",
        &args.state_db,
        &store,
        row.id,
        &identity.public_id,
        None,
    )
    .await?;
    storage.physical = original_physical;

    let retrieval = vec![
        benchmark_current_graph(&store, row.id, args.warmups, args.samples).await?,
        benchmark_set_graph(&store, row.id, args.warmups, args.samples).await?,
    ];
    ensure_equivalent_payloads(&retrieval)?;

    let report = SessionDiagnosticsReport {
        scenario: "session-inspect".to_string(),
        source: args.state_db.display().to_string(),
        config: serde_json::json!({
            "session_id": args.session_id,
            "warmups": args.warmups,
            "samples": args.samples,
            "input_was_copied": true,
        }),
        session: identity,
        stores: vec![storage],
        retrieval,
        notes: standard_notes(false),
    };

    write_reports(&args.report_dir, &report)?;
    print_summary(&report);
    Ok(())
}

fn validate_run_args(turns: usize, samples: usize) -> Result<()> {
    anyhow::ensure!(turns > 0, "--turns must be greater than zero");
    anyhow::ensure!(samples > 0, "--samples must be greater than zero");
    Ok(())
}

async fn build_synthetic_dataset(args: &SessionLabArgs) -> Result<SyntheticDataset> {
    let (temp_guard, workspace_root) = match &args.workspace_root {
        Some(root) => {
            fs::create_dir_all(root)?;
            (None, root.clone())
        }
        None => {
            let temp = tempfile::tempdir().context("failed to create session-lab workspace")?;
            let root = temp.path().to_path_buf();
            (Some(temp), root)
        }
    };
    let graph_path = workspace_root.join("graph.db");
    let flat_path = workspace_root.join("flat.db");
    remove_store_files(&graph_path)?;
    remove_store_files(&flat_path)?;

    let graph_store = StateStore::open(path_text(&graph_path)?).await?;
    checkpoint_connection(&graph_store.get_connection().await?).await?;
    let graph_baseline = file_storage(&graph_path);
    let flat_store = open_flat_store(&flat_path).await?;
    checkpoint_connection(&flat_store.connection).await?;
    let flat_baseline = file_storage(&flat_path);

    let public_id = Uuid::now_v7();
    let session_id = graph_store
        .create_session(public_id, "session-lab", Some(r#"{"title":"Session lab"}"#))
        .await?;
    flat_store
        .connection
        .execute(
            "INSERT INTO flat_sessions (id, public_id, agent_id, metadata) VALUES (?1, ?2, ?3, ?4)",
            turso::params![
                session_id,
                public_id.into_bytes().to_vec(),
                "session-lab",
                r#"{"title":"Session lab"}"#
            ],
        )
        .await?;

    let user_json = message_json("prompt", args.prompt_bytes);
    let assistant_json = message_json("response", args.response_bytes);
    let user_text = serde_json::to_string(&user_json)?;
    let assistant_text = serde_json::to_string(&assistant_json)?;
    let event = event_json(args.event_bytes);
    let event_text = serde_json::to_string(&event)?;

    for turn_index in 0..args.turns {
        let target = graph_store
            .prepare_turn_write_target(
                session_id,
                TurnWriteTarget::active_branch(turn_index as u32),
            )
            .await?
            .ok_or_else(|| anyhow!("active branch was unavailable at turn {turn_index}"))?;
        insert_graph_turn(
            &graph_store,
            session_id,
            target,
            &user_json,
            &assistant_json,
            &event,
        )
        .await?;
        insert_flat_turn(
            &flat_store.connection,
            session_id,
            turn_index,
            &user_text,
            &assistant_text,
            &event_text,
        )
        .await?;
    }

    for branch_number in 0..args.branches {
        let source_index =
            ((branch_number + 1) * args.turns / (args.branches + 1)).saturating_sub(1) as u32;
        let branch = graph_store
            .create_branch_head_from_turn_index(
                session_id,
                &format!("lab-branch-{}", branch_number + 1),
                Some(source_index),
                false,
            )
            .await?;
        for offset in 1..=args.branch_turns {
            let turn_index = source_index.saturating_add(offset as u32);
            let target = graph_store
                .prepare_turn_write_target(
                    session_id,
                    TurnWriteTarget::branch_head(Some(branch.id), turn_index),
                )
                .await?
                .ok_or_else(|| anyhow!("branch {} was unavailable", branch.name))?;
            insert_graph_turn(
                &graph_store,
                session_id,
                target,
                &user_json,
                &assistant_json,
                &event,
            )
            .await?;
        }
        insert_synthetic_graph_overlay(
            &graph_store.get_connection().await?,
            session_id,
            branch_number,
        )
        .await?;
    }

    checkpoint_connection(&graph_store.get_connection().await?).await?;
    checkpoint_connection(&flat_store.connection).await?;

    Ok(SyntheticDataset {
        _temp_guard: temp_guard,
        workspace_root,
        graph_path,
        graph_store,
        flat_store,
        session: SessionIdentity {
            public_id: public_id.simple().to_string(),
            internal_id: session_id,
            agent_id: "session-lab".to_string(),
        },
        graph_baseline,
        flat_baseline,
    })
}

async fn open_flat_store(path: &Path) -> Result<FlatStore> {
    let database = turso::Builder::new_local(path_text(path)?)
        .build()
        .await
        .with_context(|| format!("failed to open flat baseline '{}'", path.display()))?;
    let connection = database.connect()?;
    connection
        .execute("PRAGMA journal_mode = WAL;", ())
        .await
        .ok();
    connection
        .execute_batch(
            r#"
            CREATE TABLE flat_sessions (
                id INTEGER PRIMARY KEY,
                public_id BLOB NOT NULL,
                agent_id TEXT NOT NULL,
                metadata TEXT
            );
            CREATE TABLE flat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                sequence INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            );
            CREATE INDEX idx_flat_messages_session_sequence
                ON flat_messages(session_id, sequence, id);
            CREATE TABLE flat_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                sequence INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX idx_flat_events_session_sequence
                ON flat_events(session_id, sequence, id);
            "#,
        )
        .await?;
    Ok(FlatStore {
        _database: database,
        connection,
        path: path.to_path_buf(),
    })
}

async fn insert_graph_turn(
    store: &StateStore,
    session_id: i64,
    target: TurnWriteTarget,
    user: &serde_json::Value,
    assistant: &serde_json::Value,
    event: &serde_json::Value,
) -> Result<()> {
    store
        .insert_message(session_id, target, "user", user, None)
        .await?;
    store
        .insert_message(session_id, target, "assistant", assistant, None)
        .await?;
    store
        .insert_event(session_id, Some(target), "message_end", event)
        .await?;
    Ok(())
}

async fn insert_flat_turn(
    conn: &Connection,
    session_id: i64,
    turn_index: usize,
    user: &str,
    assistant: &str,
    event: &str,
) -> Result<()> {
    let sequence = turn_index.saturating_mul(2);
    conn.execute(
        "INSERT INTO flat_messages (session_id, sequence, role, content) VALUES (?1, ?2, 'user', ?3)",
        turso::params![session_id, sequence as i64, user],
    )
    .await?;
    conn.execute(
        "INSERT INTO flat_messages (session_id, sequence, role, content) VALUES (?1, ?2, 'assistant', ?3)",
        turso::params![session_id, sequence.saturating_add(1) as i64, assistant],
    )
    .await?;
    conn.execute(
        "INSERT INTO flat_events (session_id, sequence, event_type, payload) VALUES (?1, ?2, 'message_end', ?3)",
        turso::params![session_id, turn_index as i64, event],
    )
    .await?;
    Ok(())
}

async fn insert_synthetic_graph_overlay(
    conn: &Connection,
    session_id: i64,
    branch_number: usize,
) -> Result<()> {
    let node_id = Uuid::now_v7().into_bytes().to_vec();
    conn.execute(
        "INSERT INTO graph_nodes (public_id, session_id, kind, label, metadata) VALUES (?1, ?2, 'branch_note', ?3, ?4)",
        turso::params![
            node_id.clone(),
            session_id,
            format!("Branch {}", branch_number + 1),
            format!(r#"{{"branch":{}}}"#, branch_number + 1)
        ],
    )
    .await?;
    conn.execute(
        "INSERT INTO graph_edges (public_id, session_id, source_kind, source_id, target_kind, target_id, relation_kind) VALUES (?1, ?2, 'session', ?3, 'graph_node', ?4, 'explores')",
        turso::params![
            Uuid::now_v7().into_bytes().to_vec(),
            session_id,
            session_id.to_string(),
            Uuid::from_slice(&node_id)?.simple().to_string()
        ],
    )
    .await?;
    Ok(())
}

fn message_json(seed: &str, bytes: usize) -> serde_json::Value {
    encode_content_json(&[InferenceContent::Text {
        text: synthetic_text(seed, bytes),
    }])
}

fn event_json(bytes: usize) -> serde_json::Value {
    serde_json::json!({
        "type": "message_end",
        "role": "assistant",
        "input_tokens": 0,
        "output_tokens": 0,
        "padding": synthetic_text("event", bytes),
    })
}

fn synthetic_text(seed: &str, bytes: usize) -> String {
    if bytes == 0 {
        return String::new();
    }
    let mut output = String::with_capacity(bytes);
    while output.len() < bytes {
        if !output.is_empty() {
            output.push(' ');
        }
        output.push_str(seed);
    }
    output.truncate(bytes);
    output
}

async fn graph_storage(
    label: &str,
    path: &Path,
    store: &StateStore,
    session_id: i64,
    public_id: &str,
    baseline: Option<FileStorage>,
) -> Result<StoreStorage> {
    let conn = store.get_connection().await?;
    let categories = vec![
        storage_category(
            &conn,
            "session",
            "owned",
            "Session identity and metadata",
            "SELECT COUNT(*), COALESCE(SUM(length(public_id) + length(CAST(agent_id AS BLOB)) + COALESCE(length(CAST(metadata AS BLOB)), 0)), 0) FROM sessions WHERE id = ?1",
            vec![SqlValue::Integer(session_id)],
        )
        .await?,
        storage_category(
            &conn,
            "turn graph",
            "owned",
            "Turn identifiers; integer links and index pages are not included",
            "SELECT COUNT(*), COALESCE(SUM(length(public_id)), 0) FROM turns WHERE session_id = ?1",
            vec![SqlValue::Integer(session_id)],
        )
        .await?,
        storage_category(
            &conn,
            "branch heads",
            "owned",
            "Branch names, provenance, and public identifiers",
            "SELECT COUNT(*), COALESCE(SUM(length(public_id) + length(CAST(name AS BLOB)) + length(CAST(origin_kind AS BLOB)) + COALESCE(length(CAST(origin_task_id AS BLOB)), 0) + COALESCE(length(CAST(origin_execution_id AS BLOB)), 0) + COALESCE(length(CAST(origin_metadata AS BLOB)), 0)), 0) FROM branch_heads WHERE session_id = ?1",
            vec![SqlValue::Integer(session_id)],
        )
        .await?,
        storage_category(
            &conn,
            "messages",
            "owned",
            "Persisted role and content payloads across every branch",
            "SELECT COUNT(*), COALESCE(SUM(length(CAST(m.role AS BLOB)) + length(CAST(m.content AS BLOB))), 0) FROM messages m JOIN turns t ON t.id = m.turn_id WHERE t.session_id = ?1",
            vec![SqlValue::Integer(session_id)],
        )
        .await?,
        storage_category(
            &conn,
            "events / logs",
            "owned",
            "Durable event type and JSON payloads",
            "SELECT COUNT(*), COALESCE(SUM(length(CAST(event_type AS BLOB)) + length(CAST(payload AS BLOB))), 0) FROM events WHERE session_id = ?1",
            vec![SqlValue::Integer(session_id)],
        )
        .await?,
        storage_category(
            &conn,
            "tool executions",
            "owned",
            "Tool names, arguments, outputs, call identifiers, and verdicts",
            "SELECT COUNT(*), COALESCE(SUM(length(CAST(x.tool_call_id AS BLOB)) + length(CAST(x.tool_name AS BLOB)) + length(CAST(x.args AS BLOB)) + COALESCE(length(CAST(x.output AS BLOB)), 0) + length(CAST(x.verdict AS BLOB))), 0) FROM tool_executions x JOIN turns t ON t.id = x.turn_id WHERE t.session_id = ?1",
            vec![SqlValue::Integer(session_id)],
        )
        .await?,
        storage_category(
            &conn,
            "graph nodes",
            "owned",
            "Sparse semantic graph node payloads",
            "SELECT COUNT(*), COALESCE(SUM(length(public_id) + length(CAST(kind AS BLOB)) + COALESCE(length(CAST(label AS BLOB)), 0) + COALESCE(length(CAST(origin_task_id AS BLOB)), 0) + COALESCE(length(CAST(origin_execution_id AS BLOB)), 0) + COALESCE(length(CAST(metadata AS BLOB)), 0)), 0) FROM graph_nodes WHERE session_id = ?1",
            vec![SqlValue::Integer(session_id)],
        )
        .await?,
        storage_category(
            &conn,
            "graph edges",
            "owned",
            "Sparse semantic graph edge payloads",
            "SELECT COUNT(*), COALESCE(SUM(length(public_id) + length(CAST(source_kind AS BLOB)) + length(CAST(source_id AS BLOB)) + length(CAST(target_kind AS BLOB)) + length(CAST(target_id AS BLOB)) + length(CAST(relation_kind AS BLOB)) + COALESCE(length(CAST(source_role AS BLOB)), 0) + COALESCE(length(CAST(target_role AS BLOB)), 0) + COALESCE(length(CAST(origin_task_id AS BLOB)), 0) + COALESCE(length(CAST(origin_execution_id AS BLOB)), 0) + COALESCE(length(CAST(metadata AS BLOB)), 0)), 0) FROM graph_edges WHERE session_id = ?1",
            vec![SqlValue::Integer(session_id)],
        )
        .await?,
        storage_category(
            &conn,
            "session-scoped KV",
            "associated",
            "KV rows whose scope key is this bare session id",
            "SELECT COUNT(*), COALESCE(SUM(length(CAST(scope_kind AS BLOB)) + length(CAST(scope_key AS BLOB)) + length(CAST(key AS BLOB)) + length(CAST(value AS BLOB))), 0) FROM kv WHERE scope_kind = 'session' AND scope_key = ?1",
            vec![SqlValue::Text(public_id.to_string())],
        )
        .await?,
        storage_category(
            &conn,
            "session-scoped memories",
            "associated",
            "Memory rows scoped directly to this bare session id",
            "SELECT COUNT(*), COALESCE(SUM(length(public_id) + length(CAST(content AS BLOB)) + COALESCE(length(embedding), 0) + COALESCE(length(CAST(embedding_key AS BLOB)), 0) + COALESCE(length(CAST(metadata AS BLOB)), 0)), 0) FROM memories WHERE scope_kind = 'session' AND scope_key = ?1",
            vec![SqlValue::Text(public_id.to_string())],
        )
        .await?,
        storage_category(
            &conn,
            "work-item references",
            "associated",
            "Work items currently claiming this session; not owned by the session",
            "SELECT COUNT(*), COALESCE(SUM(length(public_id) + length(CAST(title AS BLOB)) + COALESCE(length(CAST(prompt AS BLOB)), 0) + COALESCE(length(CAST(content AS BLOB)), 0) + COALESCE(length(CAST(metadata AS BLOB)), 0)), 0) FROM work_items WHERE claim_session_id = ?1",
            vec![SqlValue::Text(public_id.to_string())],
        )
        .await?,
    ];

    let total_turns = scalar_usize(
        &conn,
        "SELECT COUNT(*) FROM turns WHERE session_id = ?1",
        vec![SqlValue::Integer(session_id)],
    )
    .await?;
    let active_path_turns = store.active_branch_turn_count(session_id).await? as usize;
    let branches = store.list_branch_heads(session_id).await?.len();
    let messages = category_rows(&categories, "messages");
    let events = category_rows(&categories, "events / logs");
    let physical = file_storage(path);
    let data_growth = baseline
        .as_ref()
        .map(|baseline| physical.growth_from(baseline));

    Ok(StoreStorage {
        label: label.to_string(),
        physical,
        baseline_physical: baseline,
        data_growth,
        logical_owned_bytes: categories
            .iter()
            .filter(|category| category.ownership == "owned")
            .map(|category| category.logical_bytes)
            .sum(),
        logical_associated_bytes: categories
            .iter()
            .filter(|category| category.ownership == "associated")
            .map(|category| category.logical_bytes)
            .sum(),
        structure: SessionStructure {
            total_turns,
            active_path_turns,
            off_path_turns: total_turns.saturating_sub(active_path_turns),
            branches,
            messages,
            events,
        },
        categories,
    })
}

async fn flat_storage(
    store: &FlatStore,
    session_id: i64,
    baseline: Option<FileStorage>,
) -> Result<StoreStorage> {
    let categories = vec![
        storage_category(
            &store.connection,
            "session",
            "owned",
            "Flat session identity and metadata",
            "SELECT COUNT(*), COALESCE(SUM(length(public_id) + length(CAST(agent_id AS BLOB)) + COALESCE(length(CAST(metadata AS BLOB)), 0)), 0) FROM flat_sessions WHERE id = ?1",
            vec![SqlValue::Integer(session_id)],
        )
        .await?,
        storage_category(
            &store.connection,
            "messages",
            "owned",
            "Flat role and content payloads",
            "SELECT COUNT(*), COALESCE(SUM(length(CAST(role AS BLOB)) + length(CAST(content AS BLOB))), 0) FROM flat_messages WHERE session_id = ?1",
            vec![SqlValue::Integer(session_id)],
        )
        .await?,
        storage_category(
            &store.connection,
            "events / logs",
            "owned",
            "Flat event type and JSON payloads",
            "SELECT COUNT(*), COALESCE(SUM(length(CAST(event_type AS BLOB)) + length(CAST(payload AS BLOB))), 0) FROM flat_events WHERE session_id = ?1",
            vec![SqlValue::Integer(session_id)],
        )
        .await?,
    ];
    let messages = category_rows(&categories, "messages");
    let events = category_rows(&categories, "events / logs");
    let turns = messages / 2;
    let physical = file_storage(&store.path);
    let data_growth = baseline
        .as_ref()
        .map(|baseline| physical.growth_from(baseline));

    Ok(StoreStorage {
        label: "flat-serial".to_string(),
        physical,
        baseline_physical: baseline,
        data_growth,
        logical_owned_bytes: categories
            .iter()
            .map(|category| category.logical_bytes)
            .sum(),
        logical_associated_bytes: 0,
        structure: SessionStructure {
            total_turns: turns,
            active_path_turns: turns,
            off_path_turns: 0,
            branches: 1,
            messages,
            events,
        },
        categories,
    })
}

async fn storage_category(
    conn: &Connection,
    name: &str,
    ownership: &str,
    description: &str,
    sql: &str,
    params: Vec<SqlValue>,
) -> Result<StorageCategory> {
    let mut stmt = conn.prepare(sql).await?;
    let mut rows = stmt.query(params).await?;
    let row = rows
        .next()
        .await?
        .ok_or_else(|| anyhow!("storage query for '{name}' returned no row"))?;
    Ok(StorageCategory {
        name: name.to_string(),
        ownership: ownership.to_string(),
        rows: row.get::<i64>(0)?.max(0) as usize,
        logical_bytes: row.get::<i64>(1)?.max(0) as u64,
        description: description.to_string(),
    })
}

async fn scalar_usize(conn: &Connection, sql: &str, params: Vec<SqlValue>) -> Result<usize> {
    let mut stmt = conn.prepare(sql).await?;
    let mut rows = stmt.query(params).await?;
    Ok(rows
        .next()
        .await?
        .ok_or_else(|| anyhow!("scalar query returned no row"))?
        .get::<i64>(0)?
        .max(0) as usize)
}

fn category_rows(categories: &[StorageCategory], name: &str) -> usize {
    categories
        .iter()
        .find(|category| category.name == name)
        .map_or(0, |category| category.rows)
}

async fn benchmark_current_graph(
    store: &StateStore,
    session_id: i64,
    warmups: usize,
    samples: usize,
) -> Result<RetrievalBenchmark> {
    let mut measured = Vec::with_capacity(samples);
    let mut payload_shape = (0, 0, 0);
    for iteration in 0..warmups.saturating_add(samples) {
        let total_started = Instant::now();
        let messages_started = Instant::now();
        let messages = store
            .get_messages(session_id, &SessionReadTarget::ActiveBranch)
            .await?;
        let message_load_us = messages_started.elapsed().as_micros();

        let events_started = Instant::now();
        let events = store
            .get_events(session_id, &SessionReadTarget::ActiveBranch)
            .await?;
        let event_load_us = events_started.elapsed().as_micros();

        let payloads = LoadedPayloads {
            messages: messages.into_iter().map(|row| row.content).collect(),
            events: events.into_iter().map(|row| row.payload).collect(),
        };
        let decode_started = Instant::now();
        decode_payloads(&payloads)?;
        let decode_us = decode_started.elapsed().as_micros();
        let total_us = total_started.elapsed().as_micros();
        payload_shape = payload_shape_of(&payloads);
        if iteration >= warmups {
            measured.push(RetrievalSample {
                message_load_us,
                event_load_us,
                decode_us,
                total_us,
            });
        }
    }
    Ok(retrieval_benchmark(
        "current-iterative-graph",
        "Current Turin StateStore path walk, message/event reads, and payload decoding",
        warmups,
        measured,
        payload_shape,
    ))
}

async fn benchmark_set_graph(
    store: &StateStore,
    session_id: i64,
    warmups: usize,
    samples: usize,
) -> Result<RetrievalBenchmark> {
    let conn = store.get_connection().await?;
    let mut measured = Vec::with_capacity(samples);
    let mut payload_shape = (0, 0, 0);
    for iteration in 0..warmups.saturating_add(samples) {
        let total_started = Instant::now();
        let messages_started = Instant::now();
        let messages = query_set_graph_messages(&conn, session_id).await?;
        let message_load_us = messages_started.elapsed().as_micros();

        let events_started = Instant::now();
        let events = query_set_graph_events(&conn, session_id).await?;
        let event_load_us = events_started.elapsed().as_micros();

        let payloads = LoadedPayloads { messages, events };
        let decode_started = Instant::now();
        decode_payloads(&payloads)?;
        let decode_us = decode_started.elapsed().as_micros();
        let total_us = total_started.elapsed().as_micros();
        payload_shape = payload_shape_of(&payloads);
        if iteration >= warmups {
            measured.push(RetrievalSample {
                message_load_us,
                event_load_us,
                decode_us,
                total_us,
            });
        }
    }
    Ok(retrieval_benchmark(
        "set-based-graph-probe",
        "Diagnostic one-query turn-link load plus in-memory path resolution over the same graph; production retrieval is unchanged",
        warmups,
        measured,
        payload_shape,
    ))
}

async fn benchmark_flat(
    conn: &Connection,
    session_id: i64,
    warmups: usize,
    samples: usize,
) -> Result<RetrievalBenchmark> {
    let mut measured = Vec::with_capacity(samples);
    let mut payload_shape = (0, 0, 0);
    for iteration in 0..warmups.saturating_add(samples) {
        let total_started = Instant::now();
        let messages_started = Instant::now();
        let messages = query_text_column(
            conn,
            "SELECT content FROM flat_messages WHERE session_id = ?1 ORDER BY sequence, id",
            session_id,
        )
        .await?;
        let message_load_us = messages_started.elapsed().as_micros();

        let events_started = Instant::now();
        let events = query_text_column(
            conn,
            "SELECT payload FROM flat_events WHERE session_id = ?1 ORDER BY sequence, id",
            session_id,
        )
        .await?;
        let event_load_us = events_started.elapsed().as_micros();

        let payloads = LoadedPayloads { messages, events };
        let decode_started = Instant::now();
        decode_payloads(&payloads)?;
        let decode_us = decode_started.elapsed().as_micros();
        let total_us = total_started.elapsed().as_micros();
        payload_shape = payload_shape_of(&payloads);
        if iteration >= warmups {
            measured.push(RetrievalSample {
                message_load_us,
                event_load_us,
                decode_us,
                total_us,
            });
        }
    }
    Ok(retrieval_benchmark(
        "flat-serial-baseline",
        "Indexed flat session/sequence reads over payloads identical to the active graph path",
        warmups,
        measured,
        payload_shape,
    ))
}

fn retrieval_benchmark(
    implementation: &str,
    description: &str,
    warmups: usize,
    samples: Vec<RetrievalSample>,
    payload_shape: (usize, usize, usize),
) -> RetrievalBenchmark {
    RetrievalBenchmark {
        implementation: implementation.to_string(),
        description: description.to_string(),
        warmups,
        message_load: timing_summary(samples.iter().map(|sample| sample.message_load_us)),
        event_load: timing_summary(samples.iter().map(|sample| sample.event_load_us)),
        decode: timing_summary(samples.iter().map(|sample| sample.decode_us)),
        total: timing_summary(samples.iter().map(|sample| sample.total_us)),
        samples,
        message_rows: payload_shape.0,
        event_rows: payload_shape.1,
        payload_bytes: payload_shape.2,
    }
}

fn timing_summary(values: impl Iterator<Item = u128>) -> TimingSummary {
    let mut values = values.collect::<Vec<_>>();
    if values.is_empty() {
        return TimingSummary::default();
    }
    values.sort_unstable();
    let p95_index = ((values.len() as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    TimingSummary {
        min_us: values[0],
        median_us: values[values.len() / 2],
        p95_us: values[p95_index],
        max_us: values[values.len() - 1],
    }
}

fn decode_payloads(payloads: &LoadedPayloads) -> Result<()> {
    let mut decoded_items = 0usize;
    for message in &payloads.messages {
        let value = serde_json::from_str(message).context("invalid persisted message JSON")?;
        let content = decode_content_json(value).context("invalid persisted message content")?;
        decoded_items = decoded_items.saturating_add(content.len());
    }
    for event in &payloads.events {
        let value: serde_json::Value =
            serde_json::from_str(event).context("invalid persisted event JSON")?;
        decoded_items = decoded_items.saturating_add(usize::from(!value.is_null()));
    }
    black_box(decoded_items);
    Ok(())
}

fn payload_shape_of(payloads: &LoadedPayloads) -> (usize, usize, usize) {
    (
        payloads.messages.len(),
        payloads.events.len(),
        payloads
            .messages
            .iter()
            .chain(&payloads.events)
            .map(String::len)
            .sum(),
    )
}

fn ensure_equivalent_payloads(benchmarks: &[RetrievalBenchmark]) -> Result<()> {
    let Some(first) = benchmarks.first() else {
        return Ok(());
    };
    for benchmark in &benchmarks[1..] {
        anyhow::ensure!(
            benchmark.message_rows == first.message_rows
                && benchmark.event_rows == first.event_rows
                && benchmark.payload_bytes == first.payload_bytes,
            "retrieval comparison is not equivalent: '{}' loaded {}/{} rows and {} bytes; '{}' loaded {}/{} rows and {} bytes",
            first.implementation,
            first.message_rows,
            first.event_rows,
            first.payload_bytes,
            benchmark.implementation,
            benchmark.message_rows,
            benchmark.event_rows,
            benchmark.payload_bytes,
        );
    }
    Ok(())
}

async fn query_set_graph_messages(conn: &Connection, session_id: i64) -> Result<Vec<String>> {
    let path = query_set_graph_path(conn, session_id).await?;
    let mut messages = Vec::<(u32, i64, String)>::new();
    for chunk in path.chunks(500) {
        let placeholders = sql_placeholders(chunk.len(), 1);
        let sql =
            format!("SELECT turn_id, id, content FROM messages WHERE turn_id IN ({placeholders})");
        let mut stmt = conn.prepare(&sql).await?;
        let mut rows = stmt
            .query(
                chunk
                    .iter()
                    .map(|(turn_id, _)| SqlValue::Integer(*turn_id))
                    .collect::<Vec<_>>(),
            )
            .await?;
        let depths = chunk.iter().copied().collect::<HashMap<_, _>>();
        while let Some(row) = rows.next().await? {
            let turn_id = row.get::<i64>(0)?;
            messages.push((
                *depths
                    .get(&turn_id)
                    .ok_or_else(|| anyhow!("message references unexpected turn {turn_id}"))?,
                row.get::<i64>(1)?,
                row.get::<String>(2)?,
            ));
        }
    }
    messages.sort_by_key(|(depth, id, _)| (*depth, *id));
    Ok(messages
        .into_iter()
        .map(|(_, _, content)| content)
        .collect())
}

async fn query_set_graph_events(conn: &Connection, session_id: i64) -> Result<Vec<String>> {
    let path = query_set_graph_path(conn, session_id)
        .await?
        .into_iter()
        .map(|(turn_id, _)| turn_id)
        .collect::<HashSet<_>>();
    let mut stmt = conn
        .prepare("SELECT turn_id, payload FROM events WHERE session_id = ?1 ORDER BY id")
        .await?;
    let mut rows = stmt.query([session_id]).await?;
    let mut events = Vec::new();
    while let Some(row) = rows.next().await? {
        let turn_id = row.get::<Option<i64>>(0)?;
        if turn_id.is_none_or(|turn_id| path.contains(&turn_id)) {
            events.push(row.get::<String>(1)?);
        }
    }
    Ok(events)
}

async fn query_set_graph_path(conn: &Connection, session_id: i64) -> Result<Vec<(i64, u32)>> {
    let mut head_stmt = conn
        .prepare(
            "SELECT b.head_turn_id FROM sessions s LEFT JOIN branch_heads b ON b.id = s.active_branch_head_id WHERE s.id = ?1",
        )
        .await?;
    let mut head_rows = head_stmt.query([session_id]).await?;
    let mut current = head_rows
        .next()
        .await?
        .ok_or_else(|| anyhow!("session {session_id} was not found"))?
        .get::<Option<i64>>(0)?;

    let mut turn_stmt = conn
        .prepare("SELECT id, parent_turn_id, branch_depth FROM turns WHERE session_id = ?1")
        .await?;
    let mut turn_rows = turn_stmt.query([session_id]).await?;
    let mut turns = HashMap::<i64, (Option<i64>, u32)>::new();
    while let Some(row) = turn_rows.next().await? {
        turns.insert(
            row.get::<i64>(0)?,
            (row.get::<Option<i64>>(1)?, row.get::<i64>(2)? as u32),
        );
    }

    let mut path = Vec::new();
    while let Some(turn_id) = current {
        let (parent_turn_id, depth) = turns
            .get(&turn_id)
            .copied()
            .ok_or_else(|| anyhow!("active path references missing turn {turn_id}"))?;
        path.push((turn_id, depth));
        current = parent_turn_id;
    }
    path.reverse();
    Ok(path)
}

fn sql_placeholders(count: usize, first: usize) -> String {
    (first..first.saturating_add(count))
        .map(|index| format!("?{index}"))
        .collect::<Vec<_>>()
        .join(", ")
}

async fn query_text_column(conn: &Connection, sql: &str, session_id: i64) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(sql).await?;
    let mut rows = stmt.query([session_id]).await?;
    let mut values = Vec::new();
    while let Some(row) = rows.next().await? {
        values.push(row.get::<String>(0)?);
    }
    Ok(values)
}

fn standard_notes(has_flat_baseline: bool) -> Vec<String> {
    let mut notes = vec![
        "Logical bytes sum attributable row payload columns; SQLite/Turso row headers, integer columns, indexes, page slack, and shared free pages are excluded.".to_string(),
        "Physical bytes report the complete store plus WAL and SHM. They cannot be assigned exactly to one session in a shared store.".to_string(),
        "Warm measurements include query execution, row materialization, JSON parsing, and Turin message-content reconstruction, but not runtime/provider initialization or hot-history pruning.".to_string(),
        "The current-iterative-graph measurement calls production StateStore methods. The set-based graph probe is sidecar-only and does not modify Turin retrieval.".to_string(),
        "Cold measurements require separate process runs. Reopening a connection does not clear the operating-system page cache.".to_string(),
    ];
    if has_flat_baseline {
        notes.push(
            "The flat baseline stores the same active-path message and event payloads using session_id plus sequence indexes; off-path branches exist only in the Turin graph store.".to_string(),
        );
    } else {
        notes.push(
            "Inspection runs against a private copy of the supplied state store. The original database is not opened by Turso and is not mutated.".to_string(),
        );
    }
    notes
}

fn write_reports(report_dir: &Path, report: &SessionDiagnosticsReport) -> Result<()> {
    fs::create_dir_all(report_dir)?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before unix epoch")?
        .as_secs();
    let stem = format!("{}-{stamp}", report.scenario);
    let json_path = report_dir.join(format!("{stem}.json"));
    let markdown_path = report_dir.join(format!("{stem}.md"));
    let html_path = report_dir.join(format!("{stem}.html"));
    fs::write(&json_path, serde_json::to_vec_pretty(report)?)?;
    fs::write(&markdown_path, markdown_report(report))?;
    fs::write(&html_path, html_report(report)?)?;
    println!("json_report={}", json_path.display());
    println!("markdown_report={}", markdown_path.display());
    println!("html_report={}", html_path.display());
    Ok(())
}

fn markdown_report(report: &SessionDiagnosticsReport) -> String {
    let mut out = format!(
        "# Session Diagnostics: {}\n\n- source: `{}`\n- session: `{}`\n- agent: `{}`\n- config: `{}`\n\n",
        report.scenario,
        report.source,
        report.session.public_id,
        report.session.agent_id,
        report.config
    );
    out.push_str("## Retrieval\n\n");
    out.push_str("| implementation | messages | events | payload | message median | event median | decode median | total median | total p95 |\n");
    out.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for benchmark in &report.retrieval {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            benchmark.implementation,
            benchmark.message_rows,
            benchmark.event_rows,
            human_bytes(benchmark.payload_bytes as u64),
            human_duration(benchmark.message_load.median_us),
            human_duration(benchmark.event_load.median_us),
            human_duration(benchmark.decode.median_us),
            human_duration(benchmark.total.median_us),
            human_duration(benchmark.total.p95_us),
        ));
    }

    for store in &report.stores {
        out.push_str(&format!("\n## Storage: {}\n\n", store.label));
        out.push_str(&format!(
            "- total turns: {}\n- active-path turns: {}\n- off-path turns: {}\n- branches: {}\n- logical owned payload: {}\n- logical associated payload: {}\n- store physical total: {}\n",
            store.structure.total_turns,
            store.structure.active_path_turns,
            store.structure.off_path_turns,
            store.structure.branches,
            human_bytes(store.logical_owned_bytes),
            human_bytes(store.logical_associated_bytes),
            human_bytes(store.physical.total_bytes),
        ));
        if let Some(growth) = &store.data_growth {
            out.push_str(&format!(
                "- synthetic data-file growth: {}\n",
                human_bytes(growth.total_bytes)
            ));
        }
        out.push_str("\n| category | ownership | rows | logical bytes | description |\n");
        out.push_str("|---|---|---:|---:|---|\n");
        for category in &store.categories {
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                category.name,
                category.ownership,
                category.rows,
                human_bytes(category.logical_bytes),
                category.description,
            ));
        }
    }

    out.push_str("\n## Interpretation Notes\n\n");
    for note in &report.notes {
        out.push_str(&format!("- {note}\n"));
    }
    out
}

fn html_report(report: &SessionDiagnosticsReport) -> Result<String> {
    let report_json = serde_json::to_string(report)?.replace('<', "\\u003c");
    Ok(format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Turin Session Diagnostics</title>
<style>
:root {{ font-family: Inter, ui-sans-serif, sans-serif; color: #181917; background: #f1f2ee; }}
* {{ box-sizing: border-box; }} body {{ margin: 0; }} main {{ width: min(1180px, calc(100% - 32px)); margin: 36px auto 80px; }}
header {{ display: flex; justify-content: space-between; gap: 20px; margin-bottom: 22px; }} h1 {{ margin: 0; font-size: 28px; letter-spacing: -.04em; }}
.sub {{ color: #72756e; font-size: 13px; }} .grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(230px,1fr)); gap: 12px; }}
.card {{ border: 1px solid #d9dbd4; border-radius: 14px; background: #fff; box-shadow: 0 1px 2px #1111; }} .card h2 {{ margin: 0; padding: 16px 18px; font-size: 15px; border-bottom: 1px solid #e3e4df; }}
.metric {{ padding: 16px 18px; }} .metric strong {{ display: block; font-size: 24px; letter-spacing: -.035em; }} .metric span {{ color: #777a73; font-size: 11px; }}
section {{ margin-top: 18px; }} table {{ width: 100%; border-collapse: collapse; font-size: 12px; }} th,td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #e7e8e3; }} th {{ color: #747770; font-size: 10px; text-transform: uppercase; letter-spacing: .05em; }}
.table-wrap {{ overflow: auto; }} .notes {{ color: #666a62; font-size: 12px; line-height: 1.6; }} code {{ font-family: ui-monospace, monospace; font-size: 11px; }}
@media (prefers-color-scheme: dark) {{ :root {{ color: #f1f2ed; background: #171815; }} .card {{ background:#22231f; border-color:#393b35; }} .card h2,th,td {{ border-color:#393b35; }} .sub,.metric span,th,.notes {{ color:#a6aaa0; }} }}
</style>
</head>
<body><main><header><div><h1>Session diagnostics</h1><div class="sub" id="source"></div></div><div class="sub" id="session"></div></header><div class="grid" id="summary"></div><section class="card"><h2>Retrieval comparison</h2><div class="table-wrap"><table><thead><tr><th>Implementation</th><th>Messages</th><th>Events</th><th>Payload</th><th>Message median</th><th>Event median</th><th>Decode median</th><th>Total median</th><th>Total p95</th></tr></thead><tbody id="retrieval"></tbody></table></div></section><div id="stores"></div><section class="card"><h2>Interpretation notes</h2><div class="metric notes" id="notes"></div></section></main>
<script>const r={report_json};const bytes=n=>n<1024?`${{n}} B`:n<1048576?`${{(n/1024).toFixed(1)}} KiB`:`${{(n/1048576).toFixed(2)}} MiB`;const time=n=>n<1000?`${{n}} us`:`${{(n/1000).toFixed(2)}} ms`;source.textContent=r.source;session.textContent=`Session ${{r.session.public_id}} · ${{r.session.agent_id}}`;const fastest=[...r.retrieval].sort((a,b)=>a.total.median_us-b.total.median_us)[0];summary.innerHTML=`<div class="card metric"><span>Active path</span><strong>${{r.stores[0].structure.active_path_turns.toLocaleString()}}</strong></div><div class="card metric"><span>Off-path turns</span><strong>${{r.stores[0].structure.off_path_turns.toLocaleString()}}</strong></div><div class="card metric"><span>Session payload</span><strong>${{bytes(r.stores[0].logical_owned_bytes)}}</strong></div><div class="card metric"><span>Fastest median</span><strong>${{time(fastest.total.median_us)}}</strong><span>${{fastest.implementation}}</span></div>`;retrieval.innerHTML=r.retrieval.map(x=>`<tr><td><strong>${{x.implementation}}</strong><br><span class="sub">${{x.description}}</span></td><td>${{x.message_rows.toLocaleString()}}</td><td>${{x.event_rows.toLocaleString()}}</td><td>${{bytes(x.payload_bytes)}}</td><td>${{time(x.message_load.median_us)}}</td><td>${{time(x.event_load.median_us)}}</td><td>${{time(x.decode.median_us)}}</td><td><strong>${{time(x.total.median_us)}}</strong></td><td>${{time(x.total.p95_us)}}</td></tr>`).join('');stores.innerHTML=r.stores.map(s=>`<section class="card"><h2>Storage · ${{s.label}}</h2><div class="grid"><div class="metric"><span>Logical owned</span><strong>${{bytes(s.logical_owned_bytes)}}</strong></div><div class="metric"><span>Physical store</span><strong>${{bytes(s.physical.total_bytes)}}</strong></div><div class="metric"><span>Branches</span><strong>${{s.structure.branches}}</strong></div><div class="metric"><span>Total turns</span><strong>${{s.structure.total_turns.toLocaleString()}}</strong></div></div><div class="table-wrap"><table><thead><tr><th>Category</th><th>Ownership</th><th>Rows</th><th>Logical bytes</th><th>Meaning</th></tr></thead><tbody>${{s.categories.map(c=>`<tr><td>${{c.name}}</td><td>${{c.ownership}}</td><td>${{c.rows.toLocaleString()}}</td><td>${{bytes(c.logical_bytes)}}</td><td>${{c.description}}</td></tr>`).join('')}}</tbody></table></div></section>`).join('');notes.innerHTML=`<ul>${{r.notes.map(n=>`<li>${{n}}</li>`).join('')}}</ul>`;</script></body></html>"#
    ))
}

fn print_summary(report: &SessionDiagnosticsReport) {
    println!("scenario={}", report.scenario);
    println!("session_id={}", report.session.public_id);
    for store in &report.stores {
        println!(
            "storage={} logical_owned_bytes={} physical_bytes={} active_path_turns={} off_path_turns={}",
            store.label,
            store.logical_owned_bytes,
            store.physical.total_bytes,
            store.structure.active_path_turns,
            store.structure.off_path_turns,
        );
    }
    for benchmark in &report.retrieval {
        println!(
            "retrieval={} median_us={} p95_us={} message_us={} event_us={} decode_us={}",
            benchmark.implementation,
            benchmark.total.median_us,
            benchmark.total.p95_us,
            benchmark.message_load.median_us,
            benchmark.event_load.median_us,
            benchmark.decode.median_us,
        );
    }
}

fn human_bytes(bytes: u64) -> String {
    if bytes < 1_024 {
        format!("{bytes} B")
    } else if bytes < 1_048_576 {
        format!("{:.1} KiB", bytes as f64 / 1_024.0)
    } else {
        format!("{:.2} MiB", bytes as f64 / 1_048_576.0)
    }
}

fn human_duration(microseconds: u128) -> String {
    if microseconds < 1_000 {
        format!("{microseconds} us")
    } else {
        format!("{:.2} ms", microseconds as f64 / 1_000.0)
    }
}

fn path_text(path: &Path) -> Result<&str> {
    path.to_str()
        .ok_or_else(|| anyhow!("path '{}' is not valid UTF-8", path.display()))
}

fn file_storage(path: &Path) -> FileStorage {
    let main_bytes = file_len(path);
    let wal_bytes = file_len(&PathBuf::from(format!("{}-wal", path.display())));
    let shm_bytes = file_len(&PathBuf::from(format!("{}-shm", path.display())));
    FileStorage {
        main_bytes,
        wal_bytes,
        shm_bytes,
        total_bytes: main_bytes
            .saturating_add(wal_bytes)
            .saturating_add(shm_bytes),
    }
}

fn file_len(path: &Path) -> u64 {
    path.metadata().map_or(0, |metadata| metadata.len())
}

fn remove_store_files(path: &Path) -> Result<()> {
    for candidate in store_file_paths(path) {
        if candidate.exists() {
            fs::remove_file(&candidate)
                .with_context(|| format!("failed to remove '{}'", candidate.display()))?;
        }
    }
    Ok(())
}

fn copy_store_files(source: &Path, destination: &Path) -> Result<()> {
    for (source_file, destination_file) in store_file_paths(source)
        .into_iter()
        .zip(store_file_paths(destination))
    {
        if source_file.exists() {
            fs::copy(&source_file, &destination_file).with_context(|| {
                format!(
                    "failed to copy '{}' to '{}'",
                    source_file.display(),
                    destination_file.display()
                )
            })?;
        }
    }
    Ok(())
}

fn store_file_paths(path: &Path) -> [PathBuf; 3] {
    [
        path.to_path_buf(),
        PathBuf::from(format!("{}-wal", path.display())),
        PathBuf::from(format!("{}-shm", path.display())),
    ]
}

async fn checkpoint_connection(conn: &Connection) -> Result<()> {
    let mut rows = conn
        .query("PRAGMA wal_checkpoint(TRUNCATE);", ())
        .await
        .context("failed to checkpoint synthetic store")?;
    while rows.next().await?.is_some() {}
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timing_summary_reports_sorted_percentiles() {
        let summary = timing_summary([10, 50, 20, 30, 40].into_iter());
        assert_eq!(summary.min_us, 10);
        assert_eq!(summary.median_us, 30);
        assert_eq!(summary.p95_us, 50);
        assert_eq!(summary.max_us, 50);
    }

    #[test]
    fn file_growth_saturates_when_checkpoint_shrinks_wal() {
        let baseline = FileStorage {
            main_bytes: 4_096,
            wal_bytes: 32_000,
            shm_bytes: 0,
            total_bytes: 36_096,
        };
        let current = FileStorage {
            main_bytes: 24_576,
            wal_bytes: 0,
            shm_bytes: 0,
            total_bytes: 24_576,
        };
        let growth = current.growth_from(&baseline);
        assert_eq!(growth.main_bytes, 20_480);
        assert_eq!(growth.wal_bytes, 0);
        assert_eq!(growth.total_bytes, 20_480);
    }
}
