use anyhow::{Context, Result};
use async_trait::async_trait;
use clap::{Parser, Subcommand};
use futures::future::BoxFuture;
use futures::stream;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command as StdCommand, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tempfile::TempDir;
use tokio::task::JoinHandle;
use tokio::time::{Instant as TokioInstant, sleep, timeout};
use turin::inference::content::encode_content_json;
use turin::inference::provider::{
    InferenceContent, InferenceEvent, InferenceMessage, InferenceProvider, InferenceRequest,
    InferenceStream, ProviderClient, RequestOptions, SdkError,
};
use turin::kernel::Kernel;
use turin::kernel::config::{
    AgentConfig, EmbeddingConfig, GovernanceConfig, HarnessConfig, HotHistoryConfig,
    HotHistoryProfile, InferenceConfig, KernelConfig, PersistenceConfig, ProviderConfig,
    TurinConfig,
};
use turin::kernel::session::QueuedTask;
use turin::persistence::state::{SessionReadTarget, StateStore, TurnWriteTarget};
use turin_channel_core::{
    ChannelConversationKey, ChannelKind, ChannelMessageRef, ChannelSessionScope, ChannelUser,
    InboundEvent, OutboundMessage,
};
use turin_channel_runner::{
    ChannelDriver, ChannelProgressUpdate, ChannelRunner, RunnerConfig, TaskSnapshot,
};

#[derive(Parser)]
#[command(name = "turin-perf-suite")]
#[command(about = "Local Turin performance and footprint scenarios")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Report static code and artifact footprint for refactor baselines.
    Footprint(FootprintArgs),
    /// Stress session hot history with mocked inference and large tool outputs.
    HotHistory(HotHistoryArgs),
    /// Drive the real daemon through mocked channel ingress/egress and mocked inference.
    FakeChannel(FakeChannelArgs),
    /// Measure daemon/channel cost across logical session and message-count checkpoints.
    ChannelScale(ChannelScaleArgs),
    /// Measure an already-built daemon binary as a separate process.
    BlackboxChannelScale(BlackboxChannelScaleArgs),
    /// Measure daemon task execution without the channel runner layer.
    BlackboxTaskScale(BlackboxTaskScaleArgs),
    /// Measure StateStore write/read scaling without daemon or provider runtime.
    PersistenceScale(PersistenceScaleArgs),
    /// Measure peer-runtime memory before and after idle hibernation.
    IdleRuntime(IdleRuntimeArgs),
}

#[derive(Parser)]
struct FootprintArgs {
    /// Repository root to scan.
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    /// Comma-separated source roots to include.
    #[arg(long, default_value = "src,crates")]
    roots: String,

    /// Number of largest source files to include in the report.
    #[arg(long, default_value_t = 40)]
    top_files: usize,

    /// Extra binary paths to record if they exist.
    #[arg(long = "binary")]
    binaries: Vec<PathBuf>,

    /// Report output directory.
    #[arg(long, default_value = ".workspace/perf-reports")]
    report_dir: PathBuf,
}

#[derive(Parser)]
struct HotHistoryArgs {
    /// Number of user prompts to run through one session.
    #[arg(long, default_value_t = 50)]
    turns: usize,

    /// Bytes written to each synthetic file read by the tool call.
    #[arg(long, default_value_t = 256 * 1024)]
    payload_bytes: usize,

    /// Target byte size for each mocked assistant response.
    #[arg(long, default_value_t = 1024)]
    response_bytes: usize,

    /// Request a large read_file tool call every N prompts. Use 0 to disable tool calls.
    #[arg(long, default_value_t = 4)]
    tool_every: usize,

    /// Capture a snapshot every N turns.
    #[arg(long, default_value_t = 5)]
    sample_every: usize,

    /// Optional persistent workspace. If omitted, an ephemeral temp dir is used.
    #[arg(long)]
    workspace_root: Option<PathBuf>,

    /// Report output directory.
    #[arg(long, default_value = ".workspace/perf-reports")]
    report_dir: PathBuf,

    /// Hot-history profile to apply during the run.
    #[arg(long, value_enum, default_value_t = PerfHotHistoryProfile::Default)]
    hot_history_profile: PerfHotHistoryProfile,

    /// Override hot-history resident message limit.
    #[arg(long)]
    hot_history_max_messages: Option<usize>,

    /// Override hot-history old tool-result payload byte limit.
    #[arg(long)]
    hot_history_max_tool_result_bytes: Option<usize>,

    /// Print streamed turn output while the scenario runs.
    #[arg(long)]
    verbose_turn_output: bool,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum PerfHotHistoryProfile {
    Default,
    Performance,
    Debug,
}

#[derive(Parser)]
struct FakeChannelArgs {
    /// Number of inbound channel messages to process.
    #[arg(long, default_value_t = 25)]
    messages: usize,

    /// Target byte size for each inbound channel text payload.
    #[arg(long, default_value_t = 256)]
    message_bytes: usize,

    /// Mock model response returned by the daemon's configured provider.
    #[arg(long, default_value = "PONG")]
    mock_response: String,

    /// Target byte size for each mocked assistant response.
    #[arg(long, default_value_t = 1024)]
    response_bytes: usize,

    /// Optional persistent workspace. If omitted, an ephemeral temp dir is used.
    #[arg(long)]
    workspace_root: Option<PathBuf>,

    /// Agent runtime idle timeout written to the mock daemon config.
    #[arg(long, default_value_t = 20)]
    agent_idle_timeout_seconds: u64,

    /// Milliseconds to wait after the driver finishes before taking an idle snapshot.
    #[arg(long, default_value_t = 0)]
    post_run_idle_wait_ms: u64,

    /// Run PRAGMA wal_checkpoint(TRUNCATE) after the post-run idle wait.
    #[arg(long)]
    checkpoint_state_db_after_idle: bool,

    /// Call malloc_trim(0) after the post-run idle wait and record another snapshot.
    #[arg(long)]
    trim_allocator_after_idle: bool,

    /// Report output directory.
    #[arg(long, default_value = ".workspace/perf-reports")]
    report_dir: PathBuf,

    /// Print streamed turn output while the scenario runs.
    #[arg(long)]
    verbose_turn_output: bool,
}

#[derive(Parser)]
struct ChannelScaleArgs {
    /// Number of logical channel conversations/sessions to keep active.
    #[arg(long, default_value_t = 1)]
    sessions: usize,

    /// Number of inbound messages to send to each logical session.
    #[arg(long, default_value_t = 1000)]
    messages_per_session: usize,

    /// Target byte size for each inbound channel text payload.
    #[arg(long, default_value_t = 256)]
    message_bytes: usize,

    /// Comma-separated checkpoints, measured as messages per session.
    #[arg(long, default_value = "10,100,200,1000")]
    checkpoints: String,

    /// Mock model response returned by the daemon's configured provider.
    #[arg(long, default_value = "PONG")]
    mock_response: String,

    /// Target byte size for each mocked assistant response.
    #[arg(long, default_value_t = 1024)]
    response_bytes: usize,

    /// Optional persistent workspace. If omitted, an ephemeral temp dir is used.
    #[arg(long)]
    workspace_root: Option<PathBuf>,

    /// Agent runtime idle timeout written to the mock daemon config.
    #[arg(long, default_value_t = 20)]
    agent_idle_timeout_seconds: u64,

    /// Milliseconds to wait after the driver finishes before taking an idle snapshot.
    #[arg(long, default_value_t = 0)]
    post_run_idle_wait_ms: u64,

    /// Run PRAGMA wal_checkpoint(TRUNCATE) after the post-run idle wait.
    #[arg(long)]
    checkpoint_state_db_after_idle: bool,

    /// Call malloc_trim(0) after the post-run idle wait and record another snapshot.
    #[arg(long)]
    trim_allocator_after_idle: bool,

    /// Report output directory.
    #[arg(long, default_value = ".workspace/perf-reports")]
    report_dir: PathBuf,

    /// Print streamed turn output while the scenario runs.
    #[arg(long)]
    verbose_turn_output: bool,
}

#[derive(Parser)]
struct BlackboxChannelScaleArgs {
    /// Turin binary to launch as the daemon under measurement.
    #[arg(long, default_value = "target/release/turin")]
    turin_binary: PathBuf,

    /// Number of logical channel conversations/sessions to keep active.
    #[arg(long, default_value_t = 1)]
    sessions: usize,

    /// Number of inbound messages to send to each logical session.
    #[arg(long, default_value_t = 1000)]
    messages_per_session: usize,

    /// Target byte size for each inbound channel text payload.
    #[arg(long, default_value_t = 256)]
    message_bytes: usize,

    /// Comma-separated checkpoints, measured as messages per session.
    #[arg(long, default_value = "10,100,200,1000")]
    checkpoints: String,

    /// Mock model response returned by the daemon's configured provider.
    #[arg(long, default_value = "PONG")]
    mock_response: String,

    /// Target byte size for each mocked assistant response.
    #[arg(long, default_value_t = 1024)]
    response_bytes: usize,

    /// Optional persistent workspace. If omitted, an ephemeral temp dir is used.
    #[arg(long)]
    workspace_root: Option<PathBuf>,

    /// Agent runtime idle timeout written to the mock daemon config.
    #[arg(long, default_value_t = 20)]
    agent_idle_timeout_seconds: u64,

    /// Milliseconds to wait after the driver finishes before taking an idle snapshot.
    #[arg(long, default_value_t = 0)]
    post_run_idle_wait_ms: u64,

    /// Run PRAGMA wal_checkpoint(TRUNCATE) after the post-run idle wait.
    #[arg(long)]
    checkpoint_state_db_after_idle: bool,

    /// Report output directory.
    #[arg(long, default_value = ".workspace/perf-reports")]
    report_dir: PathBuf,
}

#[derive(Parser)]
struct BlackboxTaskScaleArgs {
    /// Turin binary to launch as the daemon under measurement.
    #[arg(long, default_value = "target/release/turin")]
    turin_binary: PathBuf,

    /// Number of direct daemon tasks to submit into one live session.
    #[arg(long, default_value_t = 1000)]
    tasks: usize,

    /// Target byte size for each task prompt.
    #[arg(long, default_value_t = 256)]
    prompt_bytes: usize,

    /// Comma-separated task-count checkpoints.
    #[arg(long, default_value = "10,100,200,1000")]
    checkpoints: String,

    /// Mock model response returned by the daemon's configured provider.
    #[arg(long, default_value = "PONG")]
    mock_response: String,

    /// Target byte size for each mocked assistant response.
    #[arg(long, default_value_t = 1024)]
    response_bytes: usize,

    /// Optional persistent workspace. If omitted, an ephemeral temp dir is used.
    #[arg(long)]
    workspace_root: Option<PathBuf>,

    /// Agent runtime idle timeout written to the mock daemon config.
    #[arg(long, default_value_t = 20)]
    agent_idle_timeout_seconds: u64,

    /// Milliseconds to wait after the driver finishes before taking an idle snapshot.
    #[arg(long, default_value_t = 0)]
    post_run_idle_wait_ms: u64,

    /// Run PRAGMA wal_checkpoint(TRUNCATE) after the post-run idle wait.
    #[arg(long)]
    checkpoint_state_db_after_idle: bool,

    /// Report output directory.
    #[arg(long, default_value = ".workspace/perf-reports")]
    report_dir: PathBuf,
}

#[derive(Parser)]
struct PersistenceScaleArgs {
    /// Number of synthetic task turns to persist into one session.
    #[arg(long, default_value_t = 1000)]
    tasks: usize,

    /// Target byte size for each user prompt message.
    #[arg(long, default_value_t = 32)]
    prompt_bytes: usize,

    /// Target byte size for each assistant response message.
    #[arg(long, default_value_t = 1024)]
    response_bytes: usize,

    /// Comma-separated task-count checkpoints.
    #[arg(long, default_value = "10,100,200,1000")]
    checkpoints: String,

    /// Also materialize the active branch at each checkpoint, then drop it before sampling.
    #[arg(long)]
    read_active_branch_at_checkpoints: bool,

    /// Also persist representative daemon task, turn, and stream events for each task.
    #[arg(long)]
    include_daemon_events: bool,

    /// Optional persistent workspace. If omitted, an ephemeral temp dir is used.
    #[arg(long)]
    workspace_root: Option<PathBuf>,

    /// Report output directory.
    #[arg(long, default_value = ".workspace/perf-reports")]
    report_dir: PathBuf,
}

#[derive(Parser)]
struct IdleRuntimeArgs {
    /// Number of peer-agent requests to submit before waiting for idle release.
    #[arg(long, default_value_t = 25)]
    requests: usize,

    /// Target byte size for each mocked assistant response.
    #[arg(long, default_value_t = 4096)]
    response_bytes: usize,

    /// Agent idle timeout in seconds. Use 0 to hibernate immediately after each request.
    #[arg(long, default_value_t = 1)]
    idle_timeout_seconds: u64,

    /// Maximum milliseconds to wait for the runtime to hibernate.
    #[arg(long, default_value_t = 5000)]
    max_wait_ms: u64,

    /// Optional persistent workspace. If omitted, an ephemeral temp dir is used.
    #[arg(long)]
    workspace_root: Option<PathBuf>,

    /// Report output directory.
    #[arg(long, default_value = ".workspace/perf-reports")]
    report_dir: PathBuf,

    /// Print streamed turn output while the scenario runs.
    #[arg(long)]
    verbose_turn_output: bool,
}

#[derive(Debug, Serialize)]
struct PerfReport {
    scenario: String,
    config: serde_json::Value,
    workspace_root: String,
    state_db_path: String,
    snapshots: Vec<Snapshot>,
}

#[derive(Debug, Serialize)]
struct FootprintReport {
    scenario: String,
    config: serde_json::Value,
    totals: LineCounts,
    areas: Vec<AreaFootprint>,
    largest_files: Vec<FileFootprint>,
    binaries: Vec<BinaryFootprint>,
}

#[derive(Debug, Clone, Default, Serialize)]
struct LineCounts {
    files: usize,
    total_lines: usize,
    blank_lines: usize,
    comment_lines: usize,
    code_lines: usize,
}

#[derive(Debug, Clone, Serialize)]
struct AreaFootprint {
    area: String,
    counts: LineCounts,
}

#[derive(Debug, Clone, Serialize)]
struct FileFootprint {
    path: String,
    counts: LineCounts,
}

#[derive(Debug, Clone, Serialize)]
struct BinaryFootprint {
    path: String,
    bytes: u64,
}

#[derive(Debug, Serialize)]
struct Snapshot {
    label: String,
    elapsed_ms: u128,
    rss_kb: Option<u64>,
    pss_kb: Option<u64>,
    pss_anon_kb: Option<u64>,
    pss_file_kb: Option<u64>,
    pss_shmem_kb: Option<u64>,
    state_db_main_bytes: u64,
    state_db_wal_bytes: u64,
    state_db_shm_bytes: u64,
    state_db_bytes: u64,
    turn_index: Option<u32>,
    history_len: Option<usize>,
    outbound_messages: Option<usize>,
    active_sessions: Option<usize>,
    live_sessions: Option<usize>,
    messages_per_session: Option<usize>,
    persisted_messages: Option<usize>,
    history_message_offset: Option<usize>,
    hot_window_pruned: Option<bool>,
    history_payload_bytes: Option<usize>,
    tool_results: Option<usize>,
    tool_result_errors: Option<usize>,
    persisted_events: Option<usize>,
    persisted_event_payload_bytes: Option<usize>,
    daemon_tasks: Option<usize>,
    daemon_completed_tasks: Option<usize>,
    daemon_task_snapshot_bytes: Option<usize>,
    daemon_task_output_bytes: Option<usize>,
    daemon_task_assistant_content_bytes: Option<usize>,
}

impl From<PerfHotHistoryProfile> for HotHistoryProfile {
    fn from(value: PerfHotHistoryProfile) -> Self {
        match value {
            PerfHotHistoryProfile::Default => Self::Default,
            PerfHotHistoryProfile::Performance => Self::Performance,
            PerfHotHistoryProfile::Debug => Self::Debug,
        }
    }
}

#[derive(Debug)]
struct ProcessMemory {
    rss_kb: Option<u64>,
    pss_kb: Option<u64>,
    pss_anon_kb: Option<u64>,
    pss_file_kb: Option<u64>,
    pss_shmem_kb: Option<u64>,
}

#[derive(Debug, Default)]
struct DaemonTaskMetrics {
    tasks: usize,
    completed_tasks: usize,
    snapshot_bytes: usize,
    output_bytes: usize,
    assistant_content_bytes: usize,
}

#[derive(Debug)]
struct StateStoreSize {
    main_bytes: u64,
    wal_bytes: u64,
    shm_bytes: u64,
}

impl StateStoreSize {
    fn total_bytes(&self) -> u64 {
        self.main_bytes + self.wal_bytes + self.shm_bytes
    }
}

struct SequencePerfProvider {
    responses: Arc<Mutex<VecDeque<Vec<InferenceEvent>>>>,
}

struct ChannelDaemonHarness {
    endpoint: PathBuf,
    workspace_root: PathBuf,
    join: JoinHandle<Result<()>>,
}

struct BlackboxDaemonHarness {
    endpoint: PathBuf,
    workspace_root: PathBuf,
    child: Child,
}

#[derive(Debug, Deserialize)]
struct OpenedSessionResponse {
    session_id: String,
    slot_id: String,
}

#[derive(Debug, Deserialize)]
struct LiveSessionsResponse {
    sessions: Vec<LiveSessionResponse>,
}

#[derive(Debug, Deserialize)]
struct LiveSessionResponse {
    history: Option<LiveSessionHistoryResponse>,
}

#[derive(Debug, Deserialize)]
struct LiveSessionHistoryResponse {
    len: usize,
    message_offset: usize,
}

#[derive(Debug, Default)]
struct LiveSessionDiagnostics {
    count: usize,
    total_history_len: Option<usize>,
    max_history_message_offset: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct EventMetrics {
    count: usize,
    payload_bytes: usize,
}

#[derive(Debug, Deserialize)]
struct TaskListResponse {
    tasks: Vec<TaskSnapshot>,
}

struct MockChannelDriver {
    events: VecDeque<InboundEvent>,
    sent: Arc<Mutex<Vec<OutboundMessage>>>,
    scale_recorder: Option<ScaleRecorder>,
}

#[derive(Clone)]
struct ScaleRecorder {
    snapshots: Arc<Mutex<Vec<Snapshot>>>,
    start: Instant,
    state_db_path: PathBuf,
    sample_totals: HashSet<usize>,
    active_sessions: usize,
    daemon: turin_daemon_client::DaemonClient,
    memory_target: MemoryTarget,
}

#[derive(Debug, Clone, Copy)]
enum MemoryTarget {
    CurrentProcess,
    Pid(u32),
}

impl InferenceProvider for SequencePerfProvider {
    fn stream<'a>(
        &'a self,
        _request: InferenceRequest,
        _options: Option<RequestOptions>,
    ) -> BoxFuture<'a, std::result::Result<InferenceStream, SdkError>> {
        let responses = self.responses.clone();
        Box::pin(async move {
            let events = responses
                .lock()
                .expect("perf provider response lock poisoned")
                .pop_front()
                .unwrap_or_else(|| final_events(0, "Recorded.".len()))
                .into_iter()
                .map(Ok);
            Ok(Box::pin(stream::iter(events)) as InferenceStream)
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Footprint(args) => run_footprint(args),
        Command::HotHistory(args) => run_hot_history(args).await,
        Command::FakeChannel(args) => run_fake_channel(args).await,
        Command::ChannelScale(args) => run_channel_scale(args).await,
        Command::BlackboxChannelScale(args) => run_blackbox_channel_scale(args).await,
        Command::BlackboxTaskScale(args) => run_blackbox_task_scale(args).await,
        Command::PersistenceScale(args) => run_persistence_scale(args).await,
        Command::IdleRuntime(args) => run_idle_runtime(args).await,
    }
}

fn run_footprint(args: FootprintArgs) -> Result<()> {
    anyhow::ensure!(args.top_files > 0, "--top-files must be greater than zero");

    let repo_root = fs::canonicalize(&args.repo_root)
        .with_context(|| format!("failed to resolve repo root '{}'", args.repo_root.display()))?;
    let roots = parse_source_roots(&args.roots)?;
    let mut area_counts = BTreeMap::<String, LineCounts>::new();
    let mut file_counts = Vec::<FileFootprint>::new();
    let mut totals = LineCounts::default();

    for root in &roots {
        let root_path = repo_root.join(root);
        if !root_path.exists() {
            continue;
        }
        collect_rust_footprint(
            &repo_root,
            root,
            &root_path,
            &mut area_counts,
            &mut file_counts,
        )?;
    }

    for file in &file_counts {
        totals.add(&file.counts);
    }

    file_counts.sort_by(|left, right| {
        right
            .counts
            .code_lines
            .cmp(&left.counts.code_lines)
            .then_with(|| left.path.cmp(&right.path))
    });
    file_counts.truncate(args.top_files);

    let areas = area_counts
        .into_iter()
        .map(|(area, counts)| AreaFootprint { area, counts })
        .collect();
    let binaries = collect_binary_footprint(&repo_root, &args.binaries);
    let report = FootprintReport {
        scenario: "footprint".to_string(),
        config: serde_json::json!({
            "repo_root": repo_root.display().to_string(),
            "roots": roots,
            "top_files": args.top_files,
            "note": "Rust source scan excludes directories/files that are clearly tests, benches, examples, target artifacts, or workspace scratch data. Inline #[cfg(test)] modules are not stripped from non-test source files.",
        }),
        totals,
        areas,
        largest_files: file_counts,
        binaries,
    };

    write_footprint_reports(&args.report_dir, &report)?;
    print_footprint_summary(&report);
    Ok(())
}

async fn run_hot_history(args: HotHistoryArgs) -> Result<()> {
    anyhow::ensure!(args.turns > 0, "--turns must be greater than zero");
    anyhow::ensure!(
        args.sample_every > 0,
        "--sample-every must be greater than zero"
    );

    let (_temp_guard, workspace_root) = prepare_workspace(args.workspace_root)?;
    let harness_dir = workspace_root.join("harnesses");
    let payload_dir = workspace_root.join("payloads");
    let state_db_path = workspace_root.join("state.db");
    fs::create_dir_all(&harness_dir)?;
    fs::create_dir_all(&payload_dir)?;
    fs::write(harness_dir.join("main.lua"), "-- perf harness\n")?;

    for index in tool_payload_indices(args.turns, args.tool_every) {
        fs::write(
            payload_dir.join(format!("payload-{index}.txt")),
            synthetic_payload(index, args.payload_bytes),
        )?;
    }

    let hot_history_config = HotHistoryConfig {
        profile: args.hot_history_profile.into(),
        max_messages: args.hot_history_max_messages,
        max_tool_result_bytes: args.hot_history_max_tool_result_bytes,
    };
    let config = build_config(
        &workspace_root,
        &harness_dir,
        &state_db_path,
        args.turns,
        Some(hot_history_config.clone()),
        Some(0),
    )?;
    let responses = Arc::new(Mutex::new(build_responses(
        args.turns,
        args.tool_every,
        args.response_bytes,
    )));
    let provider = Arc::new(SequencePerfProvider { responses });

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;
    kernel.add_client("mock".to_string(), ProviderClient::new("mock", provider));

    let mut session = kernel.create_session().await;
    let start = Instant::now();
    let expected_tool_calls = tool_call_count(args.turns, args.tool_every);
    let mut snapshots = vec![hot_history_snapshot("start", start, &state_db_path, &session).await?];

    {
        let _stdout_guard = StdoutSilencer::new(!args.verbose_turn_output)?;
        for index in 0..args.turns {
            kernel
                .run(&mut session, Some(format!("Process payload {index}.")))
                .await
                .with_context(|| format!("hot-history turn {index} failed"))?;

            if (index + 1) % args.sample_every == 0 || index + 1 == args.turns {
                snapshots.push(
                    hot_history_snapshot(
                        &format!("after-turn-{}", index + 1),
                        start,
                        &state_db_path,
                        &session,
                    )
                    .await?,
                );
            }
        }

        kernel.end_session(&mut session).await?;
    }

    snapshots
        .push(hot_history_snapshot("after-end-session", start, &state_db_path, &session).await?);

    let report = PerfReport {
        scenario: "hot-history".to_string(),
        config: serde_json::json!({
            "turns": args.turns,
            "payload_bytes": args.payload_bytes,
            "response_bytes": args.response_bytes,
            "tool_every": args.tool_every,
            "expected_tool_calls": expected_tool_calls,
            "sample_every": args.sample_every,
            "hot_history_profile": format!("{:?}", args.hot_history_profile).to_ascii_lowercase(),
            "hot_history_effective_max_messages": hot_history_config.effective_max_messages(),
            "hot_history_effective_max_tool_result_bytes": hot_history_config.effective_max_tool_result_bytes(),
        }),
        workspace_root: workspace_root.display().to_string(),
        state_db_path: state_db_path.display().to_string(),
        snapshots,
    };

    write_reports(&args.report_dir, &report)?;
    print_summary(&report);
    Ok(())
}

async fn run_idle_runtime(args: IdleRuntimeArgs) -> Result<()> {
    anyhow::ensure!(args.requests > 0, "--requests must be greater than zero");
    anyhow::ensure!(
        args.max_wait_ms > 0,
        "--max-wait-ms must be greater than zero"
    );

    let (_temp_guard, workspace_root) = prepare_workspace(args.workspace_root)?;
    let harness_dir = workspace_root.join("harnesses");
    let state_db_path = workspace_root.join("state.db");
    fs::create_dir_all(&harness_dir)?;
    fs::write(
        harness_dir.join("main.lua"),
        "-- idle runtime perf harness\n",
    )?;

    let mut config = build_config(
        &workspace_root,
        &harness_dir,
        &state_db_path,
        args.requests.saturating_add(1),
        None,
        Some(args.idle_timeout_seconds),
    )?;
    if let Some(provider) = config.providers.get_mut("mock") {
        provider.base_url = Some(synthetic_text(
            "Idle runtime response.",
            args.response_bytes,
        ));
    }

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let manager = kernel.agent_manager();
    let start = Instant::now();
    let mut snapshots = vec![idle_runtime_snapshot(
        "after-kernel-init",
        start,
        &state_db_path,
        0,
    )];

    {
        let _stdout_guard = StdoutSilencer::new(!args.verbose_turn_output)?;
        for index in 0..args.requests {
            let request_id = manager
                .submit(
                    "default",
                    QueuedTask::ad_hoc(format!("idle runtime request {index}")),
                    None,
                )
                .await?;
            let result = manager.await_result(&request_id, Some(30_000)).await?;
            if let Some(err) = result.error {
                anyhow::bail!("idle-runtime request {index} failed: {err}");
            }
            if index == 0 {
                snapshots.push(idle_runtime_snapshot(
                    "after-first-request",
                    start,
                    &state_db_path,
                    manager.list_live_sessions(None).await.len(),
                ));
            }
            if args.idle_timeout_seconds == 0 && index + 1 < args.requests {
                let _ = wait_for_peer_runtime_release(
                    &manager,
                    Duration::from_millis(args.max_wait_ms),
                )
                .await;
            }
        }
    }

    let live_after_requests = manager.list_live_sessions(None).await.len();
    snapshots.push(idle_runtime_snapshot(
        "after-all-requests",
        start,
        &state_db_path,
        live_after_requests,
    ));

    let released =
        wait_for_peer_runtime_release(&manager, Duration::from_millis(args.max_wait_ms)).await;
    let live_sessions = manager.list_live_sessions(None).await.len();
    snapshots.push(idle_runtime_snapshot(
        if released {
            "after-idle-release"
        } else {
            "after-idle-wait-timeout"
        },
        start,
        &state_db_path,
        live_sessions,
    ));

    let report = PerfReport {
        scenario: "idle-runtime".to_string(),
        config: serde_json::json!({
            "requests": args.requests,
            "response_bytes": args.response_bytes,
            "idle_timeout_seconds": args.idle_timeout_seconds,
            "max_wait_ms": args.max_wait_ms,
            "released": released,
            "live_sessions_column": "live peer runtime sessions",
        }),
        workspace_root: workspace_root.display().to_string(),
        state_db_path: state_db_path.display().to_string(),
        snapshots,
    };

    write_reports(&args.report_dir, &report)?;
    print_summary(&report);
    Ok(())
}

async fn wait_for_peer_runtime_release(
    manager: &Arc<turin::kernel::agent_manager::AgentManager>,
    max_wait: Duration,
) -> bool {
    let deadline = TokioInstant::now() + max_wait;
    loop {
        let live_sessions = manager.list_live_sessions(None).await.len();
        let running = manager
            .get_status("default")
            .await
            .is_some_and(|status| status.running);
        if live_sessions == 0 && !running {
            return true;
        }
        if TokioInstant::now() >= deadline {
            return false;
        }
        sleep(Duration::from_millis(25)).await;
    }
}

async fn run_fake_channel(args: FakeChannelArgs) -> Result<()> {
    anyhow::ensure!(args.messages > 0, "--messages must be greater than zero");

    let (_temp_guard, workspace_root) = prepare_workspace(args.workspace_root)?;
    let state_db_path = workspace_root.join("state.db");
    let channel_runtime_dir = channel_runtime_dir(&workspace_root, "mock");
    fs::create_dir_all(&channel_runtime_dir)?;

    let start = Instant::now();
    let mut snapshots = vec![snapshot(
        "start",
        start,
        &state_db_path,
        None,
        None,
        Some(0),
        None,
        None,
        None,
    )];

    let mock_response = synthetic_text(&args.mock_response, args.response_bytes);
    let daemon = ChannelDaemonHarness::start(
        workspace_root.clone(),
        &state_db_path,
        &mock_response,
        args.agent_idle_timeout_seconds,
    )
    .await?;
    snapshots.push(snapshot(
        "after-daemon-start",
        start,
        &state_db_path,
        None,
        None,
        Some(0),
        None,
        Some(daemon.live_session_count().await?),
        None,
    ));

    let runner = daemon.runner();
    let mut driver = MockChannelDriver::new(sample_events(args.messages, args.message_bytes));
    let sent = Arc::clone(&driver.sent);

    {
        let _stdout_guard = StdoutSilencer::new(!args.verbose_turn_output)?;
        runner
            .run_driver("default", &mut driver, Some(30_000))
            .await?;
    }

    let outbound_count = sent.lock().expect("sent lock poisoned").len();
    snapshots.push(snapshot(
        "after-runner",
        start,
        &state_db_path,
        None,
        None,
        Some(outbound_count),
        None,
        Some(daemon.live_session_count().await?),
        None,
    ));

    if args.post_run_idle_wait_ms > 0 {
        sleep(Duration::from_millis(args.post_run_idle_wait_ms)).await;
        snapshots.push(snapshot(
            "after-idle-wait",
            start,
            &state_db_path,
            None,
            None,
            Some(outbound_count),
            None,
            Some(daemon.live_session_count().await?),
            None,
        ));
    }

    if args.checkpoint_state_db_after_idle {
        checkpoint_state_db(&state_db_path).await?;
        snapshots.push(snapshot(
            "after-db-checkpoint",
            start,
            &state_db_path,
            None,
            None,
            Some(outbound_count),
            None,
            Some(daemon.live_session_count().await?),
            None,
        ));
    }

    if args.trim_allocator_after_idle {
        let _ = trim_allocator();
        snapshots.push(snapshot(
            "after-allocator-trim",
            start,
            &state_db_path,
            None,
            None,
            Some(outbound_count),
            None,
            Some(daemon.live_session_count().await?),
            None,
        ));
    }

    daemon.stop().await?;
    snapshots.push(snapshot(
        "after-daemon-stop",
        start,
        &state_db_path,
        None,
        None,
        Some(outbound_count),
        None,
        None,
        None,
    ));

    let report = PerfReport {
        scenario: "fake-channel".to_string(),
        config: serde_json::json!({
            "messages": args.messages,
            "message_bytes": args.message_bytes,
            "mock_response_bytes": mock_response.len(),
            "agent_idle_timeout_seconds": args.agent_idle_timeout_seconds,
            "post_run_idle_wait_ms": args.post_run_idle_wait_ms,
            "checkpoint_state_db_after_idle": args.checkpoint_state_db_after_idle,
            "trim_allocator_after_idle": args.trim_allocator_after_idle,
            "allocator_trim_supported": allocator_trim_supported(),
        }),
        workspace_root: workspace_root.display().to_string(),
        state_db_path: state_db_path.display().to_string(),
        snapshots,
    };

    write_reports(&args.report_dir, &report)?;
    print_summary(&report);
    Ok(())
}

async fn run_channel_scale(args: ChannelScaleArgs) -> Result<()> {
    anyhow::ensure!(args.sessions > 0, "--sessions must be greater than zero");
    anyhow::ensure!(
        args.messages_per_session > 0,
        "--messages-per-session must be greater than zero"
    );

    let checkpoints = parse_checkpoints(&args.checkpoints, args.messages_per_session)?;
    let (_temp_guard, workspace_root) = prepare_workspace(args.workspace_root)?;
    let state_db_path = workspace_root.join("state.db");
    let channel_runtime_dir = channel_runtime_dir(&workspace_root, "mock");
    fs::create_dir_all(&channel_runtime_dir)?;

    let start = Instant::now();
    let snapshots = Arc::new(Mutex::new(vec![
        channel_scale_snapshot(
            "fresh-start",
            start,
            &state_db_path,
            Some(0),
            Some(0),
            Some(0),
            Some(0),
            MemoryTarget::CurrentProcess,
        )
        .await?,
    ]));

    let mock_response = synthetic_text(&args.mock_response, args.response_bytes);
    let daemon = ChannelDaemonHarness::start(
        workspace_root.clone(),
        &state_db_path,
        &mock_response,
        args.agent_idle_timeout_seconds,
    )
    .await?;
    let after_daemon_start = channel_scale_snapshot(
        "after-daemon-start",
        start,
        &state_db_path,
        Some(0),
        Some(0),
        Some(0),
        Some(daemon.live_session_count().await?),
        MemoryTarget::CurrentProcess,
    )
    .await?;
    snapshots
        .lock()
        .expect("scale snapshots lock poisoned")
        .push(after_daemon_start);

    let sample_totals = checkpoints
        .iter()
        .map(|checkpoint| checkpoint * args.sessions)
        .collect::<HashSet<_>>();
    let recorder = ScaleRecorder {
        snapshots: Arc::clone(&snapshots),
        start,
        state_db_path: state_db_path.clone(),
        sample_totals,
        active_sessions: args.sessions,
        daemon: daemon.client(),
        memory_target: MemoryTarget::CurrentProcess,
    };

    let runner = daemon.runner();
    let events = scale_events(args.sessions, args.messages_per_session, args.message_bytes);
    let mut driver = MockChannelDriver::with_scale_recorder(events, recorder);
    let sent = Arc::clone(&driver.sent);

    {
        let _stdout_guard = StdoutSilencer::new(!args.verbose_turn_output)?;
        runner
            .run_driver("default", &mut driver, Some(120_000))
            .await?;
    }

    let outbound_count = sent.lock().expect("sent lock poisoned").len();
    drop(driver);
    let after_runner = channel_scale_snapshot(
        "after-runner",
        start,
        &state_db_path,
        Some(outbound_count),
        Some(args.sessions),
        Some(args.messages_per_session),
        Some(daemon.live_session_count().await?),
        MemoryTarget::CurrentProcess,
    )
    .await?;
    snapshots
        .lock()
        .expect("scale snapshots lock poisoned")
        .push(after_runner);

    if args.post_run_idle_wait_ms > 0 {
        sleep(Duration::from_millis(args.post_run_idle_wait_ms)).await;
        let after_idle_wait = channel_scale_snapshot(
            "after-idle-wait",
            start,
            &state_db_path,
            Some(outbound_count),
            Some(args.sessions),
            Some(args.messages_per_session),
            Some(daemon.live_session_count().await?),
            MemoryTarget::CurrentProcess,
        )
        .await?;
        snapshots
            .lock()
            .expect("scale snapshots lock poisoned")
            .push(after_idle_wait);
    }

    if args.checkpoint_state_db_after_idle {
        checkpoint_state_db(&state_db_path).await?;
        let after_db_checkpoint = channel_scale_snapshot(
            "after-db-checkpoint",
            start,
            &state_db_path,
            Some(outbound_count),
            Some(args.sessions),
            Some(args.messages_per_session),
            Some(daemon.live_session_count().await?),
            MemoryTarget::CurrentProcess,
        )
        .await?;
        snapshots
            .lock()
            .expect("scale snapshots lock poisoned")
            .push(after_db_checkpoint);
    }

    if args.trim_allocator_after_idle {
        let _ = trim_allocator();
        let after_allocator_trim = channel_scale_snapshot(
            "after-allocator-trim",
            start,
            &state_db_path,
            Some(outbound_count),
            Some(args.sessions),
            Some(args.messages_per_session),
            Some(daemon.live_session_count().await?),
            MemoryTarget::CurrentProcess,
        )
        .await?;
        snapshots
            .lock()
            .expect("scale snapshots lock poisoned")
            .push(after_allocator_trim);
    }

    daemon.stop().await?;
    let after_daemon_stop = channel_scale_snapshot(
        "after-daemon-stop",
        start,
        &state_db_path,
        Some(outbound_count),
        Some(args.sessions),
        Some(args.messages_per_session),
        None,
        MemoryTarget::CurrentProcess,
    )
    .await?;
    snapshots
        .lock()
        .expect("scale snapshots lock poisoned")
        .push(after_daemon_stop);

    let report = PerfReport {
        scenario: "channel-scale".to_string(),
        config: serde_json::json!({
            "sessions": args.sessions,
            "messages_per_session": args.messages_per_session,
            "message_bytes": args.message_bytes,
            "checkpoints": checkpoints,
            "mock_response_bytes": mock_response.len(),
            "agent_idle_timeout_seconds": args.agent_idle_timeout_seconds,
            "post_run_idle_wait_ms": args.post_run_idle_wait_ms,
            "checkpoint_state_db_after_idle": args.checkpoint_state_db_after_idle,
            "trim_allocator_after_idle": args.trim_allocator_after_idle,
            "allocator_trim_supported": allocator_trim_supported(),
        }),
        workspace_root: workspace_root.display().to_string(),
        state_db_path: state_db_path.display().to_string(),
        snapshots: Arc::try_unwrap(snapshots)
            .map_err(|_| anyhow::anyhow!("scale snapshot recorder still has references"))?
            .into_inner()
            .expect("scale snapshots lock poisoned"),
    };

    write_reports(&args.report_dir, &report)?;
    print_summary(&report);
    Ok(())
}

async fn run_blackbox_channel_scale(args: BlackboxChannelScaleArgs) -> Result<()> {
    anyhow::ensure!(args.sessions > 0, "--sessions must be greater than zero");
    anyhow::ensure!(
        args.messages_per_session > 0,
        "--messages-per-session must be greater than zero"
    );
    anyhow::ensure!(
        args.turin_binary.exists(),
        "turin binary '{}' does not exist; build it first or pass --turin-binary",
        args.turin_binary.display()
    );

    let checkpoints = parse_checkpoints(&args.checkpoints, args.messages_per_session)?;
    let (_temp_guard, workspace_root) = prepare_workspace(args.workspace_root)?;
    let state_db_path = workspace_root.join("state.db");
    let channel_runtime_dir = channel_runtime_dir(&workspace_root, "mock");
    fs::create_dir_all(&channel_runtime_dir)?;

    let start = Instant::now();
    let mut snapshots = Vec::new();
    let mut fresh_start = channel_scale_snapshot(
        "fresh-start",
        start,
        &state_db_path,
        Some(0),
        Some(0),
        Some(0),
        Some(0),
        MemoryTarget::CurrentProcess,
    )
    .await?;
    fresh_start.rss_kb = None;
    fresh_start.pss_kb = None;
    fresh_start.pss_anon_kb = None;
    fresh_start.pss_file_kb = None;
    fresh_start.pss_shmem_kb = None;
    snapshots.push(fresh_start);

    let mock_response = synthetic_text(&args.mock_response, args.response_bytes);
    let daemon = BlackboxDaemonHarness::start(
        args.turin_binary.clone(),
        workspace_root.clone(),
        &state_db_path,
        &mock_response,
        args.agent_idle_timeout_seconds,
    )
    .await?;
    let memory_target = MemoryTarget::Pid(daemon.pid());
    snapshots.push(
        channel_scale_snapshot(
            "after-daemon-start",
            start,
            &state_db_path,
            Some(0),
            Some(0),
            Some(0),
            Some(daemon.live_session_count().await?),
            memory_target,
        )
        .await?,
    );

    let snapshots = Arc::new(Mutex::new(snapshots));
    let sample_totals = checkpoints
        .iter()
        .map(|checkpoint| checkpoint * args.sessions)
        .collect::<HashSet<_>>();
    let recorder = ScaleRecorder {
        snapshots: Arc::clone(&snapshots),
        start,
        state_db_path: state_db_path.clone(),
        sample_totals,
        active_sessions: args.sessions,
        daemon: daemon.client(),
        memory_target,
    };

    let runner = daemon.runner();
    let events = scale_events(args.sessions, args.messages_per_session, args.message_bytes);
    let mut driver = MockChannelDriver::with_scale_recorder(events, recorder);
    let sent = Arc::clone(&driver.sent);
    runner
        .run_driver("default", &mut driver, Some(120_000))
        .await?;

    let outbound_count = sent.lock().expect("sent lock poisoned").len();
    drop(driver);
    let after_runner = channel_scale_snapshot(
        "after-runner",
        start,
        &state_db_path,
        Some(outbound_count),
        Some(args.sessions),
        Some(args.messages_per_session),
        Some(daemon.live_session_count().await?),
        memory_target,
    )
    .await?;
    snapshots
        .lock()
        .expect("scale snapshots lock poisoned")
        .push(after_runner);

    if args.post_run_idle_wait_ms > 0 {
        sleep(Duration::from_millis(args.post_run_idle_wait_ms)).await;
        let mut after_idle_wait = channel_scale_snapshot(
            "after-idle-wait",
            start,
            &state_db_path,
            Some(outbound_count),
            Some(args.sessions),
            Some(args.messages_per_session),
            Some(daemon.live_session_count().await?),
            memory_target,
        )
        .await?;
        if let Ok(metrics) = daemon_task_metrics(&daemon.client()).await {
            after_idle_wait.set_daemon_task_metrics(metrics);
        }
        snapshots
            .lock()
            .expect("scale snapshots lock poisoned")
            .push(after_idle_wait);
    }

    let pid = daemon.pid();
    daemon.stop().await?;
    if args.checkpoint_state_db_after_idle {
        checkpoint_state_db(&state_db_path).await?;
    }
    let after_daemon_stop = channel_scale_snapshot(
        "after-daemon-stop",
        start,
        &state_db_path,
        Some(outbound_count),
        Some(args.sessions),
        Some(args.messages_per_session),
        None,
        MemoryTarget::Pid(pid),
    )
    .await?;
    snapshots
        .lock()
        .expect("scale snapshots lock poisoned")
        .push(after_daemon_stop);

    let report = PerfReport {
        scenario: "blackbox-channel-scale".to_string(),
        config: serde_json::json!({
            "turin_binary": args.turin_binary.display().to_string(),
            "sessions": args.sessions,
            "messages_per_session": args.messages_per_session,
            "message_bytes": args.message_bytes,
            "checkpoints": checkpoints,
            "mock_response_bytes": mock_response.len(),
            "agent_idle_timeout_seconds": args.agent_idle_timeout_seconds,
            "post_run_idle_wait_ms": args.post_run_idle_wait_ms,
            "checkpoint_state_db_after_stop": args.checkpoint_state_db_after_idle,
            "memory_target": "daemon child process",
            "turin_trim_allocator_on_peer_idle": std::env::var("TURIN_TRIM_ALLOCATOR_ON_PEER_IDLE").ok(),
        }),
        workspace_root: workspace_root.display().to_string(),
        state_db_path: state_db_path.display().to_string(),
        snapshots: Arc::try_unwrap(snapshots)
            .map_err(|_| anyhow::anyhow!("scale snapshot recorder still has references"))?
            .into_inner()
            .expect("scale snapshots lock poisoned"),
    };

    write_reports(&args.report_dir, &report)?;
    print_summary(&report);
    Ok(())
}

async fn run_blackbox_task_scale(args: BlackboxTaskScaleArgs) -> Result<()> {
    anyhow::ensure!(args.tasks > 0, "--tasks must be greater than zero");
    anyhow::ensure!(
        args.turin_binary.exists(),
        "turin binary '{}' does not exist; build it first or pass --turin-binary",
        args.turin_binary.display()
    );

    let checkpoints = parse_checkpoints(&args.checkpoints, args.tasks)?;
    let checkpoint_set = checkpoints.iter().copied().collect::<HashSet<_>>();
    let (_temp_guard, workspace_root) = prepare_workspace(args.workspace_root)?;
    let state_db_path = workspace_root.join("state.db");
    let start = Instant::now();

    let mut fresh_start = channel_scale_snapshot(
        "fresh-start",
        start,
        &state_db_path,
        Some(0),
        Some(0),
        Some(0),
        Some(0),
        MemoryTarget::CurrentProcess,
    )
    .await?;
    fresh_start.rss_kb = None;
    fresh_start.pss_kb = None;
    fresh_start.pss_anon_kb = None;
    fresh_start.pss_file_kb = None;
    fresh_start.pss_shmem_kb = None;
    let mut snapshots = vec![fresh_start];

    let mock_response = synthetic_text(&args.mock_response, args.response_bytes);
    let daemon = BlackboxDaemonHarness::start(
        args.turin_binary.clone(),
        workspace_root.clone(),
        &state_db_path,
        &mock_response,
        args.agent_idle_timeout_seconds,
    )
    .await?;
    let memory_target = MemoryTarget::Pid(daemon.pid());
    let client = daemon.client();

    snapshots.push(blackbox_task_scale_snapshot(
        &daemon,
        "after-daemon-start",
        start,
        &state_db_path,
        Some(0),
        Some(1),
        Some(0),
        memory_target,
    )
    .await?);

    let opened: OpenedSessionResponse = client
        .request_ok(
            None,
            turin_daemon_protocol::DaemonRequest::SessionOpen(
                turin_daemon_protocol::OpenSessionParams {
                    agent_id: "default".to_string(),
                    slot_id: Some("direct".to_string()),
                    channel_id: None,
                },
            ),
        )
        .await?;

    let prompt = synthetic_text("task", args.prompt_bytes);
    for completed in 1..=args.tasks {
        let submitted: TaskSnapshot = client
            .request_ok(
                None,
                turin_daemon_protocol::DaemonRequest::TaskSubmit(
                    turin_daemon_protocol::SubmitTaskParams {
                        agent_id: None,
                        session_id: Some(opened.session_id.clone()),
                        slot_id: Some(opened.slot_id.clone()),
                        prompt: prompt.clone(),
                        content: None,
                        tools: None,
                        conflict_policy: None,
                    },
                ),
            )
            .await?;
        let _: TaskSnapshot = client
            .request_ok(
                None,
                turin_daemon_protocol::DaemonRequest::TaskWait(
                    turin_daemon_protocol::WaitTaskParams {
                        request_id: submitted.request_id,
                        timeout_ms: Some(120_000),
                    },
                ),
            )
            .await?;

        if checkpoint_set.contains(&completed) {
            snapshots.push(
                blackbox_task_scale_snapshot(
                    &daemon,
                    &format!("after-{completed}-tasks"),
                    start,
                    &state_db_path,
                    Some(completed),
                    Some(1),
                    Some(completed),
                    memory_target,
                )
                .await?,
            );
        }
    }

    let mut after_runner = blackbox_task_scale_snapshot(
        &daemon,
        "after-runner",
        start,
        &state_db_path,
        Some(args.tasks),
        Some(1),
        Some(args.tasks),
        memory_target,
    )
    .await?;
    if let Ok(metrics) = daemon_task_metrics(&client).await {
        after_runner.set_daemon_task_metrics(metrics);
    }
    snapshots.push(after_runner);

    if args.post_run_idle_wait_ms > 0 {
        sleep(Duration::from_millis(args.post_run_idle_wait_ms)).await;
        let mut after_idle_wait = blackbox_task_scale_snapshot(
            &daemon,
            "after-idle-wait",
            start,
            &state_db_path,
            Some(args.tasks),
            Some(1),
            Some(args.tasks),
            memory_target,
        )
        .await?;
        if let Ok(metrics) = daemon_task_metrics(&client).await {
            after_idle_wait.set_daemon_task_metrics(metrics);
        }
        snapshots.push(after_idle_wait);
    }

    let pid = daemon.pid();
    daemon.stop().await?;
    if args.checkpoint_state_db_after_idle {
        checkpoint_state_db(&state_db_path).await?;
    }
    snapshots.push(
        channel_scale_snapshot(
            "after-daemon-stop",
            start,
            &state_db_path,
            Some(args.tasks),
            Some(1),
            Some(args.tasks),
            None,
            MemoryTarget::Pid(pid),
        )
        .await?,
    );

    let report = PerfReport {
        scenario: "blackbox-task-scale".to_string(),
        config: serde_json::json!({
            "turin_binary": args.turin_binary.display().to_string(),
            "tasks": args.tasks,
            "prompt_bytes": args.prompt_bytes,
            "checkpoints": checkpoints,
            "mock_response_bytes": mock_response.len(),
            "agent_idle_timeout_seconds": args.agent_idle_timeout_seconds,
            "post_run_idle_wait_ms": args.post_run_idle_wait_ms,
            "checkpoint_state_db_after_stop": args.checkpoint_state_db_after_idle,
            "memory_target": "daemon child process",
            "turin_trim_allocator_on_peer_idle": std::env::var("TURIN_TRIM_ALLOCATOR_ON_PEER_IDLE").ok(),
        }),
        workspace_root: workspace_root.display().to_string(),
        state_db_path: state_db_path.display().to_string(),
        snapshots,
    };

    write_reports(&args.report_dir, &report)?;
    print_summary(&report);
    Ok(())
}

async fn blackbox_task_scale_snapshot(
    daemon: &BlackboxDaemonHarness,
    label: &str,
    start: Instant,
    state_db_path: &Path,
    completed_tasks: Option<usize>,
    active_sessions: Option<usize>,
    messages_per_session: Option<usize>,
    memory_target: MemoryTarget,
) -> Result<Snapshot> {
    let diagnostics = daemon.live_session_diagnostics().await?;
    let mut snapshot = channel_scale_snapshot(
        label,
        start,
        state_db_path,
        completed_tasks,
        active_sessions,
        messages_per_session,
        Some(diagnostics.count),
        memory_target,
    )
    .await?;
    snapshot.set_live_session_diagnostics(diagnostics);
    Ok(snapshot)
}

async fn run_persistence_scale(args: PersistenceScaleArgs) -> Result<()> {
    anyhow::ensure!(args.tasks > 0, "--tasks must be greater than zero");
    let checkpoints = parse_checkpoints(&args.checkpoints, args.tasks)?;
    let checkpoint_set = checkpoints.iter().copied().collect::<HashSet<_>>();
    let (_temp_guard, workspace_root) = prepare_workspace(args.workspace_root)?;
    let state_db_path = workspace_root.join("state.db");
    let start = Instant::now();

    let mut snapshots = Vec::new();
    snapshots.push(persistence_scale_snapshot(
        "fresh-start",
        start,
        &state_db_path,
        0,
        None,
        Some(EventMetrics {
            count: 0,
            payload_bytes: 0,
        }),
    ));

    let store = StateStore::open(
        state_db_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("state DB path is not valid UTF-8"))?,
    )
    .await?;
    let session_id = store
        .create_session(uuid::Uuid::now_v7(), "default", None)
        .await?;
    snapshots.push(persistence_scale_snapshot(
        "after-session-create",
        start,
        &state_db_path,
        0,
        None,
        event_metrics_if_enabled(args.include_daemon_events, &state_db_path).await?,
    ));

    let user_content = vec![InferenceContent::Text {
        text: synthetic_text("task", args.prompt_bytes),
    }];
    let assistant_content = vec![InferenceContent::Text {
        text: synthetic_text("response", args.response_bytes),
    }];
    let user_json = encode_content_json(&user_content);
    let assistant_json = encode_content_json(&assistant_content);

    for completed in 1..=args.tasks {
        let turn_index = (completed - 1) as u32;
        let target = store
            .prepare_turn_write_target(session_id, TurnWriteTarget::active_branch(turn_index))
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!("No active branch head available for session {session_id}")
            })?;
        store
            .insert_message(session_id, target, "user", &user_json, None)
            .await?;
        store
            .insert_message(session_id, target, "assistant", &assistant_json, None)
            .await?;
        if args.include_daemon_events {
            insert_representative_daemon_task_events(
                &store,
                session_id,
                target,
                completed,
                &user_content,
                &assistant_content,
            )
            .await?;
        }

        if checkpoint_set.contains(&completed) {
            let read_len = if args.read_active_branch_at_checkpoints {
                let messages = store
                    .get_messages(session_id, &SessionReadTarget::ActiveBranch)
                    .await?;
                let len = messages.len();
                drop(messages);
                Some(len)
            } else {
                None
            };
            snapshots.push(persistence_scale_snapshot(
                &format!("after-{completed}-tasks"),
                start,
                &state_db_path,
                completed,
                read_len,
                event_metrics_if_enabled(args.include_daemon_events, &state_db_path).await?,
            ));
        }
    }

    snapshots.push(persistence_scale_snapshot(
        "after-writes",
        start,
        &state_db_path,
        args.tasks,
        None,
        event_metrics_if_enabled(args.include_daemon_events, &state_db_path).await?,
    ));

    let report = PerfReport {
        scenario: "persistence-scale".to_string(),
        config: serde_json::json!({
            "tasks": args.tasks,
            "prompt_bytes": args.prompt_bytes,
            "response_bytes": args.response_bytes,
            "checkpoints": checkpoints,
            "read_active_branch_at_checkpoints": args.read_active_branch_at_checkpoints,
            "include_daemon_events": args.include_daemon_events,
            "memory_target": "perf-suite process",
        }),
        workspace_root: workspace_root.display().to_string(),
        state_db_path: state_db_path.display().to_string(),
        snapshots,
    };

    write_reports(&args.report_dir, &report)?;
    print_summary(&report);
    Ok(())
}

fn persistence_scale_snapshot(
    label: &str,
    start: Instant,
    state_db_path: &Path,
    completed_tasks: usize,
    read_history_len: Option<usize>,
    event_metrics: Option<EventMetrics>,
) -> Snapshot {
    let mut snapshot = snapshot(
        label,
        start,
        state_db_path,
        None,
        read_history_len,
        Some(completed_tasks),
        Some(1),
        None,
        Some(completed_tasks),
    );
    snapshot.persisted_messages = Some(completed_tasks.saturating_mul(2));
    if let Some(metrics) = event_metrics {
        snapshot.persisted_events = Some(metrics.count);
        snapshot.persisted_event_payload_bytes = Some(metrics.payload_bytes);
    }
    snapshot
}

async fn insert_representative_daemon_task_events(
    store: &StateStore,
    session_id: i64,
    turn_target: TurnWriteTarget,
    task_number: usize,
    user_content: &[InferenceContent],
    assistant_content: &[InferenceContent],
) -> Result<()> {
    let turn_index = (task_number - 1) as u32;
    let task_id = format!("t_{task_number}");
    let trace_id = format!("trace-{task_number}");
    let prompt = first_text_content(user_content).unwrap_or_default();
    let response = first_text_content(assistant_content).unwrap_or_default();
    let identity = serde_json::json!({
        "session_id": format!("perf-session-{session_id}"),
        "agent_id": "default",
    });
    let execution = serde_json::json!({
        "execution_id": "ex_perf",
        "context_target": {
            "kind": "branch_head",
            "branch_head_id": null,
        },
        "visibility": "visible",
        "durability": "durable",
        "write_policy": "advance_branch_head",
    });

    store
        .insert_event(
            session_id,
            None,
            "task_start",
            &serde_json::json!({
                "type": "task_start",
                "identity": identity,
                "task_id": task_id,
                "trace_id": trace_id,
                "plan_id": null,
                "title": null,
                "prompt": prompt,
                "queue_depth": 0,
                "execution": execution,
            }),
        )
        .await?;
    store
        .insert_event(
            session_id,
            Some(turn_target),
            "turn_start",
            &serde_json::json!({
                "type": "turn_start",
                "identity": identity,
                "turn_index": turn_index,
                "task_id": task_id,
                "trace_id": trace_id,
                "task_turn_index": 0,
            }),
        )
        .await?;
    store
        .insert_event(
            session_id,
            Some(turn_target),
            "turn_prepare",
            &serde_json::json!({
                "type": "turn_prepare",
                "identity": identity,
                "turn_index": turn_index,
                "task_id": task_id,
                "trace_id": trace_id,
                "task_turn_index": 0,
            }),
        )
        .await?;
    store
        .insert_event(
            session_id,
            Some(turn_target),
            "message_start",
            &serde_json::json!({
                "type": "message_start",
                "role": "assistant",
                "model": "mock-model",
            }),
        )
        .await?;
    store
        .insert_event(
            session_id,
            Some(turn_target),
            "message_delta",
            &serde_json::json!({
                "type": "message_delta",
                "content_delta": response,
            }),
        )
        .await?;
    store
        .insert_event(
            session_id,
            Some(turn_target),
            "message_end",
            &serde_json::json!({
                "type": "message_end",
                "role": "assistant",
                "input_tokens": 10,
                "output_tokens": 5,
            }),
        )
        .await?;
    store
        .insert_event(
            session_id,
            Some(turn_target),
            "turn_end",
            &serde_json::json!({
                "type": "turn_end",
                "identity": identity,
                "turn_index": turn_index,
                "task_id": task_id,
                "trace_id": trace_id,
                "task_turn_index": 0,
                "has_tool_calls": false,
            }),
        )
        .await?;
    store
        .insert_event(
            session_id,
            None,
            "task_complete",
            &serde_json::json!({
                "type": "task_complete",
                "identity": identity,
                "task_id": task_id,
                "trace_id": trace_id,
                "plan_id": null,
                "status": "success",
                "task_turn_count": 1,
                "execution": execution,
                "error": null,
            }),
        )
        .await?;
    Ok(())
}

fn first_text_content(content: &[InferenceContent]) -> Option<&str> {
    content.iter().find_map(|part| match part {
        InferenceContent::Text { text } => Some(text.as_str()),
        _ => None,
    })
}

async fn event_metrics_if_enabled(enabled: bool, state_db_path: &Path) -> Result<Option<EventMetrics>> {
    if enabled {
        persisted_event_metrics(state_db_path).await.map(Some)
    } else {
        Ok(None)
    }
}

async fn persisted_event_metrics(state_db_path: &Path) -> Result<EventMetrics> {
    if !state_db_path.exists() {
        return Ok(EventMetrics {
            count: 0,
            payload_bytes: 0,
        });
    }
    let Some(path) = state_db_path.to_str() else {
        return Ok(EventMetrics {
            count: 0,
            payload_bytes: 0,
        });
    };
    let db = turso::Builder::new_local(path)
        .experimental_index_method(true)
        .build()
        .await
        .with_context(|| format!("failed to open '{}' for event metrics", state_db_path.display()))?;
    let conn = db.connect()?;
    conn.execute("PRAGMA busy_timeout = 5000;", ()).await.ok();
    let mut rows = conn
        .query(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(payload)), 0) FROM events",
            (),
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(EventMetrics {
            count: 0,
            payload_bytes: 0,
        });
    };
    Ok(EventMetrics {
        count: row.get::<i64>(0)? as usize,
        payload_bytes: row.get::<i64>(1)? as usize,
    })
}

fn prepare_workspace(workspace_root: Option<PathBuf>) -> Result<(Option<TempDir>, PathBuf)> {
    if let Some(path) = workspace_root {
        fs::create_dir_all(&path)?;
        return Ok((None, path));
    }

    let temp = tempfile::tempdir().context("failed to create perf workspace")?;
    let path = temp.path().to_path_buf();
    Ok((Some(temp), path))
}

#[cfg(unix)]
struct StdoutSilencer {
    saved_stdout: Option<i32>,
}

#[cfg(unix)]
impl StdoutSilencer {
    fn new(enabled: bool) -> Result<Self> {
        if !enabled {
            return Ok(Self { saved_stdout: None });
        }

        io::stdout().flush().ok();
        let saved_stdout = unsafe { libc::dup(libc::STDOUT_FILENO) };
        if saved_stdout < 0 {
            return Err(io::Error::last_os_error()).context("failed to save stdout");
        }

        let dev_null = fs::OpenOptions::new()
            .write(true)
            .open("/dev/null")
            .context("failed to open /dev/null")?;
        let result = unsafe {
            libc::dup2(
                std::os::fd::AsRawFd::as_raw_fd(&dev_null),
                libc::STDOUT_FILENO,
            )
        };
        if result < 0 {
            let err = io::Error::last_os_error();
            unsafe {
                libc::close(saved_stdout);
            }
            return Err(err).context("failed to silence stdout");
        }

        Ok(Self {
            saved_stdout: Some(saved_stdout),
        })
    }
}

#[cfg(unix)]
impl Drop for StdoutSilencer {
    fn drop(&mut self) {
        if let Some(saved_stdout) = self.saved_stdout.take() {
            io::stdout().flush().ok();
            unsafe {
                libc::dup2(saved_stdout, libc::STDOUT_FILENO);
                libc::close(saved_stdout);
            }
        }
    }
}

#[cfg(not(unix))]
struct StdoutSilencer;

#[cfg(not(unix))]
impl StdoutSilencer {
    fn new(_enabled: bool) -> Result<Self> {
        Ok(Self)
    }
}

impl ChannelDaemonHarness {
    async fn start(
        workspace_root: PathBuf,
        state_db_path: &Path,
        mock_response: &str,
        agent_idle_timeout_seconds: u64,
    ) -> Result<Self> {
        let config_path = write_mock_runtime_config(
            &workspace_root,
            state_db_path,
            mock_response,
            agent_idle_timeout_seconds,
        )?;
        let endpoint = workspace_daemon_socket(&workspace_root);
        let serve_config_path = config_path.clone();
        let join =
            tokio::spawn(async move { turin::daemon::server::serve(&serve_config_path).await });

        let deadline = TokioInstant::now() + Duration::from_secs(10);
        let client = turin_daemon_client::DaemonClient::new(&endpoint);
        loop {
            if client.handshake().await.is_ok() {
                break;
            }
            if join.is_finished() {
                let result = join
                    .await
                    .context("daemon task join failed before endpoint bind")?;
                return Err(result
                    .err()
                    .unwrap_or_else(|| anyhow::anyhow!("daemon exited before endpoint bind")));
            }
            if TokioInstant::now() >= deadline {
                join.abort();
                anyhow::bail!(
                    "timed out waiting for daemon endpoint '{}'",
                    endpoint.display()
                );
            }
            sleep(Duration::from_millis(25)).await;
        }

        Ok(Self {
            endpoint,
            workspace_root,
            join,
        })
    }

    fn runner(&self) -> ChannelRunner {
        ChannelRunner::new(
            self.client(),
            RunnerConfig {
                channel_id: "mock".to_string(),
                state_path: channel_runtime_dir(&self.workspace_root, "mock").join("bindings.json"),
                access_state_path: channel_runtime_dir(&self.workspace_root, "mock")
                    .join("access.json"),
                idle_ttl: Some(Duration::from_secs(600)),
                access_policy: Default::default(),
                tools: Default::default(),
            },
        )
    }

    fn client(&self) -> turin_daemon_client::DaemonClient {
        turin_daemon_client::DaemonClient::new(&self.endpoint)
    }

    async fn live_session_count(&self) -> Result<usize> {
        live_session_count(&self.client()).await
    }

    async fn stop(self) -> Result<()> {
        let client = self.client();
        let _: serde_json::Value = client
            .request_ok(
                None,
                turin_daemon_protocol::DaemonRequest::DaemonStop(Default::default()),
            )
            .await?;
        let _ = timeout(Duration::from_secs(5), self.join)
            .await
            .context("timed out waiting for daemon to exit")??;
        Ok(())
    }
}

impl BlackboxDaemonHarness {
    async fn start(
        turin_binary: PathBuf,
        workspace_root: PathBuf,
        state_db_path: &Path,
        mock_response: &str,
        agent_idle_timeout_seconds: u64,
    ) -> Result<Self> {
        let config_path = write_mock_runtime_config(
            &workspace_root,
            state_db_path,
            mock_response,
            agent_idle_timeout_seconds,
        )?;
        let endpoint = workspace_daemon_socket(&workspace_root);
        let child = StdCommand::new(&turin_binary)
            .arg("--log-level")
            .arg("error")
            .arg("daemon")
            .arg("start")
            .arg("--config")
            .arg(&config_path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("failed to start '{}'", turin_binary.display()))?;

        let mut harness = Self {
            endpoint,
            workspace_root,
            child,
        };
        harness.wait_until_ready().await?;
        Ok(harness)
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }

    fn client(&self) -> turin_daemon_client::DaemonClient {
        turin_daemon_client::DaemonClient::new(&self.endpoint)
    }

    fn runner(&self) -> ChannelRunner {
        ChannelRunner::new(
            self.client(),
            RunnerConfig {
                channel_id: "mock".to_string(),
                state_path: channel_runtime_dir(&self.workspace_root, "mock").join("bindings.json"),
                access_state_path: channel_runtime_dir(&self.workspace_root, "mock")
                    .join("access.json"),
                idle_ttl: Some(Duration::from_secs(600)),
                access_policy: Default::default(),
                tools: Default::default(),
            },
        )
    }

    async fn live_session_count(&self) -> Result<usize> {
        live_session_count(&self.client()).await
    }

    async fn live_session_diagnostics(&self) -> Result<LiveSessionDiagnostics> {
        live_session_diagnostics(&self.client()).await
    }

    async fn wait_until_ready(&mut self) -> Result<()> {
        let deadline = TokioInstant::now() + Duration::from_secs(10);
        let client = self.client();
        loop {
            if client.handshake().await.is_ok() {
                return Ok(());
            }
            if let Some(status) = self.child.try_wait()? {
                anyhow::bail!("daemon child exited before endpoint bind: {status}");
            }
            if TokioInstant::now() >= deadline {
                let _ = self.child.kill();
                anyhow::bail!(
                    "timed out waiting for daemon endpoint '{}'",
                    self.endpoint.display()
                );
            }
            sleep(Duration::from_millis(25)).await;
        }
    }

    async fn stop(mut self) -> Result<()> {
        let _ = self
            .client()
            .request_ok::<serde_json::Value>(
                None,
                turin_daemon_protocol::DaemonRequest::DaemonStop(Default::default()),
            )
            .await;

        let deadline = TokioInstant::now() + Duration::from_secs(30);
        loop {
            if self.child.try_wait()?.is_some() {
                return Ok(());
            }
            if TokioInstant::now() >= deadline {
                self.child.kill().context("failed to kill daemon child")?;
                let _ = self.child.wait();
                anyhow::bail!("timed out waiting for daemon child to exit");
            }
            sleep(Duration::from_millis(25)).await;
        }
    }
}

impl Drop for BlackboxDaemonHarness {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

async fn live_session_count(client: &turin_daemon_client::DaemonClient) -> Result<usize> {
    Ok(live_session_diagnostics(client).await?.count)
}

async fn live_session_diagnostics(
    client: &turin_daemon_client::DaemonClient,
) -> Result<LiveSessionDiagnostics> {
    let response: LiveSessionsResponse = client
        .request_ok(
            None,
            turin_daemon_protocol::DaemonRequest::SessionListLive(Default::default()),
        )
        .await?;
    let mut diagnostics = LiveSessionDiagnostics {
        count: response.sessions.len(),
        ..LiveSessionDiagnostics::default()
    };
    let mut saw_history = false;
    let mut total_history_len = 0usize;
    let mut max_history_message_offset = 0usize;
    for session in response.sessions {
        if let Some(history) = session.history {
            saw_history = true;
            total_history_len = total_history_len.saturating_add(history.len);
            max_history_message_offset = max_history_message_offset.max(history.message_offset);
        }
    }
    if saw_history {
        diagnostics.total_history_len = Some(total_history_len);
        diagnostics.max_history_message_offset = Some(max_history_message_offset);
    }
    Ok(diagnostics)
}

async fn daemon_task_metrics(
    client: &turin_daemon_client::DaemonClient,
) -> Result<DaemonTaskMetrics> {
    let response: TaskListResponse = client
        .request_ok(
            None,
            turin_daemon_protocol::DaemonRequest::TaskList(Default::default()),
        )
        .await?;
    let mut metrics = DaemonTaskMetrics {
        tasks: response.tasks.len(),
        ..DaemonTaskMetrics::default()
    };
    for task in response.tasks {
        metrics.snapshot_bytes = metrics
            .snapshot_bytes
            .saturating_add(serde_json::to_vec(&task).map_or(0, |json| json.len()));
        if task.state == "completed" {
            metrics.completed_tasks = metrics.completed_tasks.saturating_add(1);
        }
        metrics.output_bytes = metrics
            .output_bytes
            .saturating_add(task.output.as_deref().map_or(0, str::len));
        if let Some(content) = task.assistant_content.as_deref() {
            metrics.assistant_content_bytes = metrics
                .assistant_content_bytes
                .saturating_add(serde_json::to_string(content).map_or(0, |json| json.len()));
        }
    }
    Ok(metrics)
}

impl MockChannelDriver {
    fn new(events: Vec<InboundEvent>) -> Self {
        Self {
            events: events.into(),
            sent: Arc::new(Mutex::new(Vec::new())),
            scale_recorder: None,
        }
    }

    fn with_scale_recorder(events: Vec<InboundEvent>, scale_recorder: ScaleRecorder) -> Self {
        Self {
            events: events.into(),
            sent: Arc::new(Mutex::new(Vec::new())),
            scale_recorder: Some(scale_recorder),
        }
    }
}

#[async_trait]
impl ChannelDriver for MockChannelDriver {
    fn kind(&self) -> ChannelKind {
        ChannelKind::new("mock")
    }

    fn user_matches_selector(&self, selector: &str, user: &ChannelUser) -> bool {
        let selector = selector.trim();
        !selector.is_empty()
            && (user.id == selector
                || user
                    .username
                    .as_ref()
                    .is_some_and(|username| username.eq_ignore_ascii_case(selector)))
    }

    async fn next_event(&mut self) -> Result<Option<InboundEvent>> {
        Ok(self.events.pop_front())
    }

    async fn send(
        &mut self,
        _conversation: &ChannelConversationKey,
        message: OutboundMessage,
    ) -> Result<()> {
        let outbound_count = {
            let mut sent = self.sent.lock().expect("sent lock poisoned");
            sent.push(message);
            sent.len()
        };
        if let Some(recorder) = &self.scale_recorder {
            recorder.record_if_checkpoint(outbound_count).await?;
        }
        Ok(())
    }

    async fn send_progress(
        &mut self,
        _event: &InboundEvent,
        _update: ChannelProgressUpdate,
    ) -> Result<()> {
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

fn build_config(
    workspace_root: &Path,
    harness_dir: &Path,
    state_db_path: &Path,
    requested_turns: usize,
    hot_history: Option<HotHistoryConfig>,
    idle_timeout_seconds: Option<u64>,
) -> Result<TurinConfig> {
    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: None,
            ..ProviderConfig::default()
        },
    );

    let mut inference = InferenceConfig::default();
    if let Some(hot_history) = hot_history {
        inference.hot_history = hot_history;
    }

    Ok(TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Perf scenario. Use the requested tool and then finish.".to_string(),
            thinking: None,
            harness: None,
            idle_timeout_seconds,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: HashMap::new(),
        kernel: KernelConfig {
            workspace_root: workspace_root.to_string_lossy().to_string(),
            max_turns: requested_turns
                .checked_mul(4)
                .and_then(|value| value.try_into().ok())
                .unwrap_or(u32::MAX),
            heartbeat_interval_seconds: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference,
        persistence: PersistenceConfig::with_state_path(
            state_db_path.to_string_lossy().to_string(),
        ),
        harness: HarnessConfig {
            directory: harness_dir.to_string_lossy().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 64,
        },
        harnesses: HashMap::new(),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    })
}

fn build_responses(
    turns: usize,
    tool_every: usize,
    response_bytes: usize,
) -> VecDeque<Vec<InferenceEvent>> {
    let mut responses =
        VecDeque::with_capacity(turns.saturating_add(tool_call_count(turns, tool_every)));
    for index in 0..turns {
        if should_call_tool(index, tool_every) {
            responses.push_back(vec![
                message_start(),
                InferenceEvent::ToolCallStart {
                    id: format!("call_{index}"),
                    name: "read_file".to_string(),
                },
                InferenceEvent::ToolCallDelta {
                    delta: serde_json::json!({
                        "path": format!("payloads/payload-{index}.txt")
                    })
                    .to_string(),
                },
                InferenceEvent::MessageEnd {
                    input_tokens: 10,
                    output_tokens: 5,
                    stop_reason: None,
                },
            ]);
        }
        responses.push_back(final_events(index, response_bytes));
    }
    responses
}

fn final_events(index: usize, response_bytes: usize) -> Vec<InferenceEvent> {
    vec![
        message_start(),
        InferenceEvent::MessageDelta {
            content: synthetic_text(&format!("Recorded payload {index}."), response_bytes),
        },
        InferenceEvent::MessageEnd {
            input_tokens: 5,
            output_tokens: 2,
            stop_reason: None,
        },
    ]
}

fn message_start() -> InferenceEvent {
    InferenceEvent::MessageStart {
        role: "assistant".to_string(),
        model: "mock-model".to_string(),
        provider_id: "mock".to_string(),
    }
}

fn should_call_tool(index: usize, tool_every: usize) -> bool {
    tool_every > 0 && index % tool_every == 0
}

fn tool_call_count(turns: usize, tool_every: usize) -> usize {
    tool_payload_indices(turns, tool_every).count()
}

fn tool_payload_indices(turns: usize, tool_every: usize) -> impl Iterator<Item = usize> {
    (0..turns).filter(move |index| should_call_tool(*index, tool_every))
}

fn synthetic_payload(index: usize, bytes: usize) -> Vec<u8> {
    let prefix = format!("payload {index}\n");
    let mut payload = Vec::with_capacity(bytes.max(prefix.len()));
    while payload.len() < bytes {
        payload.extend_from_slice(prefix.as_bytes());
    }
    payload.truncate(bytes);
    payload
}

fn snapshot(
    label: &str,
    start: Instant,
    state_db_path: &Path,
    turn_index: Option<u32>,
    history_len: Option<usize>,
    outbound_messages: Option<usize>,
    active_sessions: Option<usize>,
    live_sessions: Option<usize>,
    messages_per_session: Option<usize>,
) -> Snapshot {
    let memory = read_process_memory();
    let state_store_size = state_store_size(state_db_path);
    Snapshot {
        label: label.to_string(),
        elapsed_ms: start.elapsed().as_millis(),
        rss_kb: memory.rss_kb,
        pss_kb: memory.pss_kb,
        pss_anon_kb: memory.pss_anon_kb,
        pss_file_kb: memory.pss_file_kb,
        pss_shmem_kb: memory.pss_shmem_kb,
        state_db_main_bytes: state_store_size.main_bytes,
        state_db_wal_bytes: state_store_size.wal_bytes,
        state_db_shm_bytes: state_store_size.shm_bytes,
        state_db_bytes: state_store_size.total_bytes(),
        turn_index,
        history_len,
        outbound_messages,
        active_sessions,
        live_sessions,
        messages_per_session,
        persisted_messages: None,
        history_message_offset: None,
        hot_window_pruned: None,
        history_payload_bytes: None,
        tool_results: None,
        tool_result_errors: None,
        persisted_events: None,
        persisted_event_payload_bytes: None,
        daemon_tasks: None,
        daemon_completed_tasks: None,
        daemon_task_snapshot_bytes: None,
        daemon_task_output_bytes: None,
        daemon_task_assistant_content_bytes: None,
    }
}

impl Snapshot {
    fn set_daemon_task_metrics(&mut self, metrics: DaemonTaskMetrics) {
        self.daemon_tasks = Some(metrics.tasks);
        self.daemon_completed_tasks = Some(metrics.completed_tasks);
        self.daemon_task_snapshot_bytes = Some(metrics.snapshot_bytes);
        self.daemon_task_output_bytes = Some(metrics.output_bytes);
        self.daemon_task_assistant_content_bytes = Some(metrics.assistant_content_bytes);
    }

    fn set_live_session_diagnostics(&mut self, diagnostics: LiveSessionDiagnostics) {
        self.live_sessions = Some(diagnostics.count);
        self.history_len = diagnostics.total_history_len;
        self.history_message_offset = diagnostics.max_history_message_offset;
        self.hot_window_pruned = diagnostics
            .max_history_message_offset
            .map(|offset| offset > 0);
    }
}

fn idle_runtime_snapshot(
    label: &str,
    start: Instant,
    state_db_path: &Path,
    live_sessions: usize,
) -> Snapshot {
    snapshot(
        label,
        start,
        state_db_path,
        None,
        None,
        None,
        None,
        Some(live_sessions),
        None,
    )
}

async fn hot_history_snapshot(
    label: &str,
    start: Instant,
    state_db_path: &Path,
    session: &turin::kernel::session::SessionState,
) -> Result<Snapshot> {
    let mut snapshot = snapshot(
        label,
        start,
        state_db_path,
        Some(session.turn_index),
        Some(session.history.len()),
        None,
        None,
        None,
        None,
    );
    let metrics = history_metrics(&session.history);
    snapshot.history_payload_bytes = Some(metrics.payload_bytes);
    snapshot.tool_results = Some(metrics.tool_results);
    snapshot.tool_result_errors = Some(metrics.tool_result_errors);
    snapshot.history_message_offset = Some(session.history_message_offset);
    snapshot.hot_window_pruned = Some(session.history_is_pruned());
    snapshot.persisted_messages =
        persisted_message_count(state_db_path, session.internal_id).await?;
    Ok(snapshot)
}

async fn persisted_message_count(
    state_db_path: &Path,
    session_internal_id: Option<i64>,
) -> Result<Option<usize>> {
    let Some(session_internal_id) = session_internal_id else {
        return Ok(None);
    };
    let Some(path) = state_db_path.to_str() else {
        return Ok(None);
    };
    let store = StateStore::open(path).await?;
    let messages = store
        .get_messages(session_internal_id, &SessionReadTarget::ActiveBranch)
        .await?;
    Ok(Some(messages.len()))
}

async fn channel_scale_snapshot(
    label: &str,
    start: Instant,
    state_db_path: &Path,
    outbound_messages: Option<usize>,
    active_sessions: Option<usize>,
    messages_per_session: Option<usize>,
    live_sessions: Option<usize>,
    memory_target: MemoryTarget,
) -> Result<Snapshot> {
    let mut snapshot = snapshot(
        label,
        start,
        state_db_path,
        None,
        None,
        outbound_messages,
        active_sessions,
        live_sessions,
        messages_per_session,
    );
    let memory = read_process_memory_for_target(memory_target);
    snapshot.rss_kb = memory.rss_kb;
    snapshot.pss_kb = memory.pss_kb;
    snapshot.pss_anon_kb = memory.pss_anon_kb;
    snapshot.pss_file_kb = memory.pss_file_kb;
    snapshot.pss_shmem_kb = memory.pss_shmem_kb;
    snapshot.persisted_messages = match memory_target {
        MemoryTarget::CurrentProcess => {
            persisted_message_count_for_all_sessions(state_db_path).await?
        }
        MemoryTarget::Pid(_) => persisted_message_count_for_all_sessions(state_db_path)
            .await
            .ok()
            .flatten(),
    };
    Ok(snapshot)
}

async fn persisted_message_count_for_all_sessions(state_db_path: &Path) -> Result<Option<usize>> {
    if !state_db_path.exists() {
        return Ok(Some(0));
    }
    let Some(path) = state_db_path.to_str() else {
        return Ok(None);
    };

    let store = StateStore::open(path).await?;
    let sessions = store.list_session_rows(usize::MAX, 0).await?;
    let mut total = 0usize;
    for session in sessions {
        let messages = store
            .get_messages(session.id, &SessionReadTarget::ActiveBranch)
            .await?;
        total = total.saturating_add(messages.len());
    }
    Ok(Some(total))
}

async fn checkpoint_state_db(state_db_path: &Path) -> Result<()> {
    if !state_db_path.exists() {
        return Ok(());
    }
    let Some(path) = state_db_path.to_str() else {
        return Ok(());
    };

    let db = turso::Builder::new_local(path)
        .experimental_index_method(true)
        .build()
        .await
        .with_context(|| {
            format!(
                "failed to open '{}' for WAL checkpoint",
                state_db_path.display()
            )
        })?;
    let conn = db.connect()?;
    conn.execute("PRAGMA busy_timeout = 5000;", ()).await.ok();
    let mut rows = conn
        .query("PRAGMA wal_checkpoint(TRUNCATE);", ())
        .await
        .with_context(|| format!("failed to checkpoint '{}'", state_db_path.display()))?;
    while rows.next().await?.is_some() {}
    Ok(())
}

#[derive(Debug, Default)]
struct HistoryMetrics {
    payload_bytes: usize,
    tool_results: usize,
    tool_result_errors: usize,
}

fn history_metrics(messages: &[InferenceMessage]) -> HistoryMetrics {
    let mut metrics = HistoryMetrics::default();
    for message in messages {
        for content in &message.content {
            match content {
                InferenceContent::Text { text } => {
                    metrics.payload_bytes = metrics.payload_bytes.saturating_add(text.len());
                }
                InferenceContent::Image {
                    name,
                    content_type,
                    url,
                    local_path,
                    ..
                }
                | InferenceContent::File {
                    name,
                    content_type,
                    url,
                    local_path,
                    ..
                } => {
                    metrics.payload_bytes = metrics
                        .payload_bytes
                        .saturating_add(optional_len(name.as_deref()))
                        .saturating_add(optional_len(content_type.as_deref()))
                        .saturating_add(optional_len(url.as_deref()))
                        .saturating_add(optional_len(local_path.as_deref()));
                }
                InferenceContent::ToolUse { id, name, input } => {
                    metrics.payload_bytes = metrics
                        .payload_bytes
                        .saturating_add(id.len())
                        .saturating_add(name.len())
                        .saturating_add(serde_json::to_string(input).map_or(0, |json| json.len()));
                }
                InferenceContent::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    metrics.tool_results = metrics.tool_results.saturating_add(1);
                    if *is_error {
                        metrics.tool_result_errors = metrics.tool_result_errors.saturating_add(1);
                    }
                    metrics.payload_bytes = metrics
                        .payload_bytes
                        .saturating_add(tool_use_id.len())
                        .saturating_add(content.len());
                }
                InferenceContent::Thinking { content, signature } => {
                    metrics.payload_bytes = metrics
                        .payload_bytes
                        .saturating_add(content.len())
                        .saturating_add(optional_len(signature.as_deref()));
                }
            }
        }
    }
    metrics
}

fn optional_len(value: Option<&str>) -> usize {
    value.map_or(0, str::len)
}

fn parse_source_roots(raw: &str) -> Result<Vec<String>> {
    let roots = raw
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| part.trim_matches('/').to_string())
        .collect::<Vec<_>>();
    anyhow::ensure!(!roots.is_empty(), "--roots must include at least one path");
    Ok(roots)
}

fn collect_rust_footprint(
    repo_root: &Path,
    root: &str,
    root_path: &Path,
    area_counts: &mut BTreeMap<String, LineCounts>,
    file_counts: &mut Vec<FileFootprint>,
) -> Result<()> {
    let mut pending = vec![root_path.to_path_buf()];
    while let Some(path) = pending.pop() {
        for entry in fs::read_dir(&path)
            .with_context(|| format!("failed to read directory '{}'", path.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                if should_skip_source_dir(&path) {
                    continue;
                }
                pending.push(path);
            } else if file_type.is_file() && is_counted_rust_file(&path) {
                let counts = count_rust_lines(&path)?;
                let relative = relative_path(repo_root, &path);
                area_counts
                    .entry(source_area(root, &relative))
                    .or_default()
                    .add(&counts);
                file_counts.push(FileFootprint {
                    path: relative,
                    counts,
                });
            }
        }
    }
    Ok(())
}

fn should_skip_source_dir(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    matches!(
        name,
        ".git" | ".workspace" | "target" | "tests" | "benches" | "examples"
    )
}

fn is_counted_rust_file(path: &Path) -> bool {
    if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
        return false;
    }
    !matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some("tests.rs" | "test.rs")
    )
}

fn count_rust_lines(path: &Path) -> Result<LineCounts> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read source file '{}'", path.display()))?;
    let mut counts = LineCounts {
        files: 1,
        ..LineCounts::default()
    };

    for line in raw.lines() {
        counts.total_lines += 1;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            counts.blank_lines += 1;
        } else if is_comment_line(trimmed) {
            counts.comment_lines += 1;
        } else {
            counts.code_lines += 1;
        }
    }

    Ok(counts)
}

fn is_comment_line(trimmed: &str) -> bool {
    trimmed.starts_with("//")
        || trimmed.starts_with("/*")
        || trimmed.starts_with('*')
        || trimmed.starts_with("*/")
}

fn source_area(root: &str, relative: &str) -> String {
    if root == "crates" {
        let mut parts = relative.split('/');
        if matches!(parts.next(), Some("crates"))
            && let Some(crate_name) = parts.next()
        {
            return format!("crates/{crate_name}");
        }
    }
    root.to_string()
}

fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn collect_binary_footprint(repo_root: &Path, extra_binaries: &[PathBuf]) -> Vec<BinaryFootprint> {
    let mut candidates = default_binary_candidates(repo_root);
    candidates.extend(extra_binaries.iter().map(|path| {
        if path.is_absolute() {
            path.clone()
        } else {
            repo_root.join(path)
        }
    }));

    let mut seen = HashSet::new();
    let mut binaries = Vec::new();
    for candidate in candidates {
        let path_key = candidate.display().to_string();
        if !seen.insert(path_key.clone()) {
            continue;
        }
        let bytes = file_len(&candidate);
        if bytes == 0 {
            continue;
        }
        binaries.push(BinaryFootprint {
            path: relative_path(repo_root, &candidate),
            bytes,
        });
    }
    binaries.sort_by(|left, right| left.path.cmp(&right.path));
    binaries
}

fn default_binary_candidates(repo_root: &Path) -> Vec<PathBuf> {
    [
        "target/release/turin",
        "target/release/turin-manager",
        "target/release/turin-tui",
        "target/release/turin-map",
        "target/release/turin-channel-telegram",
        "target/release/turin-channel-rocketchat",
        "target/release/turin-channel-whatsapp",
        "target/release/turin-channel-discord",
    ]
    .into_iter()
    .map(|path| repo_root.join(path))
    .collect()
}

impl LineCounts {
    fn add(&mut self, other: &Self) {
        self.files = self.files.saturating_add(other.files);
        self.total_lines = self.total_lines.saturating_add(other.total_lines);
        self.blank_lines = self.blank_lines.saturating_add(other.blank_lines);
        self.comment_lines = self.comment_lines.saturating_add(other.comment_lines);
        self.code_lines = self.code_lines.saturating_add(other.code_lines);
    }
}

impl ScaleRecorder {
    async fn record_if_checkpoint(&self, outbound_count: usize) -> Result<()> {
        if !self.sample_totals.contains(&outbound_count) {
            return Ok(());
        }

        let messages_per_session = outbound_count / self.active_sessions;
        let snapshot = channel_scale_snapshot(
            &format!(
                "after-{}-sessions-x{}-messages",
                self.active_sessions, messages_per_session
            ),
            self.start,
            &self.state_db_path,
            Some(outbound_count),
            Some(self.active_sessions),
            Some(messages_per_session),
            Some(live_session_count(&self.daemon).await?),
            self.memory_target,
        )
        .await?;
        self.snapshots
            .lock()
            .expect("scale snapshots lock poisoned")
            .push(snapshot);
        Ok(())
    }
}

fn read_process_memory() -> ProcessMemory {
    read_process_memory_for_target(MemoryTarget::CurrentProcess)
}

fn read_process_memory_for_target(target: MemoryTarget) -> ProcessMemory {
    match target {
        MemoryTarget::CurrentProcess => read_process_memory_from_proc(Path::new("/proc/self")),
        MemoryTarget::Pid(pid) => {
            read_process_memory_from_proc(&PathBuf::from(format!("/proc/{pid}")))
        }
    }
}

fn read_process_memory_from_proc(proc_path: &Path) -> ProcessMemory {
    let mut rss_kb = None;
    let mut pss_kb = None;
    let mut pss_anon_kb = None;
    let mut pss_file_kb = None;
    let mut pss_shmem_kb = None;

    if let Ok(raw) = fs::read_to_string(proc_path.join("smaps_rollup")) {
        for line in raw.lines() {
            rss_kb = rss_kb.or_else(|| parse_kb_line(line, "Rss:"));
            pss_kb = pss_kb.or_else(|| parse_kb_line(line, "Pss:"));
            pss_anon_kb = pss_anon_kb.or_else(|| parse_kb_line(line, "Pss_Anon:"));
            pss_file_kb = pss_file_kb.or_else(|| parse_kb_line(line, "Pss_File:"));
            pss_shmem_kb = pss_shmem_kb.or_else(|| parse_kb_line(line, "Pss_Shmem:"));
        }
    }

    if rss_kb.is_none() {
        if let Ok(raw) = fs::read_to_string(proc_path.join("status")) {
            for line in raw.lines() {
                rss_kb = rss_kb.or_else(|| parse_kb_line(line, "VmRSS:"));
            }
        }
    }

    ProcessMemory {
        rss_kb,
        pss_kb,
        pss_anon_kb,
        pss_file_kb,
        pss_shmem_kb,
    }
}

fn parse_kb_line(line: &str, key: &str) -> Option<u64> {
    line.strip_prefix(key)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

fn state_store_size(state_db_path: &Path) -> StateStoreSize {
    StateStoreSize {
        main_bytes: file_len(state_db_path),
        wal_bytes: sibling_len(state_db_path, "-wal"),
        shm_bytes: sibling_len(state_db_path, "-shm"),
    }
}

fn sibling_len(path: &Path, suffix: &str) -> u64 {
    let sibling = PathBuf::from(format!("{}{}", path.display(), suffix));
    file_len(&sibling)
}

fn file_len(path: &Path) -> u64 {
    path.metadata().map(|metadata| metadata.len()).unwrap_or(0)
}

fn allocator_trim_supported() -> bool {
    cfg!(all(target_os = "linux", target_env = "gnu"))
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn trim_allocator() -> bool {
    unsafe { libc::malloc_trim(0) != 0 }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn trim_allocator() -> bool {
    false
}

fn write_reports(report_dir: &Path, report: &PerfReport) -> Result<()> {
    fs::create_dir_all(report_dir)?;
    let stamp = report_stamp()?;
    let json_path = report_dir.join(format!("{}-{stamp}.json", report.scenario));
    let md_path = report_dir.join(format!("{}-{stamp}.md", report.scenario));

    fs::write(&json_path, serde_json::to_vec_pretty(report)?)?;
    fs::write(&md_path, markdown_report(report))?;

    println!("json_report={}", json_path.display());
    println!("markdown_report={}", md_path.display());
    Ok(())
}

fn write_footprint_reports(report_dir: &Path, report: &FootprintReport) -> Result<()> {
    fs::create_dir_all(report_dir)?;
    let stamp = report_stamp()?;
    let json_path = report_dir.join(format!("{}-{stamp}.json", report.scenario));
    let md_path = report_dir.join(format!("{}-{stamp}.md", report.scenario));

    fs::write(&json_path, serde_json::to_vec_pretty(report)?)?;
    fs::write(&md_path, footprint_markdown_report(report))?;

    println!("json_report={}", json_path.display());
    println!("markdown_report={}", md_path.display());
    Ok(())
}

fn report_stamp() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before unix epoch")?
        .as_secs())
}

fn markdown_report(report: &PerfReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("# Perf Report: {}\n\n", report.scenario));
    out.push_str(&format!("- config: `{}`\n", report.config));
    out.push_str(&format!("- workspace_root: `{}`\n", report.workspace_root));
    out.push_str(&format!("- state_db_path: `{}`\n\n", report.state_db_path));
    push_snapshot_summary(&mut out, &report.snapshots);
    out.push_str(
        "| label | elapsed_ms | rss_kb | pss_kb | pss_anon_kb | pss_file_kb | pss_shmem_kb | state_db_main_bytes | state_db_wal_bytes | state_db_shm_bytes | state_db_bytes | turn_index | persisted_messages | persisted_events | persisted_event_payload_bytes | history_len | history_message_offset | hot_window_pruned | history_payload_bytes | tool_results | tool_result_errors | outbound_messages | active_sessions | live_sessions | messages_per_session | daemon_tasks | daemon_completed_tasks | daemon_task_snapshot_bytes | daemon_task_output_bytes | daemon_task_assistant_content_bytes |\n",
    );
    out.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for snapshot in &report.snapshots {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            snapshot.label,
            snapshot.elapsed_ms,
            display_option(snapshot.rss_kb),
            display_option(snapshot.pss_kb),
            display_option(snapshot.pss_anon_kb),
            display_option(snapshot.pss_file_kb),
            display_option(snapshot.pss_shmem_kb),
            snapshot.state_db_main_bytes,
            snapshot.state_db_wal_bytes,
            snapshot.state_db_shm_bytes,
            snapshot.state_db_bytes,
            display_u32_option(snapshot.turn_index),
            display_usize_option(snapshot.persisted_messages),
            display_usize_option(snapshot.persisted_events),
            display_usize_option(snapshot.persisted_event_payload_bytes),
            display_usize_option(snapshot.history_len),
            display_usize_option(snapshot.history_message_offset),
            display_bool_option(snapshot.hot_window_pruned),
            display_usize_option(snapshot.history_payload_bytes),
            display_usize_option(snapshot.tool_results),
            display_usize_option(snapshot.tool_result_errors),
            display_usize_option(snapshot.outbound_messages),
            display_usize_option(snapshot.active_sessions),
            display_usize_option(snapshot.live_sessions),
            display_usize_option(snapshot.messages_per_session),
            display_usize_option(snapshot.daemon_tasks),
            display_usize_option(snapshot.daemon_completed_tasks),
            display_usize_option(snapshot.daemon_task_snapshot_bytes),
            display_usize_option(snapshot.daemon_task_output_bytes),
            display_usize_option(snapshot.daemon_task_assistant_content_bytes)
        ));
    }
    out
}

fn push_snapshot_summary(out: &mut String, snapshots: &[Snapshot]) {
    let Some((first, last)) = snapshots.first().zip(snapshots.last()) else {
        return;
    };

    out.push_str("## Summary\n\n");
    out.push_str("| metric | first | last | delta | peak |\n");
    out.push_str("|---|---:|---:|---:|---:|\n");
    push_summary_row(
        out,
        "elapsed_ms",
        first.elapsed_ms.to_string(),
        last.elapsed_ms.to_string(),
        display_i128_delta(first.elapsed_ms, last.elapsed_ms),
        display_u128_option(snapshots.iter().map(|snapshot| snapshot.elapsed_ms).max()),
    );
    push_summary_row(
        out,
        "rss_kb",
        display_option(first.rss_kb),
        display_option(last.rss_kb),
        display_optional_i128_delta(first.rss_kb, last.rss_kb),
        display_option(
            snapshots
                .iter()
                .filter_map(|snapshot| snapshot.rss_kb)
                .max(),
        ),
    );
    push_summary_row(
        out,
        "pss_kb",
        display_option(first.pss_kb),
        display_option(last.pss_kb),
        display_optional_i128_delta(first.pss_kb, last.pss_kb),
        display_option(
            snapshots
                .iter()
                .filter_map(|snapshot| snapshot.pss_kb)
                .max(),
        ),
    );
    push_summary_row(
        out,
        "pss_anon_kb",
        display_option(first.pss_anon_kb),
        display_option(last.pss_anon_kb),
        display_optional_i128_delta(first.pss_anon_kb, last.pss_anon_kb),
        display_option(
            snapshots
                .iter()
                .filter_map(|snapshot| snapshot.pss_anon_kb)
                .max(),
        ),
    );
    push_summary_row(
        out,
        "pss_file_kb",
        display_option(first.pss_file_kb),
        display_option(last.pss_file_kb),
        display_optional_i128_delta(first.pss_file_kb, last.pss_file_kb),
        display_option(
            snapshots
                .iter()
                .filter_map(|snapshot| snapshot.pss_file_kb)
                .max(),
        ),
    );
    push_summary_row(
        out,
        "pss_shmem_kb",
        display_option(first.pss_shmem_kb),
        display_option(last.pss_shmem_kb),
        display_optional_i128_delta(first.pss_shmem_kb, last.pss_shmem_kb),
        display_option(
            snapshots
                .iter()
                .filter_map(|snapshot| snapshot.pss_shmem_kb)
                .max(),
        ),
    );
    push_summary_row(
        out,
        "state_db_bytes",
        first.state_db_bytes.to_string(),
        last.state_db_bytes.to_string(),
        display_optional_i128_delta(Some(first.state_db_bytes), Some(last.state_db_bytes)),
        display_option(
            snapshots
                .iter()
                .map(|snapshot| snapshot.state_db_bytes)
                .max(),
        ),
    );
    push_summary_row(
        out,
        "persisted_messages",
        display_usize_option(first.persisted_messages),
        display_usize_option(last.persisted_messages),
        display_optional_isize_delta(first.persisted_messages, last.persisted_messages),
        display_usize_option(
            snapshots
                .iter()
                .filter_map(|snapshot| snapshot.persisted_messages)
                .max(),
        ),
    );
    push_summary_row(
        out,
        "persisted_events",
        display_usize_option(first.persisted_events),
        display_usize_option(last.persisted_events),
        display_optional_isize_delta(first.persisted_events, last.persisted_events),
        display_usize_option(
            snapshots
                .iter()
                .filter_map(|snapshot| snapshot.persisted_events)
                .max(),
        ),
    );
    push_summary_row(
        out,
        "persisted_event_payload_bytes",
        display_usize_option(first.persisted_event_payload_bytes),
        display_usize_option(last.persisted_event_payload_bytes),
        display_optional_isize_delta(
            first.persisted_event_payload_bytes,
            last.persisted_event_payload_bytes,
        ),
        display_usize_option(
            snapshots
                .iter()
                .filter_map(|snapshot| snapshot.persisted_event_payload_bytes)
                .max(),
        ),
    );
    push_summary_row(
        out,
        "history_len",
        display_usize_option(first.history_len),
        display_usize_option(last.history_len),
        display_optional_isize_delta(first.history_len, last.history_len),
        display_usize_option(
            snapshots
                .iter()
                .filter_map(|snapshot| snapshot.history_len)
                .max(),
        ),
    );
    push_summary_row(
        out,
        "history_payload_bytes",
        display_usize_option(first.history_payload_bytes),
        display_usize_option(last.history_payload_bytes),
        display_optional_isize_delta(first.history_payload_bytes, last.history_payload_bytes),
        display_usize_option(
            snapshots
                .iter()
                .filter_map(|snapshot| snapshot.history_payload_bytes)
                .max(),
        ),
    );
    push_summary_row(
        out,
        "outbound_messages",
        display_usize_option(first.outbound_messages),
        display_usize_option(last.outbound_messages),
        display_optional_isize_delta(first.outbound_messages, last.outbound_messages),
        display_usize_option(
            snapshots
                .iter()
                .filter_map(|snapshot| snapshot.outbound_messages)
                .max(),
        ),
    );
    push_summary_row(
        out,
        "active_sessions",
        display_usize_option(first.active_sessions),
        display_usize_option(last.active_sessions),
        display_optional_isize_delta(first.active_sessions, last.active_sessions),
        display_usize_option(
            snapshots
                .iter()
                .filter_map(|snapshot| snapshot.active_sessions)
                .max(),
        ),
    );
    push_summary_row(
        out,
        "live_sessions",
        display_usize_option(first.live_sessions),
        display_usize_option(last.live_sessions),
        display_optional_isize_delta(first.live_sessions, last.live_sessions),
        display_usize_option(
            snapshots
                .iter()
                .filter_map(|snapshot| snapshot.live_sessions)
                .max(),
        ),
    );
    push_summary_row(
        out,
        "daemon_tasks",
        display_usize_option(first.daemon_tasks),
        display_usize_option(last.daemon_tasks),
        display_optional_isize_delta(first.daemon_tasks, last.daemon_tasks),
        display_usize_option(
            snapshots
                .iter()
                .filter_map(|snapshot| snapshot.daemon_tasks)
                .max(),
        ),
    );
    push_summary_row(
        out,
        "daemon_task_output_bytes",
        display_usize_option(first.daemon_task_output_bytes),
        display_usize_option(last.daemon_task_output_bytes),
        display_optional_isize_delta(
            first.daemon_task_output_bytes,
            last.daemon_task_output_bytes,
        ),
        display_usize_option(
            snapshots
                .iter()
                .filter_map(|snapshot| snapshot.daemon_task_output_bytes)
                .max(),
        ),
    );
    push_summary_row(
        out,
        "daemon_task_snapshot_bytes",
        display_usize_option(first.daemon_task_snapshot_bytes),
        display_usize_option(last.daemon_task_snapshot_bytes),
        display_optional_isize_delta(
            first.daemon_task_snapshot_bytes,
            last.daemon_task_snapshot_bytes,
        ),
        display_usize_option(
            snapshots
                .iter()
                .filter_map(|snapshot| snapshot.daemon_task_snapshot_bytes)
                .max(),
        ),
    );
    push_summary_row(
        out,
        "daemon_task_assistant_content_bytes",
        display_usize_option(first.daemon_task_assistant_content_bytes),
        display_usize_option(last.daemon_task_assistant_content_bytes),
        display_optional_isize_delta(
            first.daemon_task_assistant_content_bytes,
            last.daemon_task_assistant_content_bytes,
        ),
        display_usize_option(
            snapshots
                .iter()
                .filter_map(|snapshot| snapshot.daemon_task_assistant_content_bytes)
                .max(),
        ),
    );
    out.push('\n');
}

fn push_summary_row(
    out: &mut String,
    label: &str,
    first: String,
    last: String,
    delta: String,
    peak: String,
) {
    out.push_str(&format!(
        "| {label} | {first} | {last} | {delta} | {peak} |\n"
    ));
}

fn footprint_markdown_report(report: &FootprintReport) -> String {
    let mut out = String::new();
    out.push_str("# Footprint Report\n\n");
    out.push_str(&format!("- config: `{}`\n\n", report.config));
    out.push_str("## Totals\n\n");
    out.push_str("| files | total_lines | blank_lines | comment_lines | code_lines |\n");
    out.push_str("|---:|---:|---:|---:|---:|\n");
    out.push_str(&format!("| {} |\n", line_counts_cells(&report.totals)));

    out.push_str("\n## Areas\n\n");
    out.push_str("| area | files | total_lines | blank_lines | comment_lines | code_lines |\n");
    out.push_str("|---|---:|---:|---:|---:|---:|\n");
    for area in &report.areas {
        out.push_str(&format!(
            "| {} | {} |\n",
            area.area,
            line_counts_cells(&area.counts)
        ));
    }

    out.push_str("\n## Largest Files\n\n");
    out.push_str("| path | files | total_lines | blank_lines | comment_lines | code_lines |\n");
    out.push_str("|---|---:|---:|---:|---:|---:|\n");
    for file in &report.largest_files {
        out.push_str(&format!(
            "| {} | {} |\n",
            file.path,
            line_counts_cells(&file.counts)
        ));
    }

    if !report.binaries.is_empty() {
        out.push_str("\n## Binaries\n\n");
        out.push_str("| path | bytes |\n");
        out.push_str("|---|---:|\n");
        for binary in &report.binaries {
            out.push_str(&format!("| {} | {} |\n", binary.path, binary.bytes));
        }
    }

    out
}

fn line_counts_cells(counts: &LineCounts) -> String {
    format!(
        "{} | {} | {} | {} | {}",
        counts.files,
        counts.total_lines,
        counts.blank_lines,
        counts.comment_lines,
        counts.code_lines
    )
}

fn display_option(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn display_u128_option(value: Option<u128>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn display_i128_delta(first: u128, last: u128) -> String {
    signed_delta(last as i128 - first as i128)
}

fn display_optional_i128_delta(first: Option<u64>, last: Option<u64>) -> String {
    match (first, last) {
        (Some(first), Some(last)) => signed_delta(last as i128 - first as i128),
        _ => "-".to_string(),
    }
}

fn display_optional_isize_delta(first: Option<usize>, last: Option<usize>) -> String {
    match (first, last) {
        (Some(first), Some(last)) => signed_delta(last as i128 - first as i128),
        _ => "-".to_string(),
    }
}

fn signed_delta(delta: i128) -> String {
    if delta >= 0 {
        format!("+{delta}")
    } else {
        delta.to_string()
    }
}

fn display_usize_option(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn display_u32_option(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn display_bool_option(value: Option<bool>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn workspace_turin_root(workspace_root: &Path) -> PathBuf {
    workspace_root.join(".turin")
}

fn workspace_config_path(workspace_root: &Path) -> PathBuf {
    workspace_turin_root(workspace_root).join("config.toml")
}

fn workspace_daemon_socket(workspace_root: &Path) -> PathBuf {
    workspace_turin_root(workspace_root).join("daemon.sock")
}

fn channel_runtime_dir(workspace_root: &Path, channel_id: &str) -> PathBuf {
    workspace_turin_root(workspace_root)
        .join("runtime/channels")
        .join(channel_id)
}

fn write_mock_runtime_config(
    workspace_root: &Path,
    state_db_path: &Path,
    mock_response: &str,
    agent_idle_timeout_seconds: u64,
) -> Result<PathBuf> {
    let turin_root = workspace_turin_root(workspace_root);
    let harness_dir = turin_root.join("harnesses");
    fs::create_dir_all(&harness_dir)?;
    fs::create_dir_all(turin_root.join("runtime/channels/mock"))?;
    fs::write(harness_dir.join("main.lua"), "-- perf channel harness\n")?;

    let config_path = workspace_config_path(workspace_root);
    let config_toml = format!(
        r#"[agent]
id = "default"
model = "mock-model"
provider = "mock"
system_prompt = "Fake channel perf scenario"
idle_timeout_seconds = {agent_idle_timeout_seconds}

[kernel]
workspace_root = "{workspace_root}"
max_turns = 4
heartbeat_interval_seconds = 30
initial_spawn_depth = 0

[persistence.state]
path = "{state_db_path}"

[harness]
directory = "harnesses"
fs_root = "."

[providers.mock]
type = "mock"
base_url = "{mock_response}"

[remote]
bind = "127.0.0.1:0"
"#,
        workspace_root = toml_escape_path(workspace_root),
        state_db_path = toml_escape_path(state_db_path),
        mock_response = toml_escape(mock_response),
        agent_idle_timeout_seconds = agent_idle_timeout_seconds,
    );
    fs::write(&config_path, config_toml)?;
    Ok(config_path)
}

fn toml_escape_path(path: &Path) -> String {
    toml_escape(&path.to_string_lossy())
}

fn toml_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn sample_events(count: usize, message_bytes: usize) -> Vec<InboundEvent> {
    (0..count)
        .map(|index| sample_event(index, message_bytes))
        .collect()
}

fn sample_event(index: usize, message_bytes: usize) -> InboundEvent {
    sample_event_for_session(index % 4, index, "Say pong", message_bytes)
}

fn scale_events(
    sessions: usize,
    messages_per_session: usize,
    message_bytes: usize,
) -> Vec<InboundEvent> {
    let mut events = Vec::with_capacity(sessions * messages_per_session);
    for message_index in 0..messages_per_session {
        for session_index in 0..sessions {
            events.push(sample_event_for_session(
                session_index,
                message_index,
                "Scale pong",
                message_bytes,
            ));
        }
    }
    events
}

fn sample_event_for_session(
    session_index: usize,
    message_index: usize,
    text_prefix: &str,
    message_bytes: usize,
) -> InboundEvent {
    let conversation = ChannelConversationKey {
        channel: ChannelKind::new("mock"),
        workspace_id: "perf".into(),
        room_id: Some("room".into()),
        thread_id: format!("thread-{session_index}"),
        user_id: Some(format!("user-{session_index}")),
    };
    InboundEvent {
        message: ChannelMessageRef {
            conversation: conversation.clone(),
            message_id: format!("s{session_index}-m{message_index}"),
        },
        conversation,
        user: ChannelUser {
            id: format!("user-{session_index}"),
            display_name: Some(format!("User {session_index}")),
            username: Some(format!("user_{session_index}")),
        },
        session_scope: ChannelSessionScope::User,
        text: synthetic_text(
            &format!("{text_prefix} for session {session_index} message {message_index}. "),
            message_bytes,
        ),
        attachments: vec![],
        metadata: Default::default(),
    }
}

fn synthetic_text(seed: &str, target_bytes: usize) -> String {
    if target_bytes == 0 {
        return String::new();
    }

    let seed = if seed.is_empty() { "x" } else { seed };
    let mut text = String::with_capacity(target_bytes);
    while text.len() < target_bytes {
        text.push_str(seed);
    }
    text.truncate(target_bytes);
    text
}

fn parse_checkpoints(raw: &str, max_messages: usize) -> Result<Vec<usize>> {
    let mut checkpoints = raw
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .with_context(|| format!("invalid checkpoint '{part}'"))
        })
        .collect::<Result<Vec<_>>>()?;
    checkpoints.retain(|checkpoint| *checkpoint > 0 && *checkpoint <= max_messages);
    checkpoints.sort_unstable();
    checkpoints.dedup();
    if checkpoints.last().copied() != Some(max_messages) {
        checkpoints.push(max_messages);
    }
    Ok(checkpoints)
}

fn print_summary(report: &PerfReport) {
    println!(
        "{}",
        serde_json::to_string_pretty(report).expect("perf report should serialize")
    );
}

fn print_footprint_summary(report: &FootprintReport) {
    println!(
        "{}",
        serde_json::to_string_pretty(report).expect("footprint report should serialize")
    );
}
