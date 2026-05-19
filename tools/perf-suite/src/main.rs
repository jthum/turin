use anyhow::{Context, Result};
use async_trait::async_trait;
use clap::{Parser, Subcommand};
use futures::future::BoxFuture;
use futures::stream;
use serde::Serialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tempfile::TempDir;
use tokio::task::JoinHandle;
use tokio::time::{Instant as TokioInstant, sleep, timeout};
use turin::inference::provider::{
    InferenceEvent, InferenceProvider, InferenceRequest, InferenceStream, ProviderClient,
    RequestOptions, SdkError,
};
use turin::kernel::Kernel;
use turin::kernel::config::{
    AgentConfig, EmbeddingConfig, GovernanceConfig, HarnessConfig, InferenceConfig, KernelConfig,
    PersistenceConfig, ProviderConfig, TurinConfig,
};
use turin_channel_core::{
    ChannelConversationKey, ChannelKind, ChannelMessageRef, ChannelSessionScope, ChannelUser,
    InboundEvent, OutboundMessage,
};
use turin_channel_runner::{ChannelDriver, ChannelProgressUpdate, ChannelRunner, RunnerConfig};

#[derive(Parser)]
#[command(name = "turin-perf-suite")]
#[command(about = "Local Turin performance and footprint scenarios")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Stress session hot history with mocked inference and large tool outputs.
    HotHistory(HotHistoryArgs),
    /// Drive the real daemon through mocked channel ingress/egress and mocked inference.
    FakeChannel(FakeChannelArgs),
    /// Measure daemon/channel cost across logical session and message-count checkpoints.
    ChannelScale(ChannelScaleArgs),
}

#[derive(Parser)]
struct HotHistoryArgs {
    /// Number of user prompts to run through one session.
    #[arg(long, default_value_t = 50)]
    turns: usize,

    /// Bytes written to each synthetic file read by the tool call.
    #[arg(long, default_value_t = 256 * 1024)]
    payload_bytes: usize,

    /// Capture a snapshot every N turns.
    #[arg(long, default_value_t = 5)]
    sample_every: usize,

    /// Optional persistent workspace. If omitted, an ephemeral temp dir is used.
    #[arg(long)]
    workspace_root: Option<PathBuf>,

    /// Report output directory.
    #[arg(long, default_value = ".workspace/perf-reports")]
    report_dir: PathBuf,
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

    /// Optional target byte size for each mocked assistant response.
    #[arg(long)]
    response_bytes: Option<usize>,

    /// Optional persistent workspace. If omitted, an ephemeral temp dir is used.
    #[arg(long)]
    workspace_root: Option<PathBuf>,

    /// Report output directory.
    #[arg(long, default_value = ".workspace/perf-reports")]
    report_dir: PathBuf,
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

    /// Optional target byte size for each mocked assistant response.
    #[arg(long)]
    response_bytes: Option<usize>,

    /// Optional persistent workspace. If omitted, an ephemeral temp dir is used.
    #[arg(long)]
    workspace_root: Option<PathBuf>,

    /// Report output directory.
    #[arg(long, default_value = ".workspace/perf-reports")]
    report_dir: PathBuf,
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
struct Snapshot {
    label: String,
    elapsed_ms: u128,
    rss_kb: Option<u64>,
    pss_kb: Option<u64>,
    state_db_main_bytes: u64,
    state_db_wal_bytes: u64,
    state_db_shm_bytes: u64,
    state_db_bytes: u64,
    turn_index: Option<u32>,
    history_len: Option<usize>,
    outbound_messages: Option<usize>,
    active_sessions: Option<usize>,
    messages_per_session: Option<usize>,
}

#[derive(Debug)]
struct ProcessMemory {
    rss_kb: Option<u64>,
    pss_kb: Option<u64>,
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
                .unwrap_or_else(final_events)
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
        Command::HotHistory(args) => run_hot_history(args).await,
        Command::FakeChannel(args) => run_fake_channel(args).await,
        Command::ChannelScale(args) => run_channel_scale(args).await,
    }
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

    for index in 0..args.turns {
        fs::write(
            payload_dir.join(format!("payload-{index}.txt")),
            synthetic_payload(index, args.payload_bytes),
        )?;
    }

    let config = build_config(&workspace_root, &harness_dir, &state_db_path, args.turns)?;
    let responses = Arc::new(Mutex::new(build_responses(args.turns)));
    let provider = Arc::new(SequencePerfProvider { responses });

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;
    kernel.add_client("mock".to_string(), ProviderClient::new("mock", provider));

    let mut session = kernel.create_session().await;
    let start = Instant::now();
    let mut snapshots = vec![snapshot(
        "start",
        start,
        &state_db_path,
        Some(session.turn_index),
        Some(session.history.len()),
        None,
        None,
        None,
    )];

    for index in 0..args.turns {
        kernel
            .run(&mut session, Some(format!("Read payload {index}.")))
            .await
            .with_context(|| format!("hot-history turn {index} failed"))?;

        if (index + 1) % args.sample_every == 0 || index + 1 == args.turns {
            snapshots.push(snapshot(
                &format!("after-turn-{}", index + 1),
                start,
                &state_db_path,
                Some(session.turn_index),
                Some(session.history.len()),
                None,
                None,
                None,
            ));
        }
    }

    kernel.end_session(&mut session).await?;
    snapshots.push(snapshot(
        "after-end-session",
        start,
        &state_db_path,
        Some(session.turn_index),
        Some(session.history.len()),
        None,
        None,
        None,
    ));

    let report = PerfReport {
        scenario: "hot-history".to_string(),
        config: serde_json::json!({
            "turns": args.turns,
            "payload_bytes": args.payload_bytes,
            "sample_every": args.sample_every,
        }),
        workspace_root: workspace_root.display().to_string(),
        state_db_path: state_db_path.display().to_string(),
        snapshots,
    };

    write_reports(&args.report_dir, &report)?;
    print_summary(&report);
    Ok(())
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
    )];

    let mock_response = sized_text(&args.mock_response, args.response_bytes);
    let daemon =
        ChannelDaemonHarness::start(workspace_root.clone(), &state_db_path, &mock_response).await?;
    snapshots.push(snapshot(
        "after-daemon-start",
        start,
        &state_db_path,
        None,
        None,
        Some(0),
        None,
        None,
    ));

    let runner = daemon.runner();
    let mut driver = MockChannelDriver::new(sample_events(args.messages, args.message_bytes));
    let sent = Arc::clone(&driver.sent);

    runner
        .run_driver("default", &mut driver, Some(30_000))
        .await?;

    let outbound_count = sent.lock().expect("sent lock poisoned").len();
    snapshots.push(snapshot(
        "after-runner",
        start,
        &state_db_path,
        None,
        None,
        Some(outbound_count),
        None,
        None,
    ));

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
    ));

    let report = PerfReport {
        scenario: "fake-channel".to_string(),
        config: serde_json::json!({
            "messages": args.messages,
            "message_bytes": args.message_bytes,
            "mock_response_bytes": mock_response.len(),
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
    let snapshots = Arc::new(Mutex::new(vec![snapshot(
        "fresh-start",
        start,
        &state_db_path,
        None,
        None,
        Some(0),
        Some(0),
        Some(0),
    )]));

    let mock_response = sized_text(&args.mock_response, args.response_bytes);
    let daemon =
        ChannelDaemonHarness::start(workspace_root.clone(), &state_db_path, &mock_response).await?;
    snapshots
        .lock()
        .expect("scale snapshots lock poisoned")
        .push(snapshot(
            "after-daemon-start",
            start,
            &state_db_path,
            None,
            None,
            Some(0),
            Some(0),
            Some(0),
        ));

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
    snapshots
        .lock()
        .expect("scale snapshots lock poisoned")
        .push(snapshot(
            "after-runner",
            start,
            &state_db_path,
            None,
            None,
            Some(outbound_count),
            Some(args.sessions),
            Some(args.messages_per_session),
        ));

    daemon.stop().await?;
    snapshots
        .lock()
        .expect("scale snapshots lock poisoned")
        .push(snapshot(
            "after-daemon-stop",
            start,
            &state_db_path,
            None,
            None,
            Some(outbound_count),
            Some(args.sessions),
            Some(args.messages_per_session),
        ));

    let report = PerfReport {
        scenario: "channel-scale".to_string(),
        config: serde_json::json!({
            "sessions": args.sessions,
            "messages_per_session": args.messages_per_session,
            "message_bytes": args.message_bytes,
            "checkpoints": checkpoints,
            "mock_response_bytes": mock_response.len(),
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

fn prepare_workspace(workspace_root: Option<PathBuf>) -> Result<(Option<TempDir>, PathBuf)> {
    if let Some(path) = workspace_root {
        fs::create_dir_all(&path)?;
        return Ok((None, path));
    }

    let temp = tempfile::tempdir().context("failed to create perf workspace")?;
    let path = temp.path().to_path_buf();
    Ok((Some(temp), path))
}

impl ChannelDaemonHarness {
    async fn start(
        workspace_root: PathBuf,
        state_db_path: &Path,
        mock_response: &str,
    ) -> Result<Self> {
        let config_path = write_mock_runtime_config(&workspace_root, state_db_path, mock_response)?;
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
            turin_daemon_client::DaemonClient::new(&self.endpoint),
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

    async fn stop(self) -> Result<()> {
        let client = turin_daemon_client::DaemonClient::new(&self.endpoint);
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
            recorder.record_if_checkpoint(outbound_count);
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
            idle_timeout_seconds: Some(0),
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
        inference: InferenceConfig::default(),
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

fn build_responses(turns: usize) -> VecDeque<Vec<InferenceEvent>> {
    let mut responses = VecDeque::with_capacity(turns * 2);
    for index in 0..turns {
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
        responses.push_back(final_events());
    }
    responses
}

fn final_events() -> Vec<InferenceEvent> {
    vec![
        message_start(),
        InferenceEvent::MessageDelta {
            content: "Recorded.".to_string(),
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
    messages_per_session: Option<usize>,
) -> Snapshot {
    let memory = read_process_memory();
    let state_store_size = state_store_size(state_db_path);
    Snapshot {
        label: label.to_string(),
        elapsed_ms: start.elapsed().as_millis(),
        rss_kb: memory.rss_kb,
        pss_kb: memory.pss_kb,
        state_db_main_bytes: state_store_size.main_bytes,
        state_db_wal_bytes: state_store_size.wal_bytes,
        state_db_shm_bytes: state_store_size.shm_bytes,
        state_db_bytes: state_store_size.total_bytes(),
        turn_index,
        history_len,
        outbound_messages,
        active_sessions,
        messages_per_session,
    }
}

impl ScaleRecorder {
    fn record_if_checkpoint(&self, outbound_count: usize) {
        if !self.sample_totals.contains(&outbound_count) {
            return;
        }

        let messages_per_session = outbound_count / self.active_sessions;
        self.snapshots
            .lock()
            .expect("scale snapshots lock poisoned")
            .push(snapshot(
                &format!(
                    "after-{}-sessions-x{}-messages",
                    self.active_sessions, messages_per_session
                ),
                self.start,
                &self.state_db_path,
                None,
                None,
                Some(outbound_count),
                Some(self.active_sessions),
                Some(messages_per_session),
            ));
    }
}

fn read_process_memory() -> ProcessMemory {
    let mut rss_kb = None;
    let mut pss_kb = None;

    if let Ok(raw) = fs::read_to_string("/proc/self/smaps_rollup") {
        for line in raw.lines() {
            rss_kb = rss_kb.or_else(|| parse_kb_line(line, "Rss:"));
            pss_kb = pss_kb.or_else(|| parse_kb_line(line, "Pss:"));
        }
    }

    if rss_kb.is_none() {
        if let Ok(raw) = fs::read_to_string("/proc/self/status") {
            for line in raw.lines() {
                rss_kb = rss_kb.or_else(|| parse_kb_line(line, "VmRSS:"));
            }
        }
    }

    ProcessMemory { rss_kb, pss_kb }
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

fn write_reports(report_dir: &Path, report: &PerfReport) -> Result<()> {
    fs::create_dir_all(report_dir)?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before unix epoch")?
        .as_secs();
    let json_path = report_dir.join(format!("{}-{stamp}.json", report.scenario));
    let md_path = report_dir.join(format!("{}-{stamp}.md", report.scenario));

    fs::write(&json_path, serde_json::to_vec_pretty(report)?)?;
    fs::write(&md_path, markdown_report(report))?;

    println!("json_report={}", json_path.display());
    println!("markdown_report={}", md_path.display());
    Ok(())
}

fn markdown_report(report: &PerfReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("# Perf Report: {}\n\n", report.scenario));
    out.push_str(&format!("- config: `{}`\n", report.config));
    out.push_str(&format!("- workspace_root: `{}`\n", report.workspace_root));
    out.push_str(&format!("- state_db_path: `{}`\n\n", report.state_db_path));
    out.push_str(
        "| label | elapsed_ms | rss_kb | pss_kb | state_db_main_bytes | state_db_wal_bytes | state_db_shm_bytes | state_db_bytes | turn_index | history_len | outbound_messages | active_sessions | messages_per_session |\n",
    );
    out.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for snapshot in &report.snapshots {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            snapshot.label,
            snapshot.elapsed_ms,
            display_option(snapshot.rss_kb),
            display_option(snapshot.pss_kb),
            snapshot.state_db_main_bytes,
            snapshot.state_db_wal_bytes,
            snapshot.state_db_shm_bytes,
            snapshot.state_db_bytes,
            display_u32_option(snapshot.turn_index),
            display_usize_option(snapshot.history_len),
            display_usize_option(snapshot.outbound_messages),
            display_usize_option(snapshot.active_sessions),
            display_usize_option(snapshot.messages_per_session)
        ));
    }
    out
}

fn display_option(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string())
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

fn sized_text(seed: &str, target_bytes: Option<usize>) -> String {
    match target_bytes {
        Some(bytes) => synthetic_text(seed, bytes),
        None => seed.to_string(),
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
