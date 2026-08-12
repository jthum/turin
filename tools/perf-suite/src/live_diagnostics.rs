use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{RwLock, broadcast};
use tokio::time::{MissedTickBehavior, interval};
use turin_daemon_protocol::{EventEnvelope, RuntimeEventsSubscribeParams};

const PERF_STARTED_EVENT: &str = "perf.operation.started";
const PERF_COMPLETED_EVENT: &str = "perf.operation.completed";

#[derive(Debug, Clone, Parser)]
pub struct LiveDiagnosticsArgs {
    /// Turin config used to locate the running daemon.
    #[arg(long, default_value = ".turin/config.toml")]
    config: PathBuf,

    /// Loopback address used by the optional browser diagnostics stream.
    #[arg(long, default_value = "127.0.0.1:4779")]
    bind: SocketAddr,

    /// Memory sampling period while at least one measured operation is active.
    #[arg(long, default_value_t = 50)]
    active_sample_ms: u64,

    /// Memory sampling period while all measured operations are idle.
    #[arg(long, default_value_t = 1000)]
    idle_sample_ms: u64,

    /// Number of completed operations and process samples retained for live clients.
    #[arg(long, default_value_t = 500)]
    history_limit: usize,

    /// Append operation summaries and process samples as JSONL.
    #[arg(long, default_value = ".workspace/perf-reports/live-diagnostics.jsonl")]
    output: PathBuf,

    /// Additional process IDs to include in memory trends.
    #[arg(long = "pid")]
    additional_pids: Vec<u32>,

    /// Suppress one-line operation summaries on stdout.
    #[arg(long)]
    quiet: bool,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct ProcessMemory {
    rss_kb: Option<u64>,
    pss_kb: Option<u64>,
    pss_anon_kb: Option<u64>,
    pss_file_kb: Option<u64>,
    pss_shmem_kb: Option<u64>,
}

impl ProcessMemory {
    fn observe_peak(&mut self, sample: Self) {
        self.rss_kb = option_max(self.rss_kb, sample.rss_kb);
        self.pss_kb = option_max(self.pss_kb, sample.pss_kb);
        self.pss_anon_kb = option_max(self.pss_anon_kb, sample.pss_anon_kb);
        self.pss_file_kb = option_max(self.pss_file_kb, sample.pss_file_kb);
        self.pss_shmem_kb = option_max(self.pss_shmem_kb, sample.pss_shmem_kb);
    }
}

#[derive(Debug, Clone)]
struct ActiveOperation {
    operation_id: String,
    operation: String,
    session_id: Option<String>,
    pid: u32,
    build_profile: Option<String>,
    started_at_ms: u64,
    fields: Value,
    memory_start: ProcessMemory,
    memory_peak: ProcessMemory,
}

#[derive(Debug, Clone, Serialize)]
struct ActiveOperationView {
    operation_id: String,
    operation: String,
    session_id: Option<String>,
    pid: u32,
    build_profile: Option<String>,
    started_at_ms: u64,
    fields: Value,
    memory_start: ProcessMemory,
    memory_peak: ProcessMemory,
}

impl From<&ActiveOperation> for ActiveOperationView {
    fn from(operation: &ActiveOperation) -> Self {
        Self {
            operation_id: operation.operation_id.clone(),
            operation: operation.operation.clone(),
            session_id: operation.session_id.clone(),
            pid: operation.pid,
            build_profile: operation.build_profile.clone(),
            started_at_ms: operation.started_at_ms,
            fields: operation.fields.clone(),
            memory_start: operation.memory_start,
            memory_peak: operation.memory_peak,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct OperationSummary {
    operation_id: String,
    operation: String,
    session_id: Option<String>,
    pid: u32,
    build_profile: Option<String>,
    started_at_ms: u64,
    completed_at_ms: u64,
    elapsed_us: u64,
    outcome: String,
    start_fields: Value,
    fields: Value,
    memory_start: ProcessMemory,
    memory_end: ProcessMemory,
    memory_peak: ProcessMemory,
    rss_delta_kb: Option<i64>,
    pss_delta_kb: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
struct ProcessSample {
    sampled_at_ms: u64,
    pid: u32,
    active_operations: usize,
    memory: ProcessMemory,
}

#[derive(Default)]
struct LiveState {
    active: HashMap<String, ActiveOperation>,
    completed: VecDeque<OperationSummary>,
    samples: VecDeque<ProcessSample>,
    known_pids: HashSet<u32>,
}

#[derive(Serialize)]
struct LiveSnapshot {
    active: Vec<ActiveOperationView>,
    completed: Vec<OperationSummary>,
    samples: Vec<ProcessSample>,
}

impl LiveState {
    fn snapshot(&self) -> LiveSnapshot {
        LiveSnapshot {
            active: self
                .active
                .values()
                .map(ActiveOperationView::from)
                .collect(),
            completed: self.completed.iter().cloned().collect(),
            samples: self.samples.iter().cloned().collect(),
        }
    }
}

pub async fn run_live_diagnostics(args: LiveDiagnosticsArgs) -> Result<()> {
    anyhow::ensure!(
        args.active_sample_ms > 0,
        "--active-sample-ms must be positive"
    );
    anyhow::ensure!(args.idle_sample_ms > 0, "--idle-sample-ms must be positive");
    anyhow::ensure!(args.history_limit > 0, "--history-limit must be positive");
    anyhow::ensure!(
        args.bind.ip().is_loopback(),
        "live diagnostics HTTP must bind to a loopback address"
    );

    let writer = open_jsonl_writer(&args.output)?;
    let client = turin_daemon_client::DaemonClient::from_config(&args.config)
        .await
        .with_context(|| format!("failed to resolve daemon from '{}'", args.config.display()))?;
    let mut events = client
        .subscribe_managed(RuntimeEventsSubscribeParams::default())
        .await
        .context("failed to subscribe to daemon diagnostics")?;

    let state = Arc::new(RwLock::new(LiveState::default()));
    {
        let mut guard = state.write().await;
        guard
            .known_pids
            .extend(args.additional_pids.iter().copied());
    }
    let (updates, _) = broadcast::channel::<String>(1024);
    let http_task = tokio::spawn(serve_http(args.bind, state.clone(), updates.clone()));

    println!("live_diagnostics=http://{}/events", args.bind);
    println!("jsonl={}", args.output.display());
    println!("waiting_for=perf-diagnostics enabled Turin daemon");

    let base_tick_ms = args.active_sample_ms.min(args.idle_sample_ms);
    let mut ticker = interval(Duration::from_millis(base_tick_ms));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut last_memory_sample = Instant::now()
        .checked_sub(Duration::from_millis(args.idle_sample_ms))
        .unwrap_or_else(Instant::now);

    loop {
        tokio::select! {
            event = events.next_event() => {
                let event = event.context("daemon diagnostics stream failed")?;
                handle_daemon_event(
                    event,
                    &state,
                    &updates,
                    &writer,
                    args.history_limit,
                    args.quiet,
                ).await?;
            }
            _ = ticker.tick() => {
                let active = !state.read().await.active.is_empty();
                let period = if active { args.active_sample_ms } else { args.idle_sample_ms };
                if last_memory_sample.elapsed() >= Duration::from_millis(period) {
                    sample_processes(&state, &updates, &writer, args.history_limit).await?;
                    last_memory_sample = Instant::now();
                }
            }
            result = tokio::signal::ctrl_c() => {
                result.context("failed to listen for shutdown signal")?;
                break;
            }
        }
    }

    http_task.abort();
    let _ = http_task.await;
    writer
        .lock()
        .expect("live diagnostics writer lock poisoned")
        .flush()?;
    Ok(())
}

async fn handle_daemon_event(
    event: EventEnvelope,
    state: &Arc<RwLock<LiveState>>,
    updates: &broadcast::Sender<String>,
    writer: &Arc<Mutex<BufWriter<File>>>,
    history_limit: usize,
    quiet: bool,
) -> Result<()> {
    match event.event.as_str() {
        PERF_STARTED_EVENT => {
            let operation_id = required_string(&event.data, "operation_id")?;
            let operation = required_string(&event.data, "operation")?;
            let pid = required_u32(&event.data, "pid")?;
            let memory = read_process_memory(pid);
            let active = ActiveOperation {
                operation_id: operation_id.clone(),
                operation,
                session_id: optional_string(&event.data, "session_id"),
                pid,
                build_profile: optional_string(&event.data, "build_profile"),
                started_at_ms: unix_time_ms(),
                fields: event
                    .data
                    .get("fields")
                    .cloned()
                    .unwrap_or_else(|| json!({})),
                memory_start: memory,
                memory_peak: memory,
            };
            let mut guard = state.write().await;
            guard.known_pids.insert(pid);
            guard.active.insert(operation_id, active);
        }
        PERF_COMPLETED_EVENT => {
            let operation_id = required_string(&event.data, "operation_id")?;
            let operation = required_string(&event.data, "operation")?;
            let pid = required_u32(&event.data, "pid")?;
            let memory_end = read_process_memory(pid);
            let completed_at_ms = unix_time_ms();
            let mut guard = state.write().await;
            guard.known_pids.insert(pid);
            let mut active = guard.active.remove(&operation_id).unwrap_or_else(|| {
                let memory = read_process_memory(pid);
                ActiveOperation {
                    operation_id: operation_id.clone(),
                    operation: operation.clone(),
                    session_id: optional_string(&event.data, "session_id"),
                    pid,
                    build_profile: optional_string(&event.data, "build_profile"),
                    started_at_ms: completed_at_ms,
                    fields: json!({}),
                    memory_start: memory,
                    memory_peak: memory,
                }
            });
            active.memory_peak.observe_peak(memory_end);
            let summary = OperationSummary {
                operation_id,
                operation,
                session_id: optional_string(&event.data, "session_id").or(active.session_id),
                pid,
                build_profile: optional_string(&event.data, "build_profile")
                    .or(active.build_profile),
                started_at_ms: active.started_at_ms,
                completed_at_ms,
                elapsed_us: event
                    .data
                    .get("elapsed_us")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                outcome: optional_string(&event.data, "outcome")
                    .unwrap_or_else(|| "unknown".to_string()),
                start_fields: active.fields,
                fields: event
                    .data
                    .get("fields")
                    .cloned()
                    .unwrap_or_else(|| json!({})),
                memory_start: active.memory_start,
                memory_end,
                memory_peak: active.memory_peak,
                rss_delta_kb: option_delta(active.memory_start.rss_kb, memory_end.rss_kb),
                pss_delta_kb: option_delta(active.memory_start.pss_kb, memory_end.pss_kb),
            };
            if !quiet {
                print_summary(&summary);
            }
            append_jsonl(writer, &json!({ "kind": "operation", "data": summary }))?;
            let encoded = sse_event("perf.summary", &summary)?;
            let _ = updates.send(encoded);
            push_bounded(&mut guard.completed, summary, history_limit);
        }
        _ => {}
    }
    Ok(())
}

async fn sample_processes(
    state: &Arc<RwLock<LiveState>>,
    updates: &broadcast::Sender<String>,
    writer: &Arc<Mutex<BufWriter<File>>>,
    history_limit: usize,
) -> Result<()> {
    let pids = state
        .read()
        .await
        .known_pids
        .iter()
        .copied()
        .collect::<Vec<_>>();
    for pid in pids {
        let memory = read_process_memory(pid);
        if memory.rss_kb.is_none() && memory.pss_kb.is_none() {
            continue;
        }
        let mut guard = state.write().await;
        let active_operations = guard
            .active
            .values()
            .filter(|operation| operation.pid == pid)
            .count();
        for operation in guard
            .active
            .values_mut()
            .filter(|operation| operation.pid == pid)
        {
            operation.memory_peak.observe_peak(memory);
        }
        let sample = ProcessSample {
            sampled_at_ms: unix_time_ms(),
            pid,
            active_operations,
            memory,
        };
        append_jsonl(writer, &json!({ "kind": "process_sample", "data": sample }))?;
        let encoded = sse_event("perf.memory.sample", &sample)?;
        let _ = updates.send(encoded);
        push_bounded(&mut guard.samples, sample, history_limit);
    }
    Ok(())
}

async fn serve_http(
    bind: SocketAddr,
    state: Arc<RwLock<LiveState>>,
    updates: broadcast::Sender<String>,
) -> Result<()> {
    let listener = TcpListener::bind(bind)
        .await
        .with_context(|| format!("failed to bind live diagnostics HTTP to {bind}"))?;
    loop {
        let (stream, _) = listener.accept().await?;
        let state = state.clone();
        let updates = updates.clone();
        tokio::spawn(async move {
            let _ = handle_http_connection(stream, state, updates).await;
        });
    }
}

async fn handle_http_connection(
    mut stream: TcpStream,
    state: Arc<RwLock<LiveState>>,
    updates: broadcast::Sender<String>,
) -> Result<()> {
    let mut request = vec![0u8; 8192];
    let read = stream.read(&mut request).await?;
    let request = String::from_utf8_lossy(&request[..read]);
    let first_line = request.lines().next().unwrap_or_default();
    let path = first_line.split_whitespace().nth(1).unwrap_or("/");
    match path {
        "/events" => {
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\nX-Accel-Buffering: no\r\n\r\n",
                )
                .await?;
            let snapshot = state.read().await.snapshot();
            stream
                .write_all(sse_event("perf.snapshot", &snapshot)?.as_bytes())
                .await?;
            let mut receiver = updates.subscribe();
            let mut heartbeat = interval(Duration::from_secs(15));
            loop {
                tokio::select! {
                    update = receiver.recv() => match update {
                        Ok(update) => stream.write_all(update.as_bytes()).await?,
                        Err(broadcast::error::RecvError::Lagged(_)) => {
                            let snapshot = state.read().await.snapshot();
                            stream.write_all(sse_event("perf.snapshot", &snapshot)?.as_bytes()).await?;
                        }
                        Err(broadcast::error::RecvError::Closed) => break,
                    },
                    _ = heartbeat.tick() => stream.write_all(b": keepalive\n\n").await?,
                }
            }
        }
        "/snapshot" => {
            let body = serde_json::to_vec(&state.read().await.snapshot())?;
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(header.as_bytes()).await?;
            stream.write_all(&body).await?;
        }
        _ => {
            stream
                .write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
                )
                .await?;
        }
    }
    Ok(())
}

fn open_jsonl_writer(path: &Path) -> Result<Arc<Mutex<BufWriter<File>>>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create '{}'", parent.display()))?;
    }
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open '{}'", path.display()))?;
    Ok(Arc::new(Mutex::new(BufWriter::new(file))))
}

fn append_jsonl(writer: &Arc<Mutex<BufWriter<File>>>, value: &Value) -> Result<()> {
    let mut writer = writer
        .lock()
        .expect("live diagnostics writer lock poisoned");
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn sse_event(event: &str, value: &impl Serialize) -> Result<String> {
    Ok(format!(
        "event: {event}\ndata: {}\n\n",
        serde_json::to_string(value)?
    ))
}

fn print_summary(summary: &OperationSummary) {
    let pss = match summary.pss_delta_kb {
        Some(delta) => format!(" pss_delta_kb={delta}"),
        None => String::new(),
    };
    println!(
        "operation={} elapsed_us={} outcome={} pid={}{}",
        summary.operation, summary.elapsed_us, summary.outcome, summary.pid, pss
    );
}

fn read_process_memory(pid: u32) -> ProcessMemory {
    let proc_path = PathBuf::from(format!("/proc/{pid}"));
    let mut memory = ProcessMemory::default();
    if let Ok(raw) = fs::read_to_string(proc_path.join("smaps_rollup")) {
        for line in raw.lines() {
            memory.rss_kb = memory.rss_kb.or_else(|| parse_kb_line(line, "Rss:"));
            memory.pss_kb = memory.pss_kb.or_else(|| parse_kb_line(line, "Pss:"));
            memory.pss_anon_kb = memory
                .pss_anon_kb
                .or_else(|| parse_kb_line(line, "Pss_Anon:"));
            memory.pss_file_kb = memory
                .pss_file_kb
                .or_else(|| parse_kb_line(line, "Pss_File:"));
            memory.pss_shmem_kb = memory
                .pss_shmem_kb
                .or_else(|| parse_kb_line(line, "Pss_Shmem:"));
        }
    }
    if memory.rss_kb.is_none()
        && let Ok(raw) = fs::read_to_string(proc_path.join("status"))
    {
        for line in raw.lines() {
            memory.rss_kb = memory.rss_kb.or_else(|| parse_kb_line(line, "VmRSS:"));
        }
    }
    memory
}

fn parse_kb_line(line: &str, key: &str) -> Option<u64> {
    line.strip_prefix(key)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

fn required_string(value: &Value, key: &str) -> Result<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .with_context(|| format!("diagnostic event omitted '{key}'"))
}

fn optional_string(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_string)
}

fn required_u32(value: &Value, key: &str) -> Result<u32> {
    let value = value
        .get(key)
        .and_then(Value::as_u64)
        .with_context(|| format!("diagnostic event omitted '{key}'"))?;
    u32::try_from(value).with_context(|| format!("diagnostic '{key}' exceeds u32"))
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}

fn option_max(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (left, right) => left.or(right),
    }
}

fn option_delta(start: Option<u64>, end: Option<u64>) -> Option<i64> {
    let start = i128::from(start?);
    let end = i128::from(end?);
    Some((end - start).clamp(i64::MIN as i128, i64::MAX as i128) as i64)
}

fn push_bounded<T>(values: &mut VecDeque<T>, value: T, limit: usize) {
    values.push_back(value);
    while values.len() > limit {
        values.pop_front();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_peak_keeps_largest_observation() {
        let mut peak = ProcessMemory {
            rss_kb: Some(10),
            pss_kb: Some(8),
            ..ProcessMemory::default()
        };
        peak.observe_peak(ProcessMemory {
            rss_kb: Some(9),
            pss_kb: Some(12),
            ..ProcessMemory::default()
        });
        assert_eq!(peak.rss_kb, Some(10));
        assert_eq!(peak.pss_kb, Some(12));
    }

    #[test]
    fn bounded_history_discards_oldest_value() {
        let mut values = VecDeque::new();
        push_bounded(&mut values, 1, 2);
        push_bounded(&mut values, 2, 2);
        push_bounded(&mut values, 3, 2);
        assert_eq!(values, VecDeque::from([2, 3]));
    }
}
