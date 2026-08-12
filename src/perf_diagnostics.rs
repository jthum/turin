// Removal boundary: feature builds use the collector below; ordinary builds retain only the
// no-op macros required by instrumented call sites.
#[cfg(feature = "perf-diagnostics")]
mod enabled {
    use std::collections::HashMap;
    use std::fs;
    use std::future::Future;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};
    use std::time::{Duration, Instant};

    use serde::Serialize;
    use serde_json::{Value, json};
    use tokio::sync::{Notify, broadcast};
    use tokio::task::JoinHandle;

    use crate::daemon::protocol::EventEnvelope;

    #[derive(Clone)]
    struct DiagnosticContext {
        session_id: String,
    }

    tokio::task_local! {
        static CONTEXT: DiagnosticContext;
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

    #[derive(Debug, Clone, Copy)]
    struct ActiveMemory {
        start: ProcessMemory,
        peak: ProcessMemory,
    }

    #[derive(Default)]
    struct DiagnosticsState {
        event_sink: Option<broadcast::Sender<EventEnvelope>>,
        active: HashMap<String, ActiveMemory>,
    }

    static STATE: OnceLock<Mutex<DiagnosticsState>> = OnceLock::new();
    static SAMPLER: OnceLock<Mutex<Option<JoinHandle<()>>>> = OnceLock::new();
    static ACTIVITY: OnceLock<Notify> = OnceLock::new();

    fn state() -> &'static Mutex<DiagnosticsState> {
        STATE.get_or_init(|| Mutex::new(DiagnosticsState::default()))
    }

    fn sampler() -> &'static Mutex<Option<JoinHandle<()>>> {
        SAMPLER.get_or_init(|| Mutex::new(None))
    }

    fn activity() -> &'static Notify {
        ACTIVITY.get_or_init(Notify::new)
    }

    pub(crate) fn install_event_sink(sender: broadcast::Sender<EventEnvelope>) {
        if let Some(previous) = sampler()
            .lock()
            .expect("perf diagnostics sampler lock poisoned")
            .take()
        {
            previous.abort();
        }
        {
            let mut state = state()
                .lock()
                .expect("perf diagnostics state lock poisoned");
            state.event_sink = Some(sender);
            state.active.clear();
        }
        let handle = tokio::spawn(sample_active_memory());
        *sampler()
            .lock()
            .expect("perf diagnostics sampler lock poisoned") = Some(handle);
    }

    pub(crate) async fn scope_session<F>(session_id: &str, future: F) -> F::Output
    where
        F: Future,
    {
        CONTEXT
            .scope(
                DiagnosticContext {
                    session_id: session_id.to_string(),
                },
                future,
            )
            .await
    }

    pub(crate) struct Stage {
        operation_id: String,
        operation: &'static str,
        session_id: Option<String>,
        started_at: Instant,
        completed: bool,
        enabled: bool,
        memory_tracked: bool,
    }

    impl Stage {
        pub(crate) fn start(
            operation: &'static str,
            session_id: Option<&str>,
            fields: Value,
        ) -> Self {
            if !has_event_sink() {
                return Self {
                    operation_id: String::new(),
                    operation,
                    session_id: None,
                    started_at: Instant::now(),
                    completed: false,
                    enabled: false,
                    memory_tracked: false,
                };
            }
            let operation_id = uuid::Uuid::now_v7().simple().to_string();
            let session_id = session_id
                .map(str::to_string)
                .or_else(|| CONTEXT.try_with(|context| context.session_id.clone()).ok());
            let memory_tracked = tracks_process_memory(operation);
            let memory_start = memory_tracked.then(read_process_memory);
            if let Some(memory_start) = memory_start {
                state()
                    .lock()
                    .expect("perf diagnostics state lock poisoned")
                    .active
                    .insert(
                        operation_id.clone(),
                        ActiveMemory {
                            start: memory_start,
                            peak: memory_start,
                        },
                    );
                activity().notify_one();
            }
            emit(
                "perf.operation.started",
                json!({
                    "schema_version": 1,
                    "operation_id": operation_id,
                    "operation": operation,
                    "session_id": session_id,
                    "pid": std::process::id(),
                    "build_profile": if cfg!(debug_assertions) { "debug" } else { "release" },
                    "fields": fields,
                    "memory_start": memory_start,
                }),
            );
            Self {
                operation_id,
                operation,
                session_id,
                started_at: Instant::now(),
                completed: false,
                enabled: true,
                memory_tracked,
            }
        }

        pub(crate) fn finish(mut self, outcome: &'static str, fields: Value) {
            self.completed = true;
            self.emit_completion(outcome, fields);
        }

        fn emit_completion(&self, outcome: &'static str, fields: Value) {
            if !self.enabled {
                return;
            }
            let elapsed_us = self.started_at.elapsed().as_micros().min(u64::MAX as u128) as u64;
            let mut data = json!({
                "schema_version": 1,
                "operation_id": self.operation_id,
                "operation": self.operation,
                "session_id": self.session_id,
                "pid": std::process::id(),
                "build_profile": if cfg!(debug_assertions) { "debug" } else { "release" },
                "outcome": outcome,
                "elapsed_us": elapsed_us,
                "fields": fields,
            });
            if self.memory_tracked {
                let memory_end = read_process_memory();
                let active = state()
                    .lock()
                    .expect("perf diagnostics state lock poisoned")
                    .active
                    .remove(&self.operation_id)
                    .unwrap_or(ActiveMemory {
                        start: memory_end,
                        peak: memory_end,
                    });
                let mut memory_peak = active.peak;
                memory_peak.observe_peak(memory_end);
                data["memory_start"] = json!(active.start);
                data["memory_end"] = json!(memory_end);
                data["memory_peak"] = json!(memory_peak);
                data["rss_delta_kb"] = json!(option_delta(active.start.rss_kb, memory_end.rss_kb));
                data["pss_delta_kb"] = json!(option_delta(active.start.pss_kb, memory_end.pss_kb));
            }
            emit("perf.operation.completed", data);
        }
    }

    impl Drop for Stage {
        fn drop(&mut self) {
            if !self.completed {
                self.emit_completion("incomplete", json!({}));
            }
        }
    }

    fn emit(event: &'static str, data: Value) {
        let sender = state()
            .lock()
            .ok()
            .and_then(|state| state.event_sink.clone());
        if let Some(sender) = sender {
            let _ = sender.send(EventEnvelope::new(event, data));
        }
    }

    fn has_event_sink() -> bool {
        state()
            .lock()
            .map(|state| state.event_sink.is_some())
            .unwrap_or(false)
    }

    fn tracks_process_memory(operation: &str) -> bool {
        matches!(
            operation,
            "session.projection"
                | "session.resume.materialize"
                | "session.refresh.materialize"
                | "session.history.materialize"
        )
    }

    async fn sample_active_memory() {
        loop {
            while state()
                .lock()
                .map(|state| state.active.is_empty())
                .unwrap_or(true)
            {
                activity().notified().await;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
            let memory = read_process_memory();
            let mut state = state()
                .lock()
                .expect("perf diagnostics state lock poisoned");
            for active in state.active.values_mut() {
                active.peak.observe_peak(memory);
            }
        }
    }

    fn read_process_memory() -> ProcessMemory {
        let proc_path = PathBuf::from("/proc/self");
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

    #[cfg(test)]
    mod tests {
        use super::*;

        #[tokio::test]
        async fn replacement_sink_receives_session_scoped_stage() {
            let (discarded, _) = broadcast::channel(4);
            install_event_sink(discarded);

            let (sender, mut receiver) = broadcast::channel(4);
            install_event_sink(sender);
            scope_session("session-1", async {
                Stage::start("session.projection", None, json!({ "rows": 2 }))
                    .finish("ok", json!({ "rows": 3 }));
            })
            .await;

            let started = receiver.recv().await.expect("started diagnostic event");
            let completed = receiver.recv().await.expect("completed diagnostic event");
            assert_eq!(started.event, "perf.operation.started");
            assert_eq!(completed.event, "perf.operation.completed");
            assert_eq!(started.data["session_id"], "session-1");
            assert_eq!(completed.data["session_id"], "session-1");
            assert_eq!(started.data["operation_id"], completed.data["operation_id"]);
            assert_eq!(completed.data["outcome"], "ok");
            assert!(completed.data.get("memory_start").is_some());
            assert!(completed.data.get("memory_end").is_some());
            assert!(completed.data.get("memory_peak").is_some());
        }

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
    }
}

#[cfg(feature = "perf-diagnostics")]
pub(crate) use enabled::{Stage, install_event_sink, scope_session};

#[cfg(feature = "perf-diagnostics")]
macro_rules! perf_stage {
    ($binding:ident, $operation:expr, $session_id:expr, $fields:expr) => {
        let $binding = $crate::perf_diagnostics::Stage::start($operation, $session_id, $fields);
    };
}

#[cfg(not(feature = "perf-diagnostics"))]
macro_rules! perf_stage {
    ($binding:ident, $operation:expr, $session_id:expr, $fields:expr) => {};
}

#[cfg(feature = "perf-diagnostics")]
macro_rules! perf_stage_finish {
    ($binding:ident, $outcome:expr, $fields:expr) => {
        $binding.finish($outcome, $fields);
    };
}

#[cfg(not(feature = "perf-diagnostics"))]
macro_rules! perf_stage_finish {
    ($binding:ident, $outcome:expr, $fields:expr) => {};
}

#[cfg(feature = "perf-diagnostics")]
macro_rules! perf_session_scope {
    ($session_id:expr, $future:expr) => {
        $crate::perf_diagnostics::scope_session($session_id, $future)
    };
}

#[cfg(not(feature = "perf-diagnostics"))]
macro_rules! perf_session_scope {
    ($session_id:expr, $future:expr) => {
        $future
    };
}

pub(crate) use perf_session_scope;
pub(crate) use perf_stage;
pub(crate) use perf_stage_finish;
