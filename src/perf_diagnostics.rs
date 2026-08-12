#[cfg(feature = "perf-diagnostics")]
mod enabled {
    use std::future::Future;
    use std::sync::{OnceLock, RwLock};
    use std::time::Instant;

    use serde_json::{Value, json};
    use tokio::sync::broadcast;

    use crate::daemon::protocol::EventEnvelope;

    #[derive(Clone)]
    struct DiagnosticContext {
        session_id: String,
    }

    tokio::task_local! {
        static CONTEXT: DiagnosticContext;
    }

    static EVENT_SINK: OnceLock<RwLock<Option<broadcast::Sender<EventEnvelope>>>> = OnceLock::new();

    pub(crate) fn install_event_sink(sender: broadcast::Sender<EventEnvelope>) {
        let sink = EVENT_SINK.get_or_init(|| RwLock::new(None));
        *sink.write().expect("perf diagnostics event sink poisoned") = Some(sender);
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
    }

    impl Stage {
        pub(crate) fn start(
            operation: &'static str,
            session_id: Option<&str>,
            fields: Value,
        ) -> Self {
            let operation_id = uuid::Uuid::now_v7().simple().to_string();
            let session_id = session_id
                .map(str::to_string)
                .or_else(|| CONTEXT.try_with(|context| context.session_id.clone()).ok());
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
                }),
            );
            Self {
                operation_id,
                operation,
                session_id,
                started_at: Instant::now(),
                completed: false,
            }
        }

        pub(crate) fn finish(mut self, outcome: &'static str, fields: Value) {
            self.completed = true;
            self.emit_completion(outcome, fields);
        }

        fn emit_completion(&self, outcome: &'static str, fields: Value) {
            emit(
                "perf.operation.completed",
                json!({
                    "schema_version": 1,
                    "operation_id": self.operation_id,
                    "operation": self.operation,
                    "session_id": self.session_id,
                    "pid": std::process::id(),
                    "build_profile": if cfg!(debug_assertions) { "debug" } else { "release" },
                    "outcome": outcome,
                    "elapsed_us": self.started_at.elapsed().as_micros().min(u64::MAX as u128) as u64,
                    "fields": fields,
                }),
            );
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
        if let Some(sender) = EVENT_SINK.get().and_then(|sink| sink.read().ok()?.clone()) {
            let _ = sender.send(EventEnvelope::new(event, data));
        }
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
                Stage::start("session.test", None, json!({ "rows": 2 }))
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
