use anyhow::Result;

use crate::kernel::session::QueuedTask;
use crate::persistence::schema::SignalRow;

use super::peer_runtime::PeerRuntime;

impl PeerRuntime {
    pub(super) async fn process_pending_signals(&mut self) -> Result<usize> {
        let Some(runtime_scheduler) = self.host.scheduler.as_ref() else {
            return Ok(0);
        };
        let store = runtime_scheduler.runtime_store();
        let current_session_id = self.control.current_session_id();
        let signals = store
            .list_signals_for_agent(&self.agent_id, current_session_id.as_deref(), 64)
            .await?;
        if signals.is_empty() {
            return Ok(0);
        }

        let mut processed = 0usize;
        for signal in signals {
            store.record_signal_attempt(signal.id).await?;
            match self.dispatch_signal(&signal).await {
                Ok(_) => {
                    store.delete_signal(signal.id).await?;
                    processed += 1;
                }
                Err(err) => {
                    let error_message = err.to_string();
                    store.set_signal_error(signal.id, &error_message).await?;
                    return Err(err);
                }
            }
        }

        Ok(processed)
    }

    async fn dispatch_signal(&mut self, signal: &SignalRow) -> Result<usize> {
        self.host.ensure_session_harness_engine(&mut self.session)?;
        let trace_task = QueuedTask::ad_hoc(format!("signal:{}", signal.topic));
        self.host
            .bind_harness_execution_context(&self.session, &trace_task);
        let result = {
            let harness = self
                .host
                .session_harness_engine(&self.session)
                .expect("session harness engine should be present after ensure");
            let engine = harness.lock().expect("session harness mutex poisoned");
            engine.dispatch_runtime_signal(crate::kernel::harness_contract::HarnessSignal {
                signal_id: uuid::Uuid::from_slice(&signal.public_id).ok(),
                topic: &signal.topic,
                source_agent_id: &signal.source_agent_id,
                target_agent_id: &signal.target_agent_id,
                source_session_id: signal.source_session_id.as_deref(),
                target_session_id: signal.target_session_id.as_deref(),
                payload: &signal.payload,
                created_at: &signal.created_at,
            })
        };
        self.host.unbind_harness_execution_context(&self.session);
        result
    }
}
