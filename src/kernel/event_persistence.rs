use tokio::sync::broadcast;
use tracing::{debug, instrument, warn};

use crate::kernel::Kernel;
use crate::kernel::event::KernelEvent;
use crate::kernel::session::SessionState;

impl Kernel {
    /// Persist an event to the state store in the background.
    #[instrument(skip(self, session, event), fields(event_type = %event.event_type()))]
    pub(crate) fn persist_event(&self, session: &SessionState, event: &KernelEvent) {
        let audit_persist_before_hooks = self
            .governance_manager
            .config()
            .audit
            .persist_before_hooks
            .unwrap_or(matches!(
                self.governance_manager.config().audit.mode,
                crate::kernel::config::GovernanceAuditMode::Immutable
            ));
        let protected_audit_event =
            audit_persist_before_hooks && matches!(event, KernelEvent::Audit(_));

        if protected_audit_event {
            self.persist_event_internal(
                &session.event_tx,
                session.durability_tx.as_ref(),
                session.internal_id,
                event,
            );
        }

        // Allow harness to observe/intercept any event.
        if let Ok(harness_guard) = self.harness.lock()
            && let Some(engine) = &*harness_guard
        {
            let payload = serde_json::to_value(event).unwrap_or_default();
            if let Ok(verdict) = engine.evaluate("on_kernel_event", payload)
                && verdict.is_rejected()
            {
                if protected_audit_event {
                    warn!(
                        event_type = %event.event_type(),
                        "Event REJECTED by harness on_kernel_event but already persisted (immutable audit)"
                    );
                } else {
                    warn!(event_type = %event.event_type(), "Event REJECTED by harness on_kernel_event");
                    return;
                }
            }
            // NOTE: MODIFY is intentionally ignored for general events for now.
        }

        if !protected_audit_event {
            self.persist_event_internal(
                &session.event_tx,
                session.durability_tx.as_ref(),
                session.internal_id,
                event,
            );
        }
    }

    /// Internal helper for persistence (used by parallel runners).
    pub(crate) fn persist_event_internal(
        &self,
        tx: &broadcast::Sender<(Option<i64>, KernelEvent)>,
        durability_tx: Option<&tokio::sync::mpsc::UnboundedSender<(Option<i64>, KernelEvent)>>,
        internal_id: Option<i64>,
        event: &KernelEvent,
    ) {
        if self.json {
            // In JSON mode, all events go to stdout as NDJSON.
            println!("{}", serde_json::to_string(event).unwrap_or_default());
        }
        if tx.send((internal_id, event.clone())).is_err() {
            debug!("Event broadcast skipped — no active receivers");
        }
        if let Some(durability_tx) = durability_tx
            && durability_tx.send((internal_id, event.clone())).is_err()
        {
            warn!("Event durability send failed — persistence task unavailable");
        }
    }
}
