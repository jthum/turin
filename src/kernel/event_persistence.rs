use tokio::sync::broadcast;
use tracing::{debug, instrument, warn};

use crate::kernel::event::KernelEvent;
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::session::SessionState;
use crate::kernel::session::{PersistedKernelEvent, PersistedKernelRecord};

impl ExecutionHost {
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
                session,
                event,
            );
        }

        // Allow harness to observe/intercept any event.
        {
            if let Some(harness) = self.session_harness_engine(session) {
                let engine = harness.lock().expect("session harness mutex poisoned");
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
        }

        if !protected_audit_event {
            self.persist_event_internal(
                &session.event_tx,
                session.durability_tx.as_ref(),
                session,
                event,
            );
        }
    }

    /// Internal helper for persistence (used by parallel runners).
    pub(crate) fn persist_event_internal(
        &self,
        tx: &broadcast::Sender<(Option<i64>, KernelEvent)>,
        durability_tx: Option<&tokio::sync::mpsc::UnboundedSender<PersistedKernelRecord>>,
        session: &SessionState,
        event: &KernelEvent,
    ) {
        if self.json {
            // In JSON mode, all events go to stdout as NDJSON.
            println!("{}", serde_json::to_string(event).unwrap_or_default());
        }
        if tx.send((session.internal_id, event.clone())).is_err() {
            debug!("Event broadcast skipped — no active receivers");
        }
        let Some(record) = persisted_event_record(session, event) else {
            return;
        };
        if let Some(durability_tx) = durability_tx
            && durability_tx
                .send(PersistedKernelRecord::Event(Box::new(record)))
                .is_err()
        {
            warn!("Event durability send failed — persistence task unavailable");
        }
    }
}

fn persisted_event_record(
    session: &SessionState,
    event: &KernelEvent,
) -> Option<PersistedKernelEvent> {
    if matches!(event, KernelEvent::Ui(_)) {
        return None;
    }

    if let Some(target) = branch_scoped_persistence_target(session, event) {
        return Some(PersistedKernelEvent {
            internal_id: session.internal_id,
            turn_target: Some(target),
            event: event.clone(),
        });
    }

    if event_is_branch_scoped(event) {
        return None;
    }

    Some(PersistedKernelEvent {
        internal_id: session.internal_id,
        turn_target: None,
        event: event.clone(),
    })
}

fn branch_scoped_persistence_target(
    session: &SessionState,
    event: &KernelEvent,
) -> Option<crate::persistence::state::TurnWriteTarget> {
    if event_is_branch_scoped(event) {
        session.active_turn_write_target()
    } else {
        None
    }
}

fn event_is_branch_scoped(event: &KernelEvent) -> bool {
    match event {
        KernelEvent::Lifecycle(lifecycle) => matches!(
            lifecycle,
            crate::kernel::event::LifecycleEvent::TurnStart { .. }
                | crate::kernel::event::LifecycleEvent::TurnPrepare { .. }
                | crate::kernel::event::LifecycleEvent::TurnEnd { .. }
        ),
        KernelEvent::Stream(_) => true,
        KernelEvent::Ui(_) => false,
        KernelEvent::Audit(audit) => matches!(
            audit,
            crate::kernel::event::AuditEvent::ToolResult { .. }
                | crate::kernel::event::AuditEvent::ToolExecStart { .. }
                | crate::kernel::event::AuditEvent::ToolExecEnd { .. }
                | crate::kernel::event::AuditEvent::TokenUsage { .. }
                | crate::kernel::event::AuditEvent::GovernanceDenial { .. }
                | crate::kernel::event::AuditEvent::HarnessRejection { .. }
        ),
    }
}
