use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use serde_json::Value;
use tokio::sync::{Notify, oneshot};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};

use super::peer_runtime::{PeerRuntime, SessionBootstrap};
use super::{
    AgentManager, AgentRuntimeHandle, PeerAgentTaskEnvelope, RuntimeControl, RuntimeSlotKey,
    SessionContextOverrides,
};
use crate::kernel::policy::PolicyScope;
use crate::kernel::session_refs::{parse_session_reference, session_references_match};
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::LinkedSessionCreate;

#[derive(Debug, Clone, PartialEq, Eq)]
enum TaskSchedulingKey {
    Session(String),
    PendingLinked {
        store: Option<StoreSelector>,
        parent_session_id: i64,
        thread_key: String,
    },
    Runtime,
}

impl TaskSchedulingKey {
    fn from_envelope(envelope: &PeerAgentTaskEnvelope) -> Self {
        if let Some(session_id) = envelope.session_target.session_id.as_ref() {
            return Self::Session(session_id.clone());
        }
        match (
            envelope.session_target.linked_parent_session_id,
            envelope.session_target.thread_key.as_ref(),
        ) {
            (Some(parent_session_id), Some(thread_key)) => Self::PendingLinked {
                store: envelope.session_target.store_selector.clone(),
                parent_session_id,
                thread_key: thread_key.clone(),
            },
            _ => Self::Runtime,
        }
    }

    fn matches(&self, envelope: &PeerAgentTaskEnvelope) -> bool {
        match self {
            Self::Session(session_id) => {
                envelope.session_target.session_id.as_ref() == Some(session_id)
            }
            Self::PendingLinked {
                store,
                parent_session_id,
                thread_key,
            } => {
                envelope.session_target.session_id.is_none()
                    && envelope.session_target.store_selector.as_ref() == store.as_ref()
                    && envelope.session_target.linked_parent_session_id == Some(*parent_session_id)
                    && envelope.session_target.thread_key.as_ref() == Some(thread_key)
            }
            Self::Runtime => {
                envelope.session_target.session_id.is_none()
                    && (envelope.session_target.linked_parent_session_id.is_none()
                        || envelope.session_target.thread_key.is_none())
            }
        }
    }
}

fn pop_fair_task(
    queue: &mut VecDeque<PeerAgentTaskEnvelope>,
    last_scheduled: &mut Option<TaskSchedulingKey>,
) -> Option<PeerAgentTaskEnvelope> {
    let index = last_scheduled
        .as_ref()
        .and_then(|last| queue.iter().position(|envelope| !last.matches(envelope)));
    let envelope = match index {
        Some(index) => queue.remove(index),
        None => queue.pop_front(),
    }?;
    *last_scheduled = Some(TaskSchedulingKey::from_envelope(&envelope));
    Some(envelope)
}

impl AgentManager {
    pub(super) async fn ensure_runtime(
        self: &Arc<Self>,
        agent_id: &str,
    ) -> Result<Arc<AgentRuntimeHandle>> {
        self.ensure_runtime_slot(RuntimeSlotKey::default_for(agent_id))
            .await
    }

    pub(super) async fn ensure_runtime_slot(
        self: &Arc<Self>,
        runtime_key: RuntimeSlotKey,
    ) -> Result<Arc<AgentRuntimeHandle>> {
        self.ensure_runtime_slot_in_store(
            runtime_key,
            None,
            None,
            SessionContextOverrides::default(),
        )
        .await
    }

    pub(super) async fn ensure_runtime_slot_in_store(
        self: &Arc<Self>,
        runtime_key: RuntimeSlotKey,
        initial_state_selector: Option<StoreSelector>,
        initial_default_store_selector: Option<StoreSelector>,
        session_context: SessionContextOverrides,
    ) -> Result<Arc<AgentRuntimeHandle>> {
        if self.shutting_down.load(Ordering::Acquire) {
            anyhow::bail!("Agent manager is shutting down");
        }
        {
            let runtimes = self.runtimes.read().await;
            if let Some(handle) = runtimes.get(&runtime_key)
                && handle.is_running()
            {
                return Ok(Arc::clone(handle));
            }
        }

        self.ensure_runtime_with_write_lock(
            runtime_key,
            initial_state_selector,
            initial_default_store_selector,
            session_context,
        )
        .await
    }

    pub(super) async fn ensure_runtime_slot_resumed(
        self: &Arc<Self>,
        runtime_key: RuntimeSlotKey,
        session_id: String,
        session_context: SessionContextOverrides,
    ) -> Result<Arc<AgentRuntimeHandle>> {
        if self.shutting_down.load(Ordering::Acquire) {
            anyhow::bail!("Agent manager is shutting down");
        }
        {
            let runtimes = self.runtimes.read().await;
            if let Some(handle) = runtimes.get(&runtime_key)
                && handle.is_running()
            {
                if handle
                    .control
                    .current_session_id()
                    .as_deref()
                    .is_some_and(|current| session_references_match(current, &session_id))
                {
                    return Ok(Arc::clone(handle));
                }
                handle
                    .control
                    .request_session_resume(session_id, session_context.clone());
                handle.notify.notify_one();
                return Ok(Arc::clone(handle));
            }
        }

        self.ensure_runtime_with_write_lock_and_resume(
            runtime_key,
            Some(session_id),
            session_context,
        )
        .await
    }

    pub(super) async fn ensure_runtime_slot_for_linked(
        self: &Arc<Self>,
        runtime_key: RuntimeSlotKey,
        initial_session_id: Option<&str>,
        state_selector: StoreSelector,
        default_store_selector: Option<StoreSelector>,
        session_context: SessionContextOverrides,
        link: LinkedSessionCreate,
    ) -> Result<Arc<AgentRuntimeHandle>> {
        let mut runtimes = self.runtimes.write().await;
        if self.shutting_down.load(Ordering::Acquire) {
            anyhow::bail!("Agent manager is shutting down");
        }
        if let Some(handle) = runtimes.get(&runtime_key)
            && handle.is_running()
        {
            return Ok(Arc::clone(handle));
        }

        let (initial_state_selector, initial_link) = if initial_session_id.is_some() {
            (None, None)
        } else {
            (Some(state_selector), Some(link))
        };
        let handle = Arc::new(
            self.start_agent(
                &runtime_key,
                initial_session_id,
                initial_state_selector,
                default_store_selector,
                session_context,
                initial_link,
            )
            .await?,
        );
        runtimes.insert(runtime_key, Arc::clone(&handle));
        Ok(handle)
    }

    /// Internal method to boot a background peer runtime for a specific agent profile.
    async fn start_agent(
        self: &Arc<Self>,
        runtime_key: &RuntimeSlotKey,
        initial_session_id: Option<&str>,
        initial_state_selector: Option<StoreSelector>,
        initial_default_store_selector: Option<StoreSelector>,
        session_context: SessionContextOverrides,
        initial_link: Option<LinkedSessionCreate>,
    ) -> Result<AgentRuntimeHandle> {
        let agent_id = runtime_key.agent_id.as_str();
        info!(
            agent_id = %agent_id,
            slot_id = %runtime_key.slot_id,
            "Starting background peer agent runtime"
        );

        if agent_id != self.config.agent.id && !self.config.agents.contains_key(agent_id) {
            return Err(anyhow::anyhow!("Unknown agent profile: {}", agent_id));
        }

        let queue = Arc::new(std::sync::Mutex::new(
            VecDeque::<PeerAgentTaskEnvelope>::new(),
        ));
        let notify = Arc::new(Notify::new());
        let control = Arc::new(RuntimeControl::default());
        let shutdown_token = CancellationToken::new();
        let queued_tasks = Arc::new(AtomicUsize::new(0));
        let active_tasks = Arc::new(AtomicUsize::new(0));
        let agent_id_clone = agent_id.to_string();
        let slot_id_clone = runtime_key.slot_id.clone();
        let initial_session_id = initial_session_id.map(str::to_string);
        let manager = Arc::clone(self);
        let queue_bg = Arc::clone(&queue);
        let notify_bg = Arc::clone(&notify);

        let queued_tasks_bg = queued_tasks.clone();
        let active_tasks_bg = active_tasks.clone();
        let control_bg = Arc::clone(&control);
        let idle_control = Arc::clone(&control);
        let shutdown_bg = shutdown_token.clone();
        let (startup_tx, startup_rx) = oneshot::channel::<std::result::Result<(), String>>();
        let join_handle = tokio::spawn(async move {
            debug!(agent_id = %agent_id_clone, slot_id = %slot_id_clone, "Peer agent loop initializing");

            let mut runtime = match PeerRuntime::start(
                manager.clone(),
                &agent_id_clone,
                &slot_id_clone,
                control_bg,
                SessionBootstrap {
                    initial_session_id: initial_session_id.clone(),
                    initial_state_selector,
                    initial_default_store_selector,
                    context: session_context,
                    link: initial_link,
                },
            )
            .await
            {
                Ok(runtime) => {
                    let _ = startup_tx.send(Ok(()));
                    runtime
                }
                Err(e) => {
                    let message = e.to_string();
                    let _ = startup_tx.send(Err(message));
                    error!(agent_id = %agent_id_clone, slot_id = %slot_id_clone, error = %e, "Peer agent failed to start session");
                    return;
                }
            };

            info!(agent_id = %agent_id_clone, slot_id = %slot_id_clone, "Peer agent loop ready for tasks");

            let mut processed_task = false;
            let mut last_scheduled = None;
            loop {
                if shutdown_bg.is_cancelled() {
                    break;
                }
                let envelope = {
                    let mut queue = queue_bg.lock().expect("agent runtime queue mutex poisoned");
                    pop_fair_task(&mut queue, &mut last_scheduled)
                };
                let Some(envelope) = envelope else {
                    match runtime.reset_session_if_requested().await {
                        Ok(true) => continue,
                        Ok(false) => {}
                        Err(err) => {
                            error!(
                                agent_id = %agent_id_clone,
                                slot_id = %slot_id_clone,
                                error = %err,
                                "Peer agent failed to reset session"
                            );
                            break;
                        }
                    }
                    match runtime.process_pending_signals().await {
                        Ok(processed) if processed > 0 => continue,
                        Ok(_) => {}
                        Err(err) => {
                            error!(
                                agent_id = %agent_id_clone,
                                slot_id = %slot_id_clone,
                                error = %err,
                                "Peer agent failed to process pending signal deliveries"
                            );
                        }
                    }
                    if !processed_task {
                        tokio::select! {
                            _ = shutdown_bg.cancelled() => break,
                            _ = notify_bg.notified() => {}
                        }
                        continue;
                    }
                    let idle_timeout_seconds = manager
                        .resolve_idle_timeout_seconds(
                            &agent_id_clone,
                            idle_control.current_session_id().as_deref(),
                        )
                        .await;
                    if let Some(idle_timeout_seconds) = idle_timeout_seconds {
                        let idle_timeout = if idle_timeout_seconds == 0 {
                            std::time::Duration::from_millis(1)
                        } else {
                            std::time::Duration::from_secs(idle_timeout_seconds)
                        };
                        tokio::select! {
                            _ = shutdown_bg.cancelled() => break,
                            _ = notify_bg.notified() => {}
                            _ = tokio::time::sleep(idle_timeout) => {
                                info!(
                                    agent_id = %agent_id_clone,
                                    slot_id = %slot_id_clone,
                                    idle_timeout_seconds,
                                    "Peer agent idle timeout reached; shutting down runtime"
                                );
                                break;
                            }
                        }
                    } else {
                        tokio::select! {
                            _ = shutdown_bg.cancelled() => break,
                            _ = notify_bg.notified() => {}
                        }
                    }
                    continue;
                };
                queued_tasks_bg.fetch_sub(1, Ordering::Relaxed);
                active_tasks_bg.fetch_add(1, Ordering::Relaxed);
                runtime.handle_envelope(envelope).await;
                processed_task = true;
                active_tasks_bg.fetch_sub(1, Ordering::Relaxed);
            }

            info!(agent_id = %agent_id_clone, slot_id = %slot_id_clone, "Peer agent loop terminating runtime");

            runtime.shutdown().await;
        });

        match startup_rx.await {
            Ok(Ok(())) => {}
            Ok(Err(message)) => {
                let _ = join_handle.await;
                anyhow::bail!(
                    "Peer agent '{}' [{}] failed to start: {}",
                    agent_id,
                    runtime_key.slot_id,
                    message
                );
            }
            Err(_) => {
                let join_error = join_handle.await.err();
                anyhow::bail!(
                    "Peer agent '{}' [{}] exited before reporting startup{}",
                    agent_id,
                    runtime_key.slot_id,
                    join_error
                        .map(|error| format!(": {error}"))
                        .unwrap_or_default()
                );
            }
        }

        Ok(AgentRuntimeHandle {
            queue,
            notify,
            control,
            shutdown_token,
            task: Some(join_handle),
            queued_tasks,
            active_tasks,
        })
    }

    async fn resolve_idle_timeout_seconds(
        &self,
        agent_id: &str,
        current_session_id: Option<&str>,
    ) -> Option<u64> {
        let mut effective = if agent_id == self.config.agent.id {
            self.config.agent.idle_timeout_seconds
        } else {
            self.config
                .agents
                .get(agent_id)
                .map(|agent| agent.idle_timeout_seconds)
                .unwrap_or(self.config.agent.idle_timeout_seconds)
        };

        let session_public_id = current_session_id.and_then(|raw| {
            parse_session_reference(raw)
                .ok()
                .map(|session_ref| session_ref.public_id)
        });
        let scope = PolicyScope {
            agent_id: Some(agent_id.to_string()),
            session_id: session_public_id,
            ..PolicyScope::default()
        };

        if let Some(shared_runtime) = self.shared_runtime()
            && let Ok(Some(value)) = shared_runtime
                .policy_manager
                .get("runtime.idle_timeout_seconds", &scope)
                .await
        {
            effective = match value {
                Value::Null => None,
                Value::Number(number) => number.as_u64(),
                _ => effective,
            };
        }

        effective
    }

    async fn ensure_runtime_with_write_lock(
        self: &Arc<Self>,
        runtime_key: RuntimeSlotKey,
        initial_state_selector: Option<StoreSelector>,
        initial_default_store_selector: Option<StoreSelector>,
        session_context: SessionContextOverrides,
    ) -> Result<Arc<AgentRuntimeHandle>> {
        let mut runtimes = self.runtimes.write().await;
        if self.shutting_down.load(Ordering::Acquire) {
            anyhow::bail!("Agent manager is shutting down");
        }
        if let Some(handle) = runtimes.get(&runtime_key)
            && handle.is_running()
        {
            return Ok(Arc::clone(handle));
        }

        let handle = Arc::new(
            self.start_agent(
                &runtime_key,
                None,
                initial_state_selector,
                initial_default_store_selector,
                session_context,
                None,
            )
            .await?,
        );
        runtimes.insert(runtime_key, Arc::clone(&handle));
        Ok(handle)
    }

    async fn ensure_runtime_with_write_lock_and_resume(
        self: &Arc<Self>,
        runtime_key: RuntimeSlotKey,
        initial_session_id: Option<String>,
        session_context: SessionContextOverrides,
    ) -> Result<Arc<AgentRuntimeHandle>> {
        let mut runtimes = self.runtimes.write().await;
        if self.shutting_down.load(Ordering::Acquire) {
            anyhow::bail!("Agent manager is shutting down");
        }
        if let Some(handle) = runtimes.get(&runtime_key)
            && handle.is_running()
        {
            if let Some(session_id) = initial_session_id {
                handle
                    .control
                    .request_session_resume(session_id, session_context.clone());
                handle.notify.notify_one();
            }
            return Ok(Arc::clone(handle));
        }

        let handle = Arc::new(
            self.start_agent(
                &runtime_key,
                initial_session_id.as_deref(),
                None,
                None,
                session_context,
                None,
            )
            .await?,
        );
        runtimes.insert(runtime_key, Arc::clone(&handle));
        Ok(handle)
    }
}

#[cfg(test)]
mod scheduling_tests {
    use super::*;
    use crate::kernel::session::QueuedTask;

    fn envelope(request_id: &str, session_id: &str) -> PeerAgentTaskEnvelope {
        PeerAgentTaskEnvelope {
            task: QueuedTask::ad_hoc(request_id),
            request_id: Some(request_id.to_string()),
            result_tx: None,
            delegated_capabilities: None,
            promotion_candidate: None,
            linked_session: None,
            session_target: super::super::TaskSessionTarget {
                session_id: Some(session_id.to_string()),
                ..super::super::TaskSessionTarget::default()
            },
        }
    }

    #[test]
    fn lane_scheduler_rotates_sessions_and_preserves_per_session_fifo() {
        let mut queue = VecDeque::from([
            envelope("a1", "session-a"),
            envelope("a2", "session-a"),
            envelope("b1", "session-b"),
            envelope("b2", "session-b"),
            envelope("a3", "session-a"),
        ]);
        let mut last = None;
        let mut order = Vec::new();
        while let Some(envelope) = pop_fair_task(&mut queue, &mut last) {
            order.push(envelope.request_id.expect("request id"));
        }
        assert_eq!(order, ["a1", "b1", "a2", "b2", "a3"]);
    }
}
