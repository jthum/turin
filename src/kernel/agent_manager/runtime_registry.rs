use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use serde_json::Value;
use tokio::sync::Notify;
use tracing::{debug, error, info};

use super::peer_runtime::{PeerRuntime, SessionBootstrap};
use super::{
    AgentManager, AgentRuntimeHandle, PeerAgentTaskEnvelope, RuntimeControl, RuntimeSlotKey,
    SessionContextOverrides,
};
use crate::kernel::policy::PolicyScope;
use crate::kernel::session_refs::parse_session_reference;
use crate::persistence::manager::StoreSelector;

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
        {
            let runtimes = self.runtimes.read().await;
            if let Some(handle) = runtimes.get(&runtime_key)
                && handle.is_running()
            {
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

    /// Internal method to boot a background peer runtime for a specific agent profile.
    async fn start_agent(
        self: &Arc<Self>,
        runtime_key: &RuntimeSlotKey,
        initial_session_id: Option<&str>,
        initial_state_selector: Option<StoreSelector>,
        initial_default_store_selector: Option<StoreSelector>,
        session_context: SessionContextOverrides,
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
                },
            )
            .await
            {
                Ok(runtime) => runtime,
                Err(e) => {
                    error!(agent_id = %agent_id_clone, slot_id = %slot_id_clone, error = %e, "Peer agent failed to start session");
                    return;
                }
            };

            info!(agent_id = %agent_id_clone, slot_id = %slot_id_clone, "Peer agent loop ready for tasks");

            let mut processed_task = false;
            loop {
                let envelope = {
                    let mut queue = queue_bg.lock().expect("agent runtime queue mutex poisoned");
                    queue.pop_front()
                };
                let Some(mut envelope) = envelope else {
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
                        notify_bg.notified().await;
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
                        let notified =
                            tokio::time::timeout(idle_timeout, notify_bg.notified()).await;
                        if notified.is_err() {
                            info!(
                                agent_id = %agent_id_clone,
                                slot_id = %slot_id_clone,
                                idle_timeout_seconds,
                                "Peer agent idle timeout reached; shutting down runtime"
                            );
                            break;
                        }
                    } else {
                        notify_bg.notified().await;
                    }
                    continue;
                };
                queued_tasks_bg.fetch_sub(1, Ordering::Relaxed);
                active_tasks_bg.fetch_add(1, Ordering::Relaxed);
                if let Some(request_id) = envelope.request_id.as_deref() {
                    let runtime_task_id = runtime.allocate_runtime_task_id(&mut envelope.task);
                    manager.mark_task_running(request_id, runtime_task_id).await;
                }
                runtime.handle_envelope(envelope).await;
                processed_task = true;
                active_tasks_bg.fetch_sub(1, Ordering::Relaxed);
            }

            info!(agent_id = %agent_id_clone, slot_id = %slot_id_clone, "Peer agent loop terminating runtime");

            runtime.shutdown().await;
        });

        Ok(AgentRuntimeHandle {
            queue,
            notify,
            control,
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
            )
            .await?,
        );
        runtimes.insert(runtime_key, Arc::clone(&handle));
        Ok(handle)
    }
}
