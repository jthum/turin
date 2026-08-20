use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use serde_json::Value;
use tokio::sync::{Notify, oneshot};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};

use crate::kernel::policy::PolicyScope;
use crate::kernel::session_refs::parse_session_reference;
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::LinkedSessionCreate;

use super::lane_scheduler::pop_fair_task;
use super::peer_runtime::PeerRuntime;
use super::peer_session::SessionBootstrap;
use super::{
    AgentManager, AgentRuntimeHandle, PeerAgentTaskEnvelope, RuntimeControl, RuntimeSlotKey,
    SessionContextOverrides,
};

impl AgentManager {
    /// Boot a background worker for a specific agent runtime slot.
    pub(super) async fn start_agent(
        self: &Arc<Self>,
        runtime_key: &RuntimeSlotKey,
        initial_session_id: Option<&str>,
        initial_state_selector: Option<StoreSelector>,
        initial_default_store_selector: Option<StoreSelector>,
        session_context: SessionContextOverrides,
        initial_link: Option<LinkedSessionCreate>,
    ) -> Result<AgentRuntimeHandle> {
        let config = self.config_snapshot();
        let agent_id = runtime_key.agent_id.as_str();
        info!(
            agent_id = %agent_id,
            slot_id = %runtime_key.slot_id,
            "Starting background peer agent runtime"
        );

        if agent_id != config.agent.id && !config.agents.contains_key(agent_id) {
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

            // Close task admission before the final drain. A submitter that retained this
            // handle either enqueues before cancellation and is drained below, or observes
            // cancellation while holding the queue lock and fails submission.
            shutdown_bg.cancel();
            manager
                .cancel_queued_requests_for_runtime(
                    &RuntimeSlotKey {
                        agent_id: agent_id_clone.clone(),
                        slot_id: slot_id_clone.clone(),
                    },
                    "Runtime stopped before task execution",
                )
                .await;
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
        let config = self.config_snapshot();
        let mut effective = if agent_id == config.agent.id {
            config.agent.idle_timeout_seconds
        } else {
            config
                .agents
                .get(agent_id)
                .map(|agent| agent.idle_timeout_seconds)
                .unwrap_or(config.agent.idle_timeout_seconds)
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
}
