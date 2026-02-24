use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::kernel::Kernel;
use crate::kernel::event::TaskTerminalStatus;

use super::peer_task::run_peer_task;
use super::{AgentManager, AgentRuntimeHandle, PeerAgentTaskEnvelope, PeerAgentTaskResult};

impl AgentManager {
    pub(super) async fn ensure_runtime(&self, agent_id: &str) -> Result<Arc<AgentRuntimeHandle>> {
        {
            let runtimes = self.runtimes.read().await;
            if let Some(handle) = runtimes.get(agent_id)
                && handle.is_running()
            {
                return Ok(Arc::clone(handle));
            }
        }

        self.ensure_runtime_with_write_lock(agent_id).await
    }

    /// Internal method to boot a new Kernel and background loop for a specific agent profile.
    async fn start_agent(&self, agent_id: &str) -> Result<AgentRuntimeHandle> {
        info!(agent_id = %agent_id, "Starting background peer agent runtime");

        let agent_profile = if agent_id == self.config.agent.id {
            &self.config.agent
        } else {
            self.config
                .agents
                .get(agent_id)
                .ok_or_else(|| anyhow::anyhow!("Unknown agent profile: {}", agent_id))?
        };

        let mut peer_config = (*self.config).clone();
        peer_config.agent = agent_profile.clone();
        if let Some(harness_dir) = &agent_profile.harness_dir {
            peer_config.harness.directory = harness_dir.clone();
        }

        let mut kernel = Kernel::builder(peer_config).build()?;
        kernel.store_manager = self.store_manager.clone();
        kernel.init_clients()?;

        let (tx, mut rx) = mpsc::channel::<PeerAgentTaskEnvelope>(100);
        let queued_tasks = Arc::new(AtomicUsize::new(0));
        let agent_id_clone = agent_id.to_string();
        let idle_grace_secs = agent_profile.idle_grace_secs;

        let queued_tasks_bg = queued_tasks.clone();
        let join_handle = tokio::spawn(async move {
            debug!(agent_id = %agent_id_clone, "Peer agent loop initializing");

            if let Err(e) = kernel.init_harness().await {
                error!(agent_id = %agent_id_clone, error = %e, "Peer agent failed to initialize harness");
                return;
            }

            let mut session = kernel.create_session().await;
            if let Err(e) = kernel.start_session(&mut session).await {
                error!(agent_id = %agent_id_clone, error = %e, "Peer agent failed to start session");
                return;
            }

            info!(agent_id = %agent_id_clone, "Peer agent loop ready for tasks");

            loop {
                let envelope = if let Some(idle_secs) = idle_grace_secs {
                    match tokio::time::timeout(std::time::Duration::from_secs(idle_secs), rx.recv())
                        .await
                    {
                        Ok(maybe) => maybe,
                        Err(_) => {
                            info!(
                                agent_id = %agent_id_clone,
                                idle_grace_secs = idle_secs,
                                "Peer agent idle timeout reached; shutting down runtime"
                            );
                            break;
                        }
                    }
                } else {
                    rx.recv().await
                };

                let Some(envelope) = envelope else {
                    break;
                };
                queued_tasks_bg.fetch_sub(1, Ordering::Relaxed);
                let result = run_peer_task(&mut kernel, &mut session, envelope.task).await;

                if let Some(tx_result) = envelope.result_tx {
                    let request_id = envelope
                        .request_id
                        .unwrap_or_else(|| uuid::Uuid::now_v7().simple().to_string());
                    let _ = tx_result.send(match result {
                        Ok(ok) => PeerAgentTaskResult {
                            request_id,
                            agent_id: agent_id_clone.clone(),
                            runtime_task_id: ok.runtime_task_id,
                            status: ok.status,
                            task_turn_count: ok.task_turn_count,
                            output: ok.output,
                            error: None,
                        },
                        Err(e) => PeerAgentTaskResult {
                            request_id,
                            agent_id: agent_id_clone.clone(),
                            runtime_task_id: String::new(),
                            status: TaskTerminalStatus::Error,
                            task_turn_count: 0,
                            output: None,
                            error: Some(e.to_string()),
                        },
                    });
                } else if let Err(e) = result {
                    error!(agent_id = %agent_id_clone, error = %e, "Peer agent task failed");
                }
            }

            info!(agent_id = %agent_id_clone, "Peer agent queue closed, terminating runtime");

            if let Err(e) = kernel.end_session(&mut session).await {
                warn!(agent_id = %agent_id_clone, error = %e, "Peer agent session end error");
            }
        });

        Ok(AgentRuntimeHandle {
            tx,
            task: Some(join_handle),
            queued_tasks,
        })
    }

    async fn ensure_runtime_with_write_lock(
        &self,
        agent_id: &str,
    ) -> Result<Arc<AgentRuntimeHandle>> {
        let mut runtimes = self.runtimes.write().await;
        if let Some(handle) = runtimes.get(agent_id)
            && handle.is_running()
        {
            return Ok(Arc::clone(handle));
        }

        let handle = Arc::new(self.start_agent(agent_id).await?);
        runtimes.insert(agent_id.to_string(), Arc::clone(&handle));
        Ok(handle)
    }
}
