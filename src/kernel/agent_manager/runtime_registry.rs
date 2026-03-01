use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use tokio::sync::mpsc;
use tracing::{debug, error, info};

use super::peer_runtime::PeerRuntime;
use super::{AgentManager, AgentRuntimeHandle, PeerAgentTaskEnvelope};

impl AgentManager {
    pub(super) async fn ensure_runtime(
        self: &Arc<Self>,
        agent_id: &str,
    ) -> Result<Arc<AgentRuntimeHandle>> {
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

    /// Internal method to boot a background peer runtime for a specific agent profile.
    async fn start_agent(self: &Arc<Self>, agent_id: &str) -> Result<AgentRuntimeHandle> {
        info!(agent_id = %agent_id, "Starting background peer agent runtime");

        let agent_profile = if agent_id == self.config.agent.id {
            &self.config.agent
        } else {
            self.config
                .agents
                .get(agent_id)
                .ok_or_else(|| anyhow::anyhow!("Unknown agent profile: {}", agent_id))?
        };

        let (tx, mut rx) = mpsc::channel::<PeerAgentTaskEnvelope>(100);
        let queued_tasks = Arc::new(AtomicUsize::new(0));
        let agent_id_clone = agent_id.to_string();
        let idle_grace_secs = agent_profile.idle_grace_secs;
        let manager = Arc::clone(self);

        let queued_tasks_bg = queued_tasks.clone();
        let join_handle = tokio::spawn(async move {
            debug!(agent_id = %agent_id_clone, "Peer agent loop initializing");

            let mut runtime = match PeerRuntime::start(manager.clone(), &agent_id_clone).await {
                Ok(runtime) => runtime,
                Err(e) => {
                    error!(agent_id = %agent_id_clone, error = %e, "Peer agent failed to start session");
                    return;
                }
            };

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
                runtime.handle_envelope(envelope).await;
            }

            info!(agent_id = %agent_id_clone, "Peer agent queue closed, terminating runtime");

            runtime.shutdown().await;
        });

        Ok(AgentRuntimeHandle {
            tx,
            task: Some(join_handle),
            queued_tasks,
        })
    }

    async fn ensure_runtime_with_write_lock(
        self: &Arc<Self>,
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
