use anyhow::Result;
use tracing::{error, info, warn};

use crate::kernel::event::TaskTerminalStatus;

use super::peer_task::run_peer_task;
use super::{AgentManager, PeerAgentTaskEnvelope, PeerAgentTaskResult};

pub(super) struct PeerRuntime {
    kernel: crate::kernel::Kernel,
    session: crate::kernel::session::SessionState,
    agent_id: String,
}

impl PeerRuntime {
    pub(super) async fn start(manager: &AgentManager, agent_id: &str) -> Result<Self> {
        let mut kernel = manager.build_shared_peer_kernel()?;
        if kernel.clients.is_empty() {
            kernel.init_clients()?;
        }

        let mut session = kernel.create_session_for_agent(agent_id).await;
        kernel.start_session(&mut session).await?;

        Ok(Self {
            kernel,
            session,
            agent_id: agent_id.to_string(),
        })
    }

    pub(super) async fn handle_envelope(&mut self, envelope: PeerAgentTaskEnvelope) {
        let result = run_peer_task(
            &mut self.kernel,
            &mut self.session,
            envelope.task,
            envelope.delegated_capabilities,
        )
        .await;

        if let Some(tx_result) = envelope.result_tx {
            let request_id = envelope
                .request_id
                .unwrap_or_else(|| uuid::Uuid::now_v7().simple().to_string());
            let _ = tx_result.send(match result {
                Ok(ok) => PeerAgentTaskResult {
                    request_id,
                    agent_id: self.agent_id.clone(),
                    runtime_task_id: ok.runtime_task_id,
                    status: ok.status,
                    task_turn_count: ok.task_turn_count,
                    output: ok.output,
                    error: None,
                },
                Err(e) => PeerAgentTaskResult {
                    request_id,
                    agent_id: self.agent_id.clone(),
                    runtime_task_id: String::new(),
                    status: TaskTerminalStatus::Error,
                    task_turn_count: 0,
                    output: None,
                    error: Some(e.to_string()),
                },
            });
        } else if let Err(e) = result {
            error!(agent_id = %self.agent_id, error = %e, "Peer agent task failed");
        }
    }

    pub(super) async fn shutdown(mut self) {
        if let Err(e) = self.kernel.end_session(&mut self.session).await {
            warn!(agent_id = %self.agent_id, error = %e, "Peer agent session end error");
        }
        self.kernel.shutdown_mcp_clients().await;
        info!(agent_id = %self.agent_id, "Peer runtime shut down");
    }
}
