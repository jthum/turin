use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::{RwLock, mpsc};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

use crate::kernel::config::TurinConfig;
use crate::kernel::session::QueuedTask;
use crate::kernel::Kernel;
use crate::persistence::manager::StoreManager;

/// A handle to a running peer agent.
pub struct AgentRuntimeHandle {
    /// The channel to send tasks to the background agent loop.
    pub tx: mpsc::Sender<QueuedTask>,
    /// The background task running the agent's event loop.
    pub task: Option<JoinHandle<()>>,
}

/// Orchestrates peer agents, spinning them up on demand and routing tasks to their independent runtimes.
pub struct AgentManager {
    /// The full configuration, used to look up agent profiles and instantiate kernels.
    config: Arc<TurinConfig>,
    /// Reference to the shared StoreManager for database operations.
    store_manager: Arc<StoreManager>,
    /// The list of active, running agent handles, keyed by agent_id.
    runtimes: RwLock<HashMap<String, Arc<AgentRuntimeHandle>>>,
}

impl AgentManager {
    /// Create a new AgentManager.
    pub fn new(config: Arc<TurinConfig>, store_manager: Arc<StoreManager>) -> Self {
        Self {
            config,
            store_manager,
            runtimes: RwLock::new(HashMap::new()),
        }
    }

    /// Dispatch a task to an agent by ID. If the agent isn't running, it will be started automatically.
    pub async fn send(&self, agent_id: &str, task: QueuedTask) -> Result<()> {
        let tx = {
            // Fast path: check if it's already running
            let runtimes = self.runtimes.read().await;
            if let Some(handle) = runtimes.get(agent_id) {
                handle.tx.clone()
            } else {
                // Drop read lock to acquire write lock
                drop(runtimes);
                
                let mut runtimes_write = self.runtimes.write().await;
                // Double-checked locking
                if let Some(handle) = runtimes_write.get(agent_id) {
                    handle.tx.clone()
                } else {
                    // Start the agent
                    let handle = self.start_agent(agent_id).await?;
                    let tx = handle.tx.clone();
                    runtimes_write.insert(agent_id.to_string(), Arc::new(handle));
                    tx
                }
            }
        };

        // Send the task
        tx.send(task).await.map_err(|e| anyhow::anyhow!("Failed to route task to agent queue: {}", e))
    }

    /// Internal method to boot a new Kernel and background loop for a specific agent profile.
    async fn start_agent(&self, agent_id: &str) -> Result<AgentRuntimeHandle> {
        info!(agent_id = %agent_id, "Starting background peer agent runtime");

        let agent_profile = if agent_id == self.config.agent.id {
            &self.config.agent
        } else {
            self.config.agents.get(agent_id).ok_or_else(|| {
                anyhow::anyhow!("Unknown agent profile: {}", agent_id)
            })?
        };

        // Clone the config and overwrite the active `agent` profile with this specific peer's configuration
        let mut peer_config = (*self.config).clone();
        peer_config.agent = agent_profile.clone();

        // Instantiate the Kernel for this peer
        let mut kernel = Kernel::builder(peer_config).build()?;
        
        // Pass the shared store_manager into the peer Kernel
        kernel.store_manager = self.store_manager.clone();

        // Initialize dependencies required for the background loop
        kernel.init_clients()?;
        // Note: Harness init requires async. We do this inside the background task to avoid blocking the caller.

        let (tx, mut rx) = mpsc::channel::<QueuedTask>(100);

        let agent_id_clone = agent_id.to_string();

        let join_handle = tokio::spawn(async move {
            debug!(agent_id = %agent_id_clone, "Peer agent loop initializing");

            if let Err(e) = kernel.init_harness().await {
                error!(agent_id = %agent_id_clone, error = %e, "Peer agent failed to initialize harness");
                return;
            }

            // Create a long-lived session for this peer agent to handle incoming tasks
            let mut session = kernel.create_session().await;
            if let Err(e) = kernel.start_session(&mut session).await {
                error!(agent_id = %agent_id_clone, error = %e, "Peer agent failed to start session");
                return;
            }

            info!(agent_id = %agent_id_clone, "Peer agent loop ready for tasks");

            // Consume tasks from the channel
            while let Some(task) = rx.recv().await {
                debug!(agent_id = %agent_id_clone, task_id = %task.task_id, "Peer agent received task");
                if let Err(e) = kernel.run_task(&mut session, &task).await {
                    error!(agent_id = %agent_id_clone, task_id = %task.task_id, error = %e, "Peer agent task failed");
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
        })
    }
}
