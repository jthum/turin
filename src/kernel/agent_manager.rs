use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use tokio::sync::{RwLock, mpsc, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

use crate::inference::provider::InferenceContent;
use crate::kernel::config::TurinConfig;
use crate::kernel::event::{KernelEvent, LifecycleEvent, TaskTerminalStatus};
use crate::kernel::session::{QueuedTask, SessionState};
use crate::kernel::Kernel;
use crate::persistence::manager::StoreManager;
use crate::harness::verdict::Verdict;

#[derive(Debug, Clone, serde::Serialize)]
pub struct PeerAgentTaskResult {
    pub request_id: String,
    pub agent_id: String,
    pub runtime_task_id: String,
    pub status: TaskTerminalStatus,
    pub task_turn_count: u32,
    pub output: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AgentStatusSnapshot {
    pub agent_id: String,
    pub running: bool,
    pub queued_tasks: usize,
    pub awaiting_results: usize,
}

struct PeerAgentTaskEnvelope {
    task: QueuedTask,
    request_id: Option<String>,
    result_tx: Option<oneshot::Sender<PeerAgentTaskResult>>,
}

/// A handle to a running peer agent.
pub struct AgentRuntimeHandle {
    /// The channel to send tasks to the background agent loop.
    tx: mpsc::Sender<PeerAgentTaskEnvelope>,
    /// The background task running the agent's event loop.
    task: Option<JoinHandle<()>>,
    /// Approximate number of tasks currently queued in the runtime channel.
    queued_tasks: Arc<AtomicUsize>,
}

/// Orchestrates peer agents, spinning them up on demand and routing tasks to their independent runtimes.
pub struct AgentManager {
    /// The full configuration, used to look up agent profiles and instantiate kernels.
    config: Arc<TurinConfig>,
    /// Reference to the shared StoreManager for database operations.
    store_manager: Arc<StoreManager>,
    /// The list of active, running agent handles, keyed by agent_id.
    runtimes: RwLock<HashMap<String, Arc<AgentRuntimeHandle>>>,
    /// Task result receivers keyed by runtime request id (consumed by `await_result`).
    pending_results: RwLock<HashMap<String, oneshot::Receiver<PeerAgentTaskResult>>>,
    /// Mapping of request id -> agent id for status accounting.
    pending_result_agents: RwLock<HashMap<String, String>>,
}

impl AgentManager {
    /// Create a new AgentManager.
    pub fn new(config: Arc<TurinConfig>, store_manager: Arc<StoreManager>) -> Self {
        Self {
            config,
            store_manager,
            runtimes: RwLock::new(HashMap::new()),
            pending_results: RwLock::new(HashMap::new()),
            pending_result_agents: RwLock::new(HashMap::new()),
        }
    }

    /// Dispatch a task to an agent by ID. If the agent isn't running, it will be started automatically.
    pub async fn send(&self, agent_id: &str, task: QueuedTask) -> Result<()> {
        let handle = self.ensure_runtime(agent_id).await?;
        handle.queued_tasks.fetch_add(1, Ordering::Relaxed);
        if let Err(e) = handle.tx.send(PeerAgentTaskEnvelope {
            task,
            request_id: None,
            result_tx: None,
        })
        .await
        {
            handle.queued_tasks.fetch_sub(1, Ordering::Relaxed);
            anyhow::bail!("Failed to route task to agent queue: {}", e);
        }
        Ok(())
    }

    /// Submit a task to a peer agent and return a request ID for later `await_result`.
    pub async fn submit(&self, agent_id: &str, task: QueuedTask) -> Result<String> {
        let request_id = uuid::Uuid::now_v7().simple().to_string();
        let (tx_result, rx_result) = oneshot::channel();
        {
            let mut pending = self.pending_results.write().await;
            pending.insert(request_id.clone(), rx_result);
        }
        {
            let mut map = self.pending_result_agents.write().await;
            map.insert(request_id.clone(), agent_id.to_string());
        }

        let handle = match self.ensure_runtime(agent_id).await {
            Ok(handle) => handle,
            Err(e) => {
                self.pending_results.write().await.remove(&request_id);
                self.pending_result_agents.write().await.remove(&request_id);
                return Err(e);
            }
        };

        handle.queued_tasks.fetch_add(1, Ordering::Relaxed);
        if let Err(e) = handle.tx
            .send(PeerAgentTaskEnvelope {
                task,
                request_id: Some(request_id.clone()),
                result_tx: Some(tx_result),
            })
            .await
        {
            handle.queued_tasks.fetch_sub(1, Ordering::Relaxed);
            self.pending_results.write().await.remove(&request_id);
            self.pending_result_agents.write().await.remove(&request_id);
            anyhow::bail!("Failed to route task to agent queue: {}", e);
        }

        Ok(request_id)
    }

    /// Await a previously submitted peer task result.
    pub async fn await_result(
        &self,
        request_id: &str,
        timeout_ms: Option<u64>,
    ) -> Result<PeerAgentTaskResult> {
        let mut rx = {
            let mut pending = self.pending_results.write().await;
            pending
                .remove(request_id)
                .ok_or_else(|| anyhow::anyhow!("Unknown or already-awaited peer task '{}'", request_id))?
        };

        let mut timed_out = false;
        let result = if let Some(ms) = timeout_ms {
            match tokio::time::timeout(std::time::Duration::from_millis(ms), &mut rx).await {
                Ok(Ok(res)) => Ok(res),
                Ok(Err(_)) => Err(anyhow::anyhow!("Peer task '{}' result channel closed", request_id)),
                Err(_) => {
                    timed_out = true;
                    self.pending_results
                        .write()
                        .await
                        .insert(request_id.to_string(), rx);
                    Err(anyhow::anyhow!("Timed out waiting for peer task '{}'", request_id))
                }
            }
        } else {
            match rx.await {
                Ok(res) => Ok(res),
                Err(_) => Err(anyhow::anyhow!("Peer task '{}' result channel closed", request_id)),
            }
        };

        if !timed_out {
            self.pending_result_agents.write().await.remove(request_id);
        }

        result
    }

    /// List configured agents with runtime status.
    pub async fn list_statuses(&self) -> Vec<AgentStatusSnapshot> {
        let runtimes = self.runtimes.read().await;
        let pending = self.pending_result_agents.read().await;

        let mut ids = vec![self.config.agent.id.clone()];
        ids.extend(self.config.agents.keys().cloned());
        ids.sort();
        ids.dedup();

        ids.into_iter()
            .map(|agent_id| {
                let handle = runtimes.get(&agent_id);
                let running = handle
                    .and_then(|h| h.task.as_ref())
                    .map(|jh| !jh.is_finished())
                    .unwrap_or(false);
                let awaiting_results = pending.values().filter(|a| *a == &agent_id).count();
                let queued_tasks = handle
                    .map(|h| h.queued_tasks.load(Ordering::Relaxed))
                    .unwrap_or(0);
                AgentStatusSnapshot {
                    agent_id,
                    running,
                    queued_tasks,
                    awaiting_results,
                }
            })
            .collect()
    }

    /// Get status for a single agent.
    pub async fn get_status(&self, agent_id: &str) -> Option<AgentStatusSnapshot> {
        self.list_statuses()
            .await
            .into_iter()
            .find(|s| s.agent_id == agent_id)
    }

    async fn ensure_runtime(&self, agent_id: &str) -> Result<Arc<AgentRuntimeHandle>> {
        let handle = {
            let runtimes = self.runtimes.read().await;
            if let Some(handle) = runtimes.get(agent_id) {
                let running = handle
                    .task
                    .as_ref()
                    .map(|t| !t.is_finished())
                    .unwrap_or(false);
                if running {
                    Arc::clone(handle)
                } else {
                    drop(runtimes);
                    let mut runtimes_write = self.runtimes.write().await;
                    if let Some(existing) = runtimes_write.get(agent_id) {
                        let running = existing
                            .task
                            .as_ref()
                            .map(|t| !t.is_finished())
                            .unwrap_or(false);
                        if running {
                            Arc::clone(existing)
                        } else {
                            let handle = Arc::new(self.start_agent(agent_id).await?);
                            runtimes_write.insert(agent_id.to_string(), Arc::clone(&handle));
                            handle
                        }
                    } else {
                        let handle = Arc::new(self.start_agent(agent_id).await?);
                        runtimes_write.insert(agent_id.to_string(), Arc::clone(&handle));
                        handle
                    }
                }
            } else {
                drop(runtimes);
                let mut runtimes_write = self.runtimes.write().await;
                if let Some(handle) = runtimes_write.get(agent_id) {
                    let running = handle
                        .task
                        .as_ref()
                        .map(|t| !t.is_finished())
                        .unwrap_or(false);
                    if running {
                        Arc::clone(handle)
                    } else {
                        let handle = Arc::new(self.start_agent(agent_id).await?);
                        runtimes_write.insert(agent_id.to_string(), Arc::clone(&handle));
                        handle
                    }
                } else {
                    let handle = self.start_agent(agent_id).await?;
                    let handle = Arc::new(handle);
                    runtimes_write.insert(agent_id.to_string(), Arc::clone(&handle));
                    handle
                }
            }
        };
        Ok(handle)
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
}

struct PeerRunOutcome {
    runtime_task_id: String,
    status: TaskTerminalStatus,
    task_turn_count: u32,
    output: Option<String>,
}

async fn run_peer_task(
    kernel: &mut Kernel,
    session: &mut SessionState,
    mut task: QueuedTask,
) -> Result<PeerRunOutcome> {
    if task.task_id.is_empty() {
        task.task_id = format!("t_{}", session.next_task_id);
        session.next_task_id += 1;
    }

    kernel.persist_event(
        session,
        &KernelEvent::Lifecycle(LifecycleEvent::TaskStart {
            identity: session.identity.clone(),
            task_id: task.task_id.clone(),
            plan_id: task.plan_id.clone(),
            title: task.title.clone(),
            prompt: task.prompt.clone(),
            queue_depth: 0,
        }),
    );

    let task_start_verdict = {
        let harness = kernel.lock_harness();
        if let Some(ref engine) = *harness {
            match engine.evaluate(
                "on_task_start",
                serde_json::json!({
                    "identity": session.identity.clone(),
                    "session_id": session.identity.session_id(),
                    "task_id": task.task_id.clone(),
                    "plan_id": task.plan_id.clone(),
                    "title": task.title.clone(),
                    "prompt": task.prompt.clone(),
                    "queue_depth": 0,
                }),
            ) {
                Ok(v) => v,
                Err(e) => {
                    warn!(error = %e, "Harness on_task_start error");
                    Verdict::Allow
                }
            }
        } else {
            Verdict::Allow
        }
    };

    match task_start_verdict {
        Verdict::Reject(reason) => {
            warn!(task_id = %task.task_id, reason = %reason, "Peer task rejected by on_task_start");
            kernel
                .complete_task(session, &task, TaskTerminalStatus::Rejected, 0, None)
                .await?;
            return Ok(PeerRunOutcome {
                runtime_task_id: task.task_id,
                status: TaskTerminalStatus::Rejected,
                task_turn_count: 0,
                output: None,
            });
        }
        Verdict::Modify(val) => {
            if let Some(obj) = val.as_object() {
                if let Some(prompt) = obj.get("prompt").and_then(|v| v.as_str()) {
                    task.prompt = prompt.to_string();
                }
                if let Some(title) = obj.get("title").and_then(|v| v.as_str()) {
                    task.title = Some(title.to_string());
                }
            }
        }
        Verdict::Escalate(reason) => {
            warn!(task_id = %task.task_id, reason = %reason, "Peer task escalated at on_task_start; treating as rejected");
            kernel
                .complete_task(session, &task, TaskTerminalStatus::Rejected, 0, None)
                .await?;
            return Ok(PeerRunOutcome {
                runtime_task_id: task.task_id,
                status: TaskTerminalStatus::Rejected,
                task_turn_count: 0,
                output: None,
            });
        }
        Verdict::Allow => {}
    }

    info!(task_id = %task.task_id, prompt = %task.prompt, "Running peer task");

    let run_result: crate::kernel::TaskExecutionResult = match kernel.run_task(session, &task).await {
        Ok(result) => {
            kernel
                .complete_task(session, &task, result.status, result.task_turn_count, None)
                .await?;
            result
        }
        Err(e) => {
            error!(task_id = %task.task_id, error = %e, "Peer task failed with runtime error");
            let error_message = e.to_string();
            let recovered = kernel
                .handle_inference_error(session, &task, &error_message)
                .await?;
            kernel
                .complete_task(
                    session,
                    &task,
                    TaskTerminalStatus::Error,
                    0,
                    Some(error_message),
                )
                .await?;
            if recovered {
                return Ok(PeerRunOutcome {
                    runtime_task_id: task.task_id,
                    status: TaskTerminalStatus::Error,
                    task_turn_count: 0,
                    output: None,
                });
            }
            return Err(e);
        }
    };

    let output = last_assistant_text(session);

    Ok(PeerRunOutcome {
        runtime_task_id: task.task_id,
        status: run_result.status,
        task_turn_count: run_result.task_turn_count,
        output,
    })
}

fn last_assistant_text(session: &SessionState) -> Option<String> {
    session.history.iter().rev().find_map(|msg| {
        msg.content.iter().find_map(|c| match c {
            InferenceContent::Text { text } => Some(text.clone()),
            _ => None,
        })
    })
}
