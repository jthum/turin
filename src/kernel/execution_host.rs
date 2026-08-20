use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::inference::content::{
    summarize_content_for_display, task_output_content_from_inference,
};
use crate::inference::embeddings::EmbeddingProvider;
use crate::inference::provider::{InferenceRole, ProviderClient};
use crate::kernel::agent_manager::AgentManager;
use crate::kernel::config::TurinConfig;
use crate::kernel::event::{TaskBranchOutcome, TaskTerminalStatus};
use crate::kernel::governance::GovernanceManager;
use crate::kernel::harness_manager::HarnessManager;
use crate::kernel::mcp_runtime::McpClientEntry;
use crate::kernel::policy::RuntimePolicyManager;
use crate::kernel::session::{
    ExecutionContextTarget, ExecutionStatusSnapshot, ExecutionWritePolicy, LocalTaskResult,
    QueuedTask, SessionHarnessEngine, SessionState,
};
use crate::kernel::task_promotion::TaskPromotionCandidate;
use crate::persistence::manager::{StoreManager, StorePathScope};
use crate::tools::registry::ToolRegistry;
use tracing::{error, warn};
use turin_types::TaskInputContent;

use super::TaskExecutionResult;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PersistedSessionLockKey {
    store_path: PathBuf,
    session_id: i64,
}

#[derive(Default)]
pub(crate) struct SessionPersistenceCoordinator {
    locks:
        std::sync::Mutex<HashMap<PersistedSessionLockKey, std::sync::Weak<tokio::sync::Mutex<()>>>>,
}

impl SessionPersistenceCoordinator {
    pub(crate) fn lock_for(
        &self,
        store_path: PathBuf,
        session_id: i64,
    ) -> Arc<tokio::sync::Mutex<()>> {
        let key = PersistedSessionLockKey {
            store_path,
            session_id,
        };
        let mut locks = self
            .locks
            .lock()
            .expect("session persistence coordinator mutex poisoned");
        if let Some(existing) = locks.get(&key).and_then(std::sync::Weak::upgrade) {
            return existing;
        }

        let lock = Arc::new(tokio::sync::Mutex::new(()));
        locks.insert(key, Arc::downgrade(&lock));
        lock
    }
}

/// Shared runtime execution state used by both the top-level kernel and peer runtimes.
pub struct ExecutionHost {
    pub(crate) config: Arc<TurinConfig>,
    pub(crate) json: bool,
    pub(crate) tool_registry: ToolRegistry,
    pub(crate) store_manager: Arc<StoreManager>,
    pub(crate) agent_manager: Arc<AgentManager>,
    pub(crate) policy_manager: Arc<RuntimePolicyManager>,
    pub(crate) governance_manager: Arc<GovernanceManager>,
    pub(crate) harness_manager: Arc<HarnessManager>,
    pub(crate) scheduler: Option<Arc<crate::harness::scheduler::HarnessSchedulerAccess>>,
    pub(crate) persistence_locks: Arc<SessionPersistenceCoordinator>,
    pub(crate) clients: HashMap<String, ProviderClient>,
    pub(crate) embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    pub(crate) native_harness_factory:
        Option<Arc<dyn crate::kernel::native_harness::NativeHarnessFactory>>,
    pub(crate) mcp_clients: Vec<McpClientEntry>,
}

pub(crate) enum TaskRunAttempt {
    Completed(TaskExecutionResult),
    Terminal {
        status: TaskTerminalStatus,
        error_message: String,
    },
    Error {
        error: anyhow::Error,
        error_message: String,
        recovered: bool,
    },
}

impl ExecutionHost {
    pub(crate) fn runtime_for_agent(
        &self,
        agent_id: &str,
    ) -> Arc<crate::kernel::harness_runtime::HarnessRuntime> {
        Arc::clone(self.harness_manager.resolve_harness(Some(agent_id)))
    }

    pub(crate) fn runtime_for_harness(
        &self,
        harness_id: &str,
    ) -> Option<Arc<crate::kernel::harness_runtime::HarnessRuntime>> {
        self.harness_manager.runtime_by_id(harness_id).cloned()
    }

    pub(crate) fn runtime_for_session(
        &self,
        session: &SessionState,
    ) -> Arc<crate::kernel::harness_runtime::HarnessRuntime> {
        self.runtime_for_agent(session.identity.agent_id())
    }

    pub(crate) fn session_harness_engine(
        &self,
        session: &SessionState,
    ) -> Option<SessionHarnessEngine> {
        session.harness_engine.clone()
    }

    pub(crate) fn ensure_session_harness_engine(
        &self,
        session: &mut SessionState,
    ) -> anyhow::Result<()> {
        let runtime = self.runtime_for_session(session);
        let generation = runtime.generation();
        if session.harness_engine.is_some() && session.harness_generation == generation {
            return Ok(());
        }

        let instance = runtime.create_instance(self.harness_init_context())?;
        instance.set_active_queue(Some(session.queue.clone()));
        session.harness_engine = Some(Arc::new(std::sync::Mutex::new(instance)));
        session.harness_generation = generation;
        Ok(())
    }

    pub(crate) fn clear_session_harness_engine(&self, session: &mut SessionState) {
        session.harness_engine = None;
        session.harness_generation = 0;
    }

    pub(crate) fn bind_harness_execution_context(&self, session: &SessionState, task: &QueuedTask) {
        let Some(harness) = self.session_harness_engine(session) else {
            return;
        };

        let engine = harness.lock().expect("session harness mutex poisoned");
        engine.bind_execution_context(crate::kernel::harness_contract::HarnessExecutionBinding {
            agent_id: session.identity.agent_id().to_string(),
            session_id: self.session_reference(session),
            store_selector: session.store_selector.clone(),
            default_store_selector: session.default_store_selector.clone(),
            execution: crate::kernel::harness_contract::HarnessExecutionMetadata {
                execution_id: session.execution_id().to_string(),
                context_target: session.context_target().clone(),
                visibility: session.execution.visibility,
                durability: session.execution.durability,
                write_policy: session.effective_write_policy(),
                conflict_policy: session.effective_conflict_policy(),
            },
            runtime_slot_id: session.runtime_slot_id.clone(),
            trace_id: task.trace_id.clone(),
            completed_task_results: session.completed_task_results.clone(),
            event_context: crate::kernel::harness_contract::HarnessEventContext {
                json: self.json,
                internal_id: session.internal_id,
                turn_id: session.active_history_origin().map(|origin| origin.turn_id),
                branch_head_id: session.selected_branch_head_id(),
                execution_id: session.execution_id().to_string(),
                event_tx: session.event_tx.clone(),
                durability_tx: session.durability_tx.clone(),
            },
            cancel_token: session.cancel_token.clone(),
        });
    }

    pub(crate) fn unbind_harness_execution_context(&self, session: &SessionState) {
        let Some(harness) = self.session_harness_engine(session) else {
            return;
        };

        let engine = harness.lock().expect("session harness mutex poisoned");
        engine.unbind_execution_context();
    }

    pub(crate) async fn bind_session_persistence_lock(
        &self,
        session: &mut SessionState,
    ) -> anyhow::Result<()> {
        let Some(internal_id) = session.internal_id else {
            return Ok(());
        };
        let store_path = self
            .store_manager
            .resolve_path_for_selector(&session.store_selector, StorePathScope::AllowAny)
            .await?;
        session.persistence_lock = self.persistence_locks.lock_for(store_path, internal_id);
        Ok(())
    }

    pub(crate) async fn run_task_with_conflict_handling(
        &mut self,
        session: &mut SessionState,
        task: &QueuedTask,
    ) -> anyhow::Result<TaskRunAttempt> {
        match self.run_task(session, task).await {
            Ok(result) => Ok(TaskRunAttempt::Completed(result)),
            Err(error) => {
                error!(
                    task_id = %task.task_id,
                    trace_id = %task.trace_id,
                    error = %error,
                    "Task failed with runtime error"
                );
                let error_message = error.to_string();
                if let Some(status) = terminal_status_for_runtime_error(session, &error) {
                    return Ok(TaskRunAttempt::Terminal {
                        status,
                        error_message,
                    });
                }
                if crate::persistence::state::is_turn_write_conflict(&error) {
                    match session.effective_conflict_policy() {
                        crate::kernel::session::ExecutionConflictPolicy::Reject
                        | crate::kernel::session::ExecutionConflictPolicy::ForkSibling => {
                            return Ok(TaskRunAttempt::Terminal {
                                status: TaskTerminalStatus::Conflict,
                                error_message,
                            });
                        }
                        crate::kernel::session::ExecutionConflictPolicy::Detached => {
                            warn!(
                                task_id = %task.task_id,
                                trace_id = %task.trace_id,
                                "Task write conflict downgraded to detached execution"
                            );
                            return match self.run_task(session, task).await {
                                Ok(result) => Ok(TaskRunAttempt::Completed(result)),
                                Err(detached_error) => {
                                    error!(
                                        task_id = %task.task_id,
                                        trace_id = %task.trace_id,
                                        error = %detached_error,
                                        "Detached retry failed with runtime error"
                                    );
                                    let error_message = detached_error.to_string();
                                    if let Some(status) =
                                        terminal_status_for_runtime_error(session, &detached_error)
                                    {
                                        return Ok(TaskRunAttempt::Terminal {
                                            status,
                                            error_message,
                                        });
                                    }
                                    let recovered = self
                                        .handle_inference_error(session, task, &error_message)
                                        .await?;
                                    Ok(TaskRunAttempt::Error {
                                        error: detached_error,
                                        error_message,
                                        recovered,
                                    })
                                }
                            };
                        }
                    }
                }

                let recovered = self
                    .handle_inference_error(session, task, &error_message)
                    .await?;
                Ok(TaskRunAttempt::Error {
                    error,
                    error_message,
                    recovered,
                })
            }
        }
    }

    pub(crate) async fn begin_task_execution_scope(
        &self,
        session: &mut SessionState,
        task: &QueuedTask,
    ) -> anyhow::Result<()> {
        session.set_active_delegation_budget(task.delegation_budget.clone());
        let needs_refresh = session
            .begin_task_execution_override(task.execution.as_ref())
            .map_err(anyhow::Error::msg)?;
        if needs_refresh {
            self.refresh_session_from_persistence(session).await?;
        }
        Ok(())
    }

    pub(crate) async fn finish_task_execution_scope(
        &self,
        session: &mut SessionState,
    ) -> anyhow::Result<()> {
        let needs_refresh = session.finish_task_execution_scope();
        if needs_refresh {
            self.refresh_session_from_persistence(session).await?;
        }
        Ok(())
    }

    pub(crate) fn agent_config_for(
        &self,
        agent_id: &str,
    ) -> anyhow::Result<&crate::kernel::config::AgentConfig> {
        if agent_id == self.config.agent.id {
            Ok(&self.config.agent)
        } else {
            self.config
                .agents
                .get(agent_id)
                .ok_or_else(|| anyhow::anyhow!("Unknown agent profile: {}", agent_id))
        }
    }

    pub(crate) fn agent_config_for_session(
        &self,
        session: &SessionState,
    ) -> anyhow::Result<&crate::kernel::config::AgentConfig> {
        self.agent_config_for(session.identity.agent_id())
    }

    pub(crate) fn last_assistant_text(&self, session: &SessionState) -> Option<String> {
        session.history.iter().rev().find_map(|msg| {
            if msg.role != InferenceRole::Assistant {
                return None;
            }
            let summary = summarize_content_for_display(&msg.content);
            if summary.is_empty() {
                None
            } else {
                Some(summary)
            }
        })
    }

    pub(crate) fn last_assistant_content(
        &self,
        session: &SessionState,
    ) -> Option<Vec<TaskInputContent>> {
        session.history.iter().rev().find_map(|msg| {
            if msg.role != InferenceRole::Assistant {
                return None;
            }
            let content = task_output_content_from_inference(&msg.content);
            if content.is_empty() {
                None
            } else {
                Some(content)
            }
        })
    }

    pub(crate) fn task_input_content(task: &QueuedTask) -> Vec<TaskInputContent> {
        task.content.clone().unwrap_or_else(|| {
            vec![TaskInputContent::Text {
                text: task.prompt.clone(),
            }]
        })
    }

    pub(crate) fn promotable_detached_candidate(
        &self,
        session: &SessionState,
        status: TaskTerminalStatus,
    ) -> Option<TaskPromotionCandidate> {
        if status != TaskTerminalStatus::Success
            || session.effective_write_policy() != ExecutionWritePolicy::Detached
        {
            return None;
        }

        let source_turn_id = match session.context_target() {
            ExecutionContextTarget::TurnId { turn_id } => Some(*turn_id),
            ExecutionContextTarget::SelectedPath { turn_ids } => turn_ids.last().copied(),
            ExecutionContextTarget::SummarySource { source_turn_id } => Some(*source_turn_id),
            ExecutionContextTarget::BranchHead { .. } => session
                .selected_branch_head_cursor
                .map(|cursor| cursor.turn_id),
            ExecutionContextTarget::ExternalReference { .. } => None,
        }?;

        Some(TaskPromotionCandidate {
            session_id: self.session_reference(session),
            source_turn_id,
            source_session_id: None,
        })
    }

    pub(crate) async fn record_local_completed_task(
        &self,
        session: &SessionState,
        task: &QueuedTask,
        status: TaskTerminalStatus,
        task_turn_count: u32,
        branch_outcome: Option<TaskBranchOutcome>,
        error: Option<String>,
    ) {
        let result = LocalTaskResult {
            task_id: task.task_id.clone(),
            trace_id: task.trace_id.clone(),
            execution: ExecutionStatusSnapshot::from_session(session),
            status,
            task_turn_count,
            branch_outcome,
            promotion_candidate: self.promotable_detached_candidate(session, status),
            promoted_branch: None,
            output: self.last_assistant_text(session),
            assistant_content: self.last_assistant_content(session),
            promotion_input_content: Some(Self::task_input_content(task)),
            error,
        };
        session.completed_task_results.write().await.insert(result);
    }
}

fn terminal_status_for_runtime_error(
    session: &SessionState,
    error: &anyhow::Error,
) -> Option<TaskTerminalStatus> {
    if session.cancel_token.is_cancelled() {
        return Some(TaskTerminalStatus::Cancelled);
    }
    error
        .chain()
        .any(inference_error_is_timeout)
        .then_some(TaskTerminalStatus::TimedOut)
}

fn inference_error_is_timeout(error: &(dyn std::error::Error + 'static)) -> bool {
    use crate::inference::provider::SdkError;

    let Some(error) = error.downcast_ref::<SdkError>() else {
        return false;
    };
    match error {
        SdkError::NetworkError(error) => error.is_timeout(),
        SdkError::IoError(error) => error.kind() == std::io::ErrorKind::TimedOut,
        SdkError::ApiError(message) => message.contains("total timeout budget"),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::provider::SdkError;

    #[test]
    fn runtime_error_classification_distinguishes_timeout_and_cancellation() {
        let session = SessionState::new();
        let timeout = anyhow::Error::new(SdkError::ApiError(
            "API request aborted: total timeout budget of 10s was exceeded".to_string(),
        ));
        assert_eq!(
            terminal_status_for_runtime_error(&session, &timeout),
            Some(TaskTerminalStatus::TimedOut)
        );

        session.cancel_token.cancel();
        assert_eq!(
            terminal_status_for_runtime_error(&session, &anyhow::anyhow!("stream closed")),
            Some(TaskTerminalStatus::Cancelled)
        );
    }
}
