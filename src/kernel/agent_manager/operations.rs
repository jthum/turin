use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::{Context, Result};

use crate::kernel::config::InferenceOverrideConfig;
use crate::kernel::event::KernelEvent;
use crate::kernel::session::{
    ExecutionContextTarget, ExecutionDurability, ExecutionStatusSnapshot, ExecutionVisibility,
    ExecutionWritePolicy,
};
use crate::kernel::session_refs::{
    format_session_reference, parse_session_reference, session_references_match,
};
use crate::persistence::manager::StoreSelector;

use super::{
    AgentManager, AgentRuntimeHandle, AgentStatusSnapshot, LiveSessionSnapshot,
    RuntimeControlSnapshot, RuntimeSlotKey,
};

fn live_execution_snapshot(snapshot: &RuntimeControlSnapshot) -> ExecutionStatusSnapshot {
    snapshot
        .execution
        .clone()
        .unwrap_or(ExecutionStatusSnapshot {
            execution_id: String::new(),
            context_target: ExecutionContextTarget::BranchHead {
                branch_head_id: None,
            },
            visibility: ExecutionVisibility::Visible,
            durability: ExecutionDurability::Durable,
            write_policy: ExecutionWritePolicy::AdvanceBranchHead,
        })
}

fn live_session_snapshot(
    runtime_key: &RuntimeSlotKey,
    handle: &Arc<AgentRuntimeHandle>,
    session_id: String,
) -> LiveSessionSnapshot {
    let control = handle.control.snapshot();
    live_session_snapshot_from_control(runtime_key, handle, session_id, control)
}

fn live_session_snapshot_from_control(
    runtime_key: &RuntimeSlotKey,
    handle: &Arc<AgentRuntimeHandle>,
    session_id: String,
    control: RuntimeControlSnapshot,
) -> LiveSessionSnapshot {
    let execution = live_execution_snapshot(&control);
    LiveSessionSnapshot {
        agent_id: runtime_key.agent_id.clone(),
        slot_id: runtime_key.slot_id.clone(),
        session_id,
        running: handle.is_running(),
        active_tasks: handle.active_tasks.load(Ordering::Relaxed),
        queued_tasks: handle.queued_tasks.load(Ordering::Relaxed),
        current_request_id: control.request_id,
        execution,
        conflict_policy: control.conflict_policy,
        history: control.history,
    }
}

fn runtime_slot_is_busy(handle: &Arc<AgentRuntimeHandle>) -> bool {
    handle.active_tasks.load(Ordering::Relaxed) > 0
        || handle.queued_tasks.load(Ordering::Relaxed) > 0
}

fn ensure_runtime_slot_idle(
    runtime_key: &RuntimeSlotKey,
    handle: &Arc<AgentRuntimeHandle>,
) -> Result<()> {
    if runtime_slot_is_busy(handle) {
        anyhow::bail!(
            "Runtime slot '{}' for agent '{}' is busy",
            runtime_key.slot_id,
            runtime_key.agent_id
        );
    }
    Ok(())
}

impl AgentManager {
    pub async fn wake_agent(self: &Arc<Self>, agent_id: &str) -> Result<()> {
        let handle = self.ensure_runtime(agent_id).await?;
        handle.notify.notify_one();
        Ok(())
    }

    pub async fn resolve_session_target(&self, session_id: &str) -> Result<(String, String)> {
        let session_ref = parse_session_reference(session_id)?;
        let selector = session_ref
            .store_selector
            .unwrap_or(self.config.persistence.top_level_state_selector()?);
        let public_id = uuid::Uuid::parse_str(&session_ref.public_id)?;
        let store = self.store_manager.open(&selector).await?;
        let row = store
            .get_session_row_by_public_id(public_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Session '{}' not found", session_id))?;
        Ok((
            row.agent_id,
            format_session_reference(&session_ref.public_id, &selector),
        ))
    }

    pub async fn wake_session(self: &Arc<Self>, session_id: &str) -> Result<()> {
        let live = self.find_runtimes_by_session(session_id).await;
        if !live.is_empty() {
            for (_, handle) in live {
                handle.notify.notify_one();
            }
            return Ok(());
        }
        let live = self
            .resume_session(session_id, None, None, InferenceOverrideConfig::default())
            .await?;
        if let Some((_, handle)) = self
            .find_runtimes_by_session(&live.session_id)
            .await
            .into_iter()
            .find(|(key, _)| key.slot_id == live.slot_id)
        {
            handle.notify.notify_one();
        }
        Ok(())
    }

    pub async fn open_session(
        self: &Arc<Self>,
        agent_id: &str,
        slot_id: Option<&str>,
        initial_state_selector: Option<StoreSelector>,
        initial_default_store_selector: Option<StoreSelector>,
        channel_id: Option<String>,
        initial_inference: InferenceOverrideConfig,
    ) -> Result<LiveSessionSnapshot> {
        let runtime_key = RuntimeSlotKey {
            agent_id: agent_id.to_string(),
            slot_id: slot_id
                .map(str::to_string)
                .unwrap_or_else(|| format!("sl_{}", uuid::Uuid::now_v7().simple())),
        };
        let handle = self
            .ensure_runtime_slot_in_store(
                runtime_key.clone(),
                initial_state_selector,
                initial_default_store_selector,
                super::SessionContextOverrides {
                    channel_id,
                    inference: initial_inference,
                },
            )
            .await?;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
        let session_id = loop {
            if let Some(session_id) = handle.control.current_session_id() {
                break session_id;
            }
            if tokio::time::Instant::now() >= deadline {
                anyhow::bail!(
                    "Agent runtime '{}' [{}] did not expose a live session",
                    runtime_key.agent_id,
                    runtime_key.slot_id
                );
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        };
        Ok(live_session_snapshot(&runtime_key, &handle, session_id))
    }

    pub async fn resume_session(
        self: &Arc<Self>,
        session_id: &str,
        slot_id: Option<&str>,
        channel_id: Option<String>,
        initial_inference: InferenceOverrideConfig,
    ) -> Result<LiveSessionSnapshot> {
        let live_matches = self.find_runtimes_by_session(session_id).await;
        if let Some(requested_slot_id) = slot_id {
            if let Some((runtime_key, handle)) = live_matches
                .iter()
                .find(|(runtime_key, _)| runtime_key.slot_id == requested_slot_id)
                .cloned()
            {
                return Ok(live_session_snapshot(
                    &runtime_key,
                    &handle,
                    session_id.to_string(),
                ));
            }
        } else {
            match live_matches.as_slice() {
                [] => {}
                [(runtime_key, handle)] => {
                    return Ok(live_session_snapshot(
                        runtime_key,
                        handle,
                        session_id.to_string(),
                    ));
                }
                _ => {
                    anyhow::bail!(
                        "Session '{}' is active in multiple runtime slots; specify slot_id",
                        session_id
                    );
                }
            }
        }

        let session_ref = parse_session_reference(session_id)?;
        let public_id = uuid::Uuid::parse_str(&session_ref.public_id)
            .with_context(|| format!("Invalid session id '{}'", session_ref.public_id))?;
        let store_selector = session_ref
            .store_selector
            .unwrap_or_else(|| StoreSelector::Alias("state".to_string()));
        let store = self.store_manager.open(&store_selector).await?;
        let row = store
            .get_session_row_by_public_id(public_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Session '{}' not found", session_id))?;
        let agent_id = row.agent_id.clone();

        let runtime_key = RuntimeSlotKey {
            agent_id: agent_id.clone(),
            slot_id: slot_id
                .map(str::to_string)
                .unwrap_or_else(|| format!("sl_{}", uuid::Uuid::now_v7().simple())),
        };

        if agent_id != self.config.agent.id {
            self.config
                .agents
                .get(&agent_id)
                .ok_or_else(|| anyhow::anyhow!("Unknown agent profile '{}'", agent_id))?;
        }

        let existing = {
            let runtimes = self.runtimes.read().await;
            runtimes.get(&runtime_key).cloned()
        };

        let handle = if let Some(handle) = existing {
            if handle.is_running() {
                ensure_runtime_slot_idle(&runtime_key, &handle)?;
                handle.control.request_session_resume(
                    session_id.to_string(),
                    super::SessionContextOverrides {
                        channel_id,
                        inference: initial_inference,
                    },
                );
                handle.notify.notify_one();
                handle
            } else {
                self.ensure_runtime_slot_resumed(
                    runtime_key.clone(),
                    session_id.to_string(),
                    super::SessionContextOverrides {
                        channel_id,
                        inference: initial_inference,
                    },
                )
                .await?
            }
        } else {
            self.ensure_runtime_slot_resumed(
                runtime_key.clone(),
                session_id.to_string(),
                super::SessionContextOverrides {
                    channel_id,
                    inference: initial_inference,
                },
            )
            .await?
        };

        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
        let resumed_session_id = loop {
            if let Some(current_session_id) = handle.control.current_session_id()
                && session_references_match(&current_session_id, session_id)
            {
                break current_session_id;
            }
            if tokio::time::Instant::now() >= deadline {
                anyhow::bail!(
                    "Agent runtime '{}' [{}] did not resume session '{}'",
                    runtime_key.agent_id,
                    runtime_key.slot_id,
                    session_id
                );
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        };

        Ok(live_session_snapshot(
            &runtime_key,
            &handle,
            resumed_session_id,
        ))
    }

    pub async fn reload_session(
        self: &Arc<Self>,
        session_id: &str,
        slot_id: Option<&str>,
    ) -> Result<LiveSessionSnapshot> {
        let (runtime_key, handle) = self.runtime_by_session_target(session_id, slot_id).await?;

        ensure_runtime_slot_idle(&runtime_key, &handle)?;

        let generation = handle.control.session_generation();
        let context = handle.control.current_session_context();
        handle
            .control
            .request_session_resume(session_id.to_string(), context);
        handle.notify.notify_one();

        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            let control = handle.control.snapshot();
            let current_matches = control
                .session_id
                .as_deref()
                .map(|current| session_references_match(current, session_id))
                .unwrap_or(false);
            if current_matches && control.generation > generation {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                anyhow::bail!(
                    "Agent runtime '{}' [{}] did not reload session '{}'",
                    runtime_key.agent_id,
                    runtime_key.slot_id,
                    session_id
                );
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        Ok(live_session_snapshot(
            &runtime_key,
            &handle,
            handle
                .control
                .snapshot()
                .session_id
                .unwrap_or_else(|| session_id.to_string()),
        ))
    }

    pub async fn reload_session_if_live(
        self: &Arc<Self>,
        session_id: &str,
        slot_id: Option<&str>,
    ) -> Result<bool> {
        let live_matches = self.find_runtimes_by_session(session_id).await;
        if let Some(slot_id) = slot_id {
            if !live_matches
                .iter()
                .any(|(runtime_key, _)| runtime_key.slot_id == slot_id)
            {
                return Ok(false);
            }
            self.reload_session(session_id, Some(slot_id)).await?;
            return Ok(true);
        }

        if live_matches.is_empty() {
            return Ok(false);
        }
        self.reload_session(session_id, None).await?;
        Ok(true)
    }

    pub async fn subscribe_session_events(
        &self,
        session_id: &str,
        slot_id: Option<&str>,
    ) -> Option<(
        String,
        String,
        tokio::sync::broadcast::Receiver<(Option<i64>, KernelEvent)>,
    )> {
        match self.runtime_by_session_target(session_id, slot_id).await {
            Ok((runtime_key, handle)) => handle
                .control
                .subscribe_current_session_events()
                .map(|receiver| (runtime_key.agent_id, runtime_key.slot_id, receiver)),
            Err(_) => None,
        }
    }

    /// List configured agents with runtime status.
    pub async fn list_statuses(&self) -> Vec<AgentStatusSnapshot> {
        let runtimes = self.runtimes.read().await;
        let pending = self.pending_task_states.read().await;
        let mut awaiting_by_agent: HashMap<&str, usize> = HashMap::new();
        for pending in pending.values() {
            *awaiting_by_agent
                .entry(pending.runtime_key.agent_id.as_str())
                .or_default() += 1;
        }

        let mut ids = vec![self.config.agent.id.clone()];
        ids.extend(self.config.agents.keys().cloned());
        ids.sort();
        ids.dedup();

        ids.into_iter()
            .map(|agent_id| {
                let agent = if agent_id == self.config.agent.id {
                    &self.config.agent
                } else {
                    self.config
                        .agents
                        .get(&agent_id)
                        .expect("configured agent id remains resolvable")
                };
                let effective_inference = self
                    .config
                    .effective_inference_config_for_agent(&agent_id, None)
                    .expect("validated agent inference configuration remains resolvable");
                let mut inference_contexts: Vec<_> = effective_inference
                    .contexts
                    .iter()
                    .map(|(id, context)| super::InferenceContextStatusSnapshot {
                        id: id.clone(),
                        provider: context.provider.clone(),
                        model: context.model.clone(),
                        is_default: effective_inference.default_context_name() == id,
                    })
                    .collect();
                inference_contexts.sort_by(|left, right| left.id.cmp(&right.id));
                let matching: Vec<_> = runtimes
                    .iter()
                    .filter(|(key, _)| key.agent_id == agent_id)
                    .collect();
                let running = matching.iter().any(|(_, h)| h.is_running());
                let awaiting_results = *awaiting_by_agent.get(agent_id.as_str()).unwrap_or(&0);
                let queued_tasks = matching
                    .iter()
                    .map(|(_, h)| h.queued_tasks.load(Ordering::Relaxed))
                    .sum();
                let active_tasks = matching
                    .iter()
                    .map(|(_, h)| h.active_tasks.load(Ordering::Relaxed))
                    .sum();
                let default_handle = runtimes.get(&RuntimeSlotKey::default_for(&agent_id));
                let single_handle = if matching.len() == 1 {
                    matching.first().map(|(_, h)| *h)
                } else {
                    None
                };
                let display_handle = default_handle.or(single_handle);
                let display_snapshot = display_handle.map(|h| h.control.snapshot());
                AgentStatusSnapshot {
                    agent_id,
                    provider: agent.provider.clone(),
                    model: agent.model.clone(),
                    harness_id: agent
                        .harness
                        .clone()
                        .unwrap_or_else(|| "default".to_string()),
                    inference_contexts,
                    running,
                    active_tasks,
                    queued_tasks,
                    awaiting_results,
                    current_session_id: display_snapshot
                        .as_ref()
                        .and_then(|snapshot| snapshot.session_id.clone()),
                    current_request_id: display_snapshot.and_then(|snapshot| snapshot.request_id),
                }
            })
            .collect()
    }

    pub async fn list_live_sessions(&self, agent_id: Option<&str>) -> Vec<LiveSessionSnapshot> {
        let runtimes = self.runtimes.read().await;
        let mut sessions: Vec<_> = runtimes
            .iter()
            .filter_map(|(runtime_key, handle)| {
                if agent_id.is_some_and(|wanted| runtime_key.agent_id != wanted) {
                    return None;
                }
                let control = handle.control.snapshot();
                let session_id = control.session_id.clone()?;
                Some(live_session_snapshot_from_control(
                    runtime_key,
                    handle,
                    session_id,
                    control,
                ))
            })
            .collect();
        sessions.sort_by(|a, b| {
            a.agent_id
                .cmp(&b.agent_id)
                .then_with(|| a.slot_id.cmp(&b.slot_id))
        });
        sessions
    }

    /// Get status for a single agent.
    pub async fn get_status(&self, agent_id: &str) -> Option<AgentStatusSnapshot> {
        self.list_statuses()
            .await
            .into_iter()
            .find(|s| s.agent_id == agent_id)
    }

    pub(super) async fn find_runtimes_by_session(
        &self,
        session_id: &str,
    ) -> Vec<(RuntimeSlotKey, Arc<AgentRuntimeHandle>)> {
        let runtimes = self.runtimes.read().await;
        let mut matches: Vec<_> = runtimes
            .iter()
            .filter_map(|(runtime_key, handle)| {
                let current = handle.control.current_session_id()?;
                if session_references_match(&current, session_id) {
                    Some((runtime_key.clone(), Arc::clone(handle)))
                } else {
                    None
                }
            })
            .collect();
        matches.sort_by(|(left, _), (right, _)| {
            left.agent_id
                .cmp(&right.agent_id)
                .then_with(|| left.slot_id.cmp(&right.slot_id))
        });
        matches
    }

    pub(super) async fn runtime_by_session_target(
        &self,
        session_id: &str,
        slot_id: Option<&str>,
    ) -> Result<(RuntimeSlotKey, Arc<AgentRuntimeHandle>)> {
        let matches = self.find_runtimes_by_session(session_id).await;
        if let Some(slot_id) = slot_id {
            return matches
                .into_iter()
                .find(|(runtime_key, _)| runtime_key.slot_id == slot_id)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Session '{}' is not active in runtime slot '{}'",
                        session_id,
                        slot_id
                    )
                });
        }
        match matches.len() {
            0 => anyhow::bail!(
                "Session '{}' is not an active managed runtime session",
                session_id
            ),
            1 => Ok(matches
                .into_iter()
                .next()
                .expect("single runtime match should exist")),
            _ => anyhow::bail!(
                "Session '{}' is active in multiple runtime slots; specify slot_id",
                session_id
            ),
        }
    }
}
