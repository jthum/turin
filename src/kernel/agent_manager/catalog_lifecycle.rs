use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::Result;
use tracing::warn;

use crate::kernel::config::TurinConfig;
use crate::kernel::harness_manager::HarnessManager;

use super::AgentManager;

impl AgentManager {
    pub(crate) async fn reconcile_runtime_catalog(
        &self,
        config: Arc<TurinConfig>,
        harness_manager: Arc<HarnessManager>,
        affected_agents: &HashSet<String>,
    ) -> Result<()> {
        let mut agent_ids: Vec<_> = affected_agents.iter().cloned().collect();
        agent_ids.sort();
        let mut catalog_guards = Vec::with_capacity(agent_ids.len());
        for agent_id in &agent_ids {
            catalog_guards.push(self.catalog_gate(agent_id).write_owned().await);
        }

        self.ensure_agents_reconfigurable(affected_agents).await?;

        let retiring: Vec<_> = {
            let runtimes = self.runtimes.read().await;
            runtimes
                .iter()
                .filter(|(key, _)| affected_agents.contains(&key.agent_id))
                .map(|(key, handle)| (key.clone(), Arc::clone(handle)))
                .collect()
        };

        {
            let mut runtimes = self.runtimes.write().await;
            for (key, _) in &retiring {
                runtimes.remove(key);
            }
        }
        for (_, handle) in &retiring {
            handle.shutdown_token.cancel();
            handle.notify.notify_one();
        }

        self.install_runtime_catalog(config, harness_manager);
        drop(catalog_guards);

        let deadline = tokio::time::Instant::now() + Self::SHUTDOWN_GRACE;
        while retiring.iter().any(|(_, handle)| handle.is_running()) {
            let now = tokio::time::Instant::now();
            if now >= deadline {
                break;
            }
            tokio::time::sleep((deadline - now).min(Duration::from_millis(10))).await;
        }
        for (key, handle) in retiring.iter().filter(|(_, handle)| handle.is_running()) {
            warn!(
                agent_id = %key.agent_id,
                slot_id = %key.slot_id,
                "Idle peer runtime exceeded catalog replacement grace period; aborting"
            );
            if let Some(task) = &handle.task {
                task.abort();
            }
        }

        Ok(())
    }

    pub(crate) async fn ensure_agents_reconfigurable(
        &self,
        affected_agents: &HashSet<String>,
    ) -> Result<()> {
        let pending = self.pending_task_states.read().await;
        if let Some(record) = pending
            .values()
            .find(|record| affected_agents.contains(&record.runtime_key.agent_id))
        {
            anyhow::bail!(
                "Cannot reconfigure agent '{}' while it has active, queued, or awaiting tasks",
                record.runtime_key.agent_id
            );
        }
        drop(pending);

        let runtimes = self.runtimes.read().await;
        if let Some((key, _)) = runtimes.iter().find(|(key, handle)| {
            affected_agents.contains(&key.agent_id)
                && (handle.active_tasks.load(Ordering::Relaxed) > 0
                    || handle.queued_tasks.load(Ordering::Relaxed) > 0
                    || handle.control.current_request_id().is_some())
        }) {
            anyhow::bail!(
                "Cannot reconfigure agent '{}' while runtime slot '{}' is busy",
                key.agent_id,
                key.slot_id
            );
        }
        Ok(())
    }
}
