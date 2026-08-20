use anyhow::{Result, anyhow, bail};
use turin_daemon_protocol::{HarnessActionRunParams, HarnessActionRunResult};

use super::DaemonState;

impl DaemonState {
    pub fn run_harness_action(
        &self,
        params: HarnessActionRunParams,
    ) -> Result<HarnessActionRunResult> {
        let action = params.action.trim().to_string();
        if action.is_empty() {
            bail!("Harness action name is required");
        }

        let agent_id = self.resolve_harness_action_agent(&params)?;
        self.kernel.agent_config_for(&agent_id)?;

        let runtime = match params
            .harness_id
            .as_deref()
            .filter(|value| !value.is_empty())
        {
            Some(harness_id) => self
                .kernel
                .runtime_for_harness(harness_id)
                .ok_or_else(|| anyhow!("Harness '{}' not found", harness_id))?,
            None => self.kernel.runtime_for_agent(&agent_id),
        };
        let instance = runtime.create_instance(self.kernel.harness_init_context())?;
        let ui_start = instance.ui_intent_count()?;
        let result =
            instance.invoke_action(crate::kernel::harness_contract::HarnessActionRequest {
                agent_id: &agent_id,
                name: &action,
                params: params.params,
            })?;
        let Some(result) = result else {
            bail!("Harness action '{}' is not declared", action);
        };
        let ui_intents = instance.ui_intents_from(ui_start)?;

        Ok(HarnessActionRunResult {
            action,
            agent_id,
            harness_id: params.harness_id,
            result,
            ui_intents,
        })
    }

    fn resolve_harness_action_agent(&self, params: &HarnessActionRunParams) -> Result<String> {
        if let Some(agent_id) = params.agent_id.as_ref().filter(|value| !value.is_empty()) {
            if let Some(harness_id) = params.harness_id.as_ref().filter(|value| !value.is_empty()) {
                let snapshot = self
                    .kernel
                    .harness_snapshot(harness_id)
                    .ok_or_else(|| anyhow!("Harness '{}' not found", harness_id))?;
                if !snapshot.bound_agents.is_empty()
                    && !snapshot
                        .bound_agents
                        .iter()
                        .any(|candidate| candidate == agent_id)
                {
                    bail!(
                        "Agent '{}' is not bound to harness '{}'",
                        agent_id,
                        harness_id
                    );
                }
            }
            return Ok(agent_id.clone());
        }

        let Some(harness_id) = params.harness_id.as_ref().filter(|value| !value.is_empty()) else {
            bail!("Harness action requires agent_id or harness_id");
        };
        let snapshot = self
            .kernel
            .harness_snapshot(harness_id)
            .ok_or_else(|| anyhow!("Harness '{}' not found", harness_id))?;
        match snapshot.bound_agents.as_slice() {
            [agent_id] => Ok(agent_id.clone()),
            [] => Ok(self.kernel.config().agent.id.clone()),
            agents => bail!(
                "Harness '{}' has multiple bound agents ({}); provide agent_id",
                harness_id,
                agents.join(", ")
            ),
        }
    }
}
