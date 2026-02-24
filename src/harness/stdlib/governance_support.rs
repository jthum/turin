use crate::harness::globals::HarnessAppData;
use crate::kernel::governance::CapabilityDecision;

pub(crate) fn current_agent_id(app_data: &HarnessAppData) -> &str {
    app_data.config.agent.id.as_str()
}

pub(crate) fn capability_decision(
    app_data: &HarnessAppData,
    capability: &str,
) -> CapabilityDecision {
    app_data
        .governance_manager
        .capability_decision(Some(current_agent_id(app_data)), capability)
}

pub(crate) fn require_capability(
    app_data: &HarnessAppData,
    capability: &str,
) -> Result<(), String> {
    app_data
        .governance_manager
        .require_capability(Some(current_agent_id(app_data)), capability)
}
