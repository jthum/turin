use crate::harness::globals::HarnessAppData;
use crate::kernel::governance::{CapabilityDecision, GovernanceSubject};
use std::collections::BTreeMap;

pub(crate) fn current_agent_id(app_data: &HarnessAppData) -> &str {
    app_data.config.agent.id.as_str()
}

pub(crate) fn capability_decision(
    app_data: &HarnessAppData,
    capability: &str,
) -> CapabilityDecision {
    let subject = current_subject(app_data);
    app_data
        .governance_manager
        .capability_decision_for_subject(&subject, capability)
}

pub(crate) fn require_capability(
    app_data: &HarnessAppData,
    capability: &str,
) -> Result<(), String> {
    let subject = current_subject(app_data);
    app_data
        .governance_manager
        .require_capability_for_subject(&subject, capability)
}

pub(crate) fn current_subject(app_data: &HarnessAppData) -> GovernanceSubject {
    let module_name = app_data
        .active_harness_module
        .lock()
        .ok()
        .and_then(|lock| lock.clone());
    let root_name = app_data
        .active_harness_root
        .lock()
        .ok()
        .and_then(|lock| lock.clone());
    let import_capabilities: Option<BTreeMap<String, bool>> = app_data
        .active_import_capabilities
        .lock()
        .ok()
        .and_then(|lock| lock.clone());
    GovernanceSubject {
        agent_id: Some(current_agent_id(app_data).to_string()),
        module_name,
        root_name,
        grant_id: None,
        import_capabilities,
    }
}
