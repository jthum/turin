use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use turin_daemon_protocol::UiIntentMessage;

use super::{HarnessDefinition, HarnessSourceOverlay};
use crate::harness::scheduler::HarnessSchedulerAccess;
use crate::harness::verdict::Verdict;
use crate::harness::virtual_tools::{
    DeclaredVirtualTool, VirtualToolFollowUp, VirtualToolResultResolution,
};
use crate::inference::embeddings::EmbeddingProvider;
use crate::inference::provider::ProviderClient;
use crate::kernel::agent_manager::AgentManager;
use crate::kernel::config::TurinConfig;
use crate::kernel::governance::GovernanceManager;
use crate::kernel::harness_contract::{
    HarnessActionRequest, HarnessExecutionBinding, HarnessHook, HarnessSignal, HarnessTurnRequest,
    HarnessTurnServices, SessionQueue,
};
use crate::kernel::policy::RuntimePolicyManager;
use crate::persistence::manager::StoreManager;

#[derive(Clone)]
pub struct HarnessRuntimeInitContext {
    pub config: Arc<TurinConfig>,
    pub clients: HashMap<String, ProviderClient>,
    pub store_manager: Arc<StoreManager>,
    pub agent_manager: Arc<AgentManager>,
    pub policy_manager: Arc<RuntimePolicyManager>,
    pub governance_manager: Arc<GovernanceManager>,
    pub scheduler: Option<Arc<HarnessSchedulerAccess>>,
    pub embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
}

pub trait HarnessAdapterFactory: Send + Sync {
    fn name(&self) -> &'static str;

    fn watches_sources(&self) -> bool {
        false
    }

    fn create(
        &self,
        definition: &HarnessDefinition,
        ctx: HarnessRuntimeInitContext,
    ) -> Result<Box<dyn HarnessInstance>>;

    fn validate_sources(
        &self,
        _definition: &HarnessDefinition,
        _ctx: HarnessRuntimeInitContext,
        _source_overlay: Arc<HarnessSourceOverlay>,
    ) -> Result<usize> {
        anyhow::bail!("this harness adapter does not support source validation")
    }

    fn run_source(
        &self,
        _definition: &HarnessDefinition,
        _ctx: HarnessRuntimeInitContext,
        _source: &str,
    ) -> Result<()> {
        anyhow::bail!("this harness adapter does not support direct source execution")
    }
}

pub trait HarnessInstance: Send {
    fn loaded_scripts(&self) -> Vec<String> {
        Vec::new()
    }

    fn explicit_watch_roots(&self) -> Vec<PathBuf> {
        Vec::new()
    }

    fn runtime_signal_topics(&self) -> Vec<String> {
        Vec::new()
    }

    fn ui_intents(&self) -> Vec<UiIntentMessage> {
        Vec::new()
    }

    fn ui_intent_count(&self) -> Result<usize> {
        Ok(0)
    }

    fn ui_intents_from(&self, _start_index: usize) -> Result<Vec<UiIntentMessage>> {
        Ok(Vec::new())
    }

    fn evaluate_hook(&self, hook: HarnessHook<'_>) -> Result<Verdict>;

    fn prepares_turn(&self) -> bool;

    fn prepare_turn(
        &self,
        request: &mut HarnessTurnRequest,
        services: HarnessTurnServices<'_>,
    ) -> Result<Verdict>;

    fn bind_execution_context(&self, _binding: HarnessExecutionBinding) {}

    fn unbind_execution_context(&self) {}

    fn set_active_queue(&self, _queue: Option<SessionQueue>) {}

    fn set_active_capability_delegation(&self, _capabilities: Option<BTreeMap<String, bool>>) {}

    fn take_pending_session_branch_checkout(&self) -> Option<String> {
        None
    }

    fn invoke_action(&self, request: HarnessActionRequest<'_>)
    -> Result<Option<serde_json::Value>>;

    fn declared_virtual_tools(&self) -> Result<Vec<DeclaredVirtualTool>> {
        Ok(Vec::new())
    }

    fn invoke_virtual_tool(
        &self,
        _name: &str,
        _args: serde_json::Value,
    ) -> Result<Option<VirtualToolResultResolution>> {
        Ok(None)
    }

    fn virtual_tool_follow_up(&self, _name: &str) -> Result<Option<VirtualToolFollowUp>> {
        Ok(None)
    }

    fn invoke_virtual_tool_result_handler(
        &self,
        key: &str,
        _payload: serde_json::Value,
        _default_is_error: bool,
    ) -> Result<VirtualToolResultResolution> {
        anyhow::bail!("harness has no virtual result handler '{key}'")
    }

    fn discard_virtual_tool_result_handler(&self, _key: &str) -> Result<()> {
        Ok(())
    }

    fn dispatch_runtime_signal(&self, signal: HarnessSignal<'_>) -> Result<usize>;
}
