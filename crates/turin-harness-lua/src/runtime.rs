use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use turin_daemon_protocol::UiIntentMessage;

use crate::harness::context::ContextWrapper;
use crate::harness::engine::HarnessEngine;
use crate::harness::globals::{HarnessAppData, HarnessExecutionContext};
use crate::harness::virtual_tools::{
    DeclaredVirtualTool, VirtualToolFollowUp, VirtualToolResultResolution,
};
use turin_core::kernel::harness::Verdict;
use turin_core::kernel::harness_contract::{
    HarnessActionRequest, HarnessExecutionBinding, HarnessHook, HarnessSignal, HarnessTurnRequest,
    HarnessTurnServices, SessionQueue,
};
use turin_core::kernel::harness_runtime::{
    HarnessAdapterFactory, HarnessDefinition, HarnessInstance, HarnessRuntimeInitContext,
    HarnessSourceOverlay,
};

/// Factory for session-scoped Lua harness instances.
pub struct LuaHarnessAdapterFactory;

impl HarnessAdapterFactory for LuaHarnessAdapterFactory {
    fn name(&self) -> &'static str {
        "lua"
    }

    fn watches_sources(&self) -> bool {
        true
    }

    fn create(
        &self,
        definition: &HarnessDefinition,
        ctx: HarnessRuntimeInitContext,
    ) -> Result<Box<dyn HarnessInstance>> {
        Ok(Box::new(LuaHarnessInstance {
            engine: build_engine(definition, ctx, None)?,
        }))
    }

    fn validate_sources(
        &self,
        definition: &HarnessDefinition,
        ctx: HarnessRuntimeInitContext,
        source_overlay: Arc<HarnessSourceOverlay>,
    ) -> Result<usize> {
        let engine = build_engine(definition, ctx, Some(source_overlay))?;
        Ok(engine.loaded_scripts().len())
    }

    fn run_source(
        &self,
        definition: &HarnessDefinition,
        ctx: HarnessRuntimeInitContext,
        source: &str,
    ) -> Result<()> {
        let mut engine = build_engine(definition, ctx, None)?;
        engine.load_script_str(source)
    }
}

struct LuaHarnessInstance {
    engine: HarnessEngine,
}

impl HarnessInstance for LuaHarnessInstance {
    fn loaded_scripts(&self) -> Vec<String> {
        self.engine.loaded_scripts()
    }

    fn explicit_watch_roots(&self) -> Vec<PathBuf> {
        self.engine.explicit_watch_roots()
    }

    fn runtime_signal_topics(&self) -> Vec<String> {
        self.engine.runtime_signal_topics().unwrap_or_default()
    }

    fn ui_intents(&self) -> Vec<UiIntentMessage> {
        self.engine.ui_intents().unwrap_or_default()
    }

    fn ui_intent_count(&self) -> Result<usize> {
        self.engine.ui_intent_count()
    }

    fn ui_intents_from(&self, start_index: usize) -> Result<Vec<UiIntentMessage>> {
        self.engine.ui_intents_from(start_index)
    }

    fn evaluate_hook(&self, hook: HarnessHook<'_>) -> Result<Verdict> {
        self.engine.evaluate(hook.name(), hook.to_json_value())
    }

    fn prepares_turn(&self) -> bool {
        self.engine.has_hook("on_turn_prepare")
    }

    fn prepare_turn(
        &self,
        request: &mut HarnessTurnRequest,
        services: HarnessTurnServices<'_>,
    ) -> Result<Verdict> {
        let context = ContextWrapper::from_harness_request(request, services);
        let verdict = self
            .engine
            .evaluate_userdata("on_turn_prepare", context.clone());
        context.apply_to_harness_request(request);
        verdict
    }

    fn bind_execution_context(&self, binding: HarnessExecutionBinding) {
        self.engine.bind_execution_context(binding);
    }

    fn unbind_execution_context(&self) {
        self.engine.unbind_execution_context();
    }

    fn set_active_queue(&self, queue: Option<SessionQueue>) {
        self.engine.set_active_queue(queue);
    }

    fn set_active_capability_delegation(
        &self,
        capabilities: Option<std::collections::BTreeMap<String, bool>>,
    ) {
        self.engine.set_active_capability_delegation(capabilities);
    }

    fn take_pending_session_branch_checkout(&self) -> Option<String> {
        self.engine.take_pending_session_branch_checkout()
    }

    fn invoke_action(
        &self,
        request: HarnessActionRequest<'_>,
    ) -> Result<Option<serde_json::Value>> {
        self.engine
            .invoke_declared_action_for_agent(request.agent_id, request.name, request.params)
    }

    fn declared_virtual_tools(&self) -> Result<Vec<DeclaredVirtualTool>> {
        self.engine.declared_virtual_tools()
    }

    fn invoke_virtual_tool(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<Option<VirtualToolResultResolution>> {
        self.engine.invoke_virtual_tool(name, args)
    }

    fn virtual_tool_follow_up(&self, name: &str) -> Result<Option<VirtualToolFollowUp>> {
        self.engine.virtual_tool_follow_up(name)
    }

    fn invoke_virtual_tool_result_handler(
        &self,
        key: &str,
        payload: serde_json::Value,
        default_is_error: bool,
    ) -> Result<VirtualToolResultResolution> {
        self.engine
            .invoke_virtual_tool_result_handler(key, payload, default_is_error)
    }

    fn discard_virtual_tool_result_handler(&self, key: &str) -> Result<()> {
        self.engine.discard_virtual_tool_result_handler(key)
    }

    fn dispatch_runtime_signal(&self, signal: HarnessSignal<'_>) -> Result<usize> {
        self.engine.dispatch_runtime_signal(signal)
    }
}

fn build_engine(
    definition: &HarnessDefinition,
    ctx: HarnessRuntimeInitContext,
    source_overlay: Option<Arc<HarnessSourceOverlay>>,
) -> Result<HarnessEngine> {
    let app_data = HarnessAppData {
        fs_root: definition.fs_root.clone(),
        workspace_root: definition.workspace_root.clone(),
        harness_directory: definition.directory().to_path_buf(),
        store_manager: ctx.store_manager,
        agent_manager: ctx.agent_manager,
        policy_manager: ctx.policy_manager,
        governance_manager: ctx.governance_manager,
        scheduler: ctx.scheduler,
        execution_ctx: Arc::new(std::sync::Mutex::new(HarnessExecutionContext::default())),
        embedding_provider: ctx.embedding_provider,
        config: ctx.config,
        spawn_depth: definition.spawn_depth,
        active_modules: Arc::new(std::sync::Mutex::new(Vec::new())),
        watch_roots: Arc::new(std::sync::Mutex::new(Vec::new())),
        loading_phase: Arc::new(std::sync::Mutex::new(true)),
        source_overlay,
    };
    let mut engine = HarnessEngine::new(app_data).context("Failed to create harness engine")?;
    engine.load_dir(definition.directory()).with_context(|| {
        format!(
            "Failed to load harness scripts from '{}'",
            definition.directory().display()
        )
    })?;
    engine.set_loading_phase(false);
    Ok(engine)
}
