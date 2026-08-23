use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use turin_daemon_protocol::UiIntentMessage;

use crate::harness::context::ContextWrapper;
use crate::harness::engine::HarnessEngine;
use crate::harness::globals::{HarnessAppData, HarnessExecutionContext};
use crate::harness::source::HarnessSourceOverlay;
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
};

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
        self.engine.evaluate(hook.name(), hook_payload(&hook))
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
        clients: ctx.clients,
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

fn hook_payload(hook: &HarnessHook<'_>) -> serde_json::Value {
    use serde_json::json;
    match hook {
        HarnessHook::SessionStart {
            identity,
            session_id,
            governance,
        } => json!({ "identity": identity, "session_id": session_id, "governance": governance }),
        HarnessHook::SessionEnd {
            identity,
            session_id,
            turn_count,
            total_input_tokens,
            total_output_tokens,
        } => {
            json!({ "identity": identity, "session_id": session_id, "turn_count": turn_count, "total_input_tokens": total_input_tokens, "total_output_tokens": total_output_tokens })
        }
        HarnessHook::TaskStart {
            identity,
            session_id,
            task_id,
            trace_id,
            plan_id,
            title,
            prompt,
            queue_depth,
        } => {
            json!({ "identity": identity, "session_id": session_id, "task_id": task_id, "trace_id": trace_id, "plan_id": plan_id, "title": title, "prompt": prompt, "queue_depth": queue_depth })
        }
        HarnessHook::TaskComplete {
            identity,
            session_id,
            task_id,
            trace_id,
            plan_id,
            status,
            task_turn_count,
            task_started_at_unix_ms,
            task_elapsed_ms,
            task_input_tokens,
            task_output_tokens,
            task_total_tokens,
            turn_count,
            execution,
            branch_outcome,
            error,
        } => {
            json!({ "identity": identity, "session_id": session_id, "task_id": task_id, "trace_id": trace_id, "plan_id": plan_id, "status": status, "task_turn_count": task_turn_count, "task_started_at_unix_ms": task_started_at_unix_ms, "task_elapsed_ms": task_elapsed_ms, "task_input_tokens": task_input_tokens, "task_output_tokens": task_output_tokens, "task_total_tokens": task_total_tokens, "turn_count": turn_count, "execution": execution, "branch_outcome": branch_outcome, "error": error })
        }
        HarnessHook::PlanComplete {
            identity,
            session_id,
            plan_id,
            title,
            total_tasks,
            completed_tasks,
        } => {
            json!({ "identity": identity, "session_id": session_id, "plan_id": plan_id, "title": title, "total_tasks": total_tasks, "completed_tasks": completed_tasks })
        }
        HarnessHook::AllTasksComplete {
            identity,
            session_id,
            turn_count,
        } => json!({ "identity": identity, "session_id": session_id, "turn_count": turn_count }),
        HarnessHook::InferenceError {
            identity,
            session_id,
            task_id,
            trace_id,
            plan_id,
            turn_count,
            error,
        } => {
            json!({ "identity": identity, "session_id": session_id, "task_id": task_id, "trace_id": trace_id, "plan_id": plan_id, "turn_count": turn_count, "error": error })
        }
        HarnessHook::TurnStart {
            identity,
            session_id,
            task_id,
            trace_id,
            plan_id,
            turn_index,
            task_turn_index,
        } => {
            json!({ "identity": identity, "session_id": session_id, "task_id": task_id, "trace_id": trace_id, "plan_id": plan_id, "turn_index": turn_index, "task_turn_index": task_turn_index })
        }
        HarnessHook::TurnEnd {
            identity,
            session_id,
            task_id,
            trace_id,
            plan_id,
            turn_index,
            task_turn_index,
            has_tool_calls,
        } => {
            json!({ "identity": identity, "session_id": session_id, "task_id": task_id, "trace_id": trace_id, "plan_id": plan_id, "turn_index": turn_index, "task_turn_index": task_turn_index, "has_tool_calls": has_tool_calls })
        }
        HarnessHook::ToolCall { name, id, args } => json!({ "name": name, "id": id, "args": args }),
        HarnessHook::ToolResult {
            id,
            name,
            args,
            output,
            is_error,
        } => {
            json!({ "id": id, "name": name, "args": args, "output": output, "is_error": is_error })
        }
        HarnessHook::TokenUsage {
            input_tokens,
            output_tokens,
            task_started_at_unix_ms,
            task_elapsed_ms,
            task_input_tokens,
            task_output_tokens,
            task_turn_count,
        } => {
            json!({ "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens, "task_started_at_unix_ms": task_started_at_unix_ms, "task_elapsed_ms": task_elapsed_ms, "task_input_tokens": task_input_tokens, "task_output_tokens": task_output_tokens, "task_total_tokens": task_input_tokens + task_output_tokens, "task_turn_count": task_turn_count })
        }
        HarnessHook::PlanSubmit {
            title,
            tasks,
            clear_existing,
        } => json!({ "title": title, "tasks": tasks, "clear_existing": clear_existing }),
        HarnessHook::KernelEvent(event) => serde_json::to_value(event).unwrap_or_default(),
    }
}
