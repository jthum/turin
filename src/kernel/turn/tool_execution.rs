mod finalization;
mod plan_submission;
mod result_hooks;
mod validation;
mod virtual_tools;

use anyhow::Result;
use futures::stream::{self, StreamExt};
use std::collections::{BTreeSet, HashMap};
use std::future::Future;
use std::pin::Pin;
use std::time::Instant;
use tracing::info;

use crate::harness::verdict::Verdict;
use crate::harness::virtual_tools::{VirtualToolPlan, VirtualToolResultResolution};
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::governance::{CapabilityDecision, GovernanceSubject, tool_capability_name};
use crate::kernel::session::SessionState;
use crate::tools::{ToolContext, ToolEffect, ToolError, ToolOutput};

use super::super::PendingToolCall;
use super::super::event::{AuditEvent, KernelEvent};
use super::TurnOutcome;

const MAX_VIRTUAL_TOOL_DEPTH: usize = 8;
const MAX_PARALLEL_TOOL_CALLS: usize = 8;

#[derive(Debug, Clone)]
struct FinalToolRecord {
    id: String,
    name: String,
    args: serde_json::Value,
    verdict: String,
    duration_ms: u64,
    content: String,
    is_error: bool,
    emit_exec_start: bool,
    governance_denial: Option<CapabilityDecision>,
}

enum ExecutionArtifact {
    Native(ToolEffect),
    VirtualPlan(VirtualToolPlan),
    VirtualOutput { content: String, is_error: bool },
}

impl ExecutionHost {
    /// Phase 1-3 of tool execution: verdict evaluation, parallel execution, side effects, and result collection.
    pub(super) async fn execute_tool_calls(
        &mut self,
        session: &mut SessionState,
        tool_ctx: &ToolContext,
        pending_tool_calls: Vec<PendingToolCall>,
        exposed_tool_names: &BTreeSet<String>,
    ) -> Result<TurnOutcome> {
        if session.cancel_token.is_cancelled() {
            return Ok(TurnOutcome::Cancelled);
        }

        let (immediate_records, validated_calls) = self.evaluate_pending_tool_calls(
            session,
            &pending_tool_calls,
            Some(exposed_tool_names),
        );
        let (immediate_records, validated_calls) =
            self.apply_tool_rate_limit(session, immediate_records, validated_calls);
        let final_by_id = self
            .execute_validated_tool_calls(
                session,
                tool_ctx,
                validated_calls,
                immediate_records,
                Vec::new(),
            )
            .await;
        if session.cancel_token.is_cancelled() {
            return Ok(TurnOutcome::Cancelled);
        }

        self.finalize_tool_records(session, &pending_tool_calls, final_by_id, true)
            .await?;
        Ok(TurnOutcome::Continue)
    }

    fn execute_tool_calls_hidden<'a>(
        &'a mut self,
        session: &'a mut SessionState,
        tool_ctx: &'a ToolContext,
        pending_tool_calls: Vec<PendingToolCall>,
        virtual_stack: Vec<String>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<FinalToolRecord>>> + Send + 'a>> {
        Box::pin(async move {
            if session.cancel_token.is_cancelled() {
                return Ok(Vec::new());
            }

            let (immediate_records, validated_calls) =
                self.evaluate_pending_tool_calls(session, &pending_tool_calls, None);
            let (immediate_records, validated_calls) =
                self.apply_tool_rate_limit(session, immediate_records, validated_calls);
            let final_by_id = self
                .execute_validated_tool_calls(
                    session,
                    tool_ctx,
                    validated_calls,
                    immediate_records,
                    virtual_stack,
                )
                .await;
            if session.cancel_token.is_cancelled() {
                return Ok(Vec::new());
            }

            self.finalize_tool_records(session, &pending_tool_calls, final_by_id, false)
                .await
        })
    }

    async fn execute_validated_tool_calls(
        &mut self,
        session: &mut SessionState,
        tool_ctx: &ToolContext,
        validated_calls: Vec<(PendingToolCall, Verdict)>,
        immediate_records: Vec<FinalToolRecord>,
        virtual_stack: Vec<String>,
    ) -> HashMap<String, FinalToolRecord> {
        for (tc, _) in &validated_calls {
            self.persist_event(
                session,
                &KernelEvent::Audit(AuditEvent::ToolExecStart {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                }),
            );
        }

        let kernel = &*self;
        let active_agent_id = session.identity.agent_id().to_string();
        let harness_engine = self.session_harness_engine(session);
        let futures = validated_calls.into_iter().map(|(tc, verdict)| {
            let tool_ctx = tool_ctx.clone();
            let active_agent_id = active_agent_id.clone();
            let harness_engine = harness_engine.clone();
            async move {
                let verdict_str = verdict.to_string();
                let final_args = match verdict {
                    Verdict::Modify(new_args) => {
                        info!(tool = %tc.name, "Tool arguments modified by harness");
                        new_args
                    }
                    _ => tc.args.clone(),
                };

                let start = Instant::now();
                let mut governance_denial = None;
                let effect_res = if let Some(tool) = kernel.tool_registry.get(&tc.name) {
                    if let Some(capability) =
                        tool.capability().or_else(|| tool_capability_name(&tc.name))
                    {
                        let subject = GovernanceSubject::for_agent(active_agent_id.as_str());
                        let decision = kernel
                            .governance_manager
                            .capability_decision_for_subject(&subject, capability);
                        match decision.allowed {
                            true => kernel
                                .tool_registry
                                .execute(&tc.name, final_args.clone(), &tool_ctx)
                                .await
                                .map(ExecutionArtifact::Native),
                            false => {
                                governance_denial = Some(decision.clone());
                                Err(ToolError::PermissionDenied(decision.reason.unwrap_or_else(
                                    || format!("Governance denial for capability '{}'", capability),
                                )))
                            }
                        }
                    } else {
                        kernel
                            .tool_registry
                            .execute(&tc.name, final_args.clone(), &tool_ctx)
                            .await
                            .map(ExecutionArtifact::Native)
                    }
                } else {
                    let plan_res = {
                        if let Some(harness) = &harness_engine {
                            let engine = harness.lock().expect("session harness mutex poisoned");
                            engine.invoke_virtual_tool(&tc.name, final_args.clone())
                        } else {
                            Ok(None)
                        }
                    };

                    match plan_res {
                        Ok(Some(VirtualToolResultResolution::Plan(plan))) => {
                            Ok(ExecutionArtifact::VirtualPlan(plan))
                        }
                        Ok(Some(VirtualToolResultResolution::Output(output))) => {
                            Ok(ExecutionArtifact::VirtualOutput {
                                content: output.content,
                                is_error: output.is_error,
                            })
                        }
                        Ok(None) => Err(ToolError::ExecutionError(format!(
                            "Unknown tool: {}",
                            tc.name
                        ))),
                        Err(err) => Err(ToolError::ExecutionError(format!(
                            "Virtual tool '{}' failed: {}",
                            tc.name, err
                        ))),
                    }
                };
                let duration_ms = start.elapsed().as_millis() as u64;

                let is_error = effect_res.is_err();
                let effect = effect_res.unwrap_or_else(|e| {
                    ExecutionArtifact::Native(ToolEffect::Output(ToolOutput {
                        content: format!("Error: {}", e),
                        metadata: serde_json::Value::Null,
                    }))
                });

                (
                    tc,
                    final_args,
                    verdict_str,
                    duration_ms,
                    effect,
                    is_error,
                    governance_denial,
                )
            }
        });

        let mut final_by_id: HashMap<String, FinalToolRecord> = HashMap::new();
        for record in immediate_records {
            final_by_id.insert(record.id.clone(), record);
        }

        let execution_results = tokio::select! {
            _ = session.cancel_token.cancelled() => {
                return final_by_id;
            }
            results = stream::iter(futures)
                .buffer_unordered(MAX_PARALLEL_TOOL_CALLS)
                .collect::<Vec<_>>() => results,
        };

        for (tc, final_args, verdict_str, duration_ms, effect, mut is_error, governance_denial) in
            execution_results
        {
            let mut content;
            match effect {
                ExecutionArtifact::Native(effect) => match effect {
                    ToolEffect::Output(o) => {
                        content = o.content;
                    }
                    ToolEffect::EnqueuePlan {
                        title,
                        tasks,
                        clear_existing,
                    } => {
                        let (plan_content, plan_error) = self
                            .handle_plan_submission(session, &title, tasks, clear_existing)
                            .await;
                        content = plan_content;
                        is_error = is_error || plan_error;
                    }
                    ToolEffect::SpawnMcp { command, args } => {
                        match self.spawn_mcp_server(&command, &args).await {
                            Ok(report) => {
                                content = format!(
                                    "Successfully connected to MCP server. Registered {} new tool(s) from {} listed tool(s).",
                                    report.registered_tools, report.listed_tools,
                                );
                                if report.skipped_existing_tools > 0 {
                                    content.push_str(&format!(
                                        " Skipped {} already-registered tool(s).",
                                        report.skipped_existing_tools
                                    ));
                                }
                            }
                            Err(e) => {
                                content = format!("Failed to connect to MCP server: {}", e);
                                is_error = true;
                            }
                        }
                    }
                },
                ExecutionArtifact::VirtualPlan(plan) => {
                    let result_handler_key = plan.result_handler_key.clone();
                    let mut next_virtual_stack = virtual_stack.clone();
                    next_virtual_stack.push(tc.name.clone());

                    match self.build_virtual_pending_tool_calls(
                        session,
                        &tc.id,
                        plan,
                        &next_virtual_stack,
                    ) {
                        Ok(pending_virtual_calls) => {
                            let (next_content, next_is_error) = self
                                .execute_expanded_virtual_calls(
                                    session,
                                    tool_ctx,
                                    tc.id.clone(),
                                    pending_virtual_calls,
                                    next_virtual_stack,
                                    result_handler_key,
                                )
                                .await;
                            content = next_content;
                            is_error = next_is_error;
                        }
                        Err(err) => {
                            if let Some(ref key) = result_handler_key {
                                self.discard_virtual_result_handler(session, key);
                            }
                            content = format!("Error: {}", err);
                            is_error = true;
                        }
                    }
                }
                ExecutionArtifact::VirtualOutput {
                    content: output,
                    is_error: output_is_error,
                } => {
                    content = output;
                    is_error = is_error || output_is_error;
                }
            }

            final_by_id.insert(
                tc.id.clone(),
                FinalToolRecord {
                    id: tc.id,
                    name: tc.name,
                    args: final_args,
                    verdict: verdict_str,
                    duration_ms,
                    content,
                    is_error,
                    emit_exec_start: false,
                    governance_denial,
                },
            );
        }

        final_by_id
    }
}
