mod plan_submission;
mod result_hooks;

use anyhow::Result;
use futures::future::join_all;
use std::collections::{BTreeSet, HashMap};
use std::future::Future;
use std::pin::Pin;
use std::time::Instant;
use tracing::{info, warn};

use crate::display;
use crate::harness::verdict::Verdict;
use crate::harness::virtual_tools::{VirtualToolNestedResult, VirtualToolPlan};
use crate::inference::provider::{InferenceContent, InferenceMessage, InferenceRole};
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::governance::{GovernanceSubject, tool_capability_name};
use crate::kernel::session::SessionState;
use crate::tools::{ToolContext, ToolEffect, ToolError, ToolOutput};

use super::super::PendingToolCall;
use super::super::event::{AuditEvent, KernelEvent};
use super::TurnOutcome;

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
}

enum ExecutionArtifact {
    Native(ToolEffect),
    VirtualPlan(VirtualToolPlan),
}

impl ExecutionHost {
    /// Phase 1-3 of tool execution: verdict evaluation, parallel execution, side effects, and result collection.
    pub(super) async fn execute_tool_calls(
        &mut self,
        session: &mut SessionState,
        tool_ctx: &ToolContext,
        pending_tool_calls: Vec<PendingToolCall>,
    ) -> Result<TurnOutcome> {
        if session.cancel_token.is_cancelled() {
            return Ok(TurnOutcome::Cancelled);
        }

        let (immediate_records, validated_calls) =
            self.evaluate_pending_tool_calls(session, &pending_tool_calls);
        let final_by_id = self
            .execute_validated_tool_calls(session, tool_ctx, validated_calls, immediate_records)
            .await;
        if session.cancel_token.is_cancelled() {
            return Ok(TurnOutcome::Cancelled);
        }

        self.finalize_tool_records(session, &pending_tool_calls, final_by_id, true)
            .await;
        Ok(TurnOutcome::Continue)
    }

    fn execute_tool_calls_hidden<'a>(
        &'a mut self,
        session: &'a mut SessionState,
        tool_ctx: &'a ToolContext,
        pending_tool_calls: Vec<PendingToolCall>,
    ) -> Pin<Box<dyn Future<Output = Vec<FinalToolRecord>> + Send + 'a>> {
        Box::pin(async move {
            if session.cancel_token.is_cancelled() {
                return Vec::new();
            }

            let (immediate_records, validated_calls) =
                self.evaluate_pending_tool_calls(session, &pending_tool_calls);
            let final_by_id = self
                .execute_validated_tool_calls(session, tool_ctx, validated_calls, immediate_records)
                .await;
            if session.cancel_token.is_cancelled() {
                return Vec::new();
            }

            self.finalize_tool_records(session, &pending_tool_calls, final_by_id, false)
                .await
        })
    }

    fn evaluate_pending_tool_calls(
        &self,
        session: &SessionState,
        pending_tool_calls: &[PendingToolCall],
    ) -> (Vec<FinalToolRecord>, Vec<(PendingToolCall, Verdict)>) {
        let mut immediate_records: Vec<FinalToolRecord> = Vec::new();
        let mut validated_calls: Vec<(PendingToolCall, Verdict)> = Vec::new();
        let ansi_stdout = display::stdout_ansi();

        for tc in pending_tool_calls {
            let verdict = self.evaluate_tool_call(session, &tc.name, &tc.id, &tc.args);
            match &verdict {
                Verdict::Reject(reason) => {
                    if !self.json {
                        println!(
                            "{}",
                            display::rejection_line("✗ Rejected by harness:", reason, ansi_stdout,)
                        );
                    }
                    warn!(tool = %tc.name, reason = %reason, "Tool rejected by on_tool_call");
                    let msg = format!("[HARNESS REJECTED] Tool '{}' blocked: {}", tc.name, reason);
                    immediate_records.push(FinalToolRecord {
                        id: tc.id.clone(),
                        name: tc.name.clone(),
                        args: tc.args.clone(),
                        verdict: verdict.to_string(),
                        duration_ms: 0,
                        content: msg,
                        is_error: true,
                        emit_exec_start: true,
                    });
                }
                Verdict::Escalate(reason) => {
                    warn!(tool = %tc.name, reason = %reason, "Tool requires escalation");
                    if !self.prompt_for_approval(reason) {
                        if !self.json {
                            println!("{}", display::approval_line(false, ansi_stdout));
                        }
                        let msg =
                            format!("[ESCALATION DENIED] Tool '{}' denied: {}", tc.name, reason);
                        immediate_records.push(FinalToolRecord {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            args: tc.args.clone(),
                            verdict: "escalate_denied".to_string(),
                            duration_ms: 0,
                            content: msg,
                            is_error: true,
                            emit_exec_start: true,
                        });
                    } else {
                        if !self.json {
                            println!("{}", display::approval_line(true, ansi_stdout));
                        }
                        validated_calls.push((tc.clone(), Verdict::Allow));
                    }
                }
                Verdict::Allow | Verdict::Modify(_) => {
                    validated_calls.push((tc.clone(), verdict));
                }
            }
        }

        (immediate_records, validated_calls)
    }

    async fn execute_validated_tool_calls(
        &mut self,
        session: &mut SessionState,
        tool_ctx: &ToolContext,
        validated_calls: Vec<(PendingToolCall, Verdict)>,
        immediate_records: Vec<FinalToolRecord>,
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
        let runtime = self.runtime_for_session(session);
        let futures = validated_calls.into_iter().map(|(tc, verdict)| {
            let tool_ctx = tool_ctx.clone();
            let active_agent_id = active_agent_id.clone();
            let runtime = runtime.clone();
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
                let effect_res = if kernel.tool_registry.get(&tc.name).is_some() {
                    if let Some(capability) = tool_capability_name(&tc.name) {
                        let subject = GovernanceSubject::for_agent(active_agent_id.as_str());
                        match kernel
                            .governance_manager
                            .require_capability_for_subject(&subject, capability)
                        {
                            Ok(()) => kernel
                                .tool_registry
                                .execute(&tc.name, final_args.clone(), &tool_ctx)
                                .await
                                .map(ExecutionArtifact::Native),
                            Err(err) => Err(ToolError::PermissionDenied(err)),
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
                        let harness = runtime.lock_engine();
                        if let Some(ref engine) = *harness {
                            engine.invoke_virtual_tool(&tc.name, final_args.clone())
                        } else {
                            Ok(None)
                        }
                    };

                    match plan_res {
                        Ok(Some(plan)) => Ok(ExecutionArtifact::VirtualPlan(plan)),
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

                (tc, final_args, verdict_str, duration_ms, effect, is_error)
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
            results = join_all(futures) => results,
        };

        for (tc, final_args, verdict_str, duration_ms, effect, mut is_error) in execution_results {
            let content;
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
                            Ok(count) => {
                                content = format!(
                                    "Successfully connected to MCP server. Loaded {} new tools.",
                                    count
                                );
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
                    match self.build_virtual_pending_tool_calls(session, &tc.id, plan) {
                        Ok(pending_virtual_calls) => {
                            let nested_records = self
                                .execute_tool_calls_hidden(session, tool_ctx, pending_virtual_calls)
                                .await;
                            if session.cancel_token.is_cancelled() {
                                if let Some(ref key) = result_handler_key {
                                    self.discard_virtual_result_handler(session, key);
                                }
                                content = "Virtual tool execution cancelled".to_string();
                                is_error = true;
                            } else {
                                let nested_results =
                                    self.virtual_tool_nested_results(&nested_records);
                                let nested_error =
                                    nested_results.iter().any(|record| record.is_error);
                                if let Some(ref key) = result_handler_key {
                                    match self.invoke_virtual_result_handler(
                                        session,
                                        key,
                                        &nested_results,
                                        nested_error,
                                    ) {
                                        Ok(output) => {
                                            content = output.content;
                                            is_error = output.is_error;
                                        }
                                        Err(err) => {
                                            content = format!("Error: {}", err);
                                            is_error = true;
                                        }
                                    }
                                } else {
                                    let (aggregated, nested_error) =
                                        self.aggregate_virtual_tool_records(&nested_records);
                                    content = aggregated;
                                    is_error = is_error || nested_error;
                                }
                            }
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
                },
            );
        }

        final_by_id
    }

    fn build_virtual_pending_tool_calls(
        &self,
        session: &SessionState,
        parent_tool_call_id: &str,
        plan: VirtualToolPlan,
    ) -> Result<Vec<PendingToolCall>> {
        let declared_tool_names: BTreeSet<String> = {
            let runtime = self.runtime_for_session(session);
            let harness = runtime.lock_engine();
            if let Some(ref engine) = *harness {
                engine
                    .declared_virtual_tools()?
                    .into_iter()
                    .map(|tool| tool.name)
                    .collect()
            } else {
                BTreeSet::new()
            }
        };

        let mut out = Vec::with_capacity(plan.calls.len());
        for (index, call) in plan.calls.into_iter().enumerate() {
            if declared_tool_names.contains(&call.name) {
                anyhow::bail!(
                    "virtual tool handlers cannot call other virtual tools yet ('{}')",
                    call.name
                );
            }

            out.push(PendingToolCall {
                id: format!("{}::vt{}", parent_tool_call_id, index + 1),
                name: call.name,
                args: call.args,
            });
        }
        Ok(out)
    }

    fn aggregate_virtual_tool_records(&self, records: &[FinalToolRecord]) -> (String, bool) {
        if records.is_empty() {
            return (
                "Virtual tool completed without producing any nested tool output.".to_string(),
                false,
            );
        }

        let any_error = records.iter().any(|record| record.is_error);
        if records.len() == 1 {
            return (records[0].content.clone(), any_error);
        }

        let combined = records
            .iter()
            .enumerate()
            .map(|(index, record)| {
                let status = if record.is_error { "error" } else { "ok" };
                format!(
                    "Call {}: {} [{}]\n{}",
                    index + 1,
                    record.name,
                    status,
                    record.content
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        (combined, any_error)
    }

    fn virtual_tool_nested_results(
        &self,
        records: &[FinalToolRecord],
    ) -> Vec<VirtualToolNestedResult> {
        records
            .iter()
            .map(|record| VirtualToolNestedResult {
                id: record.id.clone(),
                name: record.name.clone(),
                args: record.args.clone(),
                verdict: record.verdict.clone(),
                duration_ms: record.duration_ms,
                content: record.content.clone(),
                is_error: record.is_error,
            })
            .collect()
    }

    fn invoke_virtual_result_handler(
        &self,
        session: &SessionState,
        key: &str,
        nested_results: &[VirtualToolNestedResult],
        nested_error: bool,
    ) -> Result<crate::harness::virtual_tools::VirtualToolResultOutput> {
        let payload = if nested_results.len() == 1 {
            serde_json::to_value(&nested_results[0])?
        } else {
            serde_json::to_value(nested_results)?
        };
        let runtime = self.runtime_for_session(session);
        let harness = runtime.lock_engine();
        if let Some(ref engine) = *harness {
            engine.invoke_virtual_tool_result_handler(key, payload, nested_error)
        } else {
            anyhow::bail!(
                "virtual tool result handler '{}' could not access harness engine",
                key
            );
        }
    }

    fn discard_virtual_result_handler(&self, session: &SessionState, key: &str) {
        let runtime = self.runtime_for_session(session);
        let harness = runtime.lock_engine();
        if let Some(ref engine) = *harness {
            let _ = engine.discard_virtual_tool_result_handler(key);
        }
    }

    async fn finalize_tool_records(
        &mut self,
        session: &mut SessionState,
        pending_tool_calls: &[PendingToolCall],
        mut final_by_id: HashMap<String, FinalToolRecord>,
        publish_to_history: bool,
    ) -> Vec<FinalToolRecord> {
        let mut tool_results: Vec<InferenceContent> = Vec::new();
        let mut finalized_records = Vec::new();
        let ansi_stdout = display::stdout_ansi();

        for tc in pending_tool_calls {
            let Some(mut record) = final_by_id.remove(&tc.id) else {
                continue;
            };

            if record.emit_exec_start {
                self.persist_event(
                    session,
                    &KernelEvent::Audit(AuditEvent::ToolExecStart {
                        id: record.id.clone(),
                        name: record.name.clone(),
                    }),
                );
            }

            let (content, is_error) = self.apply_tool_result_hook(
                session,
                &record.id,
                &record.name,
                &record.args,
                record.content,
                record.is_error,
            );
            record.content = content;
            record.is_error = is_error;

            self.persist_event(
                session,
                &KernelEvent::Audit(AuditEvent::ToolResult {
                    id: record.id.clone(),
                    output: record.content.clone(),
                    is_error: record.is_error,
                }),
            );
            self.persist_event(
                session,
                &KernelEvent::Audit(AuditEvent::ToolExecEnd {
                    id: record.id.clone(),
                    success: !record.is_error,
                }),
            );

            if let Ok(store) = self.store_manager.get_default().await
                && let Some(iid) = session.internal_id
            {
                let _ = store
                    .insert_tool_execution(
                        iid,
                        session.turn_index,
                        &record.id,
                        &record.name,
                        &record.args,
                        Some(&record.content),
                        record.is_error,
                        Some(record.duration_ms),
                        &record.verdict,
                    )
                    .await;
            }

            if !self.json {
                println!(
                    "{}",
                    display::tool_status_line(&record.name, !record.is_error, ansi_stdout)
                );
            }

            if publish_to_history {
                tool_results.push(InferenceContent::ToolResult {
                    tool_use_id: record.id.clone(),
                    content: record.content.clone(),
                    is_error: record.is_error,
                });
            }

            finalized_records.push(record);
        }

        if publish_to_history && !tool_results.is_empty() {
            session.history.push(InferenceMessage {
                role: InferenceRole::Tool,
                content: tool_results.clone(),
                tool_call_id: None,
            });

            if let Ok(store) = self.store_manager.get_default().await {
                let result_content: Vec<serde_json::Value> = tool_results
                    .iter()
                    .map(|r| match r {
                        InferenceContent::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            serde_json::json!({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": content,
                                "is_error": is_error
                            })
                        }
                        _ => serde_json::json!({}),
                    })
                    .collect();
                if let Some(iid) = session.internal_id {
                    let _ = store
                        .insert_message(
                            iid,
                            session.turn_index,
                            "tool_result",
                            &serde_json::Value::Array(result_content),
                            None,
                        )
                        .await;
                }
            }
        }

        finalized_records
    }
}
