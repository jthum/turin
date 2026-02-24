use anyhow::Result;
use futures::future::join_all;
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::time::Instant;
use tracing::{error, info, warn};

use crate::harness::verdict::Verdict;
use crate::inference::provider::{InferenceContent, InferenceMessage, InferenceRole};
use crate::kernel::session::{PlanProgress, QueuedTask, SessionState};
use crate::tools::{ToolContext, ToolEffect, ToolOutput};

use super::super::event::{AuditEvent, KernelEvent};
use super::super::{Kernel, PendingToolCall};
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

impl Kernel {
    /// Phase 1-3 of tool execution: verdict evaluation, parallel execution, side effects, and result collection.
    pub(super) async fn execute_tool_calls(
        &mut self,
        session: &mut SessionState,
        tool_ctx: &ToolContext,
        pending_tool_calls: Vec<PendingToolCall>,
    ) -> Result<TurnOutcome> {
        let mut immediate_records: Vec<FinalToolRecord> = Vec::new();
        let mut validated_calls: Vec<(PendingToolCall, Verdict)> = Vec::new();

        for tc in &pending_tool_calls {
            let verdict = self.evaluate_tool_call(&tc.name, &tc.id, &tc.args);
            match &verdict {
                Verdict::Reject(reason) => {
                    if !self.json {
                        println!("\x1b[31m✗ Rejected by harness:\x1b[0m {}", reason);
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
                            println!("\x1b[31m✗ Denied by user\x1b[0m");
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
                            println!("\x1b[32m✓ Approved by user\x1b[0m");
                        }
                        validated_calls.push((tc.clone(), Verdict::Allow));
                    }
                }
                Verdict::Allow | Verdict::Modify(_) => {
                    validated_calls.push((tc.clone(), verdict));
                }
            }
        }

        for (tc, _) in &validated_calls {
            self.persist_event(
                session,
                &KernelEvent::Audit(AuditEvent::ToolExecStart {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                }),
            );
        }

        // Parallel execution for approved calls.
        let kernel = &*self;
        let futures = validated_calls.into_iter().map(|(tc, verdict)| {
            let tool_ctx = tool_ctx.clone();
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
                let effect_res = kernel
                    .tool_registry
                    .execute(&tc.name, final_args.clone(), &tool_ctx)
                    .await;
                let duration_ms = start.elapsed().as_millis() as u64;

                let is_error = effect_res.is_err();
                let effect = effect_res.unwrap_or_else(|e| {
                    ToolEffect::Output(ToolOutput {
                        content: format!("Error: {}", e),
                        metadata: serde_json::Value::Null,
                    })
                });

                (tc, final_args, verdict_str, duration_ms, effect, is_error)
            }
        });

        let execution_results = join_all(futures).await;

        let mut final_by_id: HashMap<String, FinalToolRecord> = HashMap::new();

        for record in immediate_records {
            final_by_id.insert(record.id.clone(), record);
        }

        for (tc, final_args, verdict_str, duration_ms, effect, mut is_error) in execution_results {
            let content;

            match effect {
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

        let mut tool_results: Vec<InferenceContent> = Vec::new();

        for tc in &pending_tool_calls {
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
                if record.is_error {
                    println!("\x1b[31m✗ Tool '{}' failed\x1b[0m", record.name);
                } else {
                    println!("\x1b[32m✓ Tool '{}' complete\x1b[0m", record.name);
                }
            }

            tool_results.push(InferenceContent::ToolResult {
                tool_use_id: record.id,
                content: record.content,
                is_error: record.is_error,
            });
        }

        session.history.push(InferenceMessage {
            role: InferenceRole::User,
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

        Ok(TurnOutcome::Continue)
    }

    fn prompt_for_approval(&self, reason: &str) -> bool {
        warn!(reason = %reason, "Escalation requires user approval");
        eprint!(
            "\x1b[33m\x1b[1m! Approval Required:\x1b[0m {} Allow? (y/n): ",
            reason
        );
        io::stderr().flush().ok();

        let mut input = String::new();
        io::stdin().lock().read_line(&mut input).is_ok() && input.trim().eq_ignore_ascii_case("y")
    }

    fn apply_tool_result_hook(
        &self,
        id: &str,
        name: &str,
        args: &serde_json::Value,
        content: String,
        is_error: bool,
    ) -> (String, bool) {
        let harness = self.lock_harness();
        let Some(engine) = &*harness else {
            return (content, is_error);
        };

        let payload = serde_json::json!({
            "id": id,
            "name": name,
            "args": args,
            "output": content,
            "is_error": is_error,
        });

        match engine.evaluate("on_tool_result", payload) {
            Ok(Verdict::Allow) => (content, is_error),
            Ok(Verdict::Reject(reason)) => (
                format!(
                    "[HARNESS REJECTED RESULT] Tool '{}' result blocked: {}",
                    name, reason
                ),
                true,
            ),
            Ok(Verdict::Escalate(reason)) => {
                if self.prompt_for_approval(&reason) {
                    (content, is_error)
                } else {
                    (
                        format!(
                            "[ESCALATION DENIED] Tool '{}' result denied by user: {}",
                            name, reason
                        ),
                        true,
                    )
                }
            }
            Ok(Verdict::Modify(val)) => {
                if let Some(s) = val.as_str() {
                    return (s.to_string(), is_error);
                }
                if let Some(obj) = val.as_object() {
                    let new_content = obj
                        .get("output")
                        .and_then(|v| v.as_str())
                        .or_else(|| obj.get("content").and_then(|v| v.as_str()))
                        .map(ToString::to_string)
                        .unwrap_or(content);
                    let new_is_error = obj
                        .get("is_error")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(is_error);
                    return (new_content, new_is_error);
                }
                warn!(tool = %name, "on_tool_result returned unsupported MODIFY payload; ignoring");
                (content, is_error)
            }
            Err(e) => {
                warn!(error = %e, "Harness on_tool_result error");
                (content, is_error)
            }
        }
    }

    async fn handle_plan_submission(
        &mut self,
        session: &mut SessionState,
        title: &str,
        tasks: Vec<String>,
        clear_existing: bool,
    ) -> (String, bool) {
        let mut plan_title = title.to_string();
        let mut plan_tasks = tasks;
        let mut should_clear_existing = clear_existing;

        let verdict_result = {
            let harness = self.lock_harness();
            (*harness).as_ref().map(|engine| {
                engine.evaluate(
                    "on_plan_submit",
                    serde_json::json!({
                        "title": plan_title.clone(),
                        "tasks": plan_tasks.clone(),
                        "clear_existing": should_clear_existing,
                    }),
                )
            })
        };

        match verdict_result {
            Some(Ok(Verdict::Allow)) | None => {}
            Some(Ok(Verdict::Reject(reason))) => {
                return (format!("Plan rejected by harness: {}", reason), true);
            }
            Some(Ok(Verdict::Escalate(reason))) => {
                if !self.prompt_for_approval(&reason) {
                    return (format!("Plan escalation denied by user: {}", reason), true);
                }
            }
            Some(Ok(Verdict::Modify(new_val))) => {
                if let Some(obj) = new_val.as_object() {
                    if let Some(new_title) = obj.get("title").and_then(|v| v.as_str()) {
                        plan_title = new_title.to_string();
                    }
                    if let Some(new_clear) = obj.get("clear_existing").and_then(|v| v.as_bool()) {
                        should_clear_existing = new_clear;
                    }
                    if let Some(new_tasks_val) = obj.get("tasks") {
                        plan_tasks = Kernel::parse_task_list(new_tasks_val, None, None)
                            .into_iter()
                            .map(|t| t.prompt)
                            .collect();
                    }
                } else if new_val.is_array() {
                    plan_tasks = Kernel::parse_task_list(&new_val, None, None)
                        .into_iter()
                        .map(|t| t.prompt)
                        .collect();
                }
            }
            Some(Err(e)) => {
                error!(error = %e, "Failed to evaluate on_plan_submit");
            }
        }

        if plan_tasks.is_empty() {
            return (
                "Plan submission rejected: no tasks were provided".to_string(),
                true,
            );
        }

        let cancelled_count = if should_clear_existing {
            match self.cancel_queued_tasks(session).await {
                Ok(cancelled) => cancelled,
                Err(e) => {
                    return (
                        format!("Plan submission failed while clearing queue: {}", e),
                        true,
                    );
                }
            }
        } else {
            0
        };

        let plan_id = format!("p_{}", session.next_plan_id);
        session.next_plan_id += 1;
        session.plans.insert(
            plan_id.clone(),
            PlanProgress {
                plan_id: plan_id.clone(),
                title: plan_title.clone(),
                total_tasks: plan_tasks.len(),
                completed_tasks: 0,
            },
        );

        let scheduled_tasks = plan_tasks
            .into_iter()
            .map(|prompt| {
                let mut qt =
                    QueuedTask::with_plan(prompt, plan_id.clone(), Some(plan_title.clone()));
                qt.task_id = format!("t_{}", session.next_task_id);
                session.next_task_id += 1;
                qt
            })
            .collect::<Vec<_>>();

        let queued_count = scheduled_tasks.len();
        {
            let mut q = session.queue.lock().await;
            for task in scheduled_tasks {
                q.push_back(task);
            }
        }

        if cancelled_count > 0 {
            (
                format!(
                    "Plan '{}' submitted (plan_id: {}) with {} tasks. Cancelled {} queued tasks.",
                    plan_title, plan_id, queued_count, cancelled_count
                ),
                false,
            )
        } else {
            (
                format!(
                    "Plan '{}' submitted (plan_id: {}) with {} tasks.",
                    plan_title, plan_id, queued_count
                ),
                false,
            )
        }
    }
}
