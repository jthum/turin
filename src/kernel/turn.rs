//! Agent turn execution — streaming, tool dispatch, and result collection.
//!
//! This module contains turn-level execution for the agent loop: LLM inference,
//! stream processing, hook evaluation, parallel tool execution, and side effects.

use anyhow::{Context, Result};
use futures::StreamExt;
use futures::future::join_all;
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

use crate::harness::context::{ContextWrapper, RequestOptionsOverride};
use crate::harness::verdict::Verdict;
use crate::inference::provider::{self, InferenceContent, InferenceMessage, InferenceRole};
use crate::kernel::session::{PlanProgress, QueuedTask, SessionState};
use crate::tools::{ToolContext, ToolEffect, ToolOutput};

use super::event::{AuditEvent, KernelEvent, LifecycleEvent, StreamEvent};
use super::{Kernel, PendingToolCall};

#[derive(Debug, Clone)]
pub(crate) struct TurnContext {
    pub task_id: String,
    pub plan_id: Option<String>,
    pub task_turn_index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TurnOutcome {
    Continue,
    Complete,
    Rejected,
}

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

fn merge_request_option_overrides(
    mut options: provider::RequestOptions,
    overrides: &RequestOptionsOverride,
) -> Result<provider::RequestOptions> {
    for (header_name, header_value) in &overrides.headers {
        options = options
            .with_header(header_name, header_value)
            .with_context(|| format!("invalid request header '{}'", header_name))?;
    }

    if let Some(max_retries) = overrides.max_retries {
        options = options.with_max_retries(max_retries);
    }

    if overrides.request_timeout_secs.is_some() || overrides.total_timeout_secs.is_some() {
        let mut timeout_policy = options.timeout_policy.clone().unwrap_or_default();
        if let Some(request_timeout_secs) = overrides.request_timeout_secs {
            timeout_policy.request_timeout = Some(Duration::from_secs(request_timeout_secs));
        }
        if let Some(total_timeout_secs) = overrides.total_timeout_secs {
            timeout_policy.total_timeout = Some(Duration::from_secs(total_timeout_secs));
        }
        options = options.with_timeout_policy(timeout_policy);
    }

    Ok(options)
}

impl Kernel {
    /// Execute a single turn of the agent loop.
    pub(crate) async fn execute_turn(
        &mut self,
        session: &mut SessionState,
        tool_ctx: &ToolContext,
        turn_ctx: &TurnContext,
    ) -> Result<TurnOutcome> {
        let session_id = session.identity.session_id.clone();

        // Turn-local configuration
        let mut model = self.config.agent.model.clone();
        let mut provider_name = self.config.agent.provider.clone();
        let mut system_prompt = self.config.agent.system_prompt.clone();

        if !self.json {
            println!(
                "\n\x1b[36m\x1b[1m── Turn {} ──\x1b[0m",
                session.turn_index + 1
            );
        }

        self.persist_event(
            session,
            &KernelEvent::Lifecycle(LifecycleEvent::TurnStart {
                identity: session.identity.clone(),
                turn_index: session.turn_index,
                task_id: turn_ctx.task_id.clone(),
                task_turn_index: turn_ctx.task_turn_index,
            }),
        );

        // Optional gate at turn start.
        {
            let harness = self.lock_harness();
            if let Some(ref engine) = *harness {
                match engine.evaluate(
                    "on_turn_start",
                    serde_json::json!({
                        "identity": session.identity.clone(),
                        "session_id": session.identity.session_id.clone(),
                        "task_id": turn_ctx.task_id.clone(),
                        "plan_id": turn_ctx.plan_id.clone(),
                        "turn_index": session.turn_index,
                        "task_turn_index": turn_ctx.task_turn_index,
                    }),
                ) {
                    Ok(Verdict::Reject(reason)) => {
                        warn!(reason = %reason, "Turn rejected by on_turn_start");
                        return Ok(TurnOutcome::Rejected);
                    }
                    Ok(Verdict::Escalate(reason)) => {
                        warn!(reason = %reason, "Turn escalated by on_turn_start; treating as rejected");
                        return Ok(TurnOutcome::Rejected);
                    }
                    Ok(_) => {}
                    Err(e) => {
                        warn!(error = %e, "Harness on_turn_start error");
                    }
                }
            }
        }

        self.persist_event(
            session,
            &KernelEvent::Lifecycle(LifecycleEvent::TurnPrepare {
                identity: session.identity.clone(),
                turn_index: session.turn_index,
                task_id: turn_ctx.task_id.clone(),
                task_turn_index: turn_ctx.task_turn_index,
            }),
        );

        // ─── Harness Hook: on_turn_prepare ─────────────────────────────
        let mut thinking_budget = self
            .config
            .agent
            .thinking
            .as_ref()
            .and_then(|t| if t.enabled { t.budget_tokens } else { None })
            .unwrap_or(0);
        let mut request_options_override = RequestOptionsOverride::default();

        {
            let harness = self.lock_harness();
            if let Some(ref engine) = *harness {
                let ctx = ContextWrapper::new(
                    model.clone(),
                    provider_name.clone(),
                    system_prompt.clone(),
                    session.history.clone(),
                    session.turn_index,
                    turn_ctx.task_turn_index,
                    turn_ctx.task_turn_index == 0,
                    turn_ctx.task_id.clone(),
                    turn_ctx.plan_id.clone(),
                    0,
                    128_000,
                    thinking_budget,
                    request_options_override.clone(),
                    self.clients.clone(),
                );

                match engine.evaluate_userdata("on_turn_prepare", ctx.clone()) {
                    Ok(Verdict::Reject(reason)) => {
                        warn!(reason = %reason, "Turn rejected by on_turn_prepare");
                        return Ok(TurnOutcome::Rejected);
                    }
                    Ok(Verdict::Escalate(reason)) => {
                        warn!(reason = %reason, "Turn escalated by on_turn_prepare; treating as rejected");
                        return Ok(TurnOutcome::Rejected);
                    }
                    Ok(_) => {}
                    Err(e) => {
                        warn!(error = %e, "Harness on_turn_prepare error");
                    }
                }

                let state = ctx.get_state();
                session.history = state.messages;
                system_prompt = state.system_prompt;
                model = state.model;
                provider_name = state.provider;
                thinking_budget = state.thinking_budget;
                request_options_override = state.request_options;
            }
        }

        if !self.clients.contains_key(&provider_name) {
            if let Some(config) = self.config.providers.get(&provider_name) {
                debug!(provider = %provider_name, "Lazily initializing provider");
                match self.create_client(&provider_name, config) {
                    Ok(client) => {
                        self.clients.insert(provider_name.clone(), client);
                    }
                    Err(e) => {
                        error!(provider = %provider_name, error = %e, "Failed to initialize provider");
                        anyhow::bail!("Failed to initialize provider '{}': {}", provider_name, e);
                    }
                }
            } else {
                anyhow::bail!("Provider '{}' not found in configuration", provider_name);
            }
        }

        let client = self
            .clients
            .get(&provider_name)
            .ok_or_else(|| anyhow::anyhow!("Provider '{}' not initialized", provider_name))?
            .clone();
        let provider_config = self.config.providers.get(&provider_name).ok_or_else(|| {
            anyhow::anyhow!("Provider '{}' not found in configuration", provider_name)
        })?;

        let tools = self.tool_registry.tool_definitions();

        let options = provider::InferenceOptions {
            max_tokens: None,
            temperature: None,
            thinking_budget: Some(thinking_budget),
        };
        let request_options = merge_request_option_overrides(
            provider::build_request_options(provider_config)?,
            &request_options_override,
        )?;

        let mut stream = client
            .stream(
                &model,
                &system_prompt,
                &session.history,
                &tools,
                &options,
                Some(request_options),
            )
            .await
            .with_context(|| {
                format!(
                    "failed to start inference stream (provider='{}', model='{}')",
                    provider_name, model
                )
            })?;

        let mut response_text = String::with_capacity(4096);
        let mut pending_tool_calls: Vec<PendingToolCall> = Vec::new();
        let mut is_thinking = false;

        while let Some(event_result) = stream.next().await {
            let event = event_result.with_context(|| {
                format!(
                    "inference stream event failure (provider='{}', model='{}')",
                    provider_name, model
                )
            })?;
            match &event {
                KernelEvent::Stream(e) => match e {
                    StreamEvent::ThinkingDelta { .. } => {
                        if !self.json && !is_thinking {
                            print!("\x1b[35m💭 Thinking...\x1b[0m");
                            io::stdout().flush().ok();
                            is_thinking = true;
                        }
                        self.persist_event(session, &event);
                    }
                    StreamEvent::MessageDelta { content_delta } => {
                        if is_thinking {
                            if !self.json {
                                println!();
                            }
                            is_thinking = false;
                        }
                        if !self.json {
                            print!("{}", content_delta);
                            io::stdout().flush().ok();
                        }
                        self.persist_event(session, &event);
                        response_text.push_str(content_delta);
                    }
                    StreamEvent::MessageEnd {
                        input_tokens,
                        output_tokens,
                        ..
                    } => {
                        if is_thinking {
                            if !self.json {
                                println!();
                            }
                            is_thinking = false;
                        }
                        session.total_input_tokens += *input_tokens;
                        session.total_output_tokens += *output_tokens;
                        self.persist_event(session, &event);
                    }
                    StreamEvent::ToolCall { id, name, args } => {
                        if is_thinking {
                            if !self.json {
                                println!();
                            }
                            is_thinking = false;
                        }
                        if !self.json {
                            println!(
                                "\n\x1b[33m⚒️  Tool Call:\x1b[0m \x1b[1m{}\x1b[0m({})",
                                name, args
                            );
                        }
                        self.persist_event(session, &event);
                        pending_tool_calls.push(PendingToolCall {
                            id: id.clone(),
                            name: name.clone(),
                            args: args.clone(),
                        });
                    }
                    _ => {
                        self.persist_event(session, &event);
                    }
                },
                _ => {
                    self.persist_event(session, &event);
                }
            }
        }

        if !self.json && !response_text.is_empty() && !response_text.ends_with('\n') {
            println!();
        }

        let has_tool_calls = !pending_tool_calls.is_empty();

        self.persist_event(
            session,
            &KernelEvent::Lifecycle(LifecycleEvent::TurnEnd {
                identity: session.identity.clone(),
                turn_index: session.turn_index,
                task_id: turn_ctx.task_id.clone(),
                task_turn_index: turn_ctx.task_turn_index,
                has_tool_calls,
            }),
        );

        {
            let harness = self.lock_harness();
            if let Some(ref engine) = *harness
                && let Err(e) = engine.evaluate(
                    "on_turn_end",
                    serde_json::json!({
                        "identity": session.identity.clone(),
                        "session_id": session.identity.session_id.clone(),
                        "task_id": turn_ctx.task_id.clone(),
                        "plan_id": turn_ctx.plan_id.clone(),
                        "turn_index": session.turn_index,
                        "task_turn_index": turn_ctx.task_turn_index,
                        "has_tool_calls": has_tool_calls,
                    }),
                )
            {
                warn!(error = %e, "Harness on_turn_end error");
            }
        }

        if let Some(ref store) = self.state {
            let content: Vec<serde_json::Value> = {
                let mut parts = Vec::new();
                if !response_text.is_empty() {
                    parts.push(serde_json::json!({"type": "text", "text": response_text}));
                }
                for tc in &pending_tool_calls {
                    parts.push(serde_json::json!({
                        "type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args,
                    }));
                }
                parts
            };
            let _ = store
                .insert_message(
                    &session_id,
                    session.turn_index,
                    "assistant",
                    &serde_json::Value::Array(content),
                    None,
                )
                .await;
        }

        let mut assistant_content: Vec<InferenceContent> = Vec::new();
        if !response_text.is_empty() {
            assistant_content.push(InferenceContent::Text {
                text: response_text.clone(),
            });
        }
        for tc in &pending_tool_calls {
            assistant_content.push(InferenceContent::ToolUse {
                id: tc.id.clone(),
                name: tc.name.clone(),
                input: tc.args.clone(),
            });
        }
        session.history.push(InferenceMessage {
            role: InferenceRole::Assistant,
            content: assistant_content,
            tool_call_id: None,
        });

        if !has_tool_calls {
            return Ok(TurnOutcome::Complete);
        }

        // Execute tools.
        self.execute_tool_calls(session, &session_id, tool_ctx, pending_tool_calls)
            .await
    }

    /// Phase 1-3 of tool execution: verdict evaluation, parallel execution, side effects, and result collection.
    async fn execute_tool_calls(
        &mut self,
        session: &mut SessionState,
        session_id: &str,
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

        // Parallel execution for approved calls.
        let kernel = &*self;
        let event_tx = session.event_tx.clone();
        let futures = validated_calls.into_iter().map(|(tc, verdict)| {
            let session_id = session_id.to_string();
            let tool_ctx = tool_ctx.clone();
            let event_tx = event_tx.clone();
            async move {
                let verdict_str = verdict.to_string();
                let final_args = match verdict {
                    Verdict::Modify(new_args) => {
                        info!(tool = %tc.name, "Tool arguments modified by harness");
                        new_args
                    }
                    _ => tc.args.clone(),
                };

                let _ = event_tx.send((
                    session_id.clone(),
                    KernelEvent::Audit(AuditEvent::ToolExecStart {
                        id: tc.id.clone(),
                        name: tc.name.clone(),
                    }),
                ));

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

            if let Some(ref store) = self.state {
                let _ = store
                    .insert_tool_execution(
                        session_id,
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

        if let Some(ref store) = self.state {
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
            let _ = store
                .insert_message(
                    session_id,
                    session.turn_index,
                    "tool_result",
                    &serde_json::Value::Array(result_content),
                    None,
                )
                .await;
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

        let plan_id = uuid::Uuid::new_v4().to_string();
        session.plans.insert(
            plan_id.clone(),
            PlanProgress {
                plan_id: plan_id.clone(),
                title: plan_title.clone(),
                total_tasks: plan_tasks.len(),
                completed_tasks: 0,
            },
        );

        let queued_tasks: Vec<QueuedTask> = plan_tasks
            .into_iter()
            .map(|prompt| QueuedTask::with_plan(prompt, plan_id.clone(), Some(plan_title.clone())))
            .collect();

        let queued_count = queued_tasks.len();
        {
            let mut q = session.queue.lock().await;
            for task in queued_tasks {
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
