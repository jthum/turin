//! Agent turn execution — streaming, tool dispatch, and result collection.
//!
//! This module contains the `execute_turn` method, which handles a single turn
//! of the agent loop: LLM inference, stream processing, verdict evaluation,
//! parallel tool execution, and side-effect handling.

use anyhow::Result;
use futures::StreamExt;
use futures::future::join_all;
use std::io::{self, BufRead, Write};
use std::time::Instant;
use tracing::{info, warn, error, debug};

use crate::harness::context::ContextWrapper;
use crate::harness::verdict::Verdict;
use crate::inference::provider::{
    self, InferenceContent, InferenceMessage, InferenceRole,
};
use crate::tools::ToolContext;
use super::event::{KernelEvent, LifecycleEvent, StreamEvent, AuditEvent};
use super::{Kernel, PendingToolCall};

impl Kernel {
    /// Execute a single turn of the agent loop. Returns true if loop should continue.
    pub(crate) async fn execute_turn(&mut self, session: &mut super::session::SessionState, tool_ctx: &ToolContext) -> Result<bool> {
        let session_id = session.id.clone();

        // Turn-local configuration
        let mut model = self.config.agent.model.clone();
        let mut provider_name = self.config.agent.provider.clone();
        let mut system_prompt = self.config.agent.system_prompt.clone();

        self.persist_event(session, &KernelEvent::Lifecycle(LifecycleEvent::TurnStart { turn_index: session.turn_index }));

        // ─── Harness Hook: on_before_inference ───────────────────────
        let mut thinking_budget = self.config.agent.thinking.as_ref()
            .and_then(|t| if t.enabled { t.budget_tokens } else { None })
            .unwrap_or(0);

        {
            let harness = self.lock_harness();
            if let Some(ref engine) = *harness {
                let ctx = ContextWrapper::new(
                    model.clone(),
                    provider_name.clone(),
                    system_prompt.clone(),
                    session.history.clone(),
                    0, 128_000,
                    thinking_budget,
                    self.clients.clone(),
                );
                
                match engine.evaluate_userdata("on_before_inference", ctx.clone()) {
                    Ok(verdict) => {
                         if verdict.is_rejected() {
                             warn!(reason = %verdict.reason().unwrap_or(""), "Turn rejected by harness");
                             return Ok(false);
                         }
                    }
                    Err(e) => {
                         warn!(error = %e, "Harness on_before_inference error");
                    }
                }

                let state = ctx.get_state();
                session.history = state.messages;
                system_prompt = state.system_prompt;
                model = state.model;
                provider_name = state.provider;
                thinking_budget = state.thinking_budget;
            }
        }

        if !self.clients.contains_key(&provider_name) {
             if let Some(config) = self.config.providers.get(&provider_name) {
                 debug!(provider = %provider_name, "Lazily initializing provider");
                 match self.create_client(&provider_name, config) {
                     Ok(client) => { self.clients.insert(provider_name.clone(), client); },
                     Err(e) => {
                         error!(provider = %provider_name, error = %e, "Failed to initialize provider");
                         anyhow::bail!("Failed to initialize provider '{}': {}", provider_name, e);
                     }
                 }
             } else {
                 anyhow::bail!("Provider '{}' not found in configuration", provider_name);
             }
        }

        let client = self.clients.get(&provider_name)
            .ok_or_else(|| anyhow::anyhow!("Provider '{}' not initialized", provider_name))?
            .clone();

        let tools = self.tool_registry.tool_definitions();

        let options = provider::InferenceOptions {
            max_tokens: None,
            temperature: None,
            thinking_budget: Some(thinking_budget),
        };

        let mut stream = client.stream(&model, &system_prompt, &session.history, &tools, &options).await?;
        
        let mut response_text = String::with_capacity(4096);
        let mut pending_tool_calls: Vec<PendingToolCall> = Vec::new();

        while let Some(event_result) = stream.next().await {
             let event = event_result?;
             match &event {
                KernelEvent::Stream(e) => match e {
                    StreamEvent::MessageDelta { content_delta } => {
                        if !self.json {
                            print!("{}", content_delta);
                            io::stdout().flush().ok();
                        }
                        self.persist_event(session, &event);
                        response_text.push_str(content_delta);
                    }
                    StreamEvent::ThinkingDelta { thinking: _ } => {
                        self.persist_event(session, &event);
                    }
                    StreamEvent::MessageEnd { input_tokens, output_tokens, .. } => {
                        session.total_input_tokens += *input_tokens;
                        session.total_output_tokens += *output_tokens;
                        self.persist_event(session, &event);
                    }
                    StreamEvent::ToolCall { id, name, args } => {
                        self.persist_event(session, &event);
                        pending_tool_calls.push(PendingToolCall {
                            id: id.clone(), name: name.clone(), args: args.clone()
                        });
                    }
                    _ => { self.persist_event(session, &event); }
                },
                _ => { self.persist_event(session, &event); }
             }
        }

        if !response_text.is_empty() && !response_text.ends_with('\n') { println!(); }

        let has_tool_calls = !pending_tool_calls.is_empty();

        self.persist_event(session, &KernelEvent::Lifecycle(LifecycleEvent::TurnEnd {
            turn_index: session.turn_index,
            has_tool_calls,
        }));

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
            let _ = store.insert_message(&session_id, session.turn_index, "assistant", &serde_json::Value::Array(content), None).await;
        }

        let mut assistant_content: Vec<InferenceContent> = Vec::new();
        if !response_text.is_empty() {
            assistant_content.push(InferenceContent::Text { text: response_text.clone() });
        }
        for tc in &pending_tool_calls {
            assistant_content.push(InferenceContent::ToolUse {
                id: tc.id.clone(), name: tc.name.clone(), input: tc.args.clone(),
            });
        }
        session.history.push(InferenceMessage {
            role: InferenceRole::Assistant,
            content: assistant_content,
            tool_call_id: None,
        });

        if !has_tool_calls {
            return Ok(false);
        }

        // Execute tools
        self.execute_tool_calls(session, &session_id, tool_ctx, pending_tool_calls).await
    }

    /// Phase 1-3 of tool execution: verdict evaluation, parallel execution, and side-effect handling.
    async fn execute_tool_calls(
        &mut self,
        session: &mut super::session::SessionState,
        session_id: &str,
        tool_ctx: &ToolContext,
        pending_tool_calls: Vec<PendingToolCall>,
    ) -> Result<bool> {
        // Phase 1: Evaluate verdicts
        let mut validated_calls = Vec::new();
        let mut tool_results: Vec<InferenceContent> = Vec::new();

        for tc in &pending_tool_calls {
            let verdict = self.evaluate_tool_call(&tc.name, &tc.id, &tc.args);
            match &verdict {
                Verdict::Reject(reason) => {
                     warn!(tool = %tc.name, reason = %reason, "Tool REJECTED by harness");
                     let msg = format!("[HARNESS REJECTED] Tool '{}' blocked: {}", tc.name, reason);
                     self.persist_event(session, &KernelEvent::Audit(AuditEvent::ToolExecStart { id: tc.id.clone(), name: tc.name.clone() }));
                     self.persist_event(session, &KernelEvent::Audit(AuditEvent::ToolResult { id: tc.id.clone(), output: msg.clone(), is_error: true }));
                     self.persist_event(session, &KernelEvent::Audit(AuditEvent::ToolExecEnd { id: tc.id.clone(), success: false }));
                     
                     if let Some(ref store) = self.state {
                          let _ = store.insert_tool_execution(session_id, session.turn_index, &tc.id, &tc.name, &tc.args, Some(&msg), true, Some(0), &verdict.to_string()).await;
                     }
                     tool_results.push(InferenceContent::ToolResult { tool_use_id: tc.id.clone(), content: msg, is_error: true });
                }
                Verdict::Escalate(reason) => {
                     warn!(tool = %tc.name, reason = %reason, "ESCALATION: Tool requires approval");
                     eprint!("[turin] Allow? (y/n): ");
                     io::stderr().flush().ok();
                     let mut input = String::new();
                     let approved = io::stdin().lock().read_line(&mut input).is_ok() && input.trim().eq_ignore_ascii_case("y");
                     if !approved {
                          warn!(tool = %tc.name, "Tool DENIED by user");
                          let msg = format!("[ESCALATION DENIED] Tool '{}' denied: {}", tc.name, reason);
                           self.persist_event(session, &KernelEvent::Audit(AuditEvent::ToolExecStart { id: tc.id.clone(), name: tc.name.clone() }));
                           self.persist_event(session, &KernelEvent::Audit(AuditEvent::ToolResult { id: tc.id.clone(), output: msg.clone(), is_error: true }));
                           self.persist_event(session, &KernelEvent::Audit(AuditEvent::ToolExecEnd { id: tc.id.clone(), success: false }));
                           if let Some(ref store) = self.state {
                                let _ = store.insert_tool_execution(session_id, session.turn_index, &tc.id, &tc.name, &tc.args, Some(&msg), true, Some(0), "escalate_denied").await;
                           }
                           tool_results.push(InferenceContent::ToolResult { tool_use_id: tc.id.clone(), content: msg, is_error: true });
                     } else {
                         info!(tool = %tc.name, "Tool APPROVED by user");
                         validated_calls.push((tc, verdict));
                     }
                }
                Verdict::Allow | Verdict::Modify(_) => {
                    validated_calls.push((tc, verdict));
                }
            }
        }

        // Phase 2: Parallel Execution
        let kernel = &*self;
        let event_tx = session.event_tx.clone();
        let turn_index = session.turn_index;
        let futures = validated_calls.into_iter().map(|(tc, verdict)| {
            let session_id = session_id.to_string();
            let tool_ctx = tool_ctx.clone();
            let event_tx = event_tx.clone();
            async move {
                let verdict_str = verdict.to_string();
                let final_args = match verdict {
                    Verdict::Modify(new_args) => {
                         info!(tool = %tc.name, "Tool arguments MODIFIED by harness");
                         new_args
                    },
                    _ => tc.args.clone()
                };

                let _ = event_tx.send((session_id.clone(), KernelEvent::Audit(AuditEvent::ToolExecStart { id: tc.id.clone(), name: tc.name.clone() })));
                let start = Instant::now();
                let effect_res = kernel.tool_registry.execute(&tc.name, final_args, &tool_ctx).await;
                let duration_ms = start.elapsed().as_millis() as u64;
                
                let is_error = effect_res.is_err();
                let effect = effect_res.unwrap_or_else(|e| {
                    crate::tools::ToolEffect::Output(crate::tools::ToolOutput {
                        content: format!("Error: {}", e),
                        metadata: serde_json::Value::Null,
                    })
                });

                let content = match &effect {
                    crate::tools::ToolEffect::Output(o) => o.content.clone(),
                    crate::tools::ToolEffect::EnqueueTask { title, .. } => format!("Task submitted: {}", title),
                    crate::tools::ToolEffect::SpawnMcp { command, .. } => format!("MCP spawned: {}", command),
                };

                let _ = event_tx.send((session_id.clone(), KernelEvent::Audit(AuditEvent::ToolResult { id: tc.id.clone(), output: content, is_error })));
                let _ = event_tx.send((session_id.clone(), KernelEvent::Audit(AuditEvent::ToolExecEnd { id: tc.id.clone(), success: !is_error })));

                if let Some(ref store) = kernel.state {
                     // For legacy compatibility in state store, we'll try to extract content/metadata from effect
                     let (content, _metadata) = match &effect {
                         crate::tools::ToolEffect::Output(o) => (o.content.clone(), o.metadata.clone()),
                         crate::tools::ToolEffect::EnqueueTask { title, subtasks, .. } => (format!("Task '{}' submitted with {} subtasks.", title, subtasks.len()), serde_json::json!({"action": "submit_task"})),
                         crate::tools::ToolEffect::SpawnMcp { command, .. } => (format!("Requesting MCP connection: {}", command), serde_json::json!({"action": "spawn_mcp"})),
                     };

                     let _ = store.insert_tool_execution(&session_id, turn_index, &tc.id, &tc.name, &tc.args, Some(&content), is_error, Some(duration_ms), &verdict_str).await;
                }
                (tc, effect, is_error)
            }
        });

        let execution_results = join_all(futures).await;

        // Phase 3: Side Effects & Result Collection
        for (tc, effect, mut is_error) in execution_results {
            let mut content;
            
            match effect {
                crate::tools::ToolEffect::Output(o) => {
                    content = o.content;
                }
                crate::tools::ToolEffect::EnqueueTask { title, subtasks, clear_existing } => {
                    let verdict_result = {
                        let harness = self.lock_harness();
                        if let Some(engine) = &*harness {
                            // Map ToolEffect to a metadata-like structure for the harness legacy hook
                            let metadata = serde_json::json!({
                                "action": "submit_task",
                                "title": title,
                                "subtasks": subtasks,
                                "clear_existing": clear_existing
                            });
                            Some(engine.evaluate("on_task_submit", metadata))
                        } else { None }
                    };

                    content = format!("Task '{}' submitted with {} subtasks.", title, subtasks.len());
                    
                    match verdict_result {
                        Some(Ok(Verdict::Allow)) | None => {
                            let mut q = session.queue.lock().await;
                            if clear_existing { q.clear(); }
                            for task in subtasks { q.push_back(task); }
                            debug!("Tasks queued from submit_task");
                        }
                        Some(Ok(Verdict::Modify(new_tasks_val))) => {
                            if let Some(new_tasks) = new_tasks_val.as_array() {
                                let mut q = session.queue.lock().await;
                                if clear_existing { q.clear(); }
                                for task in new_tasks {
                                    if let Some(t) = task.as_str() { q.push_back(t.to_string()); }
                                }
                                debug!("Tasks queued (MODIFIED by harness)");
                            }
                        }
                        Some(Ok(Verdict::Reject(reason))) => {
                            content = format!("Plan REJECTED by Harness: {}", reason);
                            is_error = true;
                        }
                        Some(Ok(Verdict::Escalate(reason))) => {
                            content = format!("Plan paused for approval: {}", reason);
                        }
                        Some(Err(e)) => {
                            error!(error = %e, "Failed to evaluate on_task_submit");
                        }
                    }
                }
                crate::tools::ToolEffect::SpawnMcp { command, args } => {
                     match self.spawn_mcp_server(&command, &args).await {
                         Ok(count) => {
                             content = format!("Successfully connected to MCP server. Loaded {} new tools.", count);
                         },
                         Err(e) => {
                             content = format!("Failed to connect to MCP server: {}", e);
                             is_error = true;
                         }
                     }
                }
            }
            tool_results.push(InferenceContent::ToolResult { tool_use_id: tc.id.clone(), content, is_error });
        }

        session.history.push(InferenceMessage {
            role: InferenceRole::User,
            content: tool_results.clone(),
            tool_call_id: None,
        });

         if let Some(ref store) = self.state {
             let result_content: Vec<serde_json::Value> = tool_results.iter().map(|r| match r {
                 InferenceContent::ToolResult { tool_use_id, content, is_error } => {
                     serde_json::json!({ "type": "tool_result", "tool_use_id": tool_use_id, "content": content, "is_error": is_error })
                 }
                 _ => serde_json::json!({})
             }).collect();
             let _ = store.insert_message(session_id, session.turn_index, "tool_result", &serde_json::Value::Array(result_content), None).await;
         }

         Ok(true)
    }
}
