//! Agent turn execution — streaming, tool dispatch, and result collection.
//!
//! This module contains turn-level execution for the agent loop: LLM inference,
//! stream processing, hook evaluation, parallel tool execution, and side effects.

mod tool_execution;

use anyhow::{Context, Result};
use futures::StreamExt;
use std::io::{self, Write};
use std::time::Duration;
use tracing::{debug, error, warn};

use crate::harness::context::{ContextWrapper, RequestOptionsOverride};
use crate::harness::verdict::Verdict;
use crate::inference::provider::{self, InferenceContent, InferenceMessage, InferenceRole};
use crate::kernel::session::SessionState;
use crate::tools::ToolContext;

use super::event::{KernelEvent, LifecycleEvent, StreamEvent};
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
                        "session_id": session.identity.session_id(),
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
                        "session_id": session.identity.session_id(),
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

        if let Ok(store) = self.store_manager.get_default().await {
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
            if let Some(iid) = session.internal_id {
                let _ = store
                    .insert_message(
                        iid,
                        session.turn_index,
                        "assistant",
                        &serde_json::Value::Array(content),
                        None,
                    )
                    .await;
            }
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
        self.execute_tool_calls(session, tool_ctx, pending_tool_calls)
            .await
    }
}
