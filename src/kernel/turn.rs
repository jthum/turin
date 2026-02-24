//! Agent turn execution — streaming, tool dispatch, and result collection.
//!
//! This module contains turn-level execution for the agent loop: LLM inference,
//! stream processing, hook evaluation, parallel tool execution, and side effects.

mod assistant_response;
mod streaming;
mod tool_execution;

use anyhow::{Context, Result};
use std::time::Duration;
use tracing::{debug, error, warn};

use crate::harness::context::{ContextWrapper, RequestOptionsOverride};
use crate::harness::verdict::Verdict;
use crate::inference::provider::{self};
use crate::kernel::session::SessionState;
use crate::tools::ToolContext;

use super::Kernel;
use super::event::{KernelEvent, LifecycleEvent};

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

        let stream = client
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

        let stream_output = self
            .collect_turn_stream_output(session, &provider_name, &model, stream)
            .await?;
        let response_text = stream_output.response_text;
        let pending_tool_calls = stream_output.pending_tool_calls;

        let has_tool_calls = self
            .finalize_assistant_turn_output(session, turn_ctx, &response_text, &pending_tool_calls)
            .await;

        if !has_tool_calls {
            return Ok(TurnOutcome::Complete);
        }

        // Execute tools.
        self.execute_tool_calls(session, tool_ctx, pending_tool_calls)
            .await
    }
}
