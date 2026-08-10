use std::collections::BTreeSet;
use std::pin::Pin;

use anyhow::Result;
use futures::Stream;
use tracing::{debug, error, warn};

use crate::display;
use crate::harness::context::{
    ContextWrapper, RequestOptionsOverride, build_merged_request_options,
};
use crate::harness::verdict::Verdict;
use crate::inference::provider;
use crate::kernel::config::{ProviderConfig, ResolvedInferenceCandidate, ResolvedInferenceRoute};
use crate::kernel::session::SessionState;
use crate::kernel::turn::context_window::estimate_history_input_tokens;

use super::super::event::{AuditEvent, InferenceRequestMetrics, KernelEvent, LifecycleEvent};
use super::super::execution_host::ExecutionHost;
use super::TurnContext;
use crate::kernel::turn::context_window::estimate_request_token_breakdown;

mod compaction;

pub(super) enum TurnPreflight {
    Ready(PreparedTurnStream),
    Rejected,
}

pub(super) struct PreparedTurnStream {
    pub provider_name: String,
    pub model: String,
    pub stream: Pin<Box<dyn Stream<Item = Result<KernelEvent>> + Send>>,
}

#[derive(Clone)]
pub(super) struct TurnRequestState {
    inference_context: Option<String>,
    model: String,
    provider_name: String,
    system_prompt: String,
    thinking_budget: u32,
    request_options_override: RequestOptionsOverride,
}

fn requested_context_label(route: &ResolvedInferenceRoute) -> &str {
    route.requested_context.as_deref().unwrap_or("<unset>")
}

fn resolved_context_label(candidate: &ResolvedInferenceCandidate) -> &str {
    candidate.context_name.as_deref().unwrap_or("<base>")
}

fn warn_candidate_fallback(
    requested_context: &str,
    candidate: &ResolvedInferenceCandidate,
    err: &anyhow::Error,
    message: &'static str,
) {
    warn!(
        requested_context,
        resolved_context = resolved_context_label(candidate),
        provider = %candidate.provider_name,
        model = %candidate.model,
        error = %err,
        "{}",
        message
    );
}

fn build_candidate_request_options(
    provider_config: &ProviderConfig,
    overrides: &RequestOptionsOverride,
) -> Result<provider::RequestOptions> {
    build_merged_request_options(provider_config, overrides, None)
}

impl ExecutionHost {
    pub(super) async fn prepare_turn_stream(
        &mut self,
        session: &mut SessionState,
        turn_ctx: &TurnContext,
    ) -> Result<TurnPreflight> {
        let mut req = self.default_turn_request_state(session)?;

        if self.emit_turn_start_and_gate(session, turn_ctx) {
            return Ok(TurnPreflight::Rejected);
        }

        if self
            .emit_turn_prepare_and_apply_hook(session, turn_ctx, &mut req)
            .await?
        {
            return Ok(TurnPreflight::Rejected);
        }

        let prepared = self
            .build_prepared_turn_stream(session, turn_ctx, req)
            .await?;
        Ok(TurnPreflight::Ready(prepared))
    }

    fn default_turn_request_state(&self, session: &SessionState) -> Result<TurnRequestState> {
        let agent = self.agent_config_for_session(session)?;
        Ok(TurnRequestState {
            inference_context: None,
            model: agent.model.clone(),
            provider_name: agent.provider.clone(),
            system_prompt: agent.system_prompt.clone(),
            thinking_budget: agent
                .thinking
                .as_ref()
                .and_then(|t| if t.enabled { t.budget_tokens } else { None })
                .unwrap_or(0),
            request_options_override: RequestOptionsOverride::default(),
        })
    }

    fn tool_definitions_for_session(
        &self,
        session: &SessionState,
        turn_ctx: &TurnContext,
    ) -> Result<Vec<serde_json::Value>> {
        let mut tools = self
            .tool_registry
            .tool_definitions_filtered(&turn_ctx.allowed_native_tools);
        let mut seen_names: BTreeSet<String> = tools
            .iter()
            .filter_map(|tool| tool.get("name").and_then(|value| value.as_str()))
            .map(ToOwned::to_owned)
            .collect();

        if let Some(harness) = self.session_harness_engine(session) {
            let engine = harness.lock().expect("session harness mutex poisoned");
            for tool in engine.declared_virtual_tools()? {
                if !seen_names.insert(tool.name.clone()) {
                    warn!(
                        tool = %tool.name,
                        "Skipping harness-declared virtual tool because a tool with that name already exists"
                    );
                    continue;
                }
                tools.push(tool.tool_definition());
            }
        }

        Ok(tools)
    }

    fn emit_turn_start_and_gate(&self, session: &mut SessionState, turn_ctx: &TurnContext) -> bool {
        if !self.json {
            println!(
                "\n{}",
                display::turn_header(session.turn_index + 1, display::stdout_ansi())
            );
        }

        self.persist_event(
            session,
            &KernelEvent::Lifecycle(LifecycleEvent::TurnStart {
                identity: session.identity.clone(),
                turn_index: session.turn_index,
                task_id: turn_ctx.task_id.clone(),
                trace_id: turn_ctx.trace_id.clone(),
                task_turn_index: turn_ctx.task_turn_index,
            }),
        );

        if let Some(harness) = self.session_harness_engine(session) {
            let engine = harness.lock().expect("session harness mutex poisoned");
            match engine.evaluate(
                "on_turn_start",
                serde_json::json!({
                    "identity": session.identity.clone(),
                    "session_id": self.session_reference(session),
                    "task_id": turn_ctx.task_id.clone(),
                    "trace_id": turn_ctx.trace_id.clone(),
                    "plan_id": turn_ctx.plan_id.clone(),
                    "turn_index": session.turn_index,
                    "task_turn_index": turn_ctx.task_turn_index,
                }),
            ) {
                Ok(Verdict::Reject(reason)) => {
                    warn!(reason = %reason, "Turn rejected by on_turn_start");
                    return true;
                }
                Ok(Verdict::Escalate(reason)) => {
                    warn!(reason = %reason, "Turn escalated by on_turn_start; treating as rejected");
                    return true;
                }
                Ok(_) => {}
                Err(e) => {
                    warn!(error = %e, "Harness on_turn_start error");
                }
            }
        }

        false
    }

    async fn emit_turn_prepare_and_apply_hook(
        &mut self,
        session: &mut SessionState,
        turn_ctx: &TurnContext,
        req: &mut TurnRequestState,
    ) -> Result<bool> {
        self.persist_event(
            session,
            &KernelEvent::Lifecycle(LifecycleEvent::TurnPrepare {
                identity: session.identity.clone(),
                turn_index: session.turn_index,
                task_id: turn_ctx.task_id.clone(),
                trace_id: turn_ctx.trace_id.clone(),
                task_turn_index: turn_ctx.task_turn_index,
            }),
        );

        let has_prepare_hook = self.session_harness_engine(session).is_some_and(|harness| {
            harness
                .lock()
                .expect("session harness mutex poisoned")
                .has_hook("on_turn_prepare")
        });
        if has_prepare_hook {
            self.ensure_full_history_materialized(session).await?;
        }

        if has_prepare_hook && let Some(harness) = self.session_harness_engine(session) {
            let token_count = estimate_history_input_tokens(&req.system_prompt, &session.history);
            let token_limit = self.estimate_turn_context_window_tokens(session, req)?;
            let engine = harness.lock().expect("session harness mutex poisoned");
            let ctx = ContextWrapper::new(
                req.inference_context.clone(),
                req.model.clone(),
                req.provider_name.clone(),
                req.system_prompt.clone(),
                session.history.clone(),
                session.turn_index,
                turn_ctx.task_turn_index,
                turn_ctx.task_turn_index == 0,
                turn_ctx.task_id.clone(),
                turn_ctx.plan_id.clone(),
                token_count,
                token_limit,
                req.thinking_budget,
                req.request_options_override.clone(),
                self.clients.clone(),
                self.config.clone(),
                session.identity.agent_id().to_string(),
                session.inference.clone(),
            );

            match engine.evaluate_userdata("on_turn_prepare", ctx.clone()) {
                Ok(Verdict::Reject(reason)) => {
                    warn!(reason = %reason, "Turn rejected by on_turn_prepare");
                    return Ok(true);
                }
                Ok(Verdict::Escalate(reason)) => {
                    warn!(reason = %reason, "Turn escalated by on_turn_prepare; treating as rejected");
                    return Ok(true);
                }
                Ok(_) => {}
                Err(e) => {
                    warn!(error = %e, "Harness on_turn_prepare error");
                }
            }

            let state = ctx.get_state();
            session.replace_full_history(state.messages);
            req.inference_context = state.inference;
            req.system_prompt = state.system_prompt;
            req.model = state.model;
            req.provider_name = state.provider;
            req.thinking_budget = state.thinking_budget;
            req.request_options_override = state.request_options;
        }

        Ok(false)
    }

    async fn build_prepared_turn_stream(
        &mut self,
        session: &mut SessionState,
        turn_ctx: &TurnContext,
        req: TurnRequestState,
    ) -> Result<PreparedTurnStream> {
        let tools = self.tool_definitions_for_session(session, turn_ctx)?;
        let effective_inference = self.config.effective_inference_config_for_agent(
            session.identity.agent_id(),
            Some(&session.inference),
        )?;
        let route = effective_inference.resolve_route(
            &req.provider_name,
            &req.model,
            req.thinking_budget,
            req.inference_context.as_deref(),
        );
        let requested_context = requested_context_label(&route).to_string();
        for warning in &route.warnings {
            warn!(
                requested_context = requested_context.as_str(),
                warning = %warning,
                "Inference route warning"
            );
        }
        self.maybe_refresh_context_checkpoint(session, &effective_inference, &route, &req, &tools)
            .await?;

        let mut last_error: Option<anyhow::Error> = None;
        for candidate in route.candidates {
            if let Err(err) = self.ensure_turn_provider_client(&candidate.provider_name) {
                warn_candidate_fallback(
                    &requested_context,
                    &candidate,
                    &err,
                    "Inference route failed during provider initialization; trying fallback",
                );
                last_error = Some(err);
                continue;
            }

            let Some(client) = self.clients.get(&candidate.provider_name).cloned() else {
                let err = anyhow::anyhow!(
                    "Provider '{}' was initialized but no client is available",
                    candidate.provider_name
                );
                warn_candidate_fallback(
                    &requested_context,
                    &candidate,
                    &err,
                    "Inference route failed after provider initialization; trying fallback",
                );
                last_error = Some(err);
                continue;
            };
            let Some(provider_config) = self.config.providers.get(&candidate.provider_name) else {
                let err = anyhow::anyhow!(
                    "Provider '{}' not found in configuration",
                    candidate.provider_name
                );
                warn_candidate_fallback(
                    &requested_context,
                    &candidate,
                    &err,
                    "Inference route failed because provider config is missing; trying fallback",
                );
                last_error = Some(err);
                continue;
            };

            let request_options = match build_candidate_request_options(
                provider_config,
                &req.request_options_override,
            ) {
                Ok(options) => options,
                Err(err) => {
                    warn_candidate_fallback(
                        &requested_context,
                        &candidate,
                        &err,
                        "Inference route failed while preparing request options; trying fallback",
                    );
                    last_error = Some(err);
                    continue;
                }
            };

            let options = provider::InferenceOptions {
                max_tokens: candidate.max_tokens,
                temperature: candidate.temperature,
                thinking_budget: candidate.thinking_budget,
            };
            let prepared_request = self.compact_messages_for_candidate(
                session,
                &req.system_prompt,
                &tools,
                &candidate,
                provider_config,
                &effective_inference.compaction.mode,
            );
            let effective_request = &prepared_request.context;
            let token_estimate = estimate_request_token_breakdown(
                &effective_request.system_prompt,
                &effective_request.messages,
                &tools,
            );
            let request_metrics = InferenceRequestMetrics {
                provider: candidate.provider_name.clone(),
                model: candidate.model.clone(),
                requested_context: requested_context.clone(),
                resolved_context: resolved_context_label(&candidate).to_string(),
                compaction_mode: match effective_inference.compaction.mode {
                    crate::kernel::config::InferenceCompactionMode::Hybrid => "hybrid",
                    crate::kernel::config::InferenceCompactionMode::TrimOnly => "trim_only",
                    crate::kernel::config::InferenceCompactionMode::SummaryOnly => "summary_only",
                }
                .to_string(),
                estimated_input_tokens_before_compaction: prepared_request
                    .report
                    .used_tokens_before,
                estimated_input_tokens: token_estimate.total_tokens,
                system_prompt_tokens: token_estimate.system_prompt_tokens,
                message_tokens: token_estimate.message_tokens,
                tool_definition_tokens: token_estimate.tool_definition_tokens,
                reusable_prefix_tokens: token_estimate.reusable_prefix_tokens,
                context_window_tokens: prepared_request.report.context_window_tokens,
                context_window_configured: provider_config.context_window_tokens.is_some(),
                input_budget_tokens: prepared_request.report.input_budget_tokens,
                max_output_tokens: candidate.max_tokens,
                thinking_budget_tokens: candidate.thinking_budget,
                available_message_count: session.history.len(),
                sent_message_count: effective_request.messages.len(),
                history_message_offset: session.history_message_offset,
                checkpoint_covered_message_count: session
                    .context_checkpoint
                    .as_ref()
                    .map_or(0, |checkpoint| checkpoint.covered_message_count),
                truncated_tool_results: prepared_request.report.truncated_tool_results,
                dropped_messages: prepared_request.report.dropped_messages,
                estimated_payload_bytes: token_estimate.estimated_payload_bytes,
            };

            match client
                .stream(
                    &candidate.model,
                    &effective_request.system_prompt,
                    &effective_request.messages,
                    &tools,
                    &options,
                    Some(request_options),
                )
                .await
            {
                Ok(stream) => {
                    self.persist_event(
                        session,
                        &KernelEvent::Audit(AuditEvent::InferenceRequest {
                            metrics: request_metrics,
                        }),
                    );
                    debug!(
                        requested_context = requested_context.as_str(),
                        resolved_context = resolved_context_label(&candidate),
                        provider = %candidate.provider_name,
                        model = %candidate.model,
                        "Prepared provider stream"
                    );
                    return Ok(PreparedTurnStream {
                        provider_name: candidate.provider_name,
                        model: candidate.model,
                        stream,
                    });
                }
                Err(err) => {
                    let err = err.context(format!(
                        "failed to start inference stream (provider='{}', model='{}')",
                        candidate.provider_name, candidate.model
                    ));
                    warn_candidate_fallback(
                        &requested_context,
                        &candidate,
                        &err,
                        "Inference route failed to start stream; trying fallback",
                    );
                    last_error = Some(err);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("No inference route available")))
    }

    fn ensure_turn_provider_client(&mut self, provider_name: &str) -> Result<()> {
        if self.clients.contains_key(provider_name) {
            return Ok(());
        }

        if let Some(config) = self.config.providers.get(provider_name) {
            debug!(provider = %provider_name, "Lazily initializing provider");
            match self.create_client(provider_name, config) {
                Ok(client) => {
                    self.clients.insert(provider_name.to_string(), client);
                }
                Err(e) => {
                    error!(provider = %provider_name, error = %e, "Failed to initialize provider");
                    anyhow::bail!("Failed to initialize provider '{}': {}", provider_name, e);
                }
            }
        } else {
            anyhow::bail!("Provider '{}' not found in configuration", provider_name);
        }

        Ok(())
    }
}
