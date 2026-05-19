use std::collections::BTreeSet;
use std::pin::Pin;

use anyhow::Result;
use futures::Stream;
use tracing::{debug, error, warn};

use crate::display;
use crate::harness::context::{ContextWrapper, RequestOptionsOverride};
use crate::harness::verdict::Verdict;
use crate::inference::provider;
use crate::kernel::config::{
    InferenceCompactionMode, InferenceConfig, ResolvedInferenceCandidate, ResolvedInferenceRoute,
};
use crate::kernel::event::AuditEvent;
use crate::kernel::session::SessionState;
use crate::kernel::turn::context_window::{
    build_checkpoint_summary_request, compact_messages_for_input_budget,
    effective_input_budget_tokens, effective_request_context_from_window,
    estimate_history_input_tokens, estimate_request_input_tokens, resolve_context_window_tokens,
    target_checkpoint_coverage,
};

use super::super::event::{KernelEvent, LifecycleEvent};
use super::super::execution_host::ExecutionHost;
use super::{TurnContext, merge_request_option_overrides};

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
struct TurnRequestState {
    inference_context: Option<String>,
    model: String,
    provider_name: String,
    system_prompt: String,
    thinking_budget: u32,
    request_options_override: RequestOptionsOverride,
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

    fn estimate_turn_context_window_tokens(
        &self,
        session: &SessionState,
        req: &TurnRequestState,
    ) -> Result<u32> {
        let route = self.config.resolve_inference_route(
            session.identity.agent_id(),
            &req.provider_name,
            &req.model,
            req.thinking_budget,
            req.inference_context.as_deref(),
            Some(&session.inference),
        )?;
        let provider_name = route
            .candidates
            .first()
            .map(|candidate| candidate.provider_name.as_str())
            .unwrap_or(req.provider_name.as_str());
        Ok(resolve_context_window_tokens(
            self.config.providers.get(provider_name),
        ))
    }

    async fn maybe_refresh_context_checkpoint(
        &mut self,
        session: &mut SessionState,
        effective_inference: &InferenceConfig,
        route: &ResolvedInferenceRoute,
        req: &TurnRequestState,
        tools: &[serde_json::Value],
    ) -> Result<()> {
        if !effective_inference.compaction.mode.uses_summary() {
            return Ok(());
        }

        let compaction_route =
            if let Some(inference_name) = effective_inference.compaction_inference_name() {
                let route = effective_inference.resolve_route(
                    &req.provider_name,
                    &req.model,
                    req.thinking_budget,
                    Some(inference_name),
                );
                for warning in &route.warnings {
                    warn!(
                        requested_context = inference_name,
                        warning = %warning,
                        "Context compaction route warning"
                    );
                }
                route
            } else {
                route.clone()
            };

        let Some(candidate) = compaction_route.candidates.first() else {
            return Ok(());
        };
        let Some(provider_config) = self.config.providers.get(&candidate.provider_name).cloned()
        else {
            return Ok(());
        };

        let context_window_tokens = resolve_context_window_tokens(Some(&provider_config));
        let input_budget_tokens = effective_input_budget_tokens(
            context_window_tokens,
            candidate.max_tokens,
            candidate.thinking_budget,
        );
        let effective = effective_request_context_from_window(
            &req.system_prompt,
            &session.history,
            session.history_message_offset,
            session.context_checkpoint.as_ref(),
        );
        let effective_input_tokens =
            estimate_request_input_tokens(&effective.system_prompt, &effective.messages, tools);
        let compaction_trigger_threshold = ((input_budget_tokens as f32)
            * effective_inference.compaction.trigger_ratio)
            .floor() as u32;
        if effective_input_tokens <= compaction_trigger_threshold {
            return Ok(());
        }

        let Some(target_covered_message_count) =
            target_checkpoint_coverage(session.history.len(), session.context_checkpoint.as_ref())
        else {
            return Ok(());
        };

        if self
            .ensure_turn_provider_client(&candidate.provider_name)
            .is_err()
        {
            return Ok(());
        }
        let Some(client) = self.clients.get(&candidate.provider_name).cloned() else {
            return Ok(());
        };
        let request_options = match provider::build_request_options(&provider_config) {
            Ok(options) => Some(options),
            Err(err) => {
                warn!(
                    provider = %candidate.provider_name,
                    model = %candidate.model,
                    error = %err,
                    "Skipping context auto-compaction because provider request options could not be prepared"
                );
                return Ok(());
            }
        };

        let (summary_system_prompt, summary_messages, summary_max_tokens) =
            build_checkpoint_summary_request(
                &session.history,
                session.context_checkpoint.as_ref(),
                target_covered_message_count,
                context_window_tokens,
            );
        if summary_messages.is_empty() {
            return Ok(());
        }

        let summary = match client
            .completion_with_options(
                &candidate.model,
                &summary_system_prompt,
                &summary_messages,
                &[],
                &provider::InferenceOptions {
                    temperature: candidate.temperature.or(Some(0.1)),
                    max_tokens: Some(
                        candidate
                            .max_tokens
                            .unwrap_or(summary_max_tokens)
                            .min(summary_max_tokens),
                    ),
                    thinking_budget: candidate.thinking_budget,
                },
                request_options,
            )
            .await
        {
            Ok(summary) => summary,
            Err(err) => {
                warn!(
                    provider = %candidate.provider_name,
                    model = %candidate.model,
                    error = %err,
                    "Context auto-compaction failed; continuing with structural compaction only"
                );
                return Ok(());
            }
        };
        let summary = summary.trim().to_string();
        if summary.is_empty() {
            return Ok(());
        }

        let checkpoint = crate::kernel::session::ContextCompactionCheckpoint {
            summary,
            covered_message_count: target_covered_message_count,
            generated_at_turn_index: session.turn_index,
            provider_name: candidate.provider_name.clone(),
            model: candidate.model.clone(),
        };
        session.context_checkpoint = Some(checkpoint.clone());
        self.persist_event(
            session,
            &KernelEvent::Audit(AuditEvent::ContextCompaction { checkpoint }),
        );
        Ok(())
    }

    fn compact_messages_for_candidate(
        &self,
        session: &SessionState,
        system_prompt: &str,
        tools: &[serde_json::Value],
        candidate: &ResolvedInferenceCandidate,
        provider_config: &crate::kernel::config::ProviderConfig,
        compaction_mode: &InferenceCompactionMode,
    ) -> crate::kernel::turn::context_window::EffectiveRequestContext {
        let effective = if compaction_mode.uses_summary() {
            effective_request_context_from_window(
                system_prompt,
                &session.history,
                session.history_message_offset,
                session.context_checkpoint.as_ref(),
            )
        } else {
            crate::kernel::turn::context_window::EffectiveRequestContext {
                system_prompt: system_prompt.to_string(),
                messages: session.history.clone(),
            }
        };

        let context_window_tokens = resolve_context_window_tokens(Some(provider_config));
        let input_budget_tokens = effective_input_budget_tokens(
            context_window_tokens,
            candidate.max_tokens,
            candidate.thinking_budget,
        );
        if !compaction_mode.uses_structural_trim() {
            let used_tokens =
                estimate_request_input_tokens(&effective.system_prompt, &effective.messages, tools);
            if used_tokens > input_budget_tokens {
                warn!(
                    used_tokens,
                    input_budget_tokens,
                    context_window_tokens,
                    "Turn history still exceeds the estimated provider input budget in summary_only mode"
                );
            }
            return effective;
        }

        let (messages, report) = compact_messages_for_input_budget(
            &effective.system_prompt,
            &effective.messages,
            tools,
            context_window_tokens,
            input_budget_tokens,
        );

        if report.applied() {
            warn!(
                before_tokens = report.used_tokens_before,
                after_tokens = report.used_tokens_after,
                context_window_tokens = report.context_window_tokens,
                input_budget_tokens = report.input_budget_tokens,
                truncated_tool_results = report.truncated_tool_results,
                dropped_messages = report.dropped_messages,
                "Compacted turn history to fit provider context budget"
            );
        }
        if !report.fits_budget() {
            warn!(
                used_tokens = report.used_tokens_after,
                input_budget_tokens = report.input_budget_tokens,
                context_window_tokens = report.context_window_tokens,
                "Turn history still exceeds the estimated provider input budget after compaction"
            );
        }

        crate::kernel::turn::context_window::EffectiveRequestContext {
            system_prompt: effective.system_prompt,
            messages,
        }
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
        for warning in &route.warnings {
            warn!(
                requested_context = route.requested_context.as_deref().unwrap_or("<unset>"),
                warning = %warning,
                "Inference route warning"
            );
        }
        self.maybe_refresh_context_checkpoint(session, &effective_inference, &route, &req, &tools)
            .await?;

        let mut last_error: Option<anyhow::Error> = None;
        for candidate in route.candidates {
            if let Err(err) = self.ensure_turn_provider_client(&candidate.provider_name) {
                warn!(
                    requested_context = route.requested_context.as_deref().unwrap_or("<unset>"),
                    resolved_context = candidate.context_name.as_deref().unwrap_or("<base>"),
                    provider = %candidate.provider_name,
                    model = %candidate.model,
                    error = %err,
                    "Inference route failed during provider initialization; trying fallback"
                );
                last_error = Some(err);
                continue;
            }

            let Some(client) = self.clients.get(&candidate.provider_name).cloned() else {
                let err = anyhow::anyhow!(
                    "Provider '{}' was initialized but no client is available",
                    candidate.provider_name
                );
                warn!(
                    requested_context = route.requested_context.as_deref().unwrap_or("<unset>"),
                    resolved_context = candidate.context_name.as_deref().unwrap_or("<base>"),
                    provider = %candidate.provider_name,
                    model = %candidate.model,
                    error = %err,
                    "Inference route failed after provider initialization; trying fallback"
                );
                last_error = Some(err);
                continue;
            };
            let Some(provider_config) = self.config.providers.get(&candidate.provider_name) else {
                let err = anyhow::anyhow!(
                    "Provider '{}' not found in configuration",
                    candidate.provider_name
                );
                warn!(
                    requested_context = route.requested_context.as_deref().unwrap_or("<unset>"),
                    resolved_context = candidate.context_name.as_deref().unwrap_or("<base>"),
                    provider = %candidate.provider_name,
                    model = %candidate.model,
                    error = %err,
                    "Inference route failed because provider config is missing; trying fallback"
                );
                last_error = Some(err);
                continue;
            };

            let request_options = match provider::build_request_options(provider_config) {
                Ok(options) => {
                    match merge_request_option_overrides(options, &req.request_options_override) {
                        Ok(options) => options,
                        Err(err) => {
                            warn!(
                                requested_context = route.requested_context.as_deref().unwrap_or("<unset>"),
                                resolved_context = candidate.context_name.as_deref().unwrap_or("<base>"),
                                provider = %candidate.provider_name,
                                model = %candidate.model,
                                error = %err,
                                "Inference route failed while building request options; trying fallback"
                            );
                            last_error = Some(err);
                            continue;
                        }
                    }
                }
                Err(err) => {
                    warn!(
                        requested_context = route.requested_context.as_deref().unwrap_or("<unset>"),
                        resolved_context = candidate.context_name.as_deref().unwrap_or("<base>"),
                        provider = %candidate.provider_name,
                        model = %candidate.model,
                        error = %err,
                        "Inference route failed while preparing provider options; trying fallback"
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
            let effective_request = self.compact_messages_for_candidate(
                session,
                &req.system_prompt,
                &tools,
                &candidate,
                provider_config,
                &effective_inference.compaction.mode,
            );

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
                    debug!(
                        requested_context = route.requested_context.as_deref().unwrap_or("<unset>"),
                        resolved_context = candidate.context_name.as_deref().unwrap_or("<base>"),
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
                    warn!(
                        requested_context = route.requested_context.as_deref().unwrap_or("<unset>"),
                        resolved_context = candidate.context_name.as_deref().unwrap_or("<base>"),
                        provider = %candidate.provider_name,
                        model = %candidate.model,
                        error = %err,
                        "Inference route failed to start stream; trying fallback"
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
