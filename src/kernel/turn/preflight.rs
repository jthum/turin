use std::collections::BTreeSet;
use std::pin::Pin;

use anyhow::Result;
use futures::Stream;
use tracing::{debug, error, warn};

use crate::display;
use crate::harness::context::{ContextWrapper, RequestOptionsOverride};
use crate::harness::verdict::Verdict;
use crate::inference::provider;
use crate::kernel::session::SessionState;

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

        if self.emit_turn_prepare_and_apply_hook(session, turn_ctx, &mut req) {
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

        let runtime = self.runtime_for_session(session);
        let harness = runtime.lock_engine();
        if let Some(ref engine) = *harness {
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

        let runtime = self.runtime_for_session(session);
        let harness = runtime.lock_engine();
        if let Some(ref engine) = *harness {
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

    fn emit_turn_prepare_and_apply_hook(
        &mut self,
        session: &mut SessionState,
        turn_ctx: &TurnContext,
        req: &mut TurnRequestState,
    ) -> bool {
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

        let runtime = self.runtime_for_session(session);
        let harness = runtime.lock_engine();
        if let Some(ref engine) = *harness {
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
                0,
                128_000,
                req.thinking_budget,
                req.request_options_override.clone(),
                self.clients.clone(),
            );

            match engine.evaluate_userdata("on_turn_prepare", ctx.clone()) {
                Ok(Verdict::Reject(reason)) => {
                    warn!(reason = %reason, "Turn rejected by on_turn_prepare");
                    return true;
                }
                Ok(Verdict::Escalate(reason)) => {
                    warn!(reason = %reason, "Turn escalated by on_turn_prepare; treating as rejected");
                    return true;
                }
                Ok(_) => {}
                Err(e) => {
                    warn!(error = %e, "Harness on_turn_prepare error");
                }
            }

            let state = ctx.get_state();
            session.history = state.messages;
            req.inference_context = state.inference;
            req.system_prompt = state.system_prompt;
            req.model = state.model;
            req.provider_name = state.provider;
            req.thinking_budget = state.thinking_budget;
            req.request_options_override = state.request_options;
        }

        false
    }

    async fn build_prepared_turn_stream(
        &mut self,
        session: &SessionState,
        turn_ctx: &TurnContext,
        req: TurnRequestState,
    ) -> Result<PreparedTurnStream> {
        let tools = self.tool_definitions_for_session(session, turn_ctx)?;
        let route = self.config.resolve_inference_route(
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

            match client
                .stream(
                    &candidate.model,
                    &req.system_prompt,
                    &session.history,
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
