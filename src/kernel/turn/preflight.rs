use std::pin::Pin;

use anyhow::{Context, Result};
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

        let prepared = self.build_prepared_turn_stream(session, req).await?;
        Ok(TurnPreflight::Ready(prepared))
    }

    fn default_turn_request_state(&self, session: &SessionState) -> Result<TurnRequestState> {
        let agent = self.agent_config_for_session(session)?;
        Ok(TurnRequestState {
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
                    "session_id": session.identity.session_id(),
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
        req: TurnRequestState,
    ) -> Result<PreparedTurnStream> {
        self.ensure_turn_provider_client(&req.provider_name)?;

        let client = self
            .clients
            .get(&req.provider_name)
            .ok_or_else(|| anyhow::anyhow!("Provider '{}' not initialized", req.provider_name))?
            .clone();
        let provider_config = self
            .config
            .providers
            .get(&req.provider_name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Provider '{}' not found in configuration",
                    req.provider_name
                )
            })?;

        let tools = self.tool_registry.tool_definitions();
        let options = provider::InferenceOptions {
            max_tokens: None,
            temperature: None,
            thinking_budget: Some(req.thinking_budget),
        };
        let request_options = merge_request_option_overrides(
            provider::build_request_options(provider_config)?,
            &req.request_options_override,
        )?;

        let stream = client
            .stream(
                &req.model,
                &req.system_prompt,
                &session.history,
                &tools,
                &options,
                Some(request_options),
            )
            .await
            .with_context(|| {
                format!(
                    "failed to start inference stream (provider='{}', model='{}')",
                    req.provider_name, req.model
                )
            })?;

        Ok(PreparedTurnStream {
            provider_name: req.provider_name,
            model: req.model,
            stream,
        })
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
