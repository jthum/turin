use std::pin::Pin;

use anyhow::{Context, Result};
use futures::Stream;
use tracing::{debug, error, warn};

use crate::harness::context::{ContextWrapper, RequestOptionsOverride};
use crate::harness::verdict::Verdict;
use crate::inference::provider;
use crate::kernel::session::SessionState;

use super::super::Kernel;
use super::super::event::{KernelEvent, LifecycleEvent};
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

impl Kernel {
    pub(super) async fn prepare_turn_stream(
        &mut self,
        session: &mut SessionState,
        turn_ctx: &TurnContext,
    ) -> Result<TurnPreflight> {
        let mut req = TurnRequestState {
            model: self.config.agent.model.clone(),
            provider_name: self.config.agent.provider.clone(),
            system_prompt: self.config.agent.system_prompt.clone(),
            thinking_budget: self
                .config
                .agent
                .thinking
                .as_ref()
                .and_then(|t| if t.enabled { t.budget_tokens } else { None })
                .unwrap_or(0),
            request_options_override: RequestOptionsOverride::default(),
        };

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
                        return Ok(TurnPreflight::Rejected);
                    }
                    Ok(Verdict::Escalate(reason)) => {
                        warn!(reason = %reason, "Turn escalated by on_turn_start; treating as rejected");
                        return Ok(TurnPreflight::Rejected);
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

        // Harness hook: on_turn_prepare
        {
            let harness = self.lock_harness();
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
                        return Ok(TurnPreflight::Rejected);
                    }
                    Ok(Verdict::Escalate(reason)) => {
                        warn!(reason = %reason, "Turn escalated by on_turn_prepare; treating as rejected");
                        return Ok(TurnPreflight::Rejected);
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
        }

        if !self.clients.contains_key(&req.provider_name) {
            if let Some(config) = self.config.providers.get(&req.provider_name) {
                debug!(provider = %req.provider_name, "Lazily initializing provider");
                match self.create_client(&req.provider_name, config) {
                    Ok(client) => {
                        self.clients.insert(req.provider_name.clone(), client);
                    }
                    Err(e) => {
                        error!(provider = %req.provider_name, error = %e, "Failed to initialize provider");
                        anyhow::bail!(
                            "Failed to initialize provider '{}': {}",
                            req.provider_name,
                            e
                        );
                    }
                }
            } else {
                anyhow::bail!(
                    "Provider '{}' not found in configuration",
                    req.provider_name
                );
            }
        }

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

        Ok(TurnPreflight::Ready(PreparedTurnStream {
            provider_name: req.provider_name,
            model: req.model,
            stream,
        }))
    }
}
