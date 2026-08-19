use anyhow::Result;
use tracing::warn;

use crate::inference::provider;
use crate::kernel::config::{
    InferenceCompactionMode, InferenceConfig, ResolvedInferenceCandidate, ResolvedInferenceRoute,
};
use crate::kernel::event::AuditEvent;
use crate::kernel::session::ResidentHistory;
use crate::kernel::session::SessionState;
use crate::kernel::turn::context_window::{
    CompactionReport, EffectiveRequestContext, build_checkpoint_summary_request,
    compact_messages_for_input_budget, effective_input_budget_tokens,
    effective_request_context_from_window, estimate_request_input_tokens,
    resolve_context_window_tokens, target_checkpoint_coverage,
};

use super::super::super::event::KernelEvent;
use super::super::super::execution_host::ExecutionHost;
use super::TurnRequestState;

pub(super) struct PreparedRequestContext<'a> {
    pub context: EffectiveRequestContext<'a>,
    pub report: CompactionReport,
}

impl ExecutionHost {
    pub(super) fn estimate_turn_context_window_tokens(
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

    pub(super) async fn maybe_refresh_context_checkpoint(
        &mut self,
        session: &mut SessionState,
        context_history: &ResidentHistory,
        effective_inference: &InferenceConfig,
        route: &ResolvedInferenceRoute,
        req: &TurnRequestState,
        tools: &[serde_json::Value],
    ) -> Result<()> {
        if !effective_inference.compaction.mode.uses_summary() {
            return Ok(());
        }
        if context_history.has_prior_history() && session.context_checkpoint.is_none() {
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
            context_history,
            session.context_checkpoint.as_ref(),
        );
        let effective_input_tokens =
            estimate_request_input_tokens(&effective.system_prompt, effective.messages, tools);
        let compaction_trigger_threshold = ((input_budget_tokens as f32)
            * effective_inference.compaction.trigger_ratio)
            .floor() as u32;
        if effective_input_tokens <= compaction_trigger_threshold {
            return Ok(());
        }

        let Some((summary_prefix_len, covered_origin)) =
            target_checkpoint_coverage(context_history, session.context_checkpoint.as_ref())
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
                context_history,
                session.context_checkpoint.as_ref(),
                summary_prefix_len,
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
            covered_through_turn_id: covered_origin.turn_id,
            covered_through_turn_index: covered_origin.turn_index,
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

    pub(super) fn compact_messages_for_candidate<'a>(
        &self,
        request_messages: &'a [provider::InferenceMessage],
        system_prompt: &'a str,
        tools: &[serde_json::Value],
        candidate: &ResolvedInferenceCandidate,
        provider_config: &crate::kernel::config::ProviderConfig,
        compaction_mode: &InferenceCompactionMode,
    ) -> PreparedRequestContext<'a> {
        let context_window_tokens = resolve_context_window_tokens(Some(provider_config));
        let input_budget_tokens = effective_input_budget_tokens(
            context_window_tokens,
            candidate.max_tokens,
            candidate.thinking_budget,
        );
        let used_tokens = estimate_request_input_tokens(system_prompt, request_messages, tools);
        if !compaction_mode.uses_structural_trim() || used_tokens <= input_budget_tokens {
            if used_tokens > input_budget_tokens {
                warn!(
                    used_tokens,
                    input_budget_tokens,
                    context_window_tokens,
                    "Turn history still exceeds the estimated provider input budget in summary_only mode"
                );
            }
            return PreparedRequestContext {
                context: EffectiveRequestContext {
                    system_prompt: std::borrow::Cow::Borrowed(system_prompt),
                    messages: std::borrow::Cow::Borrowed(request_messages),
                },
                report: CompactionReport {
                    used_tokens_before: used_tokens,
                    used_tokens_after: used_tokens,
                    context_window_tokens,
                    input_budget_tokens,
                    truncated_tool_results: 0,
                    dropped_messages: 0,
                },
            };
        }

        let (messages, report) = compact_messages_for_input_budget(
            system_prompt,
            request_messages,
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

        PreparedRequestContext {
            context: EffectiveRequestContext {
                system_prompt: std::borrow::Cow::Borrowed(system_prompt),
                messages: std::borrow::Cow::Owned(messages),
            },
            report,
        }
    }
}
