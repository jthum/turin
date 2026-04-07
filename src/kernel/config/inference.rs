use std::collections::{HashMap, HashSet};

use serde::Deserialize;

const DEFAULT_INFERENCE_CONTEXT_NAME: &str = "default";

#[derive(Debug, Clone, Deserialize, Default)]
pub struct InferenceConfig {
    #[serde(default)]
    pub default: Option<String>,
    #[serde(default)]
    pub contexts: HashMap<String, InferenceContextConfig>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct InferenceContextConfig {
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub fallback: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub thinking_budget: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedInferenceCandidate {
    pub context_name: Option<String>,
    pub provider_name: String,
    pub model: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub thinking_budget: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedInferenceRoute {
    pub requested_context: Option<String>,
    pub candidates: Vec<ResolvedInferenceCandidate>,
    pub warnings: Vec<String>,
}

impl ResolvedInferenceCandidate {
    fn same_effective_target(&self, other: &Self) -> bool {
        self.provider_name == other.provider_name
            && self.model == other.model
            && self.temperature == other.temperature
            && self.max_tokens == other.max_tokens
            && self.thinking_budget == other.thinking_budget
    }
}

impl InferenceConfig {
    pub fn default_context_name(&self) -> &str {
        self.default
            .as_deref()
            .unwrap_or(DEFAULT_INFERENCE_CONTEXT_NAME)
    }

    pub fn resolve_route(
        &self,
        base_provider_name: &str,
        base_model: &str,
        base_thinking_budget: u32,
        requested_context: Option<&str>,
    ) -> ResolvedInferenceRoute {
        let requested_context = requested_context
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned);
        let mut candidates = Vec::new();
        let mut warnings = Vec::new();
        let mut seen_contexts = HashSet::new();

        if let Some(requested) = requested_context.as_deref() {
            if self.contexts.contains_key(requested) {
                self.append_context_chain(
                    requested,
                    base_thinking_budget,
                    &mut seen_contexts,
                    &mut candidates,
                    &mut warnings,
                );
            } else {
                warnings.push(format!(
                    "requested inference context '{}' is not configured; falling back",
                    requested
                ));
            }
        }

        let default_context_name = self.default_context_name();
        if self.contexts.contains_key(default_context_name)
            && !seen_contexts.contains(default_context_name)
        {
            self.append_context_chain(
                default_context_name,
                base_thinking_budget,
                &mut seen_contexts,
                &mut candidates,
                &mut warnings,
            );
        }

        let base_candidate = ResolvedInferenceCandidate {
            context_name: None,
            provider_name: base_provider_name.to_string(),
            model: base_model.to_string(),
            temperature: None,
            max_tokens: None,
            thinking_budget: Some(base_thinking_budget),
        };
        if !candidates
            .last()
            .is_some_and(|last| last.same_effective_target(&base_candidate))
        {
            candidates.push(base_candidate);
        }

        ResolvedInferenceRoute {
            requested_context,
            candidates,
            warnings,
        }
    }

    fn append_context_chain(
        &self,
        start: &str,
        base_thinking_budget: u32,
        seen_contexts: &mut HashSet<String>,
        candidates: &mut Vec<ResolvedInferenceCandidate>,
        warnings: &mut Vec<String>,
    ) {
        let mut current = start.to_string();
        loop {
            if !seen_contexts.insert(current.clone()) {
                warnings.push(format!(
                    "inference context fallback cycle detected at '{}'; stopping resolution",
                    current
                ));
                break;
            }

            let Some(context) = self.contexts.get(&current) else {
                warnings.push(format!(
                    "inference context '{}' is not configured; stopping fallback chain",
                    current
                ));
                break;
            };

            let candidate = ResolvedInferenceCandidate {
                context_name: Some(current.clone()),
                provider_name: context.provider.clone(),
                model: context.model.clone(),
                temperature: context.temperature,
                max_tokens: context.max_tokens,
                thinking_budget: Some(context.thinking_budget.unwrap_or(base_thinking_budget)),
            };
            if !candidates
                .last()
                .is_some_and(|last| last.same_effective_target(&candidate))
            {
                candidates.push(candidate);
            }

            let Some(next) = context
                .fallback
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            else {
                break;
            };
            current = next.to_string();
        }
    }
}
