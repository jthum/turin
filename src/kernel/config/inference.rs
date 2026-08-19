use std::collections::{HashMap, HashSet};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::ProvidersConfig;

const DEFAULT_INFERENCE_CONTEXT_NAME: &str = "default";
const DEFAULT_COMPACTION_TRIGGER_RATIO: f32 = 0.8;

#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
pub struct InferenceConfig {
    #[serde(default)]
    pub default: Option<String>,
    #[serde(default)]
    pub contexts: HashMap<String, InferenceContextConfig>,
    #[serde(default)]
    pub compaction: InferenceCompactionConfig,
    #[serde(default)]
    pub hot_history: HotHistoryConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct HotHistoryConfig {
    #[serde(default)]
    pub profile: HotHistoryProfile,
    #[serde(default)]
    pub max_messages: Option<usize>,
    #[serde(default)]
    pub max_tool_result_bytes: Option<usize>,
}

impl Default for HotHistoryConfig {
    fn default() -> Self {
        Self {
            profile: HotHistoryProfile::Default,
            max_messages: None,
            max_tool_result_bytes: None,
        }
    }
}

impl HotHistoryConfig {
    pub fn effective_max_messages(&self) -> Option<usize> {
        self.max_messages
            .or_else(|| self.profile.default_max_messages())
    }

    pub fn effective_max_tool_result_bytes(&self) -> Option<usize> {
        self.max_tool_result_bytes
            .or_else(|| self.profile.default_max_tool_result_bytes())
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum HotHistoryProfile {
    #[default]
    Default,
    Performance,
    Debug,
}

impl HotHistoryProfile {
    fn default_max_messages(self) -> Option<usize> {
        match self {
            Self::Default => Some(64),
            Self::Performance => Some(256),
            Self::Debug => None,
        }
    }

    fn default_max_tool_result_bytes(self) -> Option<usize> {
        match self {
            Self::Default => Some(64 * 1024),
            Self::Performance => Some(256 * 1024),
            Self::Debug => None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum InferenceCompactionMode {
    #[default]
    Hybrid,
    TrimOnly,
    SummaryOnly,
}

impl InferenceCompactionMode {
    pub fn uses_summary(&self) -> bool {
        !matches!(self, Self::TrimOnly)
    }

    pub fn uses_structural_trim(&self) -> bool {
        !matches!(self, Self::SummaryOnly)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct InferenceCompactionConfig {
    #[serde(default)]
    pub mode: InferenceCompactionMode,
    #[serde(default)]
    pub inference: Option<String>,
    #[serde(default = "default_compaction_trigger_ratio")]
    pub trigger_ratio: f32,
}

impl Default for InferenceCompactionConfig {
    fn default() -> Self {
        Self {
            mode: InferenceCompactionMode::default(),
            inference: None,
            trigger_ratio: default_compaction_trigger_ratio(),
        }
    }
}

fn default_compaction_trigger_ratio() -> f32 {
    DEFAULT_COMPACTION_TRIGGER_RATIO
}

#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
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

#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
pub struct InferenceOverrideConfig {
    #[serde(default)]
    pub default: Option<String>,
    #[serde(default)]
    pub contexts: HashMap<String, InferenceContextOverrideConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
pub struct InferenceContextOverrideConfig {
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
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

impl InferenceContextConfig {
    fn apply_override(&self, override_cfg: &InferenceContextOverrideConfig) -> Self {
        Self {
            provider: override_cfg
                .provider
                .clone()
                .unwrap_or_else(|| self.provider.clone()),
            model: override_cfg
                .model
                .clone()
                .unwrap_or_else(|| self.model.clone()),
            fallback: override_cfg
                .fallback
                .clone()
                .or_else(|| self.fallback.clone()),
            temperature: override_cfg.temperature.or(self.temperature),
            max_tokens: override_cfg.max_tokens.or(self.max_tokens),
            thinking_budget: override_cfg.thinking_budget.or(self.thinking_budget),
        }
    }
}

impl InferenceOverrideConfig {
    pub fn is_empty(&self) -> bool {
        self.default.is_none() && self.contexts.is_empty()
    }

    pub fn validate_shallow(&self, providers: &ProvidersConfig, label: &str) -> Result<()> {
        if let Some(default_context) = self.default.as_deref() {
            anyhow::ensure!(
                !default_context.trim().is_empty(),
                "{label}.default must not be empty when set"
            );
        }

        for (context_name, context) in &self.contexts {
            anyhow::ensure!(
                !context_name.trim().is_empty(),
                "{label}.contexts contains an empty context name"
            );
            if let Some(provider) = context.provider.as_deref() {
                anyhow::ensure!(
                    !provider.trim().is_empty(),
                    "{label}.contexts.{context_name}.provider must not be empty when set"
                );
                anyhow::ensure!(
                    providers.contains_key(provider),
                    "{label}.contexts.{context_name}.provider '{}' not found in [providers]",
                    provider
                );
            }
            if let Some(model) = context.model.as_deref() {
                anyhow::ensure!(
                    !model.trim().is_empty(),
                    "{label}.contexts.{context_name}.model must not be empty when set"
                );
            }
            if let Some(fallback) = context.fallback.as_deref() {
                anyhow::ensure!(
                    !fallback.trim().is_empty(),
                    "{label}.contexts.{context_name}.fallback must not be empty when set"
                );
            }
            if let Some(temperature) = context.temperature {
                anyhow::ensure!(
                    temperature.is_finite(),
                    "{label}.contexts.{context_name}.temperature must be finite"
                );
            }
            if let Some(max_tokens) = context.max_tokens {
                anyhow::ensure!(
                    max_tokens > 0,
                    "{label}.contexts.{context_name}.max_tokens must be greater than 0"
                );
            }
        }

        Ok(())
    }
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

    pub fn compaction_inference_name(&self) -> Option<&str> {
        self.compaction
            .inference
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
    }

    pub fn merged_with(&self, override_cfg: &InferenceOverrideConfig) -> Self {
        let mut merged = self.clone();

        if let Some(default_context) = override_cfg.default.as_ref() {
            merged.default = Some(default_context.clone());
        }

        for (context_name, context_override) in &override_cfg.contexts {
            match merged.contexts.get(context_name) {
                Some(existing) => {
                    merged.contexts.insert(
                        context_name.clone(),
                        existing.apply_override(context_override),
                    );
                }
                None => {
                    merged.contexts.insert(
                        context_name.clone(),
                        InferenceContextConfig {
                            provider: context_override.provider.clone().unwrap_or_default(),
                            model: context_override.model.clone().unwrap_or_default(),
                            fallback: context_override.fallback.clone(),
                            temperature: context_override.temperature,
                            max_tokens: context_override.max_tokens,
                            thinking_budget: context_override.thinking_budget,
                        },
                    );
                }
            }
        }

        merged
    }

    pub fn validate_complete(&self, providers: &ProvidersConfig, label: &str) -> Result<()> {
        if let Some(default_context) = self.default.as_deref() {
            anyhow::ensure!(
                !default_context.trim().is_empty(),
                "{label}.default must not be empty when set"
            );
            anyhow::ensure!(
                self.contexts.contains_key(default_context),
                "{label}.default '{}' not found in {label}.contexts",
                default_context
            );
        }

        for (context_name, context) in &self.contexts {
            anyhow::ensure!(
                !context_name.trim().is_empty(),
                "{label}.contexts contains an empty context name"
            );
            anyhow::ensure!(
                !context.provider.trim().is_empty(),
                "{label}.contexts.{context_name}.provider must not be empty"
            );
            anyhow::ensure!(
                !context.model.trim().is_empty(),
                "{label}.contexts.{context_name}.model must not be empty"
            );
            anyhow::ensure!(
                providers.contains_key(&context.provider),
                "{label}.contexts.{context_name}.provider '{}' not found in [providers]",
                context.provider
            );
            if let Some(fallback) = context.fallback.as_deref() {
                anyhow::ensure!(
                    !fallback.trim().is_empty(),
                    "{label}.contexts.{context_name}.fallback must not be empty when set"
                );
                anyhow::ensure!(
                    self.contexts.contains_key(fallback),
                    "{label}.contexts.{context_name}.fallback '{}' not found in {label}.contexts",
                    fallback
                );
            }
            if let Some(temperature) = context.temperature {
                anyhow::ensure!(
                    temperature.is_finite(),
                    "{label}.contexts.{context_name}.temperature must be finite"
                );
            }
            if let Some(max_tokens) = context.max_tokens {
                anyhow::ensure!(
                    max_tokens > 0,
                    "{label}.contexts.{context_name}.max_tokens must be greater than 0"
                );
            }
        }

        if let Some(inference) = self.compaction.inference.as_deref() {
            anyhow::ensure!(
                !inference.trim().is_empty(),
                "{label}.compaction.inference must not be empty when set"
            );
        }
        anyhow::ensure!(
            self.compaction.trigger_ratio.is_finite()
                && self.compaction.trigger_ratio > 0.0
                && self.compaction.trigger_ratio <= 1.0,
            "{label}.compaction.trigger_ratio must be > 0 and <= 1"
        );

        for context_name in self.contexts.keys() {
            let mut seen = HashSet::new();
            let mut current = context_name.as_str();
            while let Some(next) = self
                .contexts
                .get(current)
                .and_then(|context| context.fallback.as_deref())
            {
                anyhow::ensure!(
                    seen.insert(current.to_string()),
                    "{label} context fallback cycle detected at '{}'",
                    current
                );
                current = next;
            }
        }

        Ok(())
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
