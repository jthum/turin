use anyhow::{Context, Result};
use futures::stream::{Stream, StreamExt};
use std::pin::Pin;
use std::time::Duration;
use tokio::time::sleep;

use crate::kernel::config::ProviderConfig;
use crate::kernel::event::{KernelEvent, StreamEvent};

// Use standardized types from SDK
use futures::future::BoxFuture;
pub use inference_sdk_core::{
    InferenceContent, InferenceEvent, InferenceMessage, InferenceProvider, InferenceRequest,
    InferenceResult, InferenceRole, InferenceStream, RequestOptions, SdkError, TimeoutPolicy, Tool,
    Usage,
};
use inference_sdk_registry::{ProviderInit, ProviderRegistry};

/// Options for inference execution
#[derive(Debug, Clone, Default)]
pub struct InferenceOptions {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub thinking_budget: Option<u32>,
}

/// A unified wrapper around a provider driver instance.
#[derive(Clone)]
pub struct ProviderClient {
    pub driver: String,
    pub provider: std::sync::Arc<dyn InferenceProvider>,
}

#[derive(Debug, Default)]
struct PendingToolCallEvent {
    id: Option<String>,
    name: Option<String>,
    args_json: String,
}

impl PendingToolCallEvent {
    fn on_start(&mut self, id: String, name: String) -> Result<Option<KernelEvent>> {
        let flushed = self.flush()?;
        self.id = Some(id);
        self.name = Some(name);
        self.args_json.clear();
        Ok(flushed)
    }

    fn on_delta(&mut self, delta: String) -> Result<()> {
        if self.id.is_none() || self.name.is_none() {
            anyhow::bail!("received ToolCallDelta before ToolCallStart");
        }
        self.args_json.push_str(&delta);
        Ok(())
    }

    fn flush(&mut self) -> Result<Option<KernelEvent>> {
        if self.id.is_none() && self.name.is_none() && self.args_json.is_empty() {
            return Ok(None);
        }

        let id = self
            .id
            .take()
            .ok_or_else(|| anyhow::anyhow!("missing tool call id"))?;
        let name = self
            .name
            .take()
            .ok_or_else(|| anyhow::anyhow!("missing tool call name"))?;

        let args = if self.args_json.trim().is_empty() {
            serde_json::json!({})
        } else {
            serde_json::from_str(&self.args_json)
                .with_context(|| format!("failed to parse tool call args for '{}'", name))?
        };

        self.args_json.clear();
        Ok(Some(KernelEvent::Stream(StreamEvent::ToolCall {
            id,
            name,
            args,
        })))
    }
}

impl ProviderClient {
    pub fn new(driver: impl Into<String>, provider: std::sync::Arc<dyn InferenceProvider>) -> Self {
        Self {
            driver: driver.into(),
            provider,
        }
    }

    /// Run a non-streaming completion (aggregates the stream).
    pub async fn completion(
        &self,
        model: &str,
        system_prompt: &str,
        messages: &[InferenceMessage],
    ) -> Result<String> {
        self.completion_with_options(
            model,
            system_prompt,
            messages,
            &[],
            &InferenceOptions::default(),
            None,
        )
        .await
    }

    pub async fn completion_with_options(
        &self,
        model: &str,
        system_prompt: &str,
        messages: &[InferenceMessage],
        tools: &[serde_json::Value],
        options: &InferenceOptions,
        request_options: Option<RequestOptions>,
    ) -> Result<String> {
        let req = self.build_request(model, system_prompt, messages, tools, options);
        let result = self.provider.complete(req, request_options).await?;
        Ok(result
            .content
            .iter()
            .filter_map(|c| match c {
                InferenceContent::Text { text } => Some(text.as_str()),
                // We could include thinking here if desired, but typically completion() returns just the answer.
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""))
    }

    /// Run a streaming completion.
    pub async fn stream(
        &self,
        model: &str,
        system_prompt: &str,
        messages: &[InferenceMessage],
        tools: &[serde_json::Value],
        options: &InferenceOptions,
        request_options: Option<RequestOptions>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<KernelEvent>> + Send>>> {
        let req = self.build_request(model, system_prompt, messages, tools, options);
        let sdk_stream = self.provider.stream(req, request_options).await?;
        let mut pending_tool = PendingToolCallEvent::default();

        // Map SDK InferenceEvents to Turin KernelEvents, expanding one SDK event
        // into zero or more Turin events when tool arguments are streamed as deltas.
        let kernel_stream = sdk_stream
            .map(move |res| match res {
                Ok(event) => map_sdk_event(event, &mut pending_tool),
                Err(e) => vec![Err(anyhow::anyhow!("Provider error: {}", e))],
            })
            .flat_map(futures::stream::iter);

        Ok(Box::pin(kernel_stream))
    }

    fn build_request(
        &self,
        model: &str,
        system_prompt: &str,
        messages: &[InferenceMessage],
        tools: &[serde_json::Value],
        options: &InferenceOptions,
    ) -> InferenceRequest {
        let sdk_tools: Vec<Tool> = tools
            .iter()
            .filter_map(|t| {
                Some(Tool {
                    name: t.get("name")?.as_str()?.to_string(),
                    description: t
                        .get("description")
                        .and_then(|d| d.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    input_schema: t
                        .get("input_schema")
                        .cloned()
                        .unwrap_or(serde_json::json!({"type": "object"})),
                })
            })
            .collect();

        InferenceRequest::builder()
            .model(model)
            .messages(messages.to_vec())
            .system(system_prompt)
            .maybe_tools(if sdk_tools.is_empty() {
                None
            } else {
                Some(sdk_tools)
            })
            .maybe_temperature(options.temperature)
            .maybe_max_tokens(options.max_tokens)
            .maybe_thinking_budget(options.thinking_budget)
            .build()
    }
}

// ─── Event Mapping ───────────────────────────────────────────────

fn map_sdk_event(
    event: InferenceEvent,
    pending_tool: &mut PendingToolCallEvent,
) -> Vec<Result<KernelEvent>> {
    let mut mapped = Vec::new();

    match event {
        InferenceEvent::MessageStart { role, model, .. } => {
            mapped.push(Ok(KernelEvent::Stream(StreamEvent::MessageStart {
                role,
                model,
            })));
        }
        InferenceEvent::MessageDelta { content } => {
            mapped.push(Ok(KernelEvent::Stream(StreamEvent::MessageDelta {
                content_delta: content,
            })));
        }
        InferenceEvent::ThinkingDelta { content } => {
            mapped.push(Ok(KernelEvent::Stream(StreamEvent::ThinkingDelta {
                thinking: content,
            })));
        }
        InferenceEvent::ThinkingSignatureDelta { signature } => {
            mapped.push(Ok(KernelEvent::Stream(
                StreamEvent::ThinkingSignatureDelta { signature },
            )));
        }
        InferenceEvent::ToolCallStart { id, name } => match pending_tool.on_start(id, name) {
            Ok(Some(tool_call)) => mapped.push(Ok(tool_call)),
            Ok(None) => {}
            Err(e) => mapped.push(Err(e)),
        },
        InferenceEvent::ToolCallDelta { delta } => {
            if let Err(e) = pending_tool.on_delta(delta) {
                mapped.push(Err(e));
            }
        }
        InferenceEvent::MessageEnd {
            input_tokens,
            output_tokens,
            ..
        } => {
            match pending_tool.flush() {
                Ok(Some(tool_call)) => mapped.push(Ok(tool_call)),
                Ok(None) => {}
                Err(e) => mapped.push(Err(e)),
            }
            mapped.push(Ok(KernelEvent::Stream(StreamEvent::MessageEnd {
                role: "assistant".to_string(),
                input_tokens: input_tokens as u64,
                output_tokens: output_tokens as u64,
            })));
        }
        _ => {}
    }

    mapped
}

// ─── Provider Creation ───────────────────────────────────────────

pub fn build_request_options(provider_config: &ProviderConfig) -> Result<RequestOptions> {
    let mut options = RequestOptions::default();

    for (header_name, header_value) in &provider_config.headers {
        options = options
            .with_header(header_name, header_value)
            .with_context(|| format!("invalid request header '{}'", header_name))?;
    }

    if let Some(max_retries) = provider_config.max_retries {
        options = options.with_max_retries(max_retries);
    }

    if provider_config.request_timeout_secs.is_some()
        || provider_config.total_timeout_secs.is_some()
    {
        let mut timeout_policy = TimeoutPolicy::default();
        if let Some(request_timeout_secs) = provider_config.request_timeout_secs {
            timeout_policy =
                timeout_policy.with_request_timeout(Duration::from_secs(request_timeout_secs));
        }
        if let Some(total_timeout_secs) = provider_config.total_timeout_secs {
            timeout_policy =
                timeout_policy.with_total_timeout(Duration::from_secs(total_timeout_secs));
        }
        options = options.with_timeout_policy(timeout_policy);
    }

    Ok(options)
}

pub fn create_provider_client(
    provider_config: &ProviderConfig,
) -> Result<std::sync::Arc<dyn InferenceProvider>> {
    if provider_config.kind.eq_ignore_ascii_case("mock") {
        return Ok(create_mock_client(provider_config));
    }

    let env_var = provider_config
        .api_key_env
        .as_ref()
        .context("API key environment variable not configured")?;
    let api_key = std::env::var(env_var)
        .with_context(|| format!("Missing API key in environment variable '{}'", env_var))?;

    let mut init = ProviderInit::new(api_key);
    if let Some(base_url) = &provider_config.base_url {
        init = init.with_base_url(base_url.clone());
    }

    ProviderRegistry::with_builtin_drivers()
        .create(&provider_config.kind, &init)
        .map_err(|e| {
            anyhow::anyhow!(
                "failed to create provider driver '{}': {}",
                provider_config.kind,
                e
            )
        })
}

pub fn create_mock_client(config: &ProviderConfig) -> std::sync::Arc<dyn InferenceProvider> {
    let response = config
        .base_url
        .clone()
        .unwrap_or_else(|| "Mock response".to_string());
    std::sync::Arc::new(MockProvider { response })
}

pub struct MockProvider {
    response: String,
}

impl InferenceProvider for MockProvider {
    // Use default complete() implementation

    fn stream<'a>(
        &'a self,
        _request: InferenceRequest,
        _options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        let (delay, content) = parse_mock_stream_response(&self.response);
        Box::pin(async move {
            let stream = futures::stream::unfold(
                (0u8, delay, content),
                |(step, delay, content)| async move {
                    match step {
                        0 => Some((
                            Ok(InferenceEvent::MessageStart {
                                role: "assistant".to_string(),
                                model: "mock-model".to_string(),
                                provider_id: "mock".to_string(),
                            }),
                            (1, delay, content),
                        )),
                        1 => {
                            if !delay.is_zero() {
                                sleep(delay).await;
                            }
                            Some((
                                Ok(InferenceEvent::MessageDelta { content }),
                                (2, delay, String::new()),
                            ))
                        }
                        2 => Some((
                            Ok(InferenceEvent::MessageEnd {
                                input_tokens: 10,
                                output_tokens: 5,
                                stop_reason: None,
                            }),
                            (3, delay, content),
                        )),
                        _ => None,
                    }
                },
            );
            Ok(Box::pin(stream) as InferenceStream)
        })
    }
}

fn parse_mock_stream_response(raw: &str) -> (Duration, String) {
    let Some(rest) = raw.strip_prefix("delay_ms=") else {
        return (Duration::ZERO, raw.to_string());
    };
    let Some((delay_ms, content)) = rest.split_once(';') else {
        return (Duration::ZERO, raw.to_string());
    };
    let Ok(delay_ms) = delay_ms.trim().parse::<u64>() else {
        return (Duration::ZERO, raw.to_string());
    };
    (Duration::from_millis(delay_ms), content.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_thinking_signature_delta_to_kernel_stream_event() {
        let mut pending_tool = PendingToolCallEvent::default();
        let events = map_sdk_event(
            InferenceEvent::ThinkingSignatureDelta {
                signature: "sig_part".to_string(),
            },
            &mut pending_tool,
        );

        assert_eq!(events.len(), 1);
        match &events[0] {
            Ok(KernelEvent::Stream(StreamEvent::ThinkingSignatureDelta { signature })) => {
                assert_eq!(signature, "sig_part");
            }
            other => panic!("unexpected mapped event: {other:?}"),
        }
    }
}
