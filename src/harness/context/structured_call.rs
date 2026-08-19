use mlua::LuaSerdeExt;
use serde::Deserialize;

use super::{
    ContextWrapper, RequestOptionsOverride, normalize_inference_context_name, structured_messages,
};
use crate::inference::provider::{InferenceMessage, InferenceOptions};
use crate::inference::structured::{
    fallback_system_prompt, parse_and_validate_json_response, response_format_for_schema,
};

#[derive(Debug, Deserialize)]
pub(super) struct StructuredCallArgs {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    messages: Option<Vec<InferenceMessage>>,
    #[serde(default)]
    system: Option<String>,
    schema: serde_json::Value,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    strict: Option<bool>,
    #[serde(default)]
    inference: Option<String>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    thinking_budget: Option<u32>,
    #[serde(default)]
    request_options: Option<RequestOptionsOverride>,
}

pub(super) fn parse(lua: &mlua::Lua, value: mlua::Value) -> mlua::Result<StructuredCallArgs> {
    lua.from_value(value).map_err(mlua::Error::external)
}

pub(super) async fn run(
    context: &ContextWrapper,
    args: StructuredCallArgs,
) -> Result<serde_json::Value, String> {
    let (
        current_inference,
        current_provider,
        current_model,
        current_system_prompt,
        current_messages,
        current_thinking_budget,
        current_request_options,
    ) = {
        let state = context.lock_state();
        (
            state.inference.clone(),
            state.provider.clone(),
            state.model.clone(),
            state.system_prompt.clone(),
            state.messages.clone(),
            state.thinking_budget,
            state.request_options.clone(),
        )
    };

    let requested_inference =
        normalize_inference_context_name(args.inference.or(current_inference));
    let message_set = structured_messages(args.prompt, args.messages, current_messages)?;
    let system_prompt = args.system.unwrap_or(current_system_prompt);
    let strict = args.strict.unwrap_or(true);
    let route = context
        .config
        .resolve_inference_route(
            &context.agent_id,
            &current_provider,
            &current_model,
            current_thinking_budget,
            requested_inference.as_deref(),
            Some(&context.session_inference),
        )
        .map_err(|error| error.to_string())?;
    let response_format = response_format_for_schema(
        args.name.as_deref(),
        args.description.as_deref(),
        &args.schema,
        strict,
    );
    let fallback_system_prompt = fallback_system_prompt(
        &system_prompt,
        args.name.as_deref(),
        args.description.as_deref(),
        &args.schema,
    );

    let mut last_error = None;
    for candidate in &route.candidates {
        let Some(client) = context.clients.get(&candidate.provider_name) else {
            last_error = Some(format!(
                "Provider '{}' not initialized",
                candidate.provider_name
            ));
            continue;
        };
        let Some(provider_config) = context.config.providers.get(&candidate.provider_name) else {
            last_error = Some(format!(
                "Provider '{}' not found in config",
                candidate.provider_name
            ));
            continue;
        };
        let request_options = super::build_merged_request_options(
            provider_config,
            &current_request_options,
            args.request_options.as_ref(),
        )
        .map_err(|error| error.to_string())?;
        let options = InferenceOptions {
            temperature: args.temperature.or(candidate.temperature),
            max_tokens: args.max_tokens.or(candidate.max_tokens),
            thinking_budget: Some(
                args.thinking_budget
                    .or(candidate.thinking_budget)
                    .unwrap_or(current_thinking_budget),
            ),
        };

        let raw = if client.supports_response_format(&response_format) {
            client
                .completion_with_response_format(
                    &candidate.model,
                    &system_prompt,
                    &message_set,
                    &[],
                    &options,
                    response_format.clone(),
                    Some(request_options.clone()),
                )
                .await
                .map_err(|error| error.to_string())
        } else {
            client
                .completion_with_options(
                    &candidate.model,
                    &fallback_system_prompt,
                    &message_set,
                    &[],
                    &options,
                    Some(request_options.clone()),
                )
                .await
                .map_err(|error| error.to_string())
        };
        match raw.and_then(|text| {
            parse_and_validate_json_response(&text, &args.schema).map_err(|error| error.to_string())
        }) {
            Ok(json) => return Ok(json),
            Err(error) => last_error = Some(error),
        }
    }

    Err(last_error
        .unwrap_or_else(|| "No inference route available for structured output".to_string()))
}
