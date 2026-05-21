use std::collections::HashMap;
use std::time::Duration;

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::inference::provider::{self, RequestOptions};
use crate::kernel::config::ProviderConfig;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RequestOptionsOverride {
    #[serde(default)]
    pub headers: HashMap<String, String>,
    pub max_retries: Option<u32>,
    pub request_timeout_seconds: Option<u64>,
    pub total_timeout_seconds: Option<u64>,
}

pub(crate) fn build_merged_request_options(
    provider_config: &ProviderConfig,
    current: &RequestOptionsOverride,
    override_opts: Option<&RequestOptionsOverride>,
) -> anyhow::Result<RequestOptions> {
    let mut options = provider::build_request_options(provider_config)?;
    options = apply_request_options_override(options, current)?;
    if let Some(override_opts) = override_opts {
        options = apply_request_options_override(options, override_opts)?;
    }
    Ok(options)
}

fn apply_request_options_override(
    mut options: RequestOptions,
    overrides: &RequestOptionsOverride,
) -> anyhow::Result<RequestOptions> {
    for (header_name, header_value) in &overrides.headers {
        options = options
            .with_header(header_name, header_value)
            .with_context(|| format!("invalid request header '{}'", header_name))?;
    }

    if let Some(max_retries) = overrides.max_retries {
        options = options.with_max_retries(max_retries);
    }

    if overrides.request_timeout_seconds.is_some() || overrides.total_timeout_seconds.is_some() {
        let mut timeout_policy = options.timeout_policy.clone().unwrap_or_default();
        if let Some(request_timeout_seconds) = overrides.request_timeout_seconds {
            timeout_policy.request_timeout = Some(Duration::from_secs(request_timeout_seconds));
        }
        if let Some(total_timeout_seconds) = overrides.total_timeout_seconds {
            timeout_policy.total_timeout = Some(Duration::from_secs(total_timeout_seconds));
        }
        options = options.with_timeout_policy(timeout_policy);
    }

    Ok(options)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn override_with_header(name: &str, value: &str) -> RequestOptionsOverride {
        let mut override_opts = RequestOptionsOverride::default();
        override_opts
            .headers
            .insert(name.to_string(), value.to_string());
        override_opts
    }

    #[test]
    fn request_options_override_layering_keeps_provider_defaults() {
        let mut provider = ProviderConfig {
            max_retries: Some(2),
            request_timeout_seconds: Some(30),
            ..ProviderConfig::default()
        };
        provider
            .headers
            .insert("x-provider".to_string(), "base".to_string());

        let mut current = override_with_header("x-current", "session");
        current.total_timeout_seconds = Some(90);

        let mut call = override_with_header("x-call", "structured");
        call.max_retries = Some(1);
        call.request_timeout_seconds = Some(10);

        let options = build_merged_request_options(&provider, &current, Some(&call))
            .expect("request options should merge");

        assert_header(&options, "x-provider", "base");
        assert_header(&options, "x-current", "session");
        assert_header(&options, "x-call", "structured");
        assert_eq!(options.max_retries, Some(1));

        let timeout_policy = options.timeout_policy.expect("timeout policy");
        assert_eq!(
            timeout_policy.request_timeout,
            Some(Duration::from_secs(10))
        );
        assert_eq!(timeout_policy.total_timeout, Some(Duration::from_secs(90)));
    }

    #[test]
    fn request_options_override_rejects_invalid_header_names() {
        let current = override_with_header("not a header", "value");
        let err = build_merged_request_options(&ProviderConfig::default(), &current, None)
            .expect_err("invalid header should fail");

        assert!(
            err.to_string().contains("invalid request header"),
            "unexpected error: {err:#}"
        );
    }

    fn assert_header(options: &RequestOptions, name: &str, expected: &str) {
        let actual = options
            .headers
            .get(name)
            .and_then(|value| value.to_str().ok());
        assert_eq!(actual, Some(expected));
    }
}
