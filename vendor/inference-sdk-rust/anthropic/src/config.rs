use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use std::fmt;
use std::time::Duration;

use crate::SdkError;
use inference_sdk_core::http::{RetryPolicy, TimeoutPolicy};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
pub const ANTHROPIC_VERSION: &str = "2023-06-01";
pub const DEFAULT_THINKING_BETA_HEADER: &str = "output-128k-2025-02-19";

#[derive(Clone)]
pub struct ClientConfig {
    pub(crate) base_url: String,
    pub(crate) timeout: Duration,
    pub(crate) max_retries: u32,
    pub(crate) retry_policy: RetryPolicy,
    pub(crate) timeout_policy: TimeoutPolicy,
    pub(crate) headers: HeaderMap,
    pub(crate) thinking_beta_header: Option<String>,
}

// Manually implement Debug to redact the API key
impl fmt::Debug for ClientConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClientConfig")
            .field("api_key", &"[REDACTED]")
            .field("base_url", &self.base_url)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

impl ClientConfig {
    pub fn new(api_key: String) -> Result<Self, SdkError> {
        let header_value = HeaderValue::from_str(&api_key)
            .map_err(|e| SdkError::ConfigError(format!("Invalid API key: {}", e)))?;

        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", header_value);
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            timeout: DEFAULT_TIMEOUT,
            max_retries: 2,
            retry_policy: RetryPolicy::default().with_max_retries(2),
            timeout_policy: TimeoutPolicy::default().with_request_timeout(DEFAULT_TIMEOUT),
            headers,
            thinking_beta_header: Some(DEFAULT_THINKING_BETA_HEADER.to_string()),
        })
    }

    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self.retry_policy.max_retries = retries;
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self.timeout_policy.request_timeout = Some(timeout);
        self
    }

    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.max_retries = policy.max_retries;
        self.retry_policy = policy;
        self
    }

    pub fn with_timeout_policy(mut self, policy: TimeoutPolicy) -> Self {
        if let Some(request_timeout) = policy.request_timeout {
            self.timeout = request_timeout;
        }
        self.timeout_policy = policy;
        self
    }

    /// Override the beta header used automatically when `thinking_budget` is set.
    pub fn with_thinking_beta_header(mut self, header: impl Into<String>) -> Self {
        self.thinking_beta_header = Some(header.into());
        self
    }

    /// Disable automatic beta header injection for thinking requests.
    pub fn without_thinking_beta_header(mut self) -> Self {
        self.thinking_beta_header = None;
        self
    }
}
