use inference_sdk_core::SdkError;
use inference_sdk_core::http::{RetryPolicy, TimeoutPolicy};
use reqwest::Client as HttpClient;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use crate::resources::chat::ChatResource;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone)]
pub struct ClientConfig {
    pub(crate) base_url: String,
    pub(crate) timeout: Duration,
    pub(crate) max_retries: u32,
    pub(crate) retry_policy: RetryPolicy,
    pub(crate) timeout_policy: TimeoutPolicy,
    pub(crate) headers: HeaderMap,
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
        let mut headers = HeaderMap::new();
        if !api_key.is_empty() {
            let bearer = format!("Bearer {}", api_key);
            let auth_value = HeaderValue::from_str(&bearer)
                .map_err(|e| SdkError::ConfigError(format!("Invalid API key: {}", e)))?;
            headers.insert(AUTHORIZATION, auth_value);
        }
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            timeout: DEFAULT_TIMEOUT,
            max_retries: 2,
            retry_policy: RetryPolicy::default().with_max_retries(2),
            timeout_policy: TimeoutPolicy::default().with_request_timeout(DEFAULT_TIMEOUT),
            headers,
        })
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

    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self.retry_policy.max_retries = retries;
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
}

#[derive(Clone, Debug)]
pub struct Client {
    pub(crate) http_client: HttpClient,
    pub(crate) config: Arc<ClientConfig>,
}

impl Client {
    pub fn new(api_key: impl Into<String>) -> Result<Self, SdkError> {
        let config = ClientConfig::new(api_key.into())?;
        Self::from_config(config)
    }

    pub fn from_config(config: ClientConfig) -> Result<Self, SdkError> {
        let mut builder = HttpClient::builder().default_headers(config.headers.clone());
        if let Some(timeout) = config.timeout_policy.request_timeout {
            builder = builder.timeout(timeout);
        }

        let http_client = builder
            .build()
            .map_err(|e| SdkError::ConfigError(format!("Failed to build HTTP client: {}", e)))?;

        Ok(Self {
            http_client,
            config: Arc::new(config),
        })
    }

    /// Access the Chat Completions resource.
    pub fn chat(&self) -> ChatResource {
        ChatResource::new(self.clone())
    }

    /// Access the Embeddings resource.
    pub fn embeddings(&self) -> crate::resources::embeddings::Embeddings {
        crate::resources::embeddings::Embeddings::new(self.clone())
    }
}
