use crate::error::SdkError;
use reqwest::Method;
use reqwest::StatusCode;
use reqwest::header::{HeaderMap, RETRY_AFTER};
use serde::Serialize;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::warn;

const MAX_RETRIES_CAP: u32 = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetryStatusRule {
    Code(u16),
    ServerError,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryNetworkRule {
    Timeout,
    Connect,
    Request,
}

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub jitter: Duration,
    pub retryable_statuses: Vec<RetryStatusRule>,
    pub retryable_network_errors: Vec<RetryNetworkRule>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            base_delay: Duration::from_millis(250),
            max_delay: Duration::from_millis(8_000),
            jitter: Duration::from_millis(200),
            retryable_statuses: vec![
                RetryStatusRule::Code(408),
                RetryStatusRule::Code(429),
                RetryStatusRule::ServerError,
            ],
            retryable_network_errors: vec![
                RetryNetworkRule::Timeout,
                RetryNetworkRule::Connect,
                RetryNetworkRule::Request,
            ],
        }
    }
}

impl RetryPolicy {
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    pub fn with_base_delay(mut self, base_delay: Duration) -> Self {
        self.base_delay = base_delay;
        self
    }

    pub fn with_max_delay(mut self, max_delay: Duration) -> Self {
        self.max_delay = max_delay;
        self
    }

    pub fn with_jitter(mut self, jitter: Duration) -> Self {
        self.jitter = jitter;
        self
    }

    pub fn with_retryable_statuses(mut self, statuses: Vec<RetryStatusRule>) -> Self {
        self.retryable_statuses = statuses;
        self
    }

    pub fn with_retryable_network_errors(mut self, errors: Vec<RetryNetworkRule>) -> Self {
        self.retryable_network_errors = errors;
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct TimeoutPolicy {
    pub request_timeout: Option<Duration>,
    pub total_timeout: Option<Duration>,
}

impl TimeoutPolicy {
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = Some(timeout);
        self
    }

    pub fn with_total_timeout(mut self, timeout: Duration) -> Self {
        self.total_timeout = Some(timeout);
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
    pub headers: HeaderMap,
    pub timeout: Option<std::time::Duration>,
    pub max_retries: Option<u32>,
    pub retry_policy: Option<RetryPolicy>,
    pub timeout_policy: Option<TimeoutPolicy>,
}

impl RequestOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_header(
        mut self,
        key: impl AsRef<str>,
        value: &str,
    ) -> Result<Self, crate::error::SdkError> {
        let name = reqwest::header::HeaderName::from_bytes(key.as_ref().as_bytes())
            .map_err(|e| crate::error::SdkError::ConfigError(e.to_string()))?;
        let val = reqwest::header::HeaderValue::from_str(value)
            .map_err(|e| crate::error::SdkError::ConfigError(e.to_string()))?;
        self.headers.insert(name, val);
        Ok(self)
    }

    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = Some(timeout);
        let policy = self
            .timeout_policy
            .get_or_insert_with(TimeoutPolicy::default);
        policy.request_timeout = Some(timeout);
        self
    }

    pub fn with_retries(mut self, retries: u32) -> Self {
        self.max_retries = Some(retries);
        let policy = self.retry_policy.get_or_insert_with(RetryPolicy::default);
        policy.max_retries = retries;
        self
    }

    /// Alias for `with_retries`, kept for API consistency with client configs.
    pub fn with_max_retries(self, retries: u32) -> Self {
        self.with_retries(retries)
    }

    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.max_retries = Some(policy.max_retries);
        self.retry_policy = Some(policy);
        self
    }

    pub fn with_timeout_policy(mut self, policy: TimeoutPolicy) -> Self {
        self.timeout = policy.request_timeout;
        self.timeout_policy = Some(policy);
        self
    }
}

/// Retry configuration extracted from a client's defaults and per-request options.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub base_url: String,
    pub endpoint: String,
    pub retry_policy: RetryPolicy,
    pub timeout_policy: TimeoutPolicy,
}

fn should_retry_status(status: StatusCode, retry_policy: &RetryPolicy) -> bool {
    retry_policy
        .retryable_statuses
        .iter()
        .any(|rule| match rule {
            RetryStatusRule::Code(code) => status.as_u16() == *code,
            RetryStatusRule::ServerError => status.is_server_error(),
        })
}

fn should_retry_network_error(error: &reqwest::Error, retry_policy: &RetryPolicy) -> bool {
    retry_policy
        .retryable_network_errors
        .iter()
        .any(|rule| match rule {
            RetryNetworkRule::Timeout => error.is_timeout(),
            RetryNetworkRule::Connect => error.is_connect(),
            RetryNetworkRule::Request => error.is_request(),
        })
}

fn retry_delay(attempt: u32, retry_policy: &RetryPolicy) -> Duration {
    let capped_attempt = attempt.min(10);
    let exp_multiplier = 2_u64.saturating_pow(capped_attempt.saturating_sub(1));

    let base_ms = retry_policy.base_delay.as_millis() as u64;
    let max_ms = retry_policy.max_delay.as_millis() as u64;
    let jitter_ms = retry_policy.jitter.as_millis() as u64;

    let backoff_ms = base_ms.saturating_mul(exp_multiplier).min(max_ms);
    let jitter = random_jitter(jitter_ms);
    Duration::from_millis(backoff_ms.saturating_add(jitter)).min(retry_policy.max_delay)
}

fn random_jitter(max_jitter_ms: u64) -> u64 {
    if max_jitter_ms == 0 {
        return 0;
    }

    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    seed % max_jitter_ms
}

fn retry_after_delay(headers: &HeaderMap, retry_policy: &RetryPolicy) -> Option<Duration> {
    let header = headers.get(RETRY_AFTER)?;
    let seconds = header.to_str().ok()?.trim().parse::<u64>().ok()?;
    Some(Duration::from_secs(seconds).min(retry_policy.max_delay))
}

fn resolve_retry_policy(config: &RetryConfig, options: &RequestOptions) -> RetryPolicy {
    if let Some(policy) = &options.retry_policy {
        return clamp_retry_policy(policy.clone());
    }

    let mut policy = config.retry_policy.clone();
    if let Some(retries) = options.max_retries {
        policy.max_retries = retries;
    }
    clamp_retry_policy(policy)
}

fn clamp_retry_policy(mut policy: RetryPolicy) -> RetryPolicy {
    policy.max_retries = policy.max_retries.min(MAX_RETRIES_CAP);
    policy
}

fn resolve_timeout_policy(config: &RetryConfig, options: &RequestOptions) -> TimeoutPolicy {
    if let Some(policy) = &options.timeout_policy {
        return policy.clone();
    }

    let mut policy = config.timeout_policy.clone();
    if let Some(timeout) = options.timeout {
        policy.request_timeout = Some(timeout);
    }
    policy
}

fn exceeds_total_budget(started_at: Instant, total_timeout: Duration, next_wait: Duration) -> bool {
    started_at.elapsed().saturating_add(next_wait) > total_timeout
}

/// Send an HTTP POST request with exponential backoff retry.
///
/// This is the shared "Physics" layer: every provider SDK uses this
/// to send requests and handle transient failures identically.
pub async fn send_with_retry<T: Serialize>(
    http_client: &reqwest::Client,
    config: &RetryConfig,
    request_body: &T,
    options: &RequestOptions,
) -> Result<reqwest::Response, SdkError> {
    let url = format!("{}{}", config.base_url, config.endpoint);
    let retry_policy = resolve_retry_policy(config, options);
    let timeout_policy = resolve_timeout_policy(config, options);
    let max_retries = retry_policy.max_retries;
    let started_at = Instant::now();
    let mut retries = 0;

    loop {
        if let Some(total_timeout) = timeout_policy.total_timeout
            && started_at.elapsed() > total_timeout
        {
            return Err(SdkError::ApiError(format!(
                "API request aborted: total timeout budget of {:?} was exceeded",
                total_timeout
            )));
        }

        let mut request_builder = http_client.request(Method::POST, &url).json(request_body);

        if let Some(timeout) = timeout_policy.request_timeout {
            request_builder = request_builder.timeout(timeout);
        }

        if !options.headers.is_empty() {
            request_builder = request_builder.headers(options.headers.clone());
        }

        let response_result = request_builder.send().await;

        match response_result {
            Ok(response) => {
                if response.status().is_success() {
                    return Ok(response);
                }

                let status = response.status();
                if should_retry_status(status, &retry_policy) && retries < max_retries {
                    retries += 1;
                    let wait = retry_after_delay(response.headers(), &retry_policy)
                        .unwrap_or_else(|| retry_delay(retries, &retry_policy));
                    let from_retry_after =
                        retry_after_delay(response.headers(), &retry_policy).is_some();

                    if let Some(total_timeout) = timeout_policy.total_timeout
                        && exceeds_total_budget(started_at, total_timeout, wait)
                    {
                        return Err(SdkError::ApiError(format!(
                            "API request aborted: waiting {:?} would exceed total timeout budget {:?}",
                            wait, total_timeout
                        )));
                    }

                    warn!(
                        attempt = retries,
                        max_retries,
                        status = status.as_u16(),
                        wait_ms = wait.as_millis() as u64,
                        from_retry_after,
                        %url,
                        "retrying request after retryable status"
                    );
                    tokio::time::sleep(wait).await;
                    continue;
                }

                let error_text = response.text().await.unwrap_or_default();
                return Err(SdkError::ApiError(format!(
                    "API request failed (status {}): {}",
                    status, error_text
                )));
            }
            Err(e) => {
                if should_retry_network_error(&e, &retry_policy) && retries < max_retries {
                    retries += 1;
                    let wait = retry_delay(retries, &retry_policy);

                    if let Some(total_timeout) = timeout_policy.total_timeout
                        && exceeds_total_budget(started_at, total_timeout, wait)
                    {
                        return Err(SdkError::ApiError(format!(
                            "API request aborted: waiting {:?} would exceed total timeout budget {:?}",
                            wait, total_timeout
                        )));
                    }

                    warn!(
                        attempt = retries,
                        max_retries,
                        wait_ms = wait.as_millis() as u64,
                        timeout = e.is_timeout(),
                        connect = e.is_connect(),
                        request = e.is_request(),
                        %url,
                        "retrying request after network error"
                    );
                    tokio::time::sleep(wait).await;
                    continue;
                }
                return Err(SdkError::NetworkError(e));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_retry_policy_includes_core_rules() {
        let policy = RetryPolicy::default();
        assert!(should_retry_status(StatusCode::REQUEST_TIMEOUT, &policy));
        assert!(should_retry_status(StatusCode::TOO_MANY_REQUESTS, &policy));
        assert!(should_retry_status(
            StatusCode::INTERNAL_SERVER_ERROR,
            &policy
        ));
        assert!(!should_retry_status(StatusCode::BAD_REQUEST, &policy));
    }

    #[test]
    fn test_retry_delay_respects_bounds() {
        let policy = RetryPolicy::default()
            .with_base_delay(Duration::from_millis(100))
            .with_max_delay(Duration::from_millis(400))
            .with_jitter(Duration::from_millis(50));

        let early_wait = retry_delay(1, &policy);
        assert!(early_wait >= Duration::from_millis(100));
        assert!(early_wait <= Duration::from_millis(149));

        let wait = retry_delay(10, &policy);
        assert_eq!(wait, Duration::from_millis(400));
    }

    #[test]
    fn test_retry_after_header_is_capped() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, "100".parse().expect("valid retry-after"));

        let policy = RetryPolicy::default().with_max_delay(Duration::from_secs(5));
        let delay = retry_after_delay(&headers, &policy).expect("retry-after should parse");
        assert_eq!(delay, Duration::from_secs(5));
    }

    #[test]
    fn test_clamp_retry_policy_caps_retries() {
        let policy = RetryPolicy::default().with_max_retries(999);
        let clamped = clamp_retry_policy(policy);
        assert_eq!(clamped.max_retries, MAX_RETRIES_CAP);
    }

    #[test]
    fn test_request_options_accepts_dynamic_header_names() {
        let options = RequestOptions::default()
            .with_header("x-custom-runtime-header", "value")
            .expect("dynamic header name should be accepted");

        assert_eq!(
            options
                .headers
                .get("x-custom-runtime-header")
                .and_then(|v| v.to_str().ok()),
            Some("value")
        );
    }
}
