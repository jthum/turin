use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use tokio::time::sleep;
use tracing::warn;

use crate::{MAX_API_REQUEST_ATTEMPTS, TelegramBotIdentity, TelegramChannelDriver, TelegramUser};

impl TelegramChannelDriver {
    async fn api_request_once<T: DeserializeOwned>(
        &self,
        method: &str,
        payload: &serde_json::Value,
    ) -> std::result::Result<T, TelegramApiError> {
        let url = format!(
            "{}/bot{}/{}",
            self.config.base_url, self.config.token, method
        );
        let response = self
            .client
            .post(&url)
            .json(payload)
            .send()
            .await
            .map_err(|error| TelegramApiError {
                code: "telegram_http_request_failed".to_string(),
                message: format!("Telegram {} request failed: {}", method, error),
                retriable: true,
                retry_after: None,
            })?;

        self.decode_api_response(method, response).await
    }

    async fn api_multipart_request_once<T: DeserializeOwned>(
        &self,
        method: &str,
        form: reqwest::multipart::Form,
    ) -> std::result::Result<T, TelegramApiError> {
        let url = format!(
            "{}/bot{}/{}",
            self.config.base_url, self.config.token, method
        );
        let response = self
            .client
            .post(&url)
            .multipart(form)
            .send()
            .await
            .map_err(|error| TelegramApiError {
                code: "telegram_http_request_failed".to_string(),
                message: format!("Telegram {} multipart request failed: {}", method, error),
                retriable: true,
                retry_after: None,
            })?;

        self.decode_api_response(method, response).await
    }

    async fn decode_api_response<T: DeserializeOwned>(
        &self,
        method: &str,
        response: reqwest::Response,
    ) -> std::result::Result<T, TelegramApiError> {
        let status = response.status();
        let body = response
            .text()
            .await
            .with_context(|| {
                format!(
                    "[telegram_http_decode_failed] Failed to read Telegram {} response body",
                    method
                )
            })
            .map_err(|error| TelegramApiError {
                code: "telegram_http_decode_failed".to_string(),
                message: error.to_string(),
                retriable: true,
                retry_after: None,
            })?;

        let envelope: TelegramApiEnvelope<T> = serde_json::from_str(&body)
            .with_context(|| {
                format!(
                    "[telegram_http_decode_failed] Failed to decode Telegram {} response: {}",
                    method, body
                )
            })
            .map_err(|error| TelegramApiError {
                code: "telegram_http_decode_failed".to_string(),
                message: error.to_string(),
                retriable: false,
                retry_after: None,
            })?;

        if !status.is_success() || !envelope.ok {
            let description = envelope.description.clone().unwrap_or_else(|| body.clone());
            let error_code = envelope.error_code.unwrap_or(status.as_u16() as i64);
            let code = classify_api_error(method, status.as_u16(), &description);
            let retriable = is_retriable_api_error(&code, status.as_u16())
                && !is_not_modified_description(&description);
            return Err(TelegramApiError {
                retriable,
                retry_after: envelope
                    .parameters
                    .as_ref()
                    .and_then(|parameters| parameters.retry_after)
                    .map(Duration::from_secs),
                code,
                message: format!(
                    "Telegram {} request failed with {}: {}",
                    method, error_code, description
                ),
            });
        }

        envelope
            .result
            .context(format!("Telegram {} response missing result", method))
            .map_err(|error| TelegramApiError {
                code: "telegram_missing_result".to_string(),
                message: error.to_string(),
                retriable: false,
                retry_after: None,
            })
    }

    pub(super) async fn request_with_retry<T: DeserializeOwned>(
        &self,
        method: &str,
        payload: &serde_json::Value,
    ) -> std::result::Result<T, TelegramApiError> {
        let mut attempts: u32 = 0;
        loop {
            attempts = attempts.saturating_add(1);
            match self.api_request_once(method, payload).await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if !error.retriable || attempts >= MAX_API_REQUEST_ATTEMPTS {
                        return Err(error);
                    }

                    let delay = error.retry_after.unwrap_or_else(|| retry_backoff(attempts));
                    warn!(
                        channel_runtime_id = %self.channel_runtime_id,
                        method,
                        attempt = attempts,
                        delay_ms = delay.as_millis() as u64,
                        error_code = %error.code,
                        error = %error.message,
                        "Retrying Telegram request after transient failure"
                    );
                    sleep(delay).await;
                }
            }
        }
    }

    pub(super) async fn multipart_request_with_retry<T, F>(
        &self,
        method: &str,
        form_builder: F,
    ) -> std::result::Result<T, TelegramApiError>
    where
        T: DeserializeOwned,
        F: Fn() -> Result<reqwest::multipart::Form>,
    {
        let mut attempts: u32 = 0;
        loop {
            attempts = attempts.saturating_add(1);
            let form = form_builder().map_err(|error| TelegramApiError {
                code: "telegram_multipart_build_failed".to_string(),
                message: error.to_string(),
                retriable: false,
                retry_after: None,
            })?;

            match self.api_multipart_request_once(method, form).await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if !error.retriable || attempts >= MAX_API_REQUEST_ATTEMPTS {
                        return Err(error);
                    }

                    let delay = error.retry_after.unwrap_or_else(|| retry_backoff(attempts));
                    warn!(
                        channel_runtime_id = %self.channel_runtime_id,
                        method,
                        attempt = attempts,
                        delay_ms = delay.as_millis() as u64,
                        error_code = %error.code,
                        error = %error.message,
                        "Retrying Telegram multipart request after transient failure"
                    );
                    sleep(delay).await;
                }
            }
        }
    }

    pub(super) async fn sleep_or_shutdown(&self, duration: Duration) -> bool {
        let mut shutdown_rx = self.shutdown_rx.clone();
        tokio::select! {
            changed = shutdown_rx.changed() => changed.is_ok() && *shutdown_rx.borrow(),
            _ = sleep(duration) => false,
        }
    }

    pub(super) async fn handle_transient_poll_error(
        &mut self,
        phase: &str,
        error: TelegramApiError,
    ) -> bool {
        self.consecutive_poll_failures = self.consecutive_poll_failures.saturating_add(1);
        let delay = error
            .retry_after
            .unwrap_or_else(|| retry_backoff(self.consecutive_poll_failures));
        warn!(
            channel_runtime_id = %self.channel_runtime_id,
            phase,
            error_code = %error.code,
            error = %error.message,
            delay_ms = delay.as_millis() as u64,
            "Telegram polling hit a transient failure; backing off"
        );
        self.sleep_or_shutdown(delay).await
    }

    pub(super) async fn ensure_bot_identity(
        &mut self,
    ) -> std::result::Result<(), TelegramApiError> {
        if self.bot_identity.is_some() || !self.config.respond_mode.requires_bot_identity() {
            return Ok(());
        }

        let bot: TelegramUser = self
            .request_with_retry("getMe", &serde_json::json!({}))
            .await?;
        self.bot_identity = Some(TelegramBotIdentity {
            id: bot.id,
            username: bot.username.map(|username| username.to_ascii_lowercase()),
        });
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
struct TelegramApiEnvelope<T> {
    ok: bool,
    result: Option<T>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    error_code: Option<i64>,
    #[serde(default)]
    parameters: Option<TelegramApiParameters>,
}

#[derive(Debug, Clone, Deserialize)]
struct TelegramApiParameters {
    #[serde(default)]
    retry_after: Option<u64>,
}

#[derive(Debug, Clone)]
pub(crate) struct TelegramApiError {
    pub(crate) code: String,
    pub(crate) message: String,
    pub(crate) retriable: bool,
    pub(crate) retry_after: Option<Duration>,
}

impl TelegramApiError {
    pub(crate) fn into_anyhow(self) -> anyhow::Error {
        anyhow!("[{}] {}", self.code, self.message)
    }

    pub(crate) fn is_message_not_modified(&self) -> bool {
        self.code == "telegram_edit_message_failed" && is_not_modified_description(&self.message)
    }
}

fn classify_api_error(method: &str, status_code: u16, description: &str) -> String {
    let lower = description.to_ascii_lowercase();
    if status_code == 401 || lower.contains("unauthorized") {
        return "telegram_auth_invalid_token".to_string();
    }
    if status_code == 429 || lower.contains("too many requests") {
        return "telegram_rate_limited".to_string();
    }

    match method {
        "getUpdates" => {
            if lower.contains("webhook") {
                "telegram_polling_webhook_active".to_string()
            } else if lower.contains("terminated by other getupdates request")
                || lower.contains("terminated by other long poll")
            {
                "telegram_polling_conflict".to_string()
            } else {
                "telegram_get_updates_failed".to_string()
            }
        }
        "sendMessage" => {
            if lower.contains("chat not found") {
                "telegram_send_chat_not_found".to_string()
            } else {
                "telegram_send_failed".to_string()
            }
        }
        "sendMessageDraft" => "telegram_send_draft_failed".to_string(),
        "editMessageText" => "telegram_edit_message_failed".to_string(),
        "sendChatAction" => "telegram_chat_action_failed".to_string(),
        _ => "telegram_api_failed".to_string(),
    }
}

fn is_retriable_api_error(code: &str, status_code: u16) -> bool {
    status_code == 429
        || status_code >= 500
        || matches!(
            code,
            "telegram_http_request_failed"
                | "telegram_http_decode_failed"
                | "telegram_rate_limited"
                | "telegram_send_failed"
                | "telegram_send_draft_failed"
                | "telegram_edit_message_failed"
                | "telegram_chat_action_failed"
                | "telegram_get_updates_failed"
        )
}

fn is_not_modified_description(description: &str) -> bool {
    description
        .to_ascii_lowercase()
        .contains("message is not modified")
}

fn retry_backoff(attempt: u32) -> Duration {
    let exponent = attempt.min(5);
    let millis = 250u64.saturating_mul(2u64.saturating_pow(exponent));
    Duration::from_millis(millis.min(8_000))
}
