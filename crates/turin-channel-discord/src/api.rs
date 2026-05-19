use anyhow::{Context, Result};
use serde::Deserialize;
use std::time::Duration;
use tokio::time::sleep;

use crate::{
    DiscordChannelDriver,
    render::{DiscordSendMessage, LocalAttachmentRef, discord_payload_from_message},
};

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct DiscordMessage {
    pub(crate) id: String,
    pub(crate) channel_id: String,
    #[serde(default)]
    pub(crate) guild_id: Option<String>,
    #[serde(default)]
    pub(crate) content: String,
    pub(crate) author: DiscordAuthor,
    #[serde(default)]
    pub(crate) attachments: Vec<DiscordAttachment>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct DiscordAuthor {
    pub(crate) id: String,
    pub(crate) username: String,
    #[serde(default)]
    pub(crate) global_name: Option<String>,
    #[serde(default)]
    pub(crate) bot: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct DiscordAttachment {
    pub(crate) filename: String,
    #[serde(default)]
    pub(crate) content_type: Option<String>,
    pub(crate) url: String,
}

#[derive(Debug, Clone, Deserialize)]
struct DiscordRateLimit {
    retry_after: f64,
}

#[derive(Debug, Clone)]
struct PreparedLocalFile {
    name: String,
    content_type: Option<String>,
    bytes: Vec<u8>,
}

impl DiscordChannelDriver {
    pub(crate) async fn fetch_latest_message_id(&self) -> Result<Option<String>> {
        let mut messages = self.fetch_messages(None, 1).await?;
        Ok(messages.pop().map(|msg| msg.id))
    }

    pub(crate) async fn fetch_messages(
        &self,
        after: Option<&str>,
        limit: u16,
    ) -> Result<Vec<DiscordMessage>> {
        let url = format!(
            "{}/channels/{}/messages",
            self.config.base_url, self.config.channel_id
        );
        let mut params = vec![("limit".to_string(), limit.to_string())];
        if let Some(after) = after {
            params.push(("after".to_string(), after.to_string()));
        }

        let response = self
            .request_with_retry(|| {
                self.client
                    .get(&url)
                    .header("Authorization", format!("Bot {}", self.config.token))
                    .query(&params)
                    .build()
                    .context("[discord_http_build_messages_request_failed] Failed to build Discord messages request")
            })
            .await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "[discord_http_messages_failed] Discord messages request failed with {}: {}",
                status.as_u16(),
                body
            );
        }

        response.json::<Vec<DiscordMessage>>().await.context(
            "[discord_http_decode_messages_failed] Failed to decode Discord messages response",
        )
    }

    pub(crate) async fn post_message(
        &self,
        channel_id: &str,
        message: DiscordSendMessage,
    ) -> Result<()> {
        let url = format!("{}/channels/{}/messages", self.config.base_url, channel_id);
        let payload = discord_payload_from_message(&message);

        let response = if message.files.is_empty() {
            self.request_with_retry(|| {
                self.client
                    .post(&url)
                    .header("Authorization", format!("Bot {}", self.config.token))
                    .json(&payload)
                    .build()
                    .context("[discord_http_build_send_request_failed] Failed to build Discord send request")
            })
            .await?
        } else {
            let prepared = prepare_local_files(&message.files).await?;
            self.request_with_retry(|| {
                let payload_json = serde_json::to_string(&payload)
                    .context("Failed to encode Discord multipart payload")?;
                let mut form = reqwest::multipart::Form::new().text("payload_json", payload_json);
                for (index, file) in prepared.iter().enumerate() {
                    let mut part = reqwest::multipart::Part::bytes(file.bytes.clone())
                        .file_name(file.name.clone());
                    if let Some(content_type) = &file.content_type {
                        part = part.mime_str(content_type).with_context(|| {
                            format!("Invalid content type '{}' for '{}'", content_type, file.name)
                        })?;
                    }
                    form = form.part(format!("files[{index}]"), part);
                }

                self.client
                    .post(&url)
                    .header("Authorization", format!("Bot {}", self.config.token))
                    .multipart(form)
                    .build()
                    .context("[discord_http_build_multipart_send_request_failed] Failed to build Discord multipart send request")
            })
            .await?
        };

        let status = response.status();
        if status == reqwest::StatusCode::OK || status == reqwest::StatusCode::CREATED {
            return Ok(());
        }

        let body = response.text().await.unwrap_or_default();
        anyhow::bail!(
            "[discord_send_failed] Discord send request failed with {}: {}",
            status.as_u16(),
            body
        );
    }

    async fn request_with_retry<F>(&self, request_builder: F) -> Result<reqwest::Response>
    where
        F: Fn() -> Result<reqwest::Request>,
    {
        let mut attempts = 0;
        loop {
            attempts += 1;
            let request = request_builder()?;
            let response = match self.client.execute(request).await {
                Ok(response) => response,
                Err(error) => {
                    if attempts < 5 {
                        sleep(retry_backoff(attempts)).await;
                        continue;
                    }
                    return Err(error)
                        .context("[discord_http_request_failed] Discord request failed");
                }
            };

            if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS && attempts < 6 {
                let delay = parse_rate_limit_delay(response)
                    .await
                    .unwrap_or_else(|| retry_backoff(attempts));
                sleep(delay).await;
                continue;
            }

            if response.status().is_server_error() && attempts < 5 {
                sleep(retry_backoff(attempts)).await;
                continue;
            }
            return Ok(response);
        }
    }
}

pub(crate) fn parse_snowflake(value: &str) -> Option<u64> {
    value.parse::<u64>().ok()
}

pub(crate) fn is_newer_snowflake(candidate: &str, current: &str) -> bool {
    match (parse_snowflake(candidate), parse_snowflake(current)) {
        (Some(candidate), Some(current)) => candidate > current,
        _ => candidate > current,
    }
}

async fn prepare_local_files(files: &[LocalAttachmentRef]) -> Result<Vec<PreparedLocalFile>> {
    let mut prepared = Vec::new();
    for file in files {
        let bytes = tokio::fs::read(&file.path).await.with_context(|| {
            format!("Failed to read local attachment '{}'", file.path.display())
        })?;
        prepared.push(PreparedLocalFile {
            name: file.name.clone(),
            content_type: file.content_type.clone(),
            bytes,
        });
    }
    Ok(prepared)
}

async fn parse_rate_limit_delay(response: reqwest::Response) -> Option<Duration> {
    let header_delay = response
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|raw| raw.parse::<f64>().ok())
        .map(Duration::from_secs_f64);
    if header_delay.is_some() {
        return header_delay;
    }

    let reset_after = response
        .headers()
        .get("x-ratelimit-reset-after")
        .and_then(|value| value.to_str().ok())
        .and_then(|raw| raw.parse::<f64>().ok())
        .map(Duration::from_secs_f64);
    if reset_after.is_some() {
        return reset_after;
    }

    let body_delay = response
        .text()
        .await
        .ok()
        .and_then(|raw| serde_json::from_str::<DiscordRateLimit>(&raw).ok())
        .map(|rate| Duration::from_secs_f64(rate.retry_after.max(0.1)));
    if body_delay.is_some() {
        return body_delay;
    }

    None
}

fn retry_backoff(attempt: u32) -> Duration {
    let exponent = attempt.min(5);
    let millis = 200u64.saturating_mul(2u64.saturating_pow(exponent));
    Duration::from_millis(millis.min(6_000))
}
