use anyhow::{Context, Result};
use turin_channel_core::ChannelAttachment;

use crate::{
    TelegramApiError, TelegramChannelDriver,
    inbound::{TelegramAttachmentKind, TelegramAttachmentRef, TelegramFile, TelegramMessage},
    outbound::unique_media_name,
};

impl TelegramChannelDriver {
    pub(crate) async fn collect_inbound_attachments(
        &self,
        message: &TelegramMessage,
    ) -> Result<Vec<ChannelAttachment>> {
        let refs = message.attachment_refs();
        if refs.is_empty() {
            return Ok(Vec::new());
        }

        tokio::fs::create_dir_all(&self.media_dir)
            .await
            .with_context(|| {
                format!(
                    "Failed to create Telegram media directory '{}'",
                    self.media_dir.display()
                )
            })?;

        let mut attachments = Vec::with_capacity(refs.len());
        for attachment in refs {
            attachments.push(self.download_inbound_attachment(&attachment).await?);
        }
        Ok(attachments)
    }

    async fn download_inbound_attachment(
        &self,
        attachment: &TelegramAttachmentRef,
    ) -> Result<ChannelAttachment> {
        let file: TelegramFile = self
            .request_with_retry(
                "getFile",
                &serde_json::json!({ "file_id": attachment.file_id }),
            )
            .await
            .map_err(TelegramApiError::into_anyhow)?;
        let file_path = file.file_path.context(format!(
            "Telegram getFile response missing file_path for '{}'",
            attachment.file_id
        ))?;
        let download_url = self.telegram_file_url(&file_path);
        let response = self
            .client
            .get(&download_url)
            .send()
            .await
            .with_context(|| format!("Telegram file download failed for '{}'", attachment.name))?
            .error_for_status()
            .with_context(|| {
                format!(
                    "Telegram file download returned error status for '{}'",
                    attachment.name
                )
            })?;
        let fetched_content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(|value| value.split(';').next().unwrap_or(value).trim().to_string());
        let bytes = response
            .bytes()
            .await
            .with_context(|| format!("Failed to read Telegram file '{}'", attachment.name))?;
        let target_path = self.media_dir.join(unique_media_name(
            &attachment.name,
            Some(file_path.as_str()),
        ));
        tokio::fs::write(&target_path, bytes)
            .await
            .with_context(|| {
                format!(
                    "Failed to write Telegram media attachment '{}'",
                    target_path.display()
                )
            })?;
        Ok(ChannelAttachment {
            name: attachment.name.clone(),
            content_type: attachment
                .content_type
                .clone()
                .or(fetched_content_type)
                .or_else(|| match attachment.kind {
                    TelegramAttachmentKind::Image => Some("image/jpeg".to_string()),
                    TelegramAttachmentKind::File => None,
                }),
            url: None,
            local_path: Some(target_path.display().to_string()),
        })
    }

    fn telegram_file_url(&self, file_path: &str) -> String {
        format!(
            "{}/file/bot{}/{}",
            self.config.base_url,
            self.config.token,
            file_path.trim_start_matches('/')
        )
    }
}
