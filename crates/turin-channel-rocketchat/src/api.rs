use anyhow::{Context, Result, anyhow};
use reqwest::Client;
use serde::{Deserialize, Deserializer};
use std::time::Duration;
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use crate::{
    ROCKETCHAT_HTTP_CONNECT_TIMEOUT_SECONDS, ROCKETCHAT_HTTP_TIMEOUT_SECONDS,
    RocketChatChannelDriverConfig,
};

pub(crate) fn absolute_url(base_url: &str, raw: &str) -> String {
    if raw.starts_with("http://") || raw.starts_with("https://") {
        raw.to_string()
    } else {
        format!(
            "{}/{}",
            base_url.trim_end_matches('/'),
            raw.trim_start_matches('/')
        )
    }
}

pub(crate) fn api_url(base_url: &str, path: &str) -> String {
    format!("{}/api/v1/{}", base_url.trim_end_matches('/'), path)
}

pub(crate) fn build_http_client() -> Result<Client> {
    Client::builder()
        .connect_timeout(Duration::from_secs(ROCKETCHAT_HTTP_CONNECT_TIMEOUT_SECONDS))
        .timeout(Duration::from_secs(ROCKETCHAT_HTTP_TIMEOUT_SECONDS))
        .build()
        .context("[rocketchat_http_client_build_failed] Failed to build Rocket.Chat HTTP client")
}

pub(crate) async fn fetch_bot_identity(
    client: &Client,
    config: &RocketChatChannelDriverConfig,
) -> Result<RocketChatBotIdentity> {
    let response = client
        .get(api_url(&config.base_url, "users.info"))
        .header("X-Auth-Token", &config.token)
        .header("X-User-Id", &config.user_id)
        .query(&[("userId", config.user_id.as_str())])
        .send()
        .await
        .context("Failed to query Rocket.Chat user info")?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!(
            "[rocketchat_user_info_failed] Rocket.Chat users.info failed with status {}: {}",
            status,
            body
        );
    }

    let parsed: RocketChatUserInfoResponse =
        serde_json::from_str(&body).context("Failed to decode Rocket.Chat users.info response")?;
    let username = parsed
        .user
        .username
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            anyhow!(
                "[rocketchat_user_info_missing_username] Rocket.Chat users.info did not return a username for '{}'",
                config.user_id
            )
        })?;
    let display_name = parsed
        .user
        .name
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    Ok(RocketChatBotIdentity {
        username,
        display_name,
    })
}

pub(crate) async fn fetch_rooms(
    client: &Client,
    config: &RocketChatChannelDriverConfig,
    updated_since: Option<&str>,
) -> Result<RocketChatRoomsUpdate> {
    let mut request = client
        .get(api_url(&config.base_url, "rooms.get"))
        .header("X-Auth-Token", &config.token)
        .header("X-User-Id", &config.user_id);

    if let Some(updated_since) = updated_since {
        request = request.query(&[("updatedSince", updated_since)]);
    }

    let response = request
        .send()
        .await
        .context("Failed to query Rocket.Chat rooms")?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!(
            "[rocketchat_rooms_get_failed] Rocket.Chat rooms.get failed with status {}: {}",
            status,
            body
        );
    }

    let parsed: RocketChatRoomsResponse =
        serde_json::from_str(&body).context("Failed to decode Rocket.Chat rooms.get response")?;
    let next_updated_since = parsed
        .update
        .iter()
        .filter_map(|room| room.updated_at.clone())
        .max();
    let rooms = parsed
        .update
        .into_iter()
        .map(RocketChatResolvedRoom::try_from)
        .collect::<Result<Vec<_>>>()?;

    Ok(RocketChatRoomsUpdate {
        rooms,
        remove_room_ids: parsed.remove,
        next_updated_since,
    })
}

pub(crate) async fn fetch_room_messages(
    client: &Client,
    config: &RocketChatChannelDriverConfig,
    room: &RocketChatResolvedRoom,
    cursor_ts: Option<&str>,
) -> Result<Vec<RocketChatMessage>> {
    let endpoint = match room.room_type {
        RocketChatRoomType::Channel => "channels.history",
        RocketChatRoomType::PrivateGroup => "groups.history",
        RocketChatRoomType::DirectMessage => "dm.history",
    };

    let mut request = client
        .get(api_url(&config.base_url, endpoint))
        .header("X-Auth-Token", &config.token)
        .header("X-User-Id", &config.user_id)
        .query(&[("roomId", room.id.as_str())]);

    if let Some(cursor_ts) = cursor_ts {
        request = request.query(&[("oldest", cursor_ts), ("inclusive", "true")]);
        request = request.query(&[("count", config.max_messages_per_poll.to_string())]);
        request = request.query(&[("sort", "{\"ts\":1,\"_id\":1}")]);
        request = request.query(&[("showThreadMessages", "true")]);
    } else {
        request = request.query(&[("count", config.max_messages_per_poll.to_string())]);
        request = request.query(&[("sort", "{\"ts\":-1,\"_id\":-1}")]);
        request = request.query(&[("showThreadMessages", "true")]);
    }

    let response = request
        .send()
        .await
        .context("Failed to query Rocket.Chat room history")?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!(
            "[rocketchat_history_failed] Rocket.Chat history request failed with status {}: {}",
            status,
            body
        );
    }

    let mut parsed: RocketChatHistoryResponse =
        serde_json::from_str(&body).context("Failed to decode Rocket.Chat history response")?;
    parsed
        .messages
        .sort_by(|left, right| left.ts.cmp(&right.ts).then_with(|| left.id.cmp(&right.id)));
    Ok(parsed.messages)
}

fn deserialize_rocketchat_timestamp<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    normalize_rocketchat_timestamp_value(value).map_err(serde::de::Error::custom)
}

fn deserialize_optional_rocketchat_timestamp<'de, D>(
    deserializer: D,
) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    value
        .map(normalize_rocketchat_timestamp_value)
        .transpose()
        .map_err(serde::de::Error::custom)
}

fn normalize_rocketchat_timestamp_value(value: serde_json::Value) -> Result<String> {
    match value {
        serde_json::Value::String(raw) => normalize_rocketchat_timestamp_string(&raw),
        serde_json::Value::Object(map) => {
            if let Some(inner) = map.get("$date") {
                return normalize_rocketchat_timestamp_value(inner.clone());
            }
            anyhow::bail!(
                "[rocketchat_timestamp_invalid] Rocket.Chat timestamp object must contain '$date'"
            );
        }
        serde_json::Value::Number(number) => normalize_rocketchat_timestamp_number(&number),
        other => anyhow::bail!(
            "[rocketchat_timestamp_invalid] Rocket.Chat timestamp must be a string, number, or {{$date: ...}}, got {}",
            other
        ),
    }
}

fn normalize_rocketchat_timestamp_string(raw: &str) -> Result<String> {
    match OffsetDateTime::parse(raw, &Rfc3339) {
        Ok(parsed) => parsed
            .format(&Rfc3339)
            .map_err(anyhow::Error::from)
            .context("[rocketchat_timestamp_format_failed] Failed to format Rocket.Chat timestamp"),
        Err(_) => Ok(raw.to_string()),
    }
}

fn normalize_rocketchat_timestamp_number(number: &serde_json::Number) -> Result<String> {
    let timestamp = if let Some(value) = number.as_i64() {
        value
    } else if let Some(value) = number.as_u64() {
        i64::try_from(value).context(
            "[rocketchat_timestamp_out_of_range] Rocket.Chat timestamp number does not fit in i64",
        )?
    } else {
        anyhow::bail!(
            "[rocketchat_timestamp_invalid] Rocket.Chat floating-point timestamps are not supported"
        );
    };

    let nanos = if timestamp.abs() >= 10_000_000_000 {
        i128::from(timestamp) * 1_000_000
    } else {
        i128::from(timestamp) * 1_000_000_000
    };
    let parsed = OffsetDateTime::from_unix_timestamp_nanos(nanos).context(
        "[rocketchat_timestamp_out_of_range] Rocket.Chat numeric timestamp is out of range",
    )?;
    parsed
        .format(&Rfc3339)
        .map_err(anyhow::Error::from)
        .context("[rocketchat_timestamp_format_failed] Failed to format Rocket.Chat timestamp")
}

impl RocketChatRoomType {
    fn parse(raw: &str) -> Result<Self> {
        match raw {
            "c" => Ok(Self::Channel),
            "p" => Ok(Self::PrivateGroup),
            "d" => Ok(Self::DirectMessage),
            other => anyhow::bail!(
                "[rocketchat_room_type_unsupported] Rocket.Chat room type '{}' is not supported yet",
                other
            ),
        }
    }
}

impl TryFrom<RocketChatRoomInfo> for RocketChatResolvedRoom {
    type Error = anyhow::Error;

    fn try_from(value: RocketChatRoomInfo) -> Result<Self> {
        Ok(Self {
            id: value.id,
            room_type: RocketChatRoomType::parse(&value.kind)?,
            name: value.name,
            friendly_name: value.friendly_name,
            usernames: value.usernames,
            latest_message_id: value
                .last_message
                .as_ref()
                .map(|message| message.id.clone()),
            latest_message_ts: value.last_message_at.or_else(|| {
                value
                    .last_message
                    .as_ref()
                    .map(|message| message.ts.clone())
            }),
            latest_message: value.last_message,
        })
    }
}

#[derive(Debug)]
pub(crate) struct RocketChatRoomsUpdate {
    pub(crate) rooms: Vec<RocketChatResolvedRoom>,
    pub(crate) remove_room_ids: Vec<String>,
    pub(crate) next_updated_since: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RocketChatRoomsResponse {
    #[serde(default)]
    update: Vec<RocketChatRoomInfo>,
    #[serde(default)]
    remove: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RocketChatRoomInfo {
    #[serde(rename = "_id")]
    pub(crate) id: String,
    #[serde(rename = "t")]
    pub(crate) kind: String,
    #[serde(rename = "name")]
    pub(crate) name: Option<String>,
    #[serde(rename = "fname")]
    pub(crate) friendly_name: Option<String>,
    #[serde(default)]
    pub(crate) usernames: Vec<String>,
    #[serde(
        rename = "_updatedAt",
        default,
        deserialize_with = "deserialize_optional_rocketchat_timestamp"
    )]
    pub(crate) updated_at: Option<String>,
    #[serde(
        rename = "lm",
        default,
        deserialize_with = "deserialize_optional_rocketchat_timestamp"
    )]
    pub(crate) last_message_at: Option<String>,
    #[serde(rename = "lastMessage")]
    pub(crate) last_message: Option<RocketChatMessage>,
}

#[derive(Debug, Deserialize)]
struct RocketChatHistoryResponse {
    #[serde(default)]
    messages: Vec<RocketChatMessage>,
}

#[derive(Debug, Deserialize)]
struct RocketChatUserInfoResponse {
    user: RocketChatApiUser,
}

#[derive(Debug, Deserialize)]
struct RocketChatApiUser {
    username: Option<String>,
    name: Option<String>,
}

#[derive(Debug)]
pub(crate) struct RocketChatBotIdentity {
    pub(crate) username: String,
    pub(crate) display_name: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RocketChatSendMessageResponse {
    pub(crate) message: Option<RocketChatSentMessage>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RocketChatSentMessage {
    #[serde(rename = "_id")]
    pub(crate) id: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RocketChatMessage {
    #[serde(rename = "_id")]
    pub(crate) id: String,
    #[serde(rename = "msg")]
    pub(crate) text: Option<String>,
    #[serde(deserialize_with = "deserialize_rocketchat_timestamp")]
    pub(crate) ts: String,
    #[serde(rename = "u")]
    pub(crate) user: Option<RocketChatMessageUser>,
    #[serde(rename = "t")]
    pub(crate) kind: Option<String>,
    #[serde(rename = "tmid")]
    pub(crate) thread_root_id: Option<String>,
    #[serde(default)]
    pub(crate) mentions: Vec<RocketChatMention>,
    #[serde(default)]
    pub(crate) attachments: Vec<RocketChatApiAttachment>,
    #[serde(default)]
    pub(crate) file: Option<RocketChatFileInfo>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RocketChatMessageUser {
    #[serde(rename = "_id")]
    pub(crate) id: String,
    pub(crate) username: Option<String>,
    pub(crate) name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RocketChatMention {
    #[serde(rename = "_id")]
    pub(crate) id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RocketChatApiAttachment {
    pub(crate) text: Option<String>,
    pub(crate) title: Option<String>,
    #[serde(rename = "title_link")]
    pub(crate) title_link: Option<String>,
    #[serde(rename = "message_link")]
    pub(crate) message_link: Option<String>,
    #[serde(rename = "author_name")]
    pub(crate) author_name: Option<String>,
    #[serde(rename = "image_url")]
    pub(crate) image_url: Option<String>,
    #[serde(rename = "audio_url")]
    pub(crate) audio_url: Option<String>,
    #[serde(rename = "video_url")]
    pub(crate) video_url: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RocketChatFileInfo {
    pub(crate) name: String,
    #[serde(rename = "type")]
    pub(crate) content_type: Option<String>,
    pub(crate) url: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RocketChatRoomType {
    Channel,
    PrivateGroup,
    DirectMessage,
}

#[derive(Debug, Clone)]
pub(crate) struct RocketChatResolvedRoom {
    pub(crate) id: String,
    pub(crate) room_type: RocketChatRoomType,
    pub(crate) name: Option<String>,
    pub(crate) friendly_name: Option<String>,
    pub(crate) usernames: Vec<String>,
    pub(crate) latest_message: Option<RocketChatMessage>,
    pub(crate) latest_message_id: Option<String>,
    pub(crate) latest_message_ts: Option<String>,
}

pub(crate) fn normalize_identity_label(raw: &str) -> String {
    raw.trim().trim_start_matches('@').to_ascii_lowercase()
}
