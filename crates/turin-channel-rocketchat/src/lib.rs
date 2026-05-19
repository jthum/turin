use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use reqwest::Client;
use std::collections::{HashMap, HashSet, VecDeque};
#[cfg(test)]
use std::time::Duration;
use std::time::Instant;
use tokio::sync::watch;
use tokio::time::sleep;
use tracing::warn;
use turin_channel_core::{
    ChannelCapabilities, ChannelConversationKey, ChannelKind, ChannelUser, InboundEvent,
    OutboundMessage,
};
#[cfg(test)]
use turin_channel_core::{ChannelSessionScope, DEFAULT_MAX_INBOUND_TEXT_CHARS, MessageBlock};
use turin_channel_runner::{ChannelDriver, ChannelProgressUpdate, ChannelStreamMode};

mod api;
mod inbound;
mod manifest;
mod realtime;
mod render;
mod settings;
#[cfg(test)]
use api::{RocketChatApiAttachment, RocketChatRoomInfo};
use api::{
    RocketChatMessage, RocketChatMessageUser, RocketChatResolvedRoom, RocketChatRoomType,
    RocketChatSendMessageResponse, absolute_url, api_url, build_http_client, fetch_bot_identity,
    fetch_room_messages, fetch_rooms, normalize_identity_label,
};
#[cfg(test)]
use inbound::build_rocketchat_message_link;
pub use manifest::{adapter_manifest, poll_auth_flow, start_auth_flow};
use realtime::RocketChatWsStream;
#[cfg(test)]
use realtime::{RocketChatDdpFrame, login_result_error};
#[cfg(test)]
use render::RocketChatReplyTarget;
#[cfg(test)]
use render::render_text_blocks_for_test as render_text_blocks;
use render::{
    build_rocketchat_send_payload, prepend_channel_reply_quote, render_rocketchat_message,
    reply_excerpt, resolve_reply_target, split_for_rocketchat_content,
};
pub use settings::{RocketChatChannelDriverConfig, validate_settings};
#[cfg(test)]
pub(crate) use settings::{default_websocket_url, parse_settings};

const DEFAULT_BASE_URL: &str = "http://localhost:3000";
const DEFAULT_TRANSPORT_MODE: &str = "realtime";
const DEFAULT_STREAM_MODE: &str = "typing";
const DEFAULT_POLL_INTERVAL_MS: u64 = 1_000;
const DEFAULT_MAX_MESSAGES_PER_POLL: u16 = 50;
const MAX_MESSAGES_PER_POLL: u16 = 100;
const SEEN_MESSAGE_IDS_LIMIT: usize = 1_024;
const RECENT_SENT_MESSAGE_IDS_LIMIT: usize = 256;
const DEFAULT_REALTIME_RECONNECT_DELAY_MS: u64 = 2_000;
const ROCKETCHAT_TYPING_STATUS_INTERVAL_SECONDS: u64 = 4;
const ROCKETCHAT_HTTP_TIMEOUT_SECONDS: u64 = 30;
const ROCKETCHAT_HTTP_CONNECT_TIMEOUT_SECONDS: u64 = 10;
const ROCKETCHAT_REALTIME_CONNECT_TIMEOUT_SECONDS: u64 = 15;
const ROCKETCHAT_REALTIME_HANDSHAKE_TIMEOUT_SECONDS: u64 = 15;
const ROCKETCHAT_REALTIME_KEEPALIVE_SECONDS: u64 = 15;
const ROCKETCHAT_REALTIME_STALE_SECONDS: u64 = 45;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocketChatRespondMode {
    All,
    Mentions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocketChatTransportMode {
    Realtime,
    Polling,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocketChatReplyMode {
    Thread,
    Channel,
    ThreadAndChannel,
}

#[derive(Debug, Clone)]
struct RocketChatRoomState {
    room: RocketChatResolvedRoom,
    cursor_ts: Option<String>,
}
pub struct RocketChatChannelDriver {
    channel_id: String,
    client: Client,
    config: RocketChatChannelDriverConfig,
    shutdown_rx: watch::Receiver<bool>,
    bot_username: Option<String>,
    bot_display_name: Option<String>,
    rooms: HashMap<String, RocketChatRoomState>,
    ws_stream: Option<RocketChatWsStream>,
    realtime_subscribed_room_ids: HashSet<String>,
    active_thread_keys: HashSet<String>,
    backlog: VecDeque<InboundEvent>,
    seen_message_ids: HashSet<String>,
    seen_message_order: VecDeque<String>,
    recent_sent_message_ids: HashSet<String>,
    recent_sent_message_order: VecDeque<String>,
    rooms_updated_since: Option<String>,
    last_room_refresh: Option<Instant>,
    last_typing_at: HashMap<String, Instant>,
    last_realtime_activity_at: Option<Instant>,
    last_realtime_keepalive_at: Option<Instant>,
    next_realtime_request_id: u64,
}

impl RocketChatChannelDriver {
    pub async fn from_settings(
        channel_id: impl Into<String>,
        settings: &serde_json::Value,
        allow_unconfigured_rooms: bool,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Self> {
        let config =
            RocketChatChannelDriverConfig::from_settings(settings, allow_unconfigured_rooms)?;
        let client = build_http_client()?;

        let mut driver = Self {
            channel_id: channel_id.into(),
            client,
            config,
            shutdown_rx,
            bot_username: None,
            bot_display_name: None,
            rooms: HashMap::new(),
            ws_stream: None,
            realtime_subscribed_room_ids: HashSet::new(),
            active_thread_keys: HashSet::new(),
            backlog: VecDeque::new(),
            seen_message_ids: HashSet::new(),
            seen_message_order: VecDeque::new(),
            recent_sent_message_ids: HashSet::new(),
            recent_sent_message_order: VecDeque::new(),
            rooms_updated_since: None,
            last_room_refresh: None,
            last_typing_at: HashMap::new(),
            last_realtime_activity_at: None,
            last_realtime_keepalive_at: None,
            next_realtime_request_id: 1,
        };

        if let Err(err) = driver.load_bot_identity().await {
            warn!(
                channel_id = %driver.channel_id,
                error = ?err,
                "Rocket.Chat bot username lookup failed; typing indicators disabled until it succeeds"
            );
        }
        driver.refresh_rooms(true).await?;
        if !driver.config.start_from_latest
            && matches!(
                driver.config.transport_mode,
                RocketChatTransportMode::Realtime
            )
        {
            driver.poll_messages().await?;
        }

        Ok(driver)
    }

    async fn refresh_rooms(&mut self, initial: bool) -> Result<()> {
        let update = fetch_rooms(
            &self.client,
            &self.config,
            self.rooms_updated_since.as_deref(),
        )
        .await?;

        for room_id in update.remove_room_ids {
            self.rooms.remove(&room_id);
            self.realtime_subscribed_room_ids.remove(&room_id);
        }

        for room in update.rooms {
            if !self.room_matches_filters(&room) {
                continue;
            }
            self.upsert_room(room, initial).await?;
        }

        if let Some(updated_since) = update.next_updated_since {
            let should_replace = self
                .rooms_updated_since
                .as_deref()
                .is_none_or(|current| current < updated_since.as_str());
            if should_replace {
                self.rooms_updated_since = Some(updated_since);
            }
        }
        self.last_room_refresh = Some(Instant::now());

        if initial && !self.config.accept_all_rooms && self.rooms.is_empty() {
            anyhow::bail!(
                "[rocketchat_room_not_found] Rocket.Chat could not find a room matching the configured room filter"
            );
        }

        Ok(())
    }

    fn room_matches_filters(&self, room: &RocketChatResolvedRoom) -> bool {
        if let Some(expected_room_id) = self.config.room_id.as_deref()
            && room.id != expected_room_id
        {
            return false;
        }
        if let Some(expected_room_name) = self.config.room_name.as_deref() {
            let matches = room
                .name
                .as_deref()
                .is_some_and(|value| value.eq_ignore_ascii_case(expected_room_name))
                || room
                    .friendly_name
                    .as_deref()
                    .is_some_and(|value| value.eq_ignore_ascii_case(expected_room_name));
            if !matches {
                return false;
            }
        }
        true
    }

    async fn upsert_room(&mut self, room: RocketChatResolvedRoom, initial: bool) -> Result<()> {
        let room_id = room.id.clone();
        if let Some(existing) = self.rooms.get_mut(&room_id) {
            existing.room = room;
            return Ok(());
        }

        let mut state = RocketChatRoomState {
            room,
            cursor_ts: None,
        };
        if initial && self.config.start_from_latest {
            if let Some(message_id) = state.room.latest_message_id.clone() {
                self.remember_message_id(message_id);
            }
            state.cursor_ts = state.room.latest_message_ts.clone();
        }
        self.rooms.insert(room_id.clone(), state);

        if !initial {
            if self.config.start_from_latest {
                self.seed_new_room_from_latest(&room_id)?;
            } else {
                self.poll_room_messages(&room_id).await?;
            }
        }

        Ok(())
    }

    fn seed_new_room_from_latest(&mut self, room_id: &str) -> Result<()> {
        let Some(state) = self.rooms.get(room_id).cloned() else {
            return Ok(());
        };

        if let Some(cursor_ts) = state.room.latest_message_ts.clone() {
            self.update_room_cursor(room_id, cursor_ts);
        }
        if let Some(message_id) = state.room.latest_message_id.clone() {
            self.remember_message_id(message_id);
        }
        if let Some(message) = state.room.latest_message.clone() {
            if self.seen_message_ids.contains(&message.id) {
                return Ok(());
            }
            self.remember_message_id(message.id.clone());
            self.update_room_cursor(room_id, message.ts.clone());
            if let Some(event) = self.message_to_event(&state.room, message)? {
                self.backlog.push_back(event);
            }
        }

        Ok(())
    }

    fn update_room_cursor(&mut self, room_id: &str, cursor_ts: String) {
        if let Some(state) = self.rooms.get_mut(room_id) {
            state.cursor_ts = Some(cursor_ts);
        }
    }

    async fn poll_messages(&mut self) -> Result<()> {
        self.refresh_rooms(false).await?;
        self.poll_known_rooms().await
    }

    async fn poll_known_rooms(&mut self) -> Result<()> {
        let mut room_ids: Vec<String> = self.rooms.keys().cloned().collect();
        room_ids.sort();
        for room_id in room_ids {
            self.poll_room_messages(&room_id).await?;
        }
        Ok(())
    }

    async fn poll_room_messages(&mut self, room_id: &str) -> Result<()> {
        let Some(state) = self.rooms.get(room_id).cloned() else {
            return Ok(());
        };
        let messages = fetch_room_messages(
            &self.client,
            &self.config,
            &state.room,
            state.cursor_ts.as_deref(),
        )
        .await?;

        for message in messages {
            self.update_room_cursor(room_id, message.ts.clone());
            if self.seen_message_ids.contains(&message.id) {
                continue;
            }
            self.remember_message_id(message.id.clone());

            let Some(event) = self.message_to_event(&state.room, message)? else {
                continue;
            };
            self.backlog.push_back(event);
        }

        Ok(())
    }

    fn remember_message_id(&mut self, message_id: String) {
        if self.seen_message_ids.insert(message_id.clone()) {
            self.seen_message_order.push_back(message_id);
            while self.seen_message_order.len() > SEEN_MESSAGE_IDS_LIMIT {
                if let Some(oldest) = self.seen_message_order.pop_front() {
                    self.seen_message_ids.remove(&oldest);
                }
            }
        }
    }

    fn next_request_id(&mut self) -> String {
        let id = format!("turin-{}", self.next_realtime_request_id);
        self.next_realtime_request_id += 1;
        id
    }

    fn is_bot_identity_label(&self, raw: &str) -> bool {
        let normalized = normalize_identity_label(raw);
        if normalized.is_empty() {
            return false;
        }
        self.bot_username
            .as_deref()
            .is_some_and(|value| normalize_identity_label(value) == normalized)
            || self
                .bot_display_name
                .as_deref()
                .is_some_and(|value| normalize_identity_label(value) == normalized)
    }

    async fn load_bot_identity(&mut self) -> Result<()> {
        let identity = fetch_bot_identity(&self.client, &self.config).await?;
        self.bot_username = Some(identity.username);
        self.bot_display_name = identity.display_name;
        Ok(())
    }

    fn remember_sent_message_id(&mut self, message_id: String) {
        if self.recent_sent_message_ids.insert(message_id.clone()) {
            self.recent_sent_message_order.push_back(message_id);
            while self.recent_sent_message_order.len() > RECENT_SENT_MESSAGE_IDS_LIMIT {
                if let Some(oldest) = self.recent_sent_message_order.pop_front() {
                    self.recent_sent_message_ids.remove(&oldest);
                }
            }
        }
    }
}

#[async_trait]
impl ChannelDriver for RocketChatChannelDriver {
    fn kind(&self) -> ChannelKind {
        ChannelKind::new("rocketchat")
    }

    fn user_matches_selector(&self, selector: &str, user: &ChannelUser) -> bool {
        let selector = selector.trim().trim_start_matches('@');
        if selector.is_empty() {
            return false;
        }
        user.id == selector
            || user
                .username
                .as_ref()
                .is_some_and(|username| username.eq_ignore_ascii_case(selector))
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            rich_formatting: false,
            threads: true,
            attachments: false,
            ephemeral_messages: false,
        }
    }

    async fn next_event(&mut self) -> Result<Option<InboundEvent>> {
        loop {
            if *self.shutdown_rx.borrow() {
                return Ok(None);
            }

            if let Some(event) = self.backlog.pop_front() {
                return Ok(Some(event));
            }

            if matches!(
                self.config.transport_mode,
                RocketChatTransportMode::Realtime
            ) {
                return self.next_realtime_event().await;
            }

            if let Err(err) = self.poll_messages().await {
                warn!(
                    channel_id = %self.channel_id,
                    room_count = self.rooms.len(),
                    error = ?err,
                    "Rocket.Chat polling failed"
                );
                if let Err(reset_err) = self.reset_transport_state() {
                    warn!(
                        channel_id = %self.channel_id,
                        error = ?reset_err,
                        "Rocket.Chat transport reset failed"
                    );
                }
            }

            if let Some(event) = self.backlog.pop_front() {
                return Ok(Some(event));
            }

            tokio::select! {
                changed = self.shutdown_rx.changed() => {
                    if changed.is_ok() && *self.shutdown_rx.borrow() {
                        return Ok(None);
                    }
                }
                _ = sleep(self.config.poll_interval) => {}
            }
        }
    }

    async fn send(
        &mut self,
        conversation: &ChannelConversationKey,
        message: OutboundMessage,
    ) -> Result<()> {
        let room_id = conversation
            .room_id
            .as_deref()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                anyhow!("[rocketchat_send_missing_room] outbound conversation is missing room_id")
            })?;

        let reply_target =
            resolve_reply_target(room_id, conversation, &message, self.config.reply_mode);
        let chunks = split_for_rocketchat_content(render_rocketchat_message(
            &message,
            self.config.persist_thinking,
        ));

        for (index, chunk) in chunks.into_iter().enumerate() {
            let rendered_chunk =
                if index == 0 && matches!(self.config.reply_mode, RocketChatReplyMode::Channel) {
                    prepend_channel_reply_quote(&chunk, &message)
                } else {
                    chunk
                };
            let payload =
                build_rocketchat_send_payload(room_id, &rendered_chunk, reply_target, &[]);
            if let Some(thread_id) = reply_target.thread_id
                && index == 0
            {
                self.active_thread_keys
                    .insert(active_thread_key(room_id, thread_id));
            }
            let response = self
                .client
                .post(api_url(&self.config.base_url, "chat.sendMessage"))
                .header("X-Auth-Token", &self.config.token)
                .header("X-User-Id", &self.config.user_id)
                .json(&payload)
                .send()
                .await
                .context("Failed to send Rocket.Chat message")?;
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            if !status.is_success() {
                anyhow::bail!(
                    "[rocketchat_send_failed] Rocket.Chat chat.sendMessage failed with status {}: {}",
                    status,
                    body
                );
            }
            if let Ok(parsed) = serde_json::from_str::<RocketChatSendMessageResponse>(&body)
                && let Some(sent_message) = parsed.message
            {
                self.remember_sent_message_id(sent_message.id);
            }
        }

        Ok(())
    }

    fn enrich_outbound_for_event(
        &self,
        event: &InboundEvent,
        mut outbound: OutboundMessage,
    ) -> OutboundMessage {
        if !outbound
            .metadata
            .contains_key("rocketchat_reply_to_message_id")
            && let Some(message_id) = event.metadata.get("rocketchat_message_id")
        {
            outbound.metadata.insert(
                "rocketchat_reply_to_message_id".to_string(),
                message_id.clone(),
            );
        }
        if !outbound.metadata.contains_key("rocketchat_thread_id")
            && let Some(thread_id) = event.metadata.get("rocketchat_thread_id")
        {
            outbound
                .metadata
                .insert("rocketchat_thread_id".to_string(), thread_id.clone());
        }
        if !outbound.metadata.contains_key("rocketchat_reply_to_label") {
            outbound.metadata.insert(
                "rocketchat_reply_to_label".to_string(),
                serde_json::json!(event.user.prompt_label()),
            );
        }
        if !outbound
            .metadata
            .contains_key("rocketchat_reply_to_excerpt")
            && !event.text.trim().is_empty()
        {
            outbound.metadata.insert(
                "rocketchat_reply_to_excerpt".to_string(),
                serde_json::json!(reply_excerpt(&event.text)),
            );
        }
        if !outbound
            .metadata
            .contains_key("rocketchat_reply_to_message_ts")
            && let Some(message_ts) = event.metadata.get("rocketchat_message_ts")
        {
            outbound.metadata.insert(
                "rocketchat_reply_to_message_ts".to_string(),
                message_ts.clone(),
            );
        }
        if !outbound
            .metadata
            .contains_key("rocketchat_reply_to_message_link")
            && let Some(message_link) = event.metadata.get("rocketchat_message_link")
        {
            outbound.metadata.insert(
                "rocketchat_reply_to_message_link".to_string(),
                message_link.clone(),
            );
        }
        outbound
    }

    fn stream_mode(&self) -> ChannelStreamMode {
        self.config.stream_mode
    }

    fn persist_thinking(&self) -> bool {
        self.config.persist_thinking
    }

    async fn send_progress(
        &mut self,
        event: &InboundEvent,
        update: ChannelProgressUpdate,
    ) -> Result<()> {
        match update {
            ChannelProgressUpdate::Typing => self.send_typing_status(event).await,
            ChannelProgressUpdate::StreamingPreview { .. } => Ok(()),
        }
    }

    async fn shutdown(&mut self) -> Result<()> {
        self.ws_stream = None;
        self.realtime_subscribed_room_ids.clear();
        self.last_typing_at.clear();
        Ok(())
    }
}

fn progress_key(conversation: &ChannelConversationKey) -> Result<String> {
    serde_json::to_string(conversation)
        .with_context(|| "[rocketchat_progress_key_invalid] Failed to serialize conversation key")
}

fn active_thread_key(room_id: &str, thread_id: &str) -> String {
    format!("{room_id}:{thread_id}")
}
fn subscription_request_id(room_id: &str) -> String {
    format!("room:{room_id}")
}

fn subscription_room_id(request_id: &str) -> Option<&str> {
    request_id.strip_prefix("room:")
}

#[cfg(test)]
mod tests;
