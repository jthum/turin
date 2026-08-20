use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use turin_channel_core::{ChannelConversationKey, ChannelKind, ChannelUser, string_enum_setting};

use crate::state_io;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairingMode {
    Off,
    Pending,
    Auto,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelAccessPolicy {
    pub pairing_mode: PairingMode,
    pub pairing_users: HashSet<String>,
    pub allowed_users: HashSet<String>,
    pub banned_users: HashSet<String>,
}

impl Default for ChannelAccessPolicy {
    fn default() -> Self {
        Self {
            pairing_mode: PairingMode::Off,
            pairing_users: HashSet::new(),
            allowed_users: HashSet::new(),
            banned_users: HashSet::new(),
        }
    }
}

impl ChannelAccessPolicy {
    pub fn from_settings(settings: &Value) -> Result<Self> {
        let map = settings
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("Channel settings must be a JSON object"))?;
        let pairing_mode = parse_pairing_mode(map.get("pairing_mode"))?;
        Ok(Self {
            pairing_mode,
            pairing_users: parse_string_set(map.get("pairing_users"), "pairing_users")?,
            allowed_users: parse_string_set(map.get("allowed_users"), "allowed_users")?,
            banned_users: parse_string_set(map.get("banned_users"), "banned_users")?,
        })
    }

    pub fn validate_settings(settings: &Value) -> Result<()> {
        Self::from_settings(settings).map(|_| ())
    }

    pub fn requires_unconfigured_inbound(&self) -> bool {
        !matches!(self.pairing_mode, PairingMode::Off)
    }

    pub(crate) fn is_banned(
        &self,
        user: &ChannelUser,
        matches_selector: impl FnMut(&str, &ChannelUser) -> bool,
    ) -> bool {
        !self.banned_users.is_empty() && matches_any(&self.banned_users, user, matches_selector)
    }

    pub(crate) fn allows_pairing(
        &self,
        user: &ChannelUser,
        matches_selector: impl FnMut(&str, &ChannelUser) -> bool,
    ) -> bool {
        self.pairing_users.is_empty() || matches_any(&self.pairing_users, user, matches_selector)
    }

    pub(crate) fn allows_interaction(
        &self,
        user: &ChannelUser,
        mut matches_selector: impl FnMut(&str, &ChannelUser) -> bool,
    ) -> bool {
        if self.is_banned(user, &mut matches_selector) {
            return false;
        }
        self.allowed_users.is_empty() || matches_any(&self.allowed_users, user, matches_selector)
    }
}

fn matches_any(
    selectors: &HashSet<String>,
    user: &ChannelUser,
    mut matches_selector: impl FnMut(&str, &ChannelUser) -> bool,
) -> bool {
    selectors
        .iter()
        .any(|selector| matches_selector(selector, user))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelRoomRef {
    pub channel: ChannelKind,
    pub workspace_id: String,
    pub room_id: Option<String>,
    pub thread_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovedRoomView {
    pub room: ChannelRoomRef,
    pub approved_at_unix_seconds: u64,
    pub approved_by_user_id: Option<String>,
    pub approved_by_username: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingRoomView {
    pub room: ChannelRoomRef,
    pub first_seen_unix_seconds: u64,
    pub last_seen_unix_seconds: u64,
    pub sample_user_id: Option<String>,
    pub sample_username: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelAccessSnapshot {
    pub approved_rooms: Vec<ApprovedRoomView>,
    pub pending_rooms: Vec<PendingRoomView>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct AccessStateFile {
    #[serde(default)]
    pub(crate) approved_rooms: HashMap<String, ApprovedRoom>,
    #[serde(default)]
    pub(crate) pending_rooms: HashMap<String, PendingRoom>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ApprovedRoom {
    pub(crate) room: ChannelRoomKey,
    pub(crate) approved_at_unix_seconds: u64,
    pub(crate) approved_by_user_id: Option<String>,
    pub(crate) approved_by_username: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PendingRoom {
    pub(crate) room: ChannelRoomKey,
    pub(crate) first_seen_unix_seconds: u64,
    pub(crate) last_seen_unix_seconds: u64,
    pub(crate) sample_user_id: Option<String>,
    pub(crate) sample_username: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct ChannelRoomKey {
    pub(crate) channel: ChannelKind,
    pub(crate) workspace_id: String,
    pub(crate) room_id: Option<String>,
    pub(crate) thread_id: String,
}

impl From<&ChannelConversationKey> for ChannelRoomKey {
    fn from(value: &ChannelConversationKey) -> Self {
        Self {
            channel: value.channel.clone(),
            workspace_id: value.workspace_id.clone(),
            room_id: value.room_id.clone(),
            thread_id: value.thread_id.clone(),
        }
    }
}

impl From<&ChannelRoomKey> for ChannelRoomRef {
    fn from(value: &ChannelRoomKey) -> Self {
        Self {
            channel: value.channel.clone(),
            workspace_id: value.workspace_id.clone(),
            room_id: value.room_id.clone(),
            thread_id: value.thread_id.clone(),
        }
    }
}

impl From<&ChannelRoomRef> for ChannelRoomKey {
    fn from(value: &ChannelRoomRef) -> Self {
        Self {
            channel: value.channel.clone(),
            workspace_id: value.workspace_id.clone(),
            room_id: value.room_id.clone(),
            thread_id: value.thread_id.clone(),
        }
    }
}

#[derive(Clone)]
pub struct FileAccessStateStore {
    path: PathBuf,
}

impl FileAccessStateStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub(crate) async fn load(&self) -> Result<AccessStateFile> {
        state_io::read_json(&self.path).await
    }

    #[cfg(test)]
    pub(crate) async fn save(&self, state: &AccessStateFile) -> Result<()> {
        state_io::write_json(&self.path, state).await
    }

    pub(crate) async fn update<R>(
        &self,
        update: impl FnOnce(&mut AccessStateFile) -> Result<R>,
    ) -> Result<R> {
        state_io::update_json(&self.path, update).await
    }

    pub(crate) async fn update_if<R>(
        &self,
        update: impl FnOnce(&mut AccessStateFile) -> Result<(R, bool)>,
    ) -> Result<R> {
        state_io::update_json_if(&self.path, update).await
    }

    pub async fn snapshot(&self) -> Result<ChannelAccessSnapshot> {
        let state = self.load().await?;
        Ok(channel_access_snapshot(&state))
    }

    pub async fn approve(
        &self,
        room: &ChannelRoomRef,
        approved_by_user_id: Option<String>,
        approved_by_username: Option<String>,
    ) -> Result<ChannelAccessSnapshot> {
        self.update(|state| {
            let room_key = ChannelRoomKey::from(room);
            let serialized_room = serialize_room_key(&room_key)?;
            state.pending_rooms.remove(&serialized_room);
            state.approved_rooms.insert(
                serialized_room,
                ApprovedRoom {
                    room: room_key,
                    approved_at_unix_seconds: unix_seconds(SystemTime::now()),
                    approved_by_user_id,
                    approved_by_username,
                },
            );
            Ok(channel_access_snapshot(state))
        })
        .await
    }

    pub async fn reject_pending(&self, room: &ChannelRoomRef) -> Result<ChannelAccessSnapshot> {
        self.update(|state| {
            state
                .pending_rooms
                .remove(&serialize_room_key(&ChannelRoomKey::from(room))?);
            Ok(channel_access_snapshot(state))
        })
        .await
    }

    pub async fn revoke(&self, room: &ChannelRoomRef) -> Result<ChannelAccessSnapshot> {
        self.update(|state| {
            state
                .approved_rooms
                .remove(&serialize_room_key(&ChannelRoomKey::from(room))?);
            Ok(channel_access_snapshot(state))
        })
        .await
    }
}

pub(crate) fn serialize_room_key(key: &ChannelRoomKey) -> Result<String> {
    serde_json::to_string(key).context("failed to serialize channel room key")
}

fn channel_access_snapshot(state: &AccessStateFile) -> ChannelAccessSnapshot {
    let mut approved_rooms: Vec<_> = state
        .approved_rooms
        .values()
        .map(|room| ApprovedRoomView {
            room: ChannelRoomRef::from(&room.room),
            approved_at_unix_seconds: room.approved_at_unix_seconds,
            approved_by_user_id: room.approved_by_user_id.clone(),
            approved_by_username: room.approved_by_username.clone(),
        })
        .collect();
    approved_rooms.sort_by(|left, right| {
        left.room
            .workspace_id
            .cmp(&right.room.workspace_id)
            .then_with(|| left.room.room_id.cmp(&right.room.room_id))
            .then_with(|| left.room.thread_id.cmp(&right.room.thread_id))
    });

    let mut pending_rooms: Vec<_> = state
        .pending_rooms
        .values()
        .map(|room| PendingRoomView {
            room: ChannelRoomRef::from(&room.room),
            first_seen_unix_seconds: room.first_seen_unix_seconds,
            last_seen_unix_seconds: room.last_seen_unix_seconds,
            sample_user_id: room.sample_user_id.clone(),
            sample_username: room.sample_username.clone(),
        })
        .collect();
    pending_rooms.sort_by(|left, right| {
        left.room
            .workspace_id
            .cmp(&right.room.workspace_id)
            .then_with(|| left.room.room_id.cmp(&right.room.room_id))
            .then_with(|| left.room.thread_id.cmp(&right.room.thread_id))
    });

    ChannelAccessSnapshot {
        approved_rooms,
        pending_rooms,
    }
}

pub(crate) fn unix_seconds(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn parse_pairing_mode(value: Option<&Value>) -> Result<PairingMode> {
    string_enum_setting(
        value,
        PairingMode::Off,
        |raw| match raw.trim().to_ascii_lowercase().as_str() {
            "off" => Some(PairingMode::Off),
            "pending" => Some(PairingMode::Pending),
            "auto" => Some(PairingMode::Auto),
            _ => None,
        },
        "channel setting 'pairing_mode' must be a string",
        "channel setting 'pairing_mode' must be one of: off, pending, auto",
    )
    .map_err(Into::into)
}

fn parse_string_set(value: Option<&Value>, key: &str) -> Result<HashSet<String>> {
    let mut out = HashSet::new();
    let Some(value) = value else {
        return Ok(out);
    };

    match value {
        Value::Array(values) => {
            for item in values {
                let text = item.as_str().ok_or_else(|| {
                    anyhow::anyhow!("channel setting '{}' must be an array of strings", key)
                })?;
                let normalized = normalize_string_item(text).ok_or_else(|| {
                    anyhow::anyhow!(
                        "channel setting '{}' must not contain empty string values",
                        key
                    )
                })?;
                out.insert(normalized);
            }
        }
        Value::String(text) => {
            for item in text.split(',') {
                let normalized = normalize_string_item(item).ok_or_else(|| {
                    anyhow::anyhow!(
                        "channel setting '{}' must not contain empty string values",
                        key
                    )
                })?;
                out.insert(normalized);
            }
        }
        _ => {
            anyhow::bail!(
                "channel setting '{}' must be a string or array of strings",
                key
            );
        }
    }

    Ok(out)
}

fn normalize_string_item(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}
