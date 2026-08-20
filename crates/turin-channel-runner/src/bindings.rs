use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};
use turin_channel_core::{ChannelConversationKey, ConversationBinding};

use crate::state_io;

#[derive(Debug, Default, Serialize, Deserialize)]
struct BindingFile {
    bindings: HashMap<String, ConversationBinding>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChannelBindingView {
    pub conversation: ChannelConversationKey,
    pub binding: ConversationBinding,
}

#[derive(Clone)]
pub struct FileBindingStore {
    path: PathBuf,
}

impl FileBindingStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    async fn load(&self) -> Result<HashMap<String, ConversationBinding>> {
        let file: BindingFile = state_io::read_json(&self.path).await?;
        Ok(file.bindings)
    }

    pub async fn get(&self, key: &ChannelConversationKey) -> Result<Option<ConversationBinding>> {
        Ok(self
            .load()
            .await?
            .get(&serialize_binding_key(key)?)
            .cloned())
    }

    pub async fn upsert(
        &self,
        key: &ChannelConversationKey,
        binding: ConversationBinding,
    ) -> Result<()> {
        let key = serialize_binding_key(key)?;
        state_io::update_json::<BindingFile, _>(&self.path, |file| {
            file.bindings.insert(key, binding);
            Ok(())
        })
        .await
    }

    pub async fn clear(&self, key: &ChannelConversationKey) -> Result<bool> {
        let key = serialize_binding_key(key)?;
        state_io::update_json::<BindingFile, _>(&self.path, |file| {
            Ok(file.bindings.remove(&key).is_some())
        })
        .await
    }

    pub async fn prune_expired(&self, now: SystemTime, ttl: Duration) -> Result<usize> {
        state_io::update_json::<BindingFile, _>(&self.path, |file| {
            let previous = file.bindings.len();
            file.bindings
                .retain(|_, binding| !binding.is_expired(now, ttl));
            Ok(previous - file.bindings.len())
        })
        .await
    }

    pub async fn snapshot(&self) -> Result<Vec<ChannelBindingView>> {
        let mut bindings = Vec::new();
        for (key, binding) in self.load().await? {
            let conversation = serde_json::from_str(&key)
                .with_context(|| format!("Failed to parse conversation binding key '{key}'"))?;
            bindings.push(ChannelBindingView {
                conversation,
                binding,
            });
        }
        bindings.sort_by(|left, right| {
            left.conversation
                .workspace_id
                .cmp(&right.conversation.workspace_id)
                .then_with(|| left.conversation.room_id.cmp(&right.conversation.room_id))
                .then_with(|| {
                    left.conversation
                        .thread_id
                        .cmp(&right.conversation.thread_id)
                })
                .then_with(|| left.conversation.user_id.cmp(&right.conversation.user_id))
        });
        Ok(bindings)
    }
}

pub(crate) fn serialize_binding_key(key: &ChannelConversationKey) -> Result<String> {
    serde_json::to_string(key).context("failed to serialize channel conversation key")
}
