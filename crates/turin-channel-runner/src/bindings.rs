use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use turin_channel_core::{ChannelConversationKey, ConversationBinding};

#[derive(Debug, Default, Serialize, Deserialize)]
struct BindingFile {
    bindings: HashMap<String, ConversationBinding>,
}

#[derive(Clone)]
pub struct FileBindingStore {
    path: PathBuf,
}

impl FileBindingStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub async fn load(&self) -> Result<HashMap<String, ConversationBinding>> {
        if !self.path.exists() {
            return Ok(HashMap::new());
        }
        let raw = tokio::fs::read_to_string(&self.path)
            .await
            .with_context(|| format!("Failed to read '{}'", self.path.display()))?;
        let file: BindingFile = serde_json::from_str(&raw)
            .with_context(|| format!("Failed to parse '{}'", self.path.display()))?;
        Ok(file.bindings)
    }

    pub async fn save(&self, bindings: &HashMap<String, ConversationBinding>) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let tmp = self.path.with_extension("json.tmp");
        let body = serde_json::to_string_pretty(&BindingFile {
            bindings: bindings.clone(),
        })?;
        tokio::fs::write(&tmp, body).await?;
        tokio::fs::rename(&tmp, &self.path).await?;
        Ok(())
    }
}

pub(crate) fn serialize_binding_key(key: &ChannelConversationKey) -> Result<String> {
    serde_json::to_string(key).context("failed to serialize channel conversation key")
}
