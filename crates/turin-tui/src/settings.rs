use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const DEFAULT_TRANSCRIPT_MEMORY_BUDGET_BYTES: usize = 1_048_576;
const DEFAULT_USER_LABEL: &str = "You";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatSidebarPane {
    Sessions,
    Agents,
    Channels,
    Events,
    None,
}

impl ChatSidebarPane {
    pub const ALL: [Self; 5] = [
        Self::Sessions,
        Self::Agents,
        Self::Channels,
        Self::Events,
        Self::None,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Self::Sessions => "Sessions",
            Self::Agents => "Agents",
            Self::Channels => "Channels",
            Self::Events => "Events",
            Self::None => "Hidden",
        }
    }

    pub fn next(self) -> Self {
        let idx = Self::ALL
            .iter()
            .position(|candidate| *candidate == self)
            .expect("sidebar pane exists");
        Self::ALL[(idx + 1) % Self::ALL.len()]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatInspectorPane {
    Thinking,
    Tools,
    Events,
    SessionMeta,
    None,
}

impl ChatInspectorPane {
    pub const ALL: [Self; 5] = [
        Self::Thinking,
        Self::Tools,
        Self::Events,
        Self::SessionMeta,
        Self::None,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Self::Thinking => "Thinking",
            Self::Tools => "Tools",
            Self::Events => "Events",
            Self::SessionMeta => "Session",
            Self::None => "Hidden",
        }
    }

    pub fn next(self) -> Self {
        let idx = Self::ALL
            .iter()
            .position(|candidate| *candidate == self)
            .expect("inspector pane exists");
        Self::ALL[(idx + 1) % Self::ALL.len()]
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TuiSettings {
    pub layout: TuiLayoutSettings,
    pub chat: TuiChatSettings,
}

#[derive(Debug, Clone)]
pub struct LoadedTuiSettings {
    pub settings: TuiSettings,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TuiLayoutSettings {
    pub left_pane: ChatSidebarPane,
    pub right_pane: ChatInspectorPane,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TuiChatSettings {
    pub transcript_memory_budget_bytes: usize,
    pub show_streaming_preview: bool,
    pub show_thinking: bool,
    pub follow_latest: bool,
    pub user_label: String,
}

impl Default for TuiLayoutSettings {
    fn default() -> Self {
        Self {
            left_pane: ChatSidebarPane::Sessions,
            right_pane: ChatInspectorPane::Thinking,
        }
    }
}

impl Default for TuiChatSettings {
    fn default() -> Self {
        Self {
            transcript_memory_budget_bytes: DEFAULT_TRANSCRIPT_MEMORY_BUDGET_BYTES,
            show_streaming_preview: true,
            show_thinking: true,
            follow_latest: true,
            user_label: DEFAULT_USER_LABEL.to_string(),
        }
    }
}

pub fn resolve_settings_path(explicit: Option<&Path>) -> Result<PathBuf> {
    match explicit {
        Some(path) => Ok(path.to_path_buf()),
        None => Ok(std::env::current_dir()
            .context("Failed to resolve current directory for turin-tui settings")?
            .join("turin-tui.toml")),
    }
}

pub fn load_settings(explicit: Option<&Path>) -> Result<LoadedTuiSettings> {
    let path = resolve_settings_path(explicit)?;
    if !path.exists() {
        return Ok(LoadedTuiSettings {
            settings: TuiSettings::default(),
            path,
        });
    }

    let raw = fs::read_to_string(&path)
        .with_context(|| format!("Failed to read TUI settings from '{}'", path.display()))?;
    let settings: TuiSettings = toml::from_str(&raw)
        .with_context(|| format!("Failed to parse TUI settings from '{}'", path.display()))?;
    Ok(LoadedTuiSettings { settings, path })
}

pub fn save_settings(path: &Path, settings: &TuiSettings) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "Failed to create TUI settings directory '{}'",
                parent.display()
            )
        })?;
    }
    let body = toml::to_string_pretty(settings).context("Failed to encode TUI settings as TOML")?;
    fs::write(path, body)
        .with_context(|| format!("Failed to write TUI settings to '{}'", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_are_chat_first() {
        let settings = TuiSettings::default();
        assert_eq!(settings.layout.left_pane, ChatSidebarPane::Sessions);
        assert_eq!(settings.layout.right_pane, ChatInspectorPane::Thinking);
        assert!(settings.chat.show_streaming_preview);
        assert!(settings.chat.show_thinking);
        assert!(settings.chat.follow_latest);
        assert_eq!(settings.chat.user_label, DEFAULT_USER_LABEL);
        assert_eq!(
            settings.chat.transcript_memory_budget_bytes,
            DEFAULT_TRANSCRIPT_MEMORY_BUDGET_BYTES
        );
    }
}
