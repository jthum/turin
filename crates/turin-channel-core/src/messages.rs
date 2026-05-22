use serde::{Deserialize, Deserializer, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct ChannelKind(String);

impl ChannelKind {
    pub fn parse(raw: &str) -> Result<Self, String> {
        let normalized = raw.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return Err("channel kind cannot be empty".to_string());
        }
        if !normalized.chars().all(|ch| {
            ch.is_ascii_lowercase() || ch.is_ascii_digit() || matches!(ch, '-' | '_' | '.')
        }) {
            return Err(format!(
                "channel kind '{}' must contain only lowercase letters, digits, '.', '-', or '_'",
                raw
            ));
        }
        Ok(Self(normalized))
    }

    pub fn new(raw: impl AsRef<str>) -> Self {
        Self::parse(raw.as_ref()).expect("invalid channel kind")
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ChannelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl TryFrom<String> for ChannelKind {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(&value)
    }
}

impl From<ChannelKind> for String {
    fn from(value: ChannelKind) -> Self {
        value.0
    }
}

impl<'de> Deserialize<'de> for ChannelKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Self::parse(&raw).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChannelConversationKey {
    pub channel: ChannelKind,
    pub workspace_id: String,
    pub room_id: Option<String>,
    pub thread_id: String,
    pub user_id: Option<String>,
}

impl ChannelConversationKey {
    pub fn deterministic_slot_id(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(serde_json::to_vec(self).expect("conversation key serializes"));
        let digest = hasher.finalize();
        format!("chan-{}", &hex::encode(digest)[..24])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ChannelSessionScope {
    #[default]
    User,
    Thread,
    Room,
}

impl ChannelSessionScope {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "user" => Some(Self::User),
            "thread" => Some(Self::Thread),
            "room" => Some(Self::Room),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Thread => "thread",
            Self::Room => "room",
        }
    }

    pub fn is_allowed_by(self, allowed: &[Self]) -> bool {
        allowed.contains(&self)
    }

    pub fn is_shared(self) -> bool {
        !matches!(self, Self::User)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelMessageRef {
    pub conversation: ChannelConversationKey,
    pub message_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelUser {
    pub id: String,
    pub display_name: Option<String>,
    pub username: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelAttachment {
    pub name: String,
    pub content_type: Option<String>,
    pub url: Option<String>,
    pub local_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageBlock {
    Text {
        text: String,
    },
    CodeBlock {
        language: Option<String>,
        code: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct OutboundMessage {
    #[serde(default)]
    pub blocks: Vec<MessageBlock>,
    #[serde(default)]
    pub attachments: Vec<ChannelAttachment>,
    #[serde(default)]
    pub embeds: Vec<serde_json::Value>,
    #[serde(default)]
    pub components: Vec<serde_json::Value>,
    #[serde(default)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

impl OutboundMessage {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            blocks: vec![MessageBlock::Text { text: text.into() }],
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InboundEvent {
    pub conversation: ChannelConversationKey,
    pub message: ChannelMessageRef,
    pub user: ChannelUser,
    #[serde(default)]
    pub session_scope: ChannelSessionScope,
    pub text: String,
    #[serde(default)]
    pub attachments: Vec<ChannelAttachment>,
    #[serde(default)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

impl InboundEvent {
    pub fn prompt_text(&self) -> String {
        if !self.session_scope.is_shared() {
            return self.text.clone();
        }

        format!("[Message from {}]\n{}", self.user.prompt_label(), self.text)
    }
}

pub fn bound_inbound_text(
    text: String,
    metadata: &mut serde_json::Map<String, serde_json::Value>,
    max_chars: usize,
) -> String {
    let original_chars = text.chars().count();
    if original_chars <= max_chars {
        return text;
    }

    metadata.insert(
        "turin_text_truncated".to_string(),
        serde_json::Value::Bool(true),
    );
    metadata.insert(
        "turin_original_text_chars".to_string(),
        serde_json::Value::Number(original_chars.into()),
    );
    metadata.insert(
        "turin_text_char_limit".to_string(),
        serde_json::Value::Number(max_chars.into()),
    );
    text.chars().take(max_chars).collect()
}

impl ChannelUser {
    pub fn prompt_label(&self) -> String {
        match (self.display_name.as_deref(), self.username.as_deref()) {
            (Some(display_name), Some(username))
                if !display_name.trim().is_empty()
                    && !username.trim().is_empty()
                    && !display_name.eq_ignore_ascii_case(username) =>
            {
                format!("{display_name} (@{username})")
            }
            (Some(display_name), _) if !display_name.trim().is_empty() => display_name.to_string(),
            (_, Some(username)) if !username.trim().is_empty() => format!("@{username}"),
            _ => self.id.clone(),
        }
    }
}

pub fn render_plain_text_blocks(blocks: &[MessageBlock]) -> String {
    let mut chunks = Vec::new();
    for block in blocks {
        match block {
            MessageBlock::Text { text } => {
                if !text.trim().is_empty() {
                    chunks.push(text.clone());
                }
            }
            MessageBlock::CodeBlock { language, code } => {
                let prefix = language.as_deref().unwrap_or_default();
                chunks.push(format!("```{}\n{}\n```", prefix, code));
            }
        }
    }
    chunks.join("\n\n")
}

pub fn split_text_lines_to_char_limit(content: &str, limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    let trimmed = content.trim();
    if trimmed.is_empty() || limit == 0 {
        return out;
    }

    let mut current = String::new();
    let mut current_chars = 0usize;
    for line in trimmed.lines() {
        let line_chars = line.chars().count();
        if line_chars > limit {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
                current_chars = 0;
            }
            split_long_line(line, limit, &mut out);
            continue;
        }

        let tentative_chars = if current.is_empty() {
            line_chars
        } else {
            current_chars + 1 + line_chars
        };
        if tentative_chars > limit {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            current.push_str(line);
            current_chars = line_chars;
        } else {
            if !current.is_empty() {
                current.push('\n');
            }
            current.push_str(line);
            current_chars = tentative_chars;
        }
    }

    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn split_long_line(line: &str, limit: usize, out: &mut Vec<String>) {
    let mut segment = String::new();
    let mut segment_chars = 0usize;
    for ch in line.chars() {
        segment.push(ch);
        segment_chars += 1;
        if segment_chars >= limit {
            out.push(std::mem::take(&mut segment));
            segment_chars = 0;
        }
    }
    if !segment.is_empty() {
        out.push(segment);
    }
}
