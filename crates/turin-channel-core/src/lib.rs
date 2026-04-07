use serde::{Deserialize, Deserializer, Serialize};
use std::time::{Duration, SystemTime};

pub const CHANNEL_ADAPTER_PROTOCOL_VERSION: u32 = 2;
pub const DEFAULT_MAX_INBOUND_TEXT_CHARS: usize = 16_000;

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ChannelCapabilities {
    pub rich_formatting: bool,
    pub threads: bool,
    pub attachments: bool,
    pub ephemeral_messages: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ChannelAdapterManifest {
    #[serde(default = "default_channel_protocol_version")]
    pub protocol_version: u32,
    pub kind: String,
    #[serde(default)]
    pub display_name: String,
    #[serde(default)]
    pub runtime: ChannelRuntimeManifest,
    #[serde(default)]
    pub setup: Option<ChannelSetupManifest>,
    #[serde(default)]
    pub install: Option<ChannelInstallManifest>,
}

impl ChannelAdapterManifest {
    pub fn enum_setting(&self, key: &str) -> Option<&ChannelEnumSetting> {
        self.runtime
            .enum_settings
            .iter()
            .find(|setting| setting.key == key)
    }

    pub fn display_name_or_kind(&self) -> &str {
        if self.display_name.trim().is_empty() {
            &self.kind
        } else {
            &self.display_name
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_adapter_manifest(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ChannelRuntimeManifest {
    #[serde(default)]
    pub session_scopes: Vec<String>,
    #[serde(default)]
    pub enum_settings: Vec<ChannelEnumSetting>,
    #[serde(default)]
    pub capabilities: ChannelRuntimeCapabilities,
    #[serde(default)]
    pub identity_selectors: ChannelIdentitySelectors,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelEnumSetting {
    pub key: String,
    #[serde(default)]
    pub options: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ChannelRuntimeCapabilities {
    #[serde(default)]
    pub dm: bool,
    #[serde(default)]
    pub groups: bool,
    #[serde(default)]
    pub threads: bool,
    #[serde(default)]
    pub attachments: bool,
    #[serde(default)]
    pub streaming: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ChannelIdentitySelectors {
    #[serde(default)]
    pub matching_rules: Vec<String>,
    #[serde(default)]
    pub examples: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ChannelSetupManifest {
    #[serde(default)]
    pub required_secrets: Vec<ChannelSecretRequirement>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub setup_url: Option<String>,
    #[serde(default)]
    pub validation_checks: Vec<ChannelValidationCheck>,
    #[serde(default)]
    pub config_fields: Vec<ChannelConfigField>,
    #[serde(default)]
    pub auth_flows: Vec<ChannelAuthFlow>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ChannelSecretRequirement {
    pub name: String,
    pub env_var: String,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub help: Option<String>,
    #[serde(default)]
    pub optional: bool,
    #[serde(default)]
    pub hints: Vec<String>,
    #[serde(default)]
    pub target: Option<ChannelConfigTarget>,
    #[serde(default)]
    pub validate: Option<ChannelValidationCheck>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ChannelValidationCheck {
    pub kind: String,
    #[serde(default)]
    pub url_template: Option<String>,
    #[serde(default)]
    pub message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ChannelConfigField {
    pub key: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(rename = "type")]
    pub field_type: String,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub help: Option<String>,
    #[serde(default)]
    pub hint: Option<String>,
    #[serde(default)]
    pub example: Option<String>,
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub advanced: bool,
    #[serde(default)]
    pub default: Option<serde_json::Value>,
    #[serde(default)]
    pub options: Vec<ChannelConfigFieldOption>,
    #[serde(default)]
    pub visible_if: Option<ChannelFieldVisibilityRule>,
    #[serde(default)]
    pub target: Option<ChannelConfigTarget>,
    #[serde(default)]
    pub validate: Option<ChannelValidationCheck>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ChannelConfigFieldOption {
    pub value: String,
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelConfigTargetKind {
    ChannelSetting,
    RootConfig,
    AgentConfig,
    EnvVar,
    LocalSecretStore,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelConfigTarget {
    pub kind: ChannelConfigTargetKind,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelFieldVisibilityRule {
    pub key: String,
    pub equals: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelAuthFlowKind {
    OauthDeviceCode,
    QrPairing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelAuthFlow {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: ChannelAuthFlowKind,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub help: Option<String>,
    #[serde(default)]
    pub hint: Option<String>,
    #[serde(default)]
    pub advanced: bool,
    #[serde(default)]
    pub visible_if: Option<ChannelFieldVisibilityRule>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelAuthFlowResolvedValue {
    pub target: ChannelConfigTarget,
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ChannelAuthFlowDisplay {
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub verification_uri: Option<String>,
    #[serde(default)]
    pub verification_uri_complete: Option<String>,
    #[serde(default)]
    pub user_code: Option<String>,
    #[serde(default)]
    pub qr_text: Option<String>,
    #[serde(default)]
    pub pairing_code: Option<String>,
    #[serde(default)]
    pub expires_in_secs: Option<u64>,
    #[serde(default)]
    pub poll_interval_secs: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelAuthFlowStartRequest {
    pub flow_id: String,
    #[serde(default)]
    pub current_settings: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelAuthFlowStartResponse {
    pub session: serde_json::Value,
    pub display: ChannelAuthFlowDisplay,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelAuthFlowPollRequest {
    pub flow_id: String,
    pub session: serde_json::Value,
    #[serde(default)]
    pub current_settings: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ChannelAuthFlowPollResponse {
    Pending {
        display: ChannelAuthFlowDisplay,
    },
    Complete {
        #[serde(default)]
        values: Vec<ChannelAuthFlowResolvedValue>,
        #[serde(default)]
        message: Option<String>,
    },
    Failed {
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ChannelInstallManifest {
    #[serde(default)]
    pub binary_name: Option<String>,
}

fn default_channel_protocol_version() -> u32 {
    CHANNEL_ADAPTER_PROTOCOL_VERSION
}

pub fn validate_adapter_manifest(manifest: &ChannelAdapterManifest) -> Result<(), String> {
    ChannelKind::parse(&manifest.kind).map_err(|err| {
        format!(
            "adapter manifest kind '{}' is invalid: {}",
            manifest.kind, err
        )
    })?;

    if manifest.protocol_version != CHANNEL_ADAPTER_PROTOCOL_VERSION {
        return Err(format!(
            "adapter manifest for '{}' uses protocol_version={} but Turin expects {}",
            manifest.kind, manifest.protocol_version, CHANNEL_ADAPTER_PROTOCOL_VERSION
        ));
    }

    let mut enum_keys = std::collections::BTreeSet::new();
    for setting in &manifest.runtime.enum_settings {
        if setting.key.trim().is_empty() {
            return Err(format!(
                "adapter manifest for '{}' contains an enum setting with an empty key",
                manifest.kind
            ));
        }
        if !enum_keys.insert(setting.key.clone()) {
            return Err(format!(
                "adapter manifest for '{}' contains duplicate enum setting '{}'",
                manifest.kind, setting.key
            ));
        }
    }

    if let Some(setup) = &manifest.setup {
        let mut field_keys = std::collections::BTreeSet::new();
        for field in &setup.config_fields {
            if field.key.trim().is_empty() {
                return Err(format!(
                    "adapter manifest for '{}' contains a config field with an empty key",
                    manifest.kind
                ));
            }
            if !field_keys.insert(field.key.clone()) {
                return Err(format!(
                    "adapter manifest for '{}' contains duplicate config field '{}'",
                    manifest.kind, field.key
                ));
            }
        }

        let mut flow_ids = std::collections::BTreeSet::new();
        for flow in &setup.auth_flows {
            if flow.id.trim().is_empty() {
                return Err(format!(
                    "adapter manifest for '{}' contains an auth flow with an empty id",
                    manifest.kind
                ));
            }
            if !flow_ids.insert(flow.id.clone()) {
                return Err(format!(
                    "adapter manifest for '{}' contains duplicate auth flow '{}'",
                    manifest.kind, flow.id
                ));
            }
        }

        for secret in &setup.required_secrets {
            if secret.name.trim().is_empty() {
                return Err(format!(
                    "adapter manifest for '{}' contains a secret with an empty name",
                    manifest.kind
                ));
            }
            if secret.env_var.trim().is_empty() {
                return Err(format!(
                    "adapter manifest for '{}' contains secret '{}' without env_var",
                    manifest.kind, secret.name
                ));
            }
        }
    }

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationBinding {
    pub agent_id: String,
    pub slot_id: String,
    pub session_id: String,
    pub updated_at_unix_secs: u64,
}

impl ConversationBinding {
    pub fn new(
        agent_id: impl Into<String>,
        session_id: impl Into<String>,
        key: &ChannelConversationKey,
        now: SystemTime,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            slot_id: key.deterministic_slot_id(),
            session_id: session_id.into(),
            updated_at_unix_secs: unix_secs(now),
        }
    }

    pub fn touch(&mut self, now: SystemTime) {
        self.updated_at_unix_secs = unix_secs(now);
    }

    pub fn is_expired(&self, now: SystemTime, ttl: Duration) -> bool {
        unix_secs(now).saturating_sub(self.updated_at_unix_secs) > ttl.as_secs()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingDecision {
    Reuse { slot_id: String, session_id: String },
    StartFresh { slot_id: String },
}

pub fn decide_routing(
    key: &ChannelConversationKey,
    binding: Option<&ConversationBinding>,
    now: SystemTime,
    ttl: Option<Duration>,
    reset_requested: bool,
) -> RoutingDecision {
    let slot_id = key.deterministic_slot_id();
    if reset_requested {
        return RoutingDecision::StartFresh { slot_id };
    }

    match binding {
        Some(binding) => {
            if ttl.is_some_and(|ttl| binding.is_expired(now, ttl)) {
                RoutingDecision::StartFresh { slot_id }
            } else {
                RoutingDecision::Reuse {
                    slot_id,
                    session_id: binding.session_id.clone(),
                }
            }
        }
        None => RoutingDecision::StartFresh { slot_id },
    }
}

fn unix_secs(time: SystemTime) -> u64 {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> ChannelConversationKey {
        ChannelConversationKey {
            channel: ChannelKind::new("discord"),
            workspace_id: "guild-1".into(),
            room_id: Some("room-2".into()),
            thread_id: "thread-3".into(),
            user_id: Some("user-4".into()),
        }
    }

    #[test]
    fn slot_id_is_stable() {
        let key = key();
        assert_eq!(key.deterministic_slot_id(), key.deterministic_slot_id());
    }

    #[test]
    fn reset_forces_fresh_session() {
        let key = key();
        let binding = ConversationBinding::new("writer", "sess-1", &key, SystemTime::UNIX_EPOCH);
        let decision = decide_routing(&key, Some(&binding), SystemTime::UNIX_EPOCH, None, true);
        assert!(matches!(decision, RoutingDecision::StartFresh { .. }));
    }

    #[test]
    fn ttl_expiry_forces_fresh_session() {
        let key = key();
        let binding = ConversationBinding::new("writer", "sess-1", &key, SystemTime::UNIX_EPOCH);
        let decision = decide_routing(
            &key,
            Some(&binding),
            SystemTime::UNIX_EPOCH + Duration::from_secs(120),
            Some(Duration::from_secs(60)),
            false,
        );
        assert!(matches!(decision, RoutingDecision::StartFresh { .. }));
    }

    #[test]
    fn structured_outbound_message_keeps_code_blocks() {
        let message = OutboundMessage {
            blocks: vec![
                MessageBlock::Text {
                    text: "Here is code".into(),
                },
                MessageBlock::CodeBlock {
                    language: Some("rust".into()),
                    code: "fn main() {}".into(),
                },
            ],
            ..OutboundMessage::default()
        };
        let value = serde_json::to_value(&message).expect("serialize outbound message");
        assert_eq!(value["blocks"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn shared_scope_prompt_includes_sender_label() {
        let key = key();
        let event = InboundEvent {
            conversation: key.clone(),
            message: ChannelMessageRef {
                conversation: key,
                message_id: "m-1".into(),
            },
            user: ChannelUser {
                id: "user-4".into(),
                display_name: Some("Jay".into()),
                username: Some("jthum".into()),
            },
            session_scope: ChannelSessionScope::Thread,
            text: "hello".into(),
            attachments: vec![],
            metadata: serde_json::Map::new(),
        };

        assert_eq!(event.prompt_text(), "[Message from Jay (@jthum)]\nhello");
    }

    #[test]
    fn bound_inbound_text_leaves_short_messages_unchanged() {
        let mut metadata = serde_json::Map::new();
        let text = bound_inbound_text(
            "hello".into(),
            &mut metadata,
            DEFAULT_MAX_INBOUND_TEXT_CHARS,
        );
        assert_eq!(text, "hello");
        assert!(metadata.is_empty());
    }

    #[test]
    fn bound_inbound_text_truncates_and_marks_metadata() {
        let mut metadata = serde_json::Map::new();
        let input = "a".repeat(DEFAULT_MAX_INBOUND_TEXT_CHARS + 5);
        let text = bound_inbound_text(input, &mut metadata, DEFAULT_MAX_INBOUND_TEXT_CHARS);
        assert_eq!(text.chars().count(), DEFAULT_MAX_INBOUND_TEXT_CHARS);
        assert_eq!(
            metadata.get("turin_text_truncated"),
            Some(&serde_json::Value::Bool(true))
        );
        assert_eq!(
            metadata.get("turin_original_text_chars"),
            Some(&serde_json::Value::Number(
                (DEFAULT_MAX_INBOUND_TEXT_CHARS + 5).into()
            ))
        );
    }

    #[test]
    fn bound_inbound_text_uses_custom_limit() {
        let mut metadata = serde_json::Map::new();
        let text = bound_inbound_text("abcdef".into(), &mut metadata, 3);
        assert_eq!(text, "abc");
        assert_eq!(
            metadata.get("turin_text_char_limit"),
            Some(&serde_json::Value::Number(3usize.into()))
        );
    }

    #[test]
    fn channel_kind_normalizes_to_lowercase() {
        let kind = ChannelKind::parse("TeLeGrAm").expect("normalized");
        assert_eq!(kind.as_str(), "telegram");
    }

    #[test]
    fn channel_kind_rejects_invalid_characters() {
        let err = ChannelKind::parse("telegram!").expect_err("invalid");
        assert!(err.contains("channel kind"));
    }

    #[test]
    fn manifest_validation_rejects_wrong_protocol_version() {
        let manifest = ChannelAdapterManifest {
            protocol_version: CHANNEL_ADAPTER_PROTOCOL_VERSION + 1,
            kind: "telegram".to_string(),
            ..ChannelAdapterManifest::default()
        };
        let err = manifest.validate().expect_err("invalid protocol");
        assert!(err.contains("protocol_version"));
    }

    #[test]
    fn manifest_validation_rejects_duplicate_auth_flows() {
        let manifest = ChannelAdapterManifest {
            protocol_version: CHANNEL_ADAPTER_PROTOCOL_VERSION,
            kind: "telegram".to_string(),
            setup: Some(ChannelSetupManifest {
                auth_flows: vec![
                    ChannelAuthFlow {
                        id: "pair".to_string(),
                        kind: ChannelAuthFlowKind::QrPairing,
                        label: None,
                        prompt: None,
                        help: None,
                        hint: None,
                        advanced: false,
                        visible_if: None,
                    },
                    ChannelAuthFlow {
                        id: "pair".to_string(),
                        kind: ChannelAuthFlowKind::QrPairing,
                        label: None,
                        prompt: None,
                        help: None,
                        hint: None,
                        advanced: false,
                        visible_if: None,
                    },
                ],
                ..ChannelSetupManifest::default()
            }),
            ..ChannelAdapterManifest::default()
        };
        let err = manifest.validate().expect_err("duplicate auth flow");
        assert!(err.contains("duplicate auth flow"));
    }
}
