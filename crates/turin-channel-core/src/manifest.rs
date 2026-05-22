use serde::{Deserialize, Serialize};

use crate::auth::ChannelAuthFlow;
use crate::messages::ChannelKind;
use crate::{CHANNEL_ADAPTER_PROTOCOL_VERSION, DEFAULT_MAX_INBOUND_TEXT_CHARS};

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

pub fn channel_enum_setting<I, S>(key: impl Into<String>, options: I) -> ChannelEnumSetting
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    ChannelEnumSetting {
        key: key.into(),
        options: options.into_iter().map(Into::into).collect(),
    }
}

pub fn channel_setting_target(name: impl Into<String>) -> ChannelConfigTarget {
    ChannelConfigTarget {
        kind: ChannelConfigTargetKind::ChannelSetting,
        name: name.into(),
    }
}

pub fn channel_setting_target_opt(name: impl Into<String>) -> Option<ChannelConfigTarget> {
    Some(channel_setting_target(name))
}

pub fn config_field_option(
    value: impl Into<String>,
    label: impl Into<String>,
) -> ChannelConfigFieldOption {
    ChannelConfigFieldOption {
        value: value.into(),
        label: Some(label.into()),
    }
}

pub fn config_field_options<I, V, L>(options: I) -> Vec<ChannelConfigFieldOption>
where
    I: IntoIterator<Item = (V, L)>,
    V: Into<String>,
    L: Into<String>,
{
    options
        .into_iter()
        .map(|(value, label)| config_field_option(value, label))
        .collect()
}

pub fn max_inbound_text_chars_field(help: impl Into<String>) -> ChannelConfigField {
    ChannelConfigField {
        key: "max_inbound_text_chars".to_string(),
        label: Some("Max Inbound Text Chars".to_string()),
        field_type: "number".to_string(),
        help: Some(help.into()),
        default: Some(serde_json::json!(DEFAULT_MAX_INBOUND_TEXT_CHARS)),
        advanced: true,
        target: channel_setting_target_opt("max_inbound_text_chars"),
        ..ChannelConfigField::default()
    }
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
