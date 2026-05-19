use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Map, Value};
use turin_channel_core::{ChannelSessionScope, DEFAULT_MAX_INBOUND_TEXT_CHARS};

use crate::{
    DEFAULT_PERSONAL_TRIGGER_PREFIX, DEFAULT_RUNTIME_STORE_BASENAME, DEFAULT_WORKSPACE_ID,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WhatsAppAccountMode {
    Personal,
    Dedicated,
}

#[derive(Debug, Clone)]
pub struct WhatsAppChannelDriverConfig {
    pub workspace_id: String,
    pub(crate) account_mode: WhatsAppAccountMode,
    pub session_scope: ChannelSessionScope,
    pub session_store_path: PathBuf,
    pub(crate) media_dir: PathBuf,
    pub(crate) max_inbound_text_chars: usize,
    pub(crate) trigger_prefix: Option<String>,
    pub(crate) allowed_chats: Vec<String>,
    pub(crate) banned_chats: Vec<String>,
}

pub fn validate_settings(settings: &Value, _allow_unconfigured_chats: bool) -> Result<()> {
    parse_settings(settings, None).map(|_| ())
}

pub(crate) fn parse_settings(
    settings: &Value,
    runtime_dir: Option<&Path>,
) -> Result<WhatsAppChannelDriverConfig> {
    let map = settings_object(settings)?;
    let workspace_id = optional_nonempty_string(map, "workspace_id")?
        .unwrap_or_else(|| DEFAULT_WORKSPACE_ID.to_string());
    let account_mode = parse_account_mode(map.get("account_mode"))?;
    let session_scope = parse_session_scope(map.get("session_scope"))?;
    let trigger_prefix = optional_nonempty_string(map, "trigger_prefix")?;
    let trigger_prefix = match (account_mode, trigger_prefix) {
        (WhatsAppAccountMode::Personal, None) => Some(DEFAULT_PERSONAL_TRIGGER_PREFIX.to_string()),
        (_, value) => value,
    };
    let allowed_chats = parse_string_list(map.get("allowed_chats"), "allowed_chats")?;
    let banned_chats = parse_string_list(map.get("banned_chats"), "banned_chats")?;

    let pair_code_phone_number = optional_nonempty_string(map, "pair_code_phone_number")?;
    let pair_code_custom_code = optional_nonempty_string(map, "pair_code_custom_code")?;
    validate_pair_code_fields(
        pair_code_phone_number.as_deref(),
        pair_code_custom_code.as_deref(),
    )?;

    let session_store_path = match optional_nonempty_string(map, "session_store_path")? {
        Some(raw) => resolve_runtime_store_path(&raw, runtime_dir),
        None => runtime_dir
            .map(|dir| dir.join(DEFAULT_RUNTIME_STORE_BASENAME))
            .unwrap_or_else(|| {
                default_auth_store_path(
                    &std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
                    &workspace_id,
                )
            }),
    };

    let max_inbound_text_chars = match map.get("max_inbound_text_chars") {
        None => DEFAULT_MAX_INBOUND_TEXT_CHARS,
        Some(value) => {
            let max = value.as_u64().ok_or_else(|| {
                anyhow!(
                    "[whatsapp_config_invalid_max_inbound_text_chars] WhatsApp channel setting 'max_inbound_text_chars' must be a positive integer"
                )
            })?;
            let max = usize::try_from(max).map_err(|_| {
                anyhow!(
                    "[whatsapp_config_invalid_max_inbound_text_chars] WhatsApp channel setting 'max_inbound_text_chars' is too large"
                )
            })?;
            if max == 0 {
                bail!(
                    "[whatsapp_config_invalid_max_inbound_text_chars] WhatsApp channel setting 'max_inbound_text_chars' must be > 0"
                );
            }
            max
        }
    };

    Ok(WhatsAppChannelDriverConfig {
        workspace_id: workspace_id.clone(),
        account_mode,
        session_scope,
        session_store_path,
        media_dir: runtime_dir.map(|dir| dir.join("media")).unwrap_or_else(|| {
            std::env::temp_dir()
                .join("turin")
                .join("channels")
                .join("whatsapp")
                .join(sanitize_component(&workspace_id))
                .join("media")
        }),
        max_inbound_text_chars,
        trigger_prefix,
        allowed_chats,
        banned_chats,
    })
}

pub(crate) fn settings_object(settings: &Value) -> Result<&Map<String, Value>> {
    settings
        .as_object()
        .ok_or_else(|| anyhow!("Channel settings must be a JSON object"))
}

pub(crate) fn optional_nonempty_string(
    map: &Map<String, Value>,
    key: &str,
) -> Result<Option<String>> {
    match map.get(key) {
        Some(Value::String(value)) if !value.trim().is_empty() => {
            Ok(Some(value.trim().to_string()))
        }
        Some(Value::String(_)) | Some(Value::Null) | None => Ok(None),
        Some(_) => bail!("channel setting '{key}' must be a string"),
    }
}

fn parse_session_scope(value: Option<&Value>) -> Result<ChannelSessionScope> {
    let scope = match value {
        None | Some(Value::Null) => return Ok(ChannelSessionScope::User),
        Some(Value::String(value)) => value.as_str(),
        Some(_) => bail!("channel setting 'session_scope' must be a string"),
    };

    match scope {
        "user" => Ok(ChannelSessionScope::User),
        "room" => Ok(ChannelSessionScope::Room),
        other => {
            bail!("channel setting 'session_scope' must be one of: user, room (got '{other}')")
        }
    }
}

fn parse_account_mode(value: Option<&Value>) -> Result<WhatsAppAccountMode> {
    let mode = match value {
        None | Some(Value::Null) => "personal",
        Some(Value::String(value)) => value.as_str(),
        Some(_) => bail!("channel setting 'account_mode' must be a string"),
    };

    match mode {
        "personal" => Ok(WhatsAppAccountMode::Personal),
        "dedicated" => Ok(WhatsAppAccountMode::Dedicated),
        other => bail!(
            "channel setting 'account_mode' must be one of: personal, dedicated (got '{other}')"
        ),
    }
}

fn parse_string_list(value: Option<&Value>, key: &str) -> Result<Vec<String>> {
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(values)) => {
            let mut out = Vec::with_capacity(values.len());
            for value in values {
                let Some(value) = value.as_str() else {
                    bail!("channel setting '{key}' must contain only strings");
                };
                let trimmed = value.trim();
                if !trimmed.is_empty() {
                    out.push(trimmed.to_string());
                }
            }
            Ok(out)
        }
        Some(_) => bail!("channel setting '{key}' must be an array of strings"),
    }
}

pub(crate) fn validate_pair_code_fields(
    phone_number: Option<&str>,
    custom_code: Option<&str>,
) -> Result<()> {
    if custom_code.is_some() && phone_number.is_none() {
        bail!("channel setting 'pair_code_custom_code' requires 'pair_code_phone_number'");
    }

    if let Some(custom_code) = custom_code {
        if custom_code.len() != 8 {
            bail!("channel setting 'pair_code_custom_code' must be exactly 8 characters");
        }
        if !custom_code
            .chars()
            .all(|ch| matches!(ch, '1'..='9' | 'A'..='H' | 'J'..='N' | 'P'..='T' | 'V'..='Z' | 'a'..='h' | 'j'..='n' | 'p'..='t' | 'v'..='z'))
        {
            bail!("channel setting 'pair_code_custom_code' must use Crockford Base32 characters");
        }
    }

    Ok(())
}

fn resolve_runtime_store_path(raw: &str, runtime_dir: Option<&Path>) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else if let Some(runtime_dir) = runtime_dir {
        runtime_dir.join(path)
    } else {
        path
    }
}

pub(crate) fn resolve_auth_store_path(settings: &Map<String, Value>) -> Result<PathBuf> {
    let cwd =
        std::env::current_dir().context("Failed to resolve current directory for auth flow")?;
    if let Some(raw) = optional_nonempty_string(settings, "session_store_path")? {
        let configured = PathBuf::from(raw);
        if configured.is_absolute() {
            return Ok(configured);
        }
        return Ok(cwd.join(configured));
    }

    let workspace_id = optional_nonempty_string(settings, "workspace_id")?
        .unwrap_or_else(|| DEFAULT_WORKSPACE_ID.to_string());
    Ok(default_auth_store_path(&cwd, &workspace_id))
}

fn default_auth_store_path(cwd: &Path, workspace_id: &str) -> PathBuf {
    let workspace_component = sanitize_component(workspace_id);
    cwd.join(".turin")
        .join("data")
        .join("channels")
        .join(format!("whatsapp-{workspace_component}.db"))
}

pub(crate) fn sanitize_component(raw: &str) -> String {
    let mut out = String::new();
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('-');
        }
    }
    let trimmed = out.trim_matches('-');
    if trimmed.is_empty() {
        "default".to_string()
    } else {
        trimmed.to_string()
    }
}
