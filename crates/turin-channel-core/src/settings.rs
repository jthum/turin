use crate::ChannelSessionScope;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct ChannelConfigError {
    message: String,
}

impl ChannelConfigError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

pub fn required_non_empty_setting<'a>(
    settings: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
    missing_message: impl Into<String>,
    invalid_message: impl Into<String>,
) -> Result<&'a str, ChannelConfigError> {
    let invalid_message = invalid_message.into();
    let value = settings
        .get(key)
        .ok_or_else(|| ChannelConfigError::new(missing_message))?
        .as_str()
        .ok_or_else(|| ChannelConfigError::new(invalid_message.clone()))?;
    if value.trim().is_empty() {
        return Err(ChannelConfigError::new(invalid_message));
    }
    Ok(value)
}

pub fn optional_non_empty_setting<'a>(
    settings: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
    invalid_message: impl Into<String>,
) -> Result<Option<&'a str>, ChannelConfigError> {
    let invalid_message = invalid_message.into();
    let Some(value) = settings.get(key) else {
        return Ok(None);
    };
    let value = value
        .as_str()
        .ok_or_else(|| ChannelConfigError::new(invalid_message.clone()))?;
    if value.trim().is_empty() {
        return Err(ChannelConfigError::new(invalid_message));
    }
    Ok(Some(value))
}

pub fn optional_bool_setting(
    value: Option<&serde_json::Value>,
    default: bool,
    invalid_message: impl Into<String>,
) -> Result<bool, ChannelConfigError> {
    match value {
        None => Ok(default),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| ChannelConfigError::new(invalid_message)),
    }
}

pub fn u64_setting_with_min(
    value: Option<&serde_json::Value>,
    default: u64,
    min: u64,
    invalid_message: impl Into<String>,
) -> Result<u64, ChannelConfigError> {
    let invalid_message = invalid_message.into();
    match value {
        None => Ok(default),
        Some(value) => {
            let parsed = value
                .as_u64()
                .ok_or_else(|| ChannelConfigError::new(invalid_message.clone()))?;
            if parsed < min {
                return Err(ChannelConfigError::new(invalid_message));
            }
            Ok(parsed)
        }
    }
}

pub fn positive_usize_setting(
    value: Option<&serde_json::Value>,
    default: usize,
    invalid_message: impl Into<String>,
    too_large_message: impl Into<String>,
) -> Result<usize, ChannelConfigError> {
    let invalid_message = invalid_message.into();
    let max = match value {
        None => return Ok(default),
        Some(value) => value
            .as_u64()
            .ok_or_else(|| ChannelConfigError::new(invalid_message.clone()))?,
    };
    let max = usize::try_from(max).map_err(|_| ChannelConfigError::new(too_large_message))?;
    if max == 0 {
        return Err(ChannelConfigError::new(invalid_message));
    }
    Ok(max)
}

pub fn session_scope_setting(
    value: Option<&serde_json::Value>,
    default: ChannelSessionScope,
    allowed: &[ChannelSessionScope],
    invalid_type_message: impl Into<String>,
    invalid_value_message: impl Into<String>,
) -> Result<ChannelSessionScope, ChannelConfigError> {
    let Some(value) = value else {
        return Ok(default);
    };
    let scope = value
        .as_str()
        .ok_or_else(|| ChannelConfigError::new(invalid_type_message))?;
    ChannelSessionScope::parse(scope)
        .filter(|scope| scope.is_allowed_by(allowed))
        .ok_or_else(|| ChannelConfigError::new(invalid_value_message))
}

pub fn optional_session_scope_setting(
    value: Option<&serde_json::Value>,
    allowed: &[ChannelSessionScope],
    invalid_type_message: impl Into<String>,
    invalid_value_message: impl Into<String>,
) -> Result<Option<ChannelSessionScope>, ChannelConfigError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let scope = value
        .as_str()
        .ok_or_else(|| ChannelConfigError::new(invalid_type_message))?;
    ChannelSessionScope::parse(scope)
        .filter(|scope| scope.is_allowed_by(allowed))
        .map(Some)
        .ok_or_else(|| ChannelConfigError::new(invalid_value_message))
}

pub fn string_enum_setting<T>(
    value: Option<&serde_json::Value>,
    default: T,
    parse: impl FnOnce(&str) -> Option<T>,
    invalid_type_message: impl Into<String>,
    invalid_value_message: impl Into<String>,
) -> Result<T, ChannelConfigError> {
    let Some(value) = value else {
        return Ok(default);
    };
    let raw = value
        .as_str()
        .ok_or_else(|| ChannelConfigError::new(invalid_type_message))?;
    parse(raw).ok_or_else(|| ChannelConfigError::new(invalid_value_message))
}
