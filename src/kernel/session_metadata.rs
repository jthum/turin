use crate::kernel::config::StoreTargetConfig;
use crate::persistence::manager::StoreSelector;

pub(crate) fn create_session_metadata(
    default_store_selector: Option<&StoreSelector>,
    channel_id: Option<&str>,
) -> Option<String> {
    let mut turin_meta = serde_json::Map::new();
    if let Some(default_store) = default_store_selector.and_then(store_target_from_selector) {
        turin_meta.insert(
            "default_store".to_string(),
            serde_json::json!(default_store),
        );
    }
    if let Some(channel_id) = channel_id {
        turin_meta.insert("channel_id".to_string(), serde_json::json!(channel_id));
    }
    if turin_meta.is_empty() {
        return None;
    }
    Some(
        serde_json::json!({
            "_turin": turin_meta,
        })
        .to_string(),
    )
}

pub(crate) fn session_default_store_selector_from_metadata(
    metadata: Option<&str>,
) -> Option<StoreSelector> {
    let target = turin_metadata(metadata)
        .and_then(|value| value.get("default_store").cloned())
        .and_then(|value| serde_json::from_value::<StoreTargetConfig>(value).ok());

    target.and_then(store_selector_from_target)
}

pub(crate) fn session_channel_id_from_metadata(metadata: Option<&str>) -> Option<String> {
    turin_metadata(metadata)
        .and_then(|value| value.get("channel_id").cloned())
        .and_then(|value| value.as_str().map(ToString::to_string))
}

pub(crate) fn session_title_from_metadata(metadata: Option<&str>) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(metadata?).ok()?;
    value
        .get("title")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|title| !title.is_empty())
        .map(str::to_string)
}

fn turin_metadata(metadata: Option<&str>) -> Option<serde_json::Value> {
    metadata
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
        .and_then(|value| value.get("_turin").cloned())
}

fn store_target_from_selector(selector: &StoreSelector) -> Option<StoreTargetConfig> {
    match selector {
        StoreSelector::Alias(alias) => Some(StoreTargetConfig::from_alias(alias.clone())),
        StoreSelector::Path(path) => Some(StoreTargetConfig::from_path(path.clone())),
        StoreSelector::Handle(_) => None,
    }
}

fn store_selector_from_target(target: StoreTargetConfig) -> Option<StoreSelector> {
    if let Some(path) = target.path {
        Some(StoreSelector::Path(path))
    } else {
        target.alias.map(StoreSelector::Alias)
    }
}
