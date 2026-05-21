use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use serde::Deserialize;
use turin_types::layout::resolve_relative_to;

use crate::persistence::manager::StoreSelector;

use super::{ResolvedLayout, default_state_path, default_state_target};

#[derive(Debug, Clone, Deserialize)]
pub struct PersistenceConfig {
    /// Primary owning state target for session/runtime data.
    #[serde(default = "default_state_target")]
    pub state: StoreTargetConfig,
    /// Optional default auxiliary store for scoped non-session data.
    #[serde(default)]
    pub store: Option<StoreTargetConfig>,
    /// Optional named state stores that can be referenced by alias.
    #[serde(default)]
    pub states: HashMap<String, NamedStoreConfig>,
    /// Optional named auxiliary stores that can be referenced by alias.
    #[serde(default)]
    pub stores: HashMap<String, NamedStoreConfig>,
    /// Optional Level 1 placement rules for scoped memory/KV when no store is provided.
    #[serde(default)]
    pub placements: Vec<ScopedStorePlacementConfig>,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            state: default_state_target(),
            store: None,
            states: HashMap::new(),
            stores: HashMap::new(),
            placements: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, serde::Serialize, Default, PartialEq, Eq)]
pub struct StoreTargetConfig {
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub alias: Option<String>,
}

impl StoreTargetConfig {
    pub fn from_path(path: impl Into<String>) -> Self {
        Self {
            path: Some(path.into()),
            alias: None,
        }
    }

    pub fn from_alias(alias: impl Into<String>) -> Self {
        Self {
            path: None,
            alias: Some(alias.into()),
        }
    }

    pub(super) fn validate(&self, label: &str) -> Result<()> {
        let has_path = self
            .path
            .as_ref()
            .is_some_and(|value| !value.trim().is_empty());
        let has_alias = self
            .alias
            .as_ref()
            .is_some_and(|value| !value.trim().is_empty());

        anyhow::ensure!(
            has_path || has_alias,
            "{} requires either 'path' or 'alias'",
            label
        );
        anyhow::ensure!(
            !(has_path && has_alias),
            "{} cannot set both 'path' and 'alias'",
            label
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, serde::Serialize, Default, PartialEq, Eq)]
pub struct ContextPersistenceConfig {
    #[serde(default)]
    pub state: Option<StoreTargetConfig>,
    #[serde(default)]
    pub store: Option<StoreTargetConfig>,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize, Default)]
pub struct NamedStoreConfig {
    pub path: String,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize, Default)]
pub struct ScopedStorePlacementConfig {
    pub scope_kind: String,
    #[serde(default)]
    pub scope_key: Option<String>,
    #[serde(default)]
    pub namespace: Option<String>,
    pub store: String,
}

impl PersistenceConfig {
    pub fn with_state_path(path: impl Into<String>) -> Self {
        Self {
            state: StoreTargetConfig::from_path(path),
            ..Self::default()
        }
    }

    pub fn top_level_state_selector(&self) -> Result<StoreSelector> {
        self.resolve_state_target(&self.state)
    }

    pub fn top_level_store_selector(&self) -> Result<StoreSelector> {
        match &self.store {
            Some(target) => self.resolve_store_target(target),
            None => self.top_level_state_selector(),
        }
    }

    pub fn resolve_context_state_selector(
        &self,
        context: Option<&ContextPersistenceConfig>,
    ) -> Result<StoreSelector> {
        if let Some(target) = context.and_then(|context| context.state.as_ref()) {
            return self.resolve_state_target(target);
        }
        self.top_level_state_selector()
    }

    pub fn resolve_context_store_selector(
        &self,
        context: Option<&ContextPersistenceConfig>,
    ) -> Result<StoreSelector> {
        if let Some(target) = context.and_then(|context| context.store.as_ref()) {
            return self.resolve_store_target(target);
        }
        if let Some(target) = context.and_then(|context| context.state.as_ref()) {
            return self.resolve_state_target(target);
        }
        self.top_level_store_selector()
    }

    pub fn resolve_state_target(&self, target: &StoreTargetConfig) -> Result<StoreSelector> {
        target.validate("persistence.state")?;
        if let Some(path) = &target.path {
            return Ok(StoreSelector::Path(path.clone()));
        }
        let alias = target
            .alias
            .as_deref()
            .expect("validated state target has alias");
        anyhow::ensure!(
            alias == "state" || self.states.contains_key(alias),
            "persistence.state alias '{}' not found in persistence.states",
            alias
        );
        Ok(StoreSelector::Alias(alias.to_string()))
    }

    pub fn resolve_store_target(&self, target: &StoreTargetConfig) -> Result<StoreSelector> {
        target.validate("persistence.store")?;
        if let Some(path) = &target.path {
            return Ok(StoreSelector::Path(path.clone()));
        }
        let alias = target
            .alias
            .as_deref()
            .expect("validated store target has alias");
        anyhow::ensure!(
            alias == "state" || self.states.contains_key(alias) || self.stores.contains_key(alias),
            "persistence.store alias '{}' not found in persistence.states or persistence.stores",
            alias
        );
        Ok(StoreSelector::Alias(alias.to_string()))
    }

    pub fn resolve_store_alias_for_scope(
        &self,
        scope_kind: &str,
        raw_scope_key: Option<&str>,
        namespace: &str,
    ) -> Option<&str> {
        let exact = self.placements.iter().find(|placement| {
            placement.scope_kind == scope_kind
                && placement.scope_key.as_deref() == raw_scope_key
                && placement.namespace.as_deref().unwrap_or("default") == namespace
        });
        if let Some(exact) = exact {
            return Some(exact.store.as_str());
        }

        self.placements
            .iter()
            .find(|placement| placement.scope_kind == scope_kind && placement.scope_key.is_none())
            .map(|placement| placement.store.as_str())
    }

    pub fn resolve_store_selector_for_scope(
        &self,
        scope_kind: &str,
        raw_scope_key: Option<&str>,
        namespace: &str,
    ) -> Option<StoreSelector> {
        if let Some(alias) =
            self.resolve_store_alias_for_scope(scope_kind, raw_scope_key, namespace)
        {
            return Some(StoreSelector::Alias(alias.to_string()));
        }
        self.top_level_store_selector().ok()
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedPersistenceConfig {
    pub state: StoreTargetConfig,
    pub store: Option<StoreTargetConfig>,
    pub states: HashMap<String, NamedStoreConfig>,
    pub stores: HashMap<String, NamedStoreConfig>,
}

impl ResolvedPersistenceConfig {
    pub fn from_parts(
        workspace_root: &Path,
        layout: &ResolvedLayout,
        persistence: &PersistenceConfig,
    ) -> Self {
        Self {
            state: resolve_store_target_path(
                workspace_root,
                &layout.default_state_db,
                default_state_path().as_str(),
                &persistence.state,
            ),
            store: persistence.store.as_ref().map(|target| {
                resolve_store_target_path(workspace_root, &layout.stores_dir, "", target)
            }),
            states: persistence
                .states
                .iter()
                .map(|(alias, store)| {
                    (
                        alias.clone(),
                        resolve_named_store_path(workspace_root, store),
                    )
                })
                .collect(),
            stores: persistence
                .stores
                .iter()
                .map(|(alias, store)| {
                    (
                        alias.clone(),
                        resolve_named_store_path(workspace_root, store),
                    )
                })
                .collect(),
        }
    }
}

fn resolve_store_target_path(
    workspace_root: &Path,
    layout_default: &Path,
    default_value: &str,
    target: &StoreTargetConfig,
) -> StoreTargetConfig {
    let mut resolved = target.clone();
    if let Some(path) = resolved.path.as_mut() {
        let resolved_path = if !default_value.is_empty() && path == default_value {
            layout_default.to_path_buf()
        } else {
            resolve_relative_to(workspace_root, Path::new(path))
        };
        *path = resolved_path.display().to_string();
    }
    resolved
}

fn resolve_named_store_path(workspace_root: &Path, store: &NamedStoreConfig) -> NamedStoreConfig {
    let mut resolved = store.clone();
    if Path::new(&resolved.path).is_relative() {
        resolved.path = workspace_root.join(&resolved.path).display().to_string();
    }
    resolved
}
