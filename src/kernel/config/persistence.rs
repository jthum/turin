use std::collections::HashMap;
use std::path::Path;

use turin_types::layout::resolve_relative_to;

use super::{
    NamedStoreConfig, PersistenceConfig, ResolvedLayout, StoreTargetConfig, default_state_path,
};

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
