use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;

use crate::persistence::manager::{StoreManager, StorePathScope, StoreSelector};

impl StoreManager {
    pub(super) fn default_alias_path(&self, alias: &str) -> PathBuf {
        let file_name = format!("{}.db", sanitize_alias(alias));
        self.store_root.join(file_name)
    }

    pub(super) async fn resolve_selector_with_alias(
        &self,
        selector: &StoreSelector,
        path_scope: StorePathScope,
    ) -> Result<(PathBuf, Option<String>)> {
        match selector {
            StoreSelector::Alias(alias) => {
                let path = self.resolve_alias_path(alias).await?;
                Ok((path, Some(alias.clone())))
            }
            StoreSelector::Path(p) => {
                let path = self.resolve_path_scoped(Path::new(p), path_scope)?;
                Ok((path, None))
            }
            StoreSelector::Handle(handle) => {
                let mut handles = self.handles.write().await;
                let entry = handles
                    .get_mut(handle)
                    .ok_or_else(|| anyhow::anyhow!("Unknown db handle '{}'", handle))?;
                entry.open_count = entry.open_count.saturating_add(1);
                entry.last_access = Instant::now();
                Ok((entry.path.clone(), entry.alias.clone()))
            }
        }
    }

    pub(super) async fn resolve_selector_path(
        &self,
        selector: &StoreSelector,
        path_scope: StorePathScope,
    ) -> Result<PathBuf> {
        let (path, _) = self
            .resolve_selector_with_alias(selector, path_scope)
            .await?;
        Ok(path)
    }

    async fn resolve_alias_path(&self, alias: &str) -> Result<PathBuf> {
        if let Some(path) = self.aliases.read().await.get(alias).cloned() {
            return Ok(path);
        }

        let path = self.default_alias_path(alias);
        let path = self.resolve_path_unchecked(&path)?;
        let mut aliases = self.aliases.write().await;
        Ok(aliases
            .entry(alias.to_string())
            .or_insert_with(|| path.clone())
            .clone())
    }

    pub(super) fn resolve_path_unchecked(&self, requested: &Path) -> Result<PathBuf> {
        if requested
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            anyhow::bail!("Path traversal (..) not allowed: {}", requested.display());
        }
        let resolved = if requested.is_absolute() {
            requested.to_path_buf()
        } else {
            self.workspace_root.join(requested)
        };
        Ok(resolved)
    }

    fn resolve_path_scoped(&self, requested: &Path, path_scope: StorePathScope) -> Result<PathBuf> {
        match path_scope {
            StorePathScope::AllowAny => self.resolve_path_unchecked(requested),
            StorePathScope::WorkspaceOnly => {
                crate::tools::is_safe_path(&self.workspace_root, requested)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))
            }
        }
    }
}

fn sanitize_alias(alias: &str) -> String {
    let mut out = String::with_capacity(alias.len());
    for ch in alias.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "store".to_string()
    } else {
        out
    }
}
