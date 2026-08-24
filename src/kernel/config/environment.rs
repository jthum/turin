use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};

/// Environment values captured from a config-adjacent file at load time.
#[derive(Debug, Clone, Default)]
pub(crate) struct EnvironmentSnapshot {
    values: BTreeMap<String, String>,
}

impl EnvironmentSnapshot {
    pub(super) fn from_file(path: &Path) -> Result<Self> {
        if !path.is_file() {
            return Ok(Self::default());
        }

        let mut values = BTreeMap::new();
        for item in dotenvy::from_path_iter(path)
            .with_context(|| format!("Failed to parse '{}'", path.display()))?
        {
            let (key, value) =
                item.with_context(|| format!("Failed to parse '{}'", path.display()))?;
            values.insert(key, value);
        }
        Ok(Self { values })
    }

    pub(super) fn get(&self, key: &str) -> Option<String> {
        self.values.get(key).cloned()
    }
}
