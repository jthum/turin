use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use sha2::{Digest, Sha256};
use turin_daemon_protocol::{
    HarnessSourceEntry, HarnessSourceFile, HarnessSourceListResult, HarnessSourceOverlay,
    HarnessSourceSaveChange, HarnessSourceSaveResult, HarnessSourceValidationResult,
};

use super::DaemonState;

const MAX_HARNESS_SOURCE_BYTES: usize = 10 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct HarnessSourceConflict(String);

impl DaemonState {
    pub fn list_harness_sources(&self, harness_id: &str) -> Result<HarnessSourceListResult> {
        let root = self.require_harness_source_root(harness_id)?;
        let mut paths = Vec::new();
        collect_lua_sources(&root, &root, &mut paths)?;
        paths.sort();

        let files = paths
            .into_iter()
            .map(|path| source_entry(&root, &path))
            .collect::<Result<Vec<_>>>()?;
        Ok(HarnessSourceListResult {
            harness_id: harness_id.to_string(),
            files,
        })
    }

    pub fn get_harness_source(&self, harness_id: &str, path: &str) -> Result<HarnessSourceFile> {
        let root = self.require_harness_source_root(harness_id)?;
        let relative = normalize_source_path(path)?;
        let target = resolve_source_target(&root, &relative)?;
        let source = read_source(&target)?;
        let bytes = source.len();
        Ok(HarnessSourceFile {
            path: path_string(&relative)?,
            hash: source_hash(source.as_bytes()),
            source,
            bytes,
        })
    }

    pub fn validate_harness_sources(
        &self,
        harness_id: &str,
        changes: Vec<HarnessSourceOverlay>,
    ) -> Result<HarnessSourceValidationResult> {
        let detail = self
            .harness_detail(harness_id)
            .ok_or_else(|| anyhow!("Harness '{}' not found", harness_id))?;
        let mut overlay = crate::harness::source::HarnessSourceOverlay::default();
        let mut seen = HashSet::new();
        for change in changes {
            let relative = normalize_source_path(&change.path)?;
            let display = path_string(&relative)?;
            if !seen.insert(relative.clone()) {
                anyhow::bail!("Harness source path '{}' appears more than once", display);
            }
            validate_source_size(change.source.as_deref(), &display)?;
            overlay.insert(relative, change.source);
        }

        let script_count = self
            .kernel
            .validate_named_harness_sources(harness_id, overlay)?;
        Ok(HarnessSourceValidationResult {
            harness_id: harness_id.to_string(),
            directory: detail.directory,
            script_count,
            valid: true,
        })
    }

    pub fn save_harness_sources(
        &self,
        harness_id: &str,
        changes: Vec<HarnessSourceSaveChange>,
    ) -> Result<HarnessSourceSaveResult> {
        if changes.is_empty() {
            anyhow::bail!("Harness source save requires at least one change");
        }
        let root = self.require_harness_source_root(harness_id)?;
        let mut seen = HashSet::new();
        let mut prepared = Vec::with_capacity(changes.len());

        for change in changes {
            let relative = normalize_source_path(&change.path)?;
            let display = path_string(&relative)?;
            if !seen.insert(relative.clone()) {
                anyhow::bail!("Harness source path '{}' appears more than once", display);
            }
            validate_source_size(change.source.as_deref(), &display)?;
            let target = resolve_source_target(&root, &relative)?;
            ensure_expected_hash(&target, change.expected_hash.as_deref(), &display)?;
            prepared.push(PreparedSourceChange {
                relative,
                target,
                source: change.source,
                expected_hash: change.expected_hash,
            });
        }

        let mut staged = Vec::new();
        let stage_result = (|| -> Result<()> {
            for change in &prepared {
                let Some(source) = &change.source else {
                    continue;
                };
                let parent = change
                    .target
                    .parent()
                    .ok_or_else(|| anyhow!("Harness source target has no parent"))?;
                std::fs::create_dir_all(parent).with_context(|| {
                    format!(
                        "Failed to create harness source directory '{}'",
                        parent.display()
                    )
                })?;
                let target = resolve_source_target(&root, &change.relative)?;
                let temp = temporary_source_path(&target);
                let mut file = OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(&temp)
                    .with_context(|| {
                        format!("Failed to stage harness source '{}'", temp.display())
                    })?;
                file.write_all(source.as_bytes())?;
                if target.is_file() {
                    std::fs::set_permissions(&temp, std::fs::metadata(&target)?.permissions())?;
                }
                file.sync_all()?;
                staged.push((target, temp));
            }
            Ok(())
        })();
        if let Err(err) = stage_result {
            for (_, temp) in &staged {
                let _ = std::fs::remove_file(temp);
            }
            return Err(err);
        }

        let apply_result = (|| -> Result<()> {
            // Recheck after staging so a concurrent editor cannot silently lose
            // an update during a large batch preparation.
            for change in &prepared {
                ensure_expected_hash(
                    &change.target,
                    change.expected_hash.as_deref(),
                    &path_string(&change.relative)?,
                )?;
            }
            for (target, temp) in &staged {
                std::fs::rename(temp, target).with_context(|| {
                    format!("Failed to replace harness source '{}'", target.display())
                })?;
            }
            for change in &prepared {
                if change.source.is_none() && change.target.exists() {
                    std::fs::remove_file(&change.target).with_context(|| {
                        format!(
                            "Failed to delete harness source '{}'",
                            change.target.display()
                        )
                    })?;
                }
            }
            Ok(())
        })();
        for (_, temp) in &staged {
            let _ = std::fs::remove_file(temp);
        }
        apply_result?;

        let mut saved = Vec::new();
        let mut deleted = Vec::new();
        for change in prepared {
            if change.source.is_some() {
                saved.push(source_entry(&root, &change.target)?);
            } else {
                deleted.push(path_string(&change.relative)?);
            }
        }
        saved.sort_by(|left, right| left.path.cmp(&right.path));
        deleted.sort();
        Ok(HarnessSourceSaveResult {
            harness_id: harness_id.to_string(),
            saved,
            deleted,
        })
    }

    fn require_harness_source_root(&self, harness_id: &str) -> Result<PathBuf> {
        let root = self
            .resolve_harness_root(harness_id)
            .ok_or_else(|| anyhow!("Harness '{}' not found", harness_id))?;
        if !root.is_dir() {
            anyhow::bail!("Harness directory '{}' does not exist", root.display());
        }
        Ok(root)
    }
}

struct PreparedSourceChange {
    relative: PathBuf,
    target: PathBuf,
    source: Option<String>,
    expected_hash: Option<String>,
}

fn collect_lua_sources(root: &Path, directory: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(directory)
        .with_context(|| format!("Failed to read harness directory '{}'", directory.display()))?
    {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_dir() {
            collect_lua_sources(root, &path, paths)?;
        } else if file_type.is_file()
            && path.extension().is_some_and(|extension| extension == "lua")
        {
            paths.push(path.strip_prefix(root)?.to_path_buf());
        }
    }
    Ok(())
}

fn normalize_source_path(raw: &str) -> Result<PathBuf> {
    if raw.is_empty() {
        anyhow::bail!("Harness source path must not be empty");
    }
    let mut normalized = PathBuf::new();
    for component in Path::new(raw).components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => normalized.push(part),
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                anyhow::bail!(
                    "Harness source path '{}' must stay under the harness root",
                    raw
                )
            }
        }
    }
    if normalized.as_os_str().is_empty()
        || !normalized
            .extension()
            .is_some_and(|extension| extension == "lua")
    {
        anyhow::bail!("Harness source path '{}' must identify a .lua file", raw);
    }
    Ok(normalized)
}

fn resolve_source_target(root: &Path, relative: &Path) -> Result<PathBuf> {
    crate::tools::is_safe_path(root, relative).map_err(|err| {
        anyhow!(
            "Unsafe harness source path '{}': {}",
            relative.display(),
            err
        )
    })
}

fn read_source(path: &Path) -> Result<String> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("Failed to inspect harness source '{}'", path.display()))?;
    if !metadata.is_file() {
        anyhow::bail!("Harness source '{}' is not a file", path.display());
    }
    if metadata.len() > MAX_HARNESS_SOURCE_BYTES as u64 {
        anyhow::bail!(
            "Harness source '{}' exceeds the {} byte limit",
            path.display(),
            MAX_HARNESS_SOURCE_BYTES
        );
    }
    std::fs::read_to_string(path).with_context(|| {
        format!(
            "Failed to read harness source '{}' as UTF-8",
            path.display()
        )
    })
}

fn source_entry(root: &Path, relative_or_target: &Path) -> Result<HarnessSourceEntry> {
    let target = if relative_or_target.starts_with(root) {
        relative_or_target.to_path_buf()
    } else {
        resolve_source_target(root, relative_or_target)?
    };
    let source = read_source(&target)?;
    let relative = target.strip_prefix(root)?;
    Ok(HarnessSourceEntry {
        path: path_string(relative)?,
        hash: source_hash(source.as_bytes()),
        bytes: source.len(),
    })
}

fn ensure_expected_hash(path: &Path, expected: Option<&str>, display: &str) -> Result<()> {
    let actual = if path.exists() {
        Some(source_hash(read_source(path)?.as_bytes()))
    } else {
        None
    };
    if actual.as_deref() != expected {
        return Err(HarnessSourceConflict(format!(
            "Harness source '{}' changed since it was read (expected {}, found {})",
            display,
            expected.unwrap_or("missing"),
            actual.as_deref().unwrap_or("missing")
        ))
        .into());
    }
    Ok(())
}

fn validate_source_size(source: Option<&str>, path: &str) -> Result<()> {
    if source.is_some_and(|source| source.len() > MAX_HARNESS_SOURCE_BYTES) {
        anyhow::bail!(
            "Harness source '{}' exceeds the {} byte limit",
            path,
            MAX_HARNESS_SOURCE_BYTES
        );
    }
    Ok(())
}

fn source_hash(source: &[u8]) -> String {
    let digest = Sha256::digest(source);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn path_string(path: &Path) -> Result<String> {
    path.to_str()
        .map(|path| path.replace(std::path::MAIN_SEPARATOR, "/"))
        .ok_or_else(|| {
            anyhow!(
                "Harness source path '{}' is not valid UTF-8",
                path.display()
            )
        })
}

fn temporary_source_path(target: &Path) -> PathBuf {
    let name = target
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("source.lua");
    target.with_file_name(format!(".{name}.turin-{}.tmp", uuid::Uuid::new_v4()))
}
