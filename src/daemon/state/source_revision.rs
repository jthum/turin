use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

use super::DaemonWatchPaths;

pub(super) type SourceRevision = [u8; 32];

pub(super) fn calculate_source_revision(
    config_path: &Path,
    watch_paths: &DaemonWatchPaths,
) -> Result<SourceRevision> {
    let mut hasher = Sha256::new();
    hash_file(&mut hasher, config_path, config_path)?;
    for root in [&watch_paths.agents_dir, &watch_paths.harnesses_dir] {
        hash_tree(&mut hasher, root, root)?;
    }
    Ok(hasher.finalize().into())
}

fn hash_tree(hasher: &mut Sha256, root: &Path, path: &Path) -> Result<()> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            hash_entry(hasher, root, path, b'm');
            return Ok(());
        }
        Err(err) => {
            return Err(err).with_context(|| format!("Failed to inspect '{}'", path.display()));
        }
    };

    if metadata.file_type().is_symlink() {
        hash_entry(hasher, root, path, b'l');
        let target = fs::read_link(path)
            .with_context(|| format!("Failed to read symlink '{}'", path.display()))?;
        hasher.update(target.as_os_str().as_encoded_bytes());
        return Ok(());
    }
    if metadata.is_file() {
        return hash_file(hasher, root, path);
    }

    hash_entry(hasher, root, path, b'd');
    let mut children = fs::read_dir(path)
        .with_context(|| format!("Failed to read registry directory '{}'", path.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    children.sort_by_key(|entry| entry.file_name());
    for child in children {
        hash_tree(hasher, root, &child.path())?;
    }
    Ok(())
}

fn hash_file(hasher: &mut Sha256, root: &Path, path: &Path) -> Result<()> {
    hash_entry(hasher, root, path, b'f');
    if is_runtime_source(path) {
        let bytes = fs::read(path)
            .with_context(|| format!("Failed to read runtime source '{}'", path.display()))?;
        hasher.update(bytes);
    }
    Ok(())
}

fn hash_entry(hasher: &mut Sha256, root: &Path, path: &Path, kind: u8) {
    hasher.update([kind]);
    let relative = path.strip_prefix(root).unwrap_or(path);
    hasher.update(relative.as_os_str().as_encoded_bytes());
    hasher.update([0]);
}

fn is_runtime_source(path: &Path) -> bool {
    path.file_name().is_some_and(|name| name == "config.toml")
        || path.extension().is_some_and(|extension| extension == "lua")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn watch_paths(root: &Path) -> DaemonWatchPaths {
        DaemonWatchPaths {
            config_path: root.join("config.toml"),
            agents_dir: root.join("agents"),
            harnesses_dir: root.join("harnesses"),
        }
    }

    #[test]
    fn revision_tracks_runtime_sources_and_registry_shape() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let paths = watch_paths(temp.path());
        fs::write(&paths.config_path, "[agent]\n")?;
        fs::create_dir_all(paths.agents_dir.join("reviewer/harness"))?;
        fs::write(
            paths.agents_dir.join("reviewer/config.toml"),
            "model = \"test\"\n",
        )?;
        fs::write(
            paths.agents_dir.join("reviewer/harness/main.lua"),
            "return {}\n",
        )?;

        let initial = calculate_source_revision(&paths.config_path, &paths)?;
        assert_eq!(
            initial,
            calculate_source_revision(&paths.config_path, &paths)?
        );

        fs::write(
            paths.agents_dir.join("reviewer/harness/main.lua"),
            "return { changed = true }\n",
        )?;
        assert_ne!(
            initial,
            calculate_source_revision(&paths.config_path, &paths)?
        );

        Ok(())
    }

    #[test]
    fn revision_ignores_unread_non_source_file_contents() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let paths = watch_paths(temp.path());
        fs::write(&paths.config_path, "[agent]\n")?;
        fs::create_dir_all(&paths.harnesses_dir)?;
        let asset = paths.harnesses_dir.join("notes.txt");
        fs::write(&asset, "one")?;
        let initial = calculate_source_revision(&paths.config_path, &paths)?;

        fs::write(asset, "two")?;
        assert_eq!(
            initial,
            calculate_source_revision(&paths.config_path, &paths)?
        );
        Ok(())
    }
}
