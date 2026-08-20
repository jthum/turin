use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;
use serde::de::DeserializeOwned;

pub(crate) async fn read_json<T>(path: &Path) -> Result<T>
where
    T: Default + DeserializeOwned,
{
    let _lock = acquire_lock(path).await?;
    read_json_unlocked(path).await
}

#[cfg(test)]
pub(crate) async fn write_json<T>(path: &Path, value: &T) -> Result<()>
where
    T: Serialize,
{
    let _lock = acquire_lock(path).await?;
    write_json_unlocked(path, value).await
}

pub(crate) async fn update_json<T, R>(
    path: &Path,
    update: impl FnOnce(&mut T) -> Result<R>,
) -> Result<R>
where
    T: Default + DeserializeOwned + Serialize,
{
    let _lock = acquire_lock(path).await?;
    let mut value = read_json_unlocked(path).await?;
    let result = update(&mut value)?;
    write_json_unlocked(path, &value).await?;
    Ok(result)
}

pub(crate) async fn update_json_if<T, R>(
    path: &Path,
    update: impl FnOnce(&mut T) -> Result<(R, bool)>,
) -> Result<R>
where
    T: Default + DeserializeOwned + Serialize,
{
    let _lock = acquire_lock(path).await?;
    let mut value = read_json_unlocked(path).await?;
    let (result, changed) = update(&mut value)?;
    if changed {
        write_json_unlocked(path, &value).await?;
    }
    Ok(result)
}

async fn acquire_lock(path: &Path) -> Result<File> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let lock_path = lock_path(path);
    tokio::task::spawn_blocking(move || {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .with_context(|| format!("Failed to open state lock '{}'", lock_path.display()))?;
        file.lock()
            .with_context(|| format!("Failed to lock channel state '{}'", lock_path.display()))?;
        Ok(file)
    })
    .await
    .context("Channel state lock task failed")?
}

async fn read_json_unlocked<T>(path: &Path) -> Result<T>
where
    T: Default + DeserializeOwned,
{
    if !path.exists() {
        return Ok(T::default());
    }
    let raw = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("Failed to read '{}'", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("Failed to parse '{}'", path.display()))
}

async fn write_json_unlocked<T>(path: &Path, value: &T) -> Result<()>
where
    T: Serialize,
{
    let tmp = path.with_extension("json.tmp");
    let body = serde_json::to_string_pretty(value)?;
    tokio::fs::write(&tmp, body)
        .await
        .with_context(|| format!("Failed to stage '{}'", tmp.display()))?;
    tokio::fs::rename(&tmp, path)
        .await
        .with_context(|| format!("Failed to replace '{}'", path.display()))
}

fn lock_path(path: &Path) -> PathBuf {
    path.with_extension("lock")
}
