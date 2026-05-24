use anyhow::{Context, Result, bail};
use ignore::{DirEntry, WalkBuilder};
use sha2::{Digest, Sha256};
use std::path::{Component, Path, PathBuf};

use super::IndexableFileContent;

const MAX_FILE_BYTES: usize = 512 * 1024;

pub(super) fn collect_indexable_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut builder = WalkBuilder::new(root);
    builder
        .hidden(false)
        .require_git(false)
        .sort_by_file_path(|left, right| left.cmp(right))
        .filter_entry(should_walk_entry);

    let mut out = Vec::new();
    for entry in builder.build() {
        let entry = entry?;
        if !entry
            .file_type()
            .is_some_and(|file_type| file_type.is_file())
        {
            continue;
        }
        let path = entry.into_path();
        if detect_language(&path).is_some() {
            out.push(path);
        }
    }
    Ok(out)
}

pub(super) fn normalize_relative_path(root: &Path, path: &Path) -> Result<String> {
    let relative = if path.is_absolute() {
        let stripped = path
            .strip_prefix(root)
            .with_context(|| format!("'{}' is outside '{}'", path.display(), root.display()))?
            .to_path_buf();
        normalize_relative_components(root, path, &stripped)?
    } else {
        normalize_relative_components(root, path, path)?
    };
    let normalized = relative
        .to_str()
        .with_context(|| format!("path '{}' is not valid UTF-8", relative.display()))?
        .replace('\\', "/");
    if normalized.is_empty() {
        bail!("path must not be empty");
    }
    Ok(normalized)
}

fn normalize_relative_components(root: &Path, original: &Path, relative: &Path) -> Result<PathBuf> {
    let mut normalized = PathBuf::new();
    for component in relative.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => normalized.push(part),
            Component::ParentDir => {
                if !normalized.pop() {
                    bail!(
                        "path '{}' escapes root '{}'",
                        original.display(),
                        root.display()
                    );
                }
            }
            Component::Prefix(_) | Component::RootDir => {
                bail!(
                    "path '{}' is outside root '{}'",
                    original.display(),
                    root.display()
                );
            }
        }
    }
    Ok(normalized)
}

pub(super) fn read_indexable_file(
    relative_path: String,
    file: &Path,
) -> Result<Option<IndexableFileContent>> {
    let language = match detect_language(file) {
        Some(language) => language,
        None => return Ok(None),
    };
    let bytes = std::fs::read(file)?;
    if bytes.len() > MAX_FILE_BYTES {
        return Ok(None);
    }
    let content_hash = file_content_hash(&bytes);
    let content = match String::from_utf8(bytes) {
        Ok(content) => content,
        Err(_) => return Ok(None),
    };
    if content.trim().is_empty() {
        return Ok(None);
    }

    Ok(Some(IndexableFileContent {
        path: relative_path,
        language: language.to_string(),
        content_hash,
        content,
    }))
}

fn should_walk_entry(entry: &DirEntry) -> bool {
    if entry.depth() == 0 {
        return true;
    }
    match entry.path().file_name().and_then(|name| name.to_str()) {
        Some(name) => !should_skip_dir(name),
        None => true,
    }
}

fn should_skip_dir(name: &str) -> bool {
    matches!(
        name,
        ".git"
            | ".turin"
            | ".next"
            | ".nuxt"
            | ".svelte-kit"
            | "build"
            | "dist"
            | "node_modules"
            | "target"
            | "vendor"
    )
}

fn detect_language(path: &Path) -> Option<&'static str> {
    match path.extension().and_then(|ext| ext.to_str())? {
        "rs" => Some("rust"),
        "lua" => Some("lua"),
        "py" => Some("python"),
        "go" => Some("go"),
        "php" => Some("php"),
        "js" | "mjs" | "cjs" => Some("javascript"),
        "ts" | "tsx" => Some("typescript"),
        _ => None,
    }
}

fn file_content_hash(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}
