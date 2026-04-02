use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const DEFAULT_CONFIG_FILE: &str = ".turin/config.toml";

#[derive(Debug, Clone)]
pub(crate) struct LoadedTurinMapConfig {
    pub path: PathBuf,
    pub providers: HashMap<String, MapProviderConfig>,
    pub embeddings: Option<MapEmbeddingConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct MapEmbeddingConfig {
    pub provider: String,
    #[serde(default = "default_embedding_model")]
    pub model: String,
    #[serde(default = "default_embedding_dimensions")]
    pub dimensions: usize,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub(crate) struct MapProviderConfig {
    #[serde(rename = "type")]
    pub kind: String,
    pub api_key_env: Option<String>,
    pub base_url: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct RawTurinMapConfig {
    #[serde(default)]
    providers: HashMap<String, MapProviderConfig>,
    embeddings: Option<MapEmbeddingConfig>,
}

pub(crate) fn load_turin_map_config(
    cwd: &Path,
    explicit_path: Option<&Path>,
) -> Result<Option<LoadedTurinMapConfig>> {
    let path = match explicit_path {
        Some(path) => resolve_path(cwd, path),
        None => {
            let candidate = cwd.join(DEFAULT_CONFIG_FILE);
            if !candidate.exists() {
                return Ok(None);
            }
            candidate
        }
    };

    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read Turin config '{}'", path.display()))?;
    let parsed: RawTurinMapConfig = toml::from_str(&raw)
        .with_context(|| format!("failed to parse Turin config '{}'", path.display()))?;
    Ok(Some(LoadedTurinMapConfig {
        path,
        providers: parsed.providers,
        embeddings: parsed.embeddings,
    }))
}

fn resolve_path(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn default_embedding_model() -> String {
    "text-embedding-3-small".to_string()
}

fn default_embedding_dimensions() -> usize {
    turin_code_index::embeddings::CODE_INDEX_VECTOR_DIM
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values_match_runtime_expectations() {
        let raw = r#"
[providers.local_embeddings]
type = "openai"
base_url = "http://127.0.0.1:11434/v1"

[embeddings]
provider = "local_embeddings"
"#;

        let parsed: RawTurinMapConfig = toml::from_str(raw).expect("config parses");
        let embeddings = parsed.embeddings.expect("embeddings present");
        assert_eq!(embeddings.model, "text-embedding-3-small");
        assert_eq!(
            embeddings.dimensions,
            turin_code_index::embeddings::CODE_INDEX_VECTOR_DIM
        );
        assert_eq!(
            parsed
                .providers
                .get("local_embeddings")
                .and_then(|provider| provider.base_url.as_deref()),
            Some("http://127.0.0.1:11434/v1")
        );
    }
}
