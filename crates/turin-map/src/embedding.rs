use anyhow::{Context, Result};
use async_trait::async_trait;
use clap::{Args, ValueEnum};
use inference_sdk_core::{
    EmbeddingProvider as SdkEmbeddingProvider, EmbeddingRequest as SdkEmbeddingRequest,
};
use inference_sdk_registry::{ProviderInit, create_embedding_provider as create_sdk_provider};
use std::borrow::Cow;
use std::sync::Arc;
use turin_code_index::code_index_writer::CodeIndexBuildOptions;
use turin_code_index::embeddings::{CODE_INDEX_VECTOR_DIM, CodeEmbeddingProvider};

use crate::config::{LoadedTurinMapConfig, MapEmbeddingConfig, MapProviderConfig};

const DEFAULT_OPENAI_MODEL: &str = "text-embedding-3-small";
const DEFAULT_OPENAI_API_KEY_ENV: &str = "OPENAI_API_KEY";

#[derive(Copy, Clone, Debug, ValueEnum)]
pub(crate) enum EmbeddingProviderKind {
    Openai,
    Noop,
}

#[derive(Args, Debug, Clone, Default)]
pub(crate) struct EmbeddingArgs {
    #[arg(
        long,
        value_enum,
        help = "Override the embedding driver for this run instead of using [embeddings] from turin.toml"
    )]
    pub embedding_provider: Option<EmbeddingProviderKind>,

    #[arg(long, help = "Embedding model identifier to use for this indexing run")]
    pub embedding_model: Option<String>,

    #[arg(
        long,
        help = "Embedding output dimensions; must match the model's actual output size"
    )]
    pub embedding_dimensions: Option<usize>,

    #[arg(
        long,
        help = "Base URL for a local or proxied OpenAI-compatible embeddings endpoint"
    )]
    pub embedding_base_url: Option<String>,

    #[arg(
        long,
        help = "Environment variable holding the embedding API key, when required"
    )]
    pub embedding_api_key_env: Option<String>,
}

pub(crate) fn build_embedding_provider(
    args: &EmbeddingArgs,
    config: Option<&LoadedTurinMapConfig>,
) -> Result<Option<Arc<dyn CodeEmbeddingProvider>>> {
    let provider = match resolve_embedding_profile(args, config)? {
        None => None,
        Some(profile) => {
            Some(Arc::new(SdkCodeEmbeddingProvider::new(profile)?) as Arc<dyn CodeEmbeddingProvider>)
        }
    };
    Ok(provider)
}

pub(crate) fn build_options(
    args: &EmbeddingArgs,
    config: Option<&LoadedTurinMapConfig>,
) -> Result<CodeIndexBuildOptions> {
    Ok(CodeIndexBuildOptions {
        embedding_provider: build_embedding_provider(args, config)?,
    })
}

#[derive(Debug, Clone)]
struct ResolvedEmbeddingProfile {
    driver: String,
    model: String,
    dimensions: usize,
    base_url: Option<String>,
    api_key_env: Option<String>,
}

struct SdkCodeEmbeddingProvider {
    provider: Arc<dyn SdkEmbeddingProvider>,
    config_key: String,
    model: String,
    dimensions: usize,
}

impl SdkCodeEmbeddingProvider {
    fn new(profile: ResolvedEmbeddingProfile) -> Result<Self> {
        let api_key = resolve_api_key(&profile)?;

        let mut init = ProviderInit::new(api_key);
        if let Some(base_url) = &profile.base_url {
            init = init.with_base_url(base_url.clone());
        }
        let provider = create_sdk_provider(&profile.driver, &init).with_context(|| {
            format!(
                "failed to initialize embedding provider '{}'",
                profile.driver
            )
        })?;

        Ok(Self {
            provider,
            config_key: format!(
                "{}:{}:{}:{}",
                profile.driver,
                profile.base_url.as_deref().unwrap_or("default"),
                profile.model,
                profile.dimensions
            ),
            model: profile.model,
            dimensions: profile.dimensions,
        })
    }
}

#[async_trait]
impl CodeEmbeddingProvider for SdkCodeEmbeddingProvider {
    fn config_key(&self) -> String {
        self.config_key.clone()
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let response = self
            .provider
            .embed(
                SdkEmbeddingRequest::builder()
                    .input(vec![text.to_string()])
                    .model(self.model.clone())
                    .dimensions(self.dimensions as u32)
                    .build(),
                None,
            )
            .await?;

        let vector = response
            .data
            .first()
            .map(|row| row.embedding.clone())
            .context("no embedding data returned from provider")?;
        if vector.len() != self.dimensions {
            anyhow::bail!(
                "embedding model '{}' returned {} dimensions; expected {}",
                self.model,
                vector.len(),
                self.dimensions
            );
        }
        Ok(vector)
    }
}

fn resolve_embedding_profile(
    args: &EmbeddingArgs,
    config: Option<&LoadedTurinMapConfig>,
) -> Result<Option<ResolvedEmbeddingProfile>> {
    let config_embeddings = config.and_then(|loaded| loaded.embeddings.as_ref());
    let has_cli_embedding_overrides = args.has_cli_embedding_overrides();
    let has_cli_provider_overrides = args.embedding_provider.is_some();

    if !has_cli_embedding_overrides && config_embeddings.is_none() {
        return Ok(None);
    }

    let model = args
        .embedding_model
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_string())
        .or_else(|| config_embeddings.map(|embedding| embedding.model.clone()))
        .unwrap_or_else(|| DEFAULT_OPENAI_MODEL.to_string());

    let dimensions = args
        .embedding_dimensions
        .or_else(|| config_embeddings.map(|embedding| embedding.dimensions))
        .unwrap_or(CODE_INDEX_VECTOR_DIM);
    if dimensions == 0 {
        anyhow::bail!("embedding dimensions must be greater than 0");
    }

    let driver = resolve_driver(args, config_embeddings, config)?.to_string();
    let provider_defaults = config_embeddings
        .and_then(|embedding| provider_defaults_for(driver.as_str(), embedding, config).transpose())
        .transpose()?;
    let base_url = normalize_optional(
        args.embedding_base_url
            .as_ref()
            .map(|value| Cow::Borrowed(value.as_str()))
            .or_else(|| {
                provider_defaults
                    .as_ref()
                    .and_then(|provider| provider.base_url.as_deref().map(Cow::Borrowed))
            }),
    );
    let api_key_env = normalize_optional(
        args.embedding_api_key_env
            .as_ref()
            .map(|value| Cow::Borrowed(value.as_str()))
            .or_else(|| {
                provider_defaults
                    .as_ref()
                    .and_then(|provider| provider.api_key_env.as_deref().map(Cow::Borrowed))
            })
            .or_else(|| {
                if driver == "openai" {
                    Some(Cow::Borrowed(DEFAULT_OPENAI_API_KEY_ENV))
                } else {
                    None
                }
            }),
    );

    if !has_cli_provider_overrides && config_embeddings.is_none() && driver == "openai" {
        // CLI-only embedding overrides without an explicit provider default to the OpenAI-compatible driver.
    }

    Ok(Some(ResolvedEmbeddingProfile {
        driver,
        model,
        dimensions,
        base_url,
        api_key_env,
    }))
}

fn resolve_driver<'a>(
    args: &EmbeddingArgs,
    config_embeddings: Option<&'a MapEmbeddingConfig>,
    config: Option<&'a LoadedTurinMapConfig>,
) -> Result<&'a str> {
    if let Some(kind) = args.embedding_provider {
        return Ok(match kind {
            EmbeddingProviderKind::Openai => "openai",
            EmbeddingProviderKind::Noop => "noop",
        });
    }

    if let Some(embedding) = config_embeddings {
        let provider_name = embedding.provider.trim();
        if provider_name.is_empty() {
            anyhow::bail!("embeddings.provider must not be empty");
        }
        if provider_name == "noop" {
            return Ok("noop");
        }
        let loaded = config.context("embedding config requested without loaded Turin config")?;
        let provider = loaded.providers.get(provider_name).with_context(|| {
            format!(
                "embeddings.provider '{}' not found in [providers] of '{}'",
                provider_name,
                loaded.path.display()
            )
        })?;
        let driver = provider.kind.trim();
        if driver.is_empty() {
            anyhow::bail!(
                "providers.{}.type must not be empty in '{}'",
                provider_name,
                loaded.path.display()
            );
        }
        return Ok(driver);
    }

    Ok("openai")
}

fn provider_defaults_for<'a>(
    driver: &str,
    embeddings: &'a MapEmbeddingConfig,
    config: Option<&'a LoadedTurinMapConfig>,
) -> Result<Option<&'a MapProviderConfig>> {
    let provider_name = embeddings.provider.trim();
    if provider_name.is_empty() || provider_name == "noop" {
        return Ok(None);
    }

    let loaded = config.context("embedding defaults requested without loaded Turin config")?;
    let provider = loaded.providers.get(provider_name).with_context(|| {
        format!(
            "embeddings.provider '{}' not found in [providers] of '{}'",
            provider_name,
            loaded.path.display()
        )
    })?;
    if provider.kind.trim() == driver {
        Ok(Some(provider))
    } else {
        Ok(None)
    }
}

fn normalize_optional(value: Option<Cow<'_, str>>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn resolve_api_key(profile: &ResolvedEmbeddingProfile) -> Result<String> {
    if profile.driver == "noop" {
        return Ok(String::new());
    }

    match profile.api_key_env.as_deref() {
        Some(env) => match std::env::var(env) {
            Ok(value) => Ok(value),
            Err(_) if profile.base_url.is_some() => Ok(String::new()),
            Err(_) => Err(anyhow::anyhow!(
                "embedding provider {} requires env var '{}' unless --embedding-base-url is set for a local OpenAI-compatible endpoint",
                profile.driver,
                env
            )),
        },
        None if profile.driver == "openai" && profile.base_url.is_none() => Err(anyhow::anyhow!(
            "embedding provider openai requires an API key env unless --embedding-base-url is set for a local OpenAI-compatible endpoint"
        )),
        None => Ok(String::new()),
    }
}

impl EmbeddingArgs {
    fn has_cli_embedding_overrides(&self) -> bool {
        self.embedding_provider.is_some()
            || self.embedding_model.is_some()
            || self.embedding_dimensions.is_some()
            || self.embedding_base_url.is_some()
            || self.embedding_api_key_env.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{LoadedTurinMapConfig, MapEmbeddingConfig, MapProviderConfig};
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn config_with_local_embeddings() -> LoadedTurinMapConfig {
        let mut providers = HashMap::new();
        providers.insert(
            "local_embeddings".to_string(),
            MapProviderConfig {
                kind: "openai".to_string(),
                api_key_env: None,
                base_url: Some("http://127.0.0.1:11434/v1".to_string()),
            },
        );
        LoadedTurinMapConfig {
            path: PathBuf::from("turin.toml"),
            providers,
            embeddings: Some(MapEmbeddingConfig {
                provider: "local_embeddings".to_string(),
                model: "nomic-embed-text".to_string(),
                dimensions: 768,
            }),
        }
    }

    #[test]
    fn config_embeddings_enable_semantic_indexing_without_cli_flags() {
        let config = config_with_local_embeddings();
        let resolved = resolve_embedding_profile(&EmbeddingArgs::default(), Some(&config))
            .expect("profile resolves")
            .expect("profile present");

        assert_eq!(resolved.driver, "openai");
        assert_eq!(resolved.model, "nomic-embed-text");
        assert_eq!(resolved.dimensions, 768);
        assert_eq!(
            resolved.base_url.as_deref(),
            Some("http://127.0.0.1:11434/v1")
        );
        assert_eq!(resolved.api_key_env.as_deref(), Some("OPENAI_API_KEY"));
    }

    #[test]
    fn cli_overrides_config_fields_without_requiring_provider_duplication() {
        let config = config_with_local_embeddings();
        let resolved = resolve_embedding_profile(
            &EmbeddingArgs {
                embedding_dimensions: Some(384),
                ..EmbeddingArgs::default()
            },
            Some(&config),
        )
        .expect("profile resolves")
        .expect("profile present");

        assert_eq!(resolved.driver, "openai");
        assert_eq!(resolved.model, "nomic-embed-text");
        assert_eq!(resolved.dimensions, 384);
        assert_eq!(
            resolved.base_url.as_deref(),
            Some("http://127.0.0.1:11434/v1")
        );
    }

    #[test]
    fn cli_only_overrides_default_to_openai_compatible_provider() {
        let resolved = resolve_embedding_profile(
            &EmbeddingArgs {
                embedding_base_url: Some("http://127.0.0.1:11434/v1".to_string()),
                embedding_model: Some("nomic-embed-text".to_string()),
                embedding_dimensions: Some(768),
                ..EmbeddingArgs::default()
            },
            None,
        )
        .expect("profile resolves")
        .expect("profile present");

        assert_eq!(resolved.driver, "openai");
        assert_eq!(
            resolved.base_url.as_deref(),
            Some("http://127.0.0.1:11434/v1")
        );
        assert_eq!(resolved.api_key_env.as_deref(), Some("OPENAI_API_KEY"));
    }
}
