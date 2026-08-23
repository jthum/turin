use anyhow::{Context, Result};
use async_trait::async_trait;
use inference_sdk_core::{
    EmbeddingProvider as SdkEmbeddingProvider, EmbeddingRequest as SdkEmbeddingRequest,
};
use inference_sdk_registry::{ProviderInit, create_embedding_provider as create_sdk_provider};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::kernel::config::{EmbeddingConfig, ProviderConfig, ProvidersConfig, TurinConfig};

/// A vector embedding of a text string.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    pub content: String,
    pub vector: Vec<f32>,
    pub model: String,
}

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    fn config_key(&self) -> String;

    fn dimensions(&self) -> usize;

    async fn embed(&self, text: &str) -> Result<Embedding>;

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            embeddings.push(self.embed(text).await?);
        }
        Ok(embeddings)
    }
}

pub struct SdkEmbeddingProviderAdapter {
    provider: Arc<dyn SdkEmbeddingProvider>,
    config_key: String,
    model: String,
    dimensions: usize,
}

impl SdkEmbeddingProviderAdapter {
    fn new(
        provider_name: &str,
        provider_config: Option<&ProviderConfig>,
        embedding_config: &EmbeddingConfig,
        config: &TurinConfig,
    ) -> Result<Self> {
        let driver = provider_config
            .map(|config| config.kind.as_str())
            .unwrap_or(provider_name);
        let api_key = resolve_api_key(driver, provider_config, config)?;

        let mut init = ProviderInit::new(api_key);
        if let Some(base_url) = provider_config.and_then(|config| config.base_url.as_ref()) {
            init = init.with_base_url(base_url.clone());
        }

        let provider = create_sdk_provider(driver, &init).with_context(|| {
            format!(
                "failed to initialize embedding provider '{}'",
                provider_name
            )
        })?;
        let config_key = build_config_key(driver, provider_config, embedding_config);

        Ok(Self {
            provider,
            config_key,
            model: embedding_config.model.clone(),
            dimensions: embedding_config.dimensions,
        })
    }
}

#[async_trait]
impl EmbeddingProvider for SdkEmbeddingProviderAdapter {
    fn config_key(&self) -> String {
        self.config_key.clone()
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn embed(&self, text: &str) -> Result<Embedding> {
        let response = self
            .provider
            .embed(
                SdkEmbeddingRequest::builder()
                    .model(self.model.clone())
                    .input(vec![text.to_string()])
                    .dimensions(self.dimensions as u32)
                    .build(),
                None,
            )
            .await?;

        let embedding = response
            .data
            .into_iter()
            .next()
            .context("embedding provider returned no embedding rows")?;
        validate_dimensions(&embedding.embedding, self.dimensions, &self.model)?;

        Ok(Embedding {
            content: text.to_string(),
            vector: embedding.embedding,
            model: response.model,
        })
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        let response = self
            .provider
            .embed(
                SdkEmbeddingRequest::builder()
                    .model(self.model.clone())
                    .input(texts.to_vec())
                    .dimensions(self.dimensions as u32)
                    .build(),
                None,
            )
            .await?;

        if response.data.len() != texts.len() {
            anyhow::bail!(
                "embedding provider returned {} embeddings for {} inputs",
                response.data.len(),
                texts.len()
            );
        }

        let mut rows = response.data;
        rows.sort_by_key(|row| row.index);
        let mut out = Vec::with_capacity(texts.len());
        for (index, row) in rows.into_iter().enumerate() {
            validate_dimensions(&row.embedding, self.dimensions, &self.model)?;
            out.push(Embedding {
                content: texts[index].clone(),
                vector: row.embedding,
                model: response.model.clone(),
            });
        }
        Ok(out)
    }
}

pub fn create_embedding_provider(
    embedding_config: &EmbeddingConfig,
    providers: &ProvidersConfig,
    config: &TurinConfig,
) -> Result<Arc<dyn EmbeddingProvider>> {
    let provider_name = embedding_config.provider.trim();
    if provider_name.is_empty() {
        anyhow::bail!("embeddings.provider must not be empty");
    }

    let provider_config = if provider_name == "noop" {
        None
    } else {
        Some(providers.get(provider_name).with_context(|| {
            format!(
                "Embedding provider '{}' not found in [providers] configuration",
                provider_name
            )
        })?)
    };

    Ok(Arc::new(SdkEmbeddingProviderAdapter::new(
        provider_name,
        provider_config,
        embedding_config,
        config,
    )?))
}

fn build_config_key(
    driver: &str,
    provider_config: Option<&ProviderConfig>,
    embedding_config: &EmbeddingConfig,
) -> String {
    let base_url = provider_config
        .and_then(|config| config.base_url.as_deref())
        .unwrap_or("default");
    format!(
        "{driver}:{base_url}:{}:{}",
        embedding_config.model, embedding_config.dimensions
    )
}

fn resolve_api_key(
    driver: &str,
    provider_config: Option<&ProviderConfig>,
    config: &TurinConfig,
) -> Result<String> {
    let Some(provider_config) = provider_config else {
        return Ok(String::new());
    };

    let base_url_present = provider_config.base_url.is_some();

    match provider_config.api_key_env.as_deref() {
        Some(env) => match config.environment_value(env) {
            Some(value) => Ok(value),
            None if base_url_present => Ok(String::new()),
            None => Err(anyhow::anyhow!("Environment variable '{}' not set", env)),
        },
        None if driver == "openai" && !base_url_present => Err(anyhow::anyhow!(
            "Embedding provider '{}' requires api_key_env unless base_url is set for a local OpenAI-compatible endpoint",
            driver
        )),
        None => Ok(String::new()),
    }
}

fn validate_dimensions(vector: &[f32], expected: usize, model: &str) -> Result<()> {
    if vector.len() != expected {
        anyhow::bail!(
            "embedding model '{}' returned {} dimensions; expected {}",
            model,
            vector.len(),
            expected
        );
    }
    Ok(())
}
