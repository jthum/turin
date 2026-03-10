use anyhow::{Context, Result};
use async_trait::async_trait;
use clap::{Args, ValueEnum};
use inference_sdk_core::{
    EmbeddingProvider as SdkEmbeddingProvider, EmbeddingRequest as SdkEmbeddingRequest,
};
use inference_sdk_registry::{ProviderInit, create_embedding_provider as create_sdk_provider};
use std::sync::Arc;
use turin_code_index::code_index_writer::CodeIndexBuildOptions;
use turin_code_index::embeddings::{CODE_INDEX_VECTOR_DIM, CodeEmbeddingProvider};

const DEFAULT_OPENAI_MODEL: &str = "text-embedding-3-small";
const DEFAULT_OPENAI_API_KEY_ENV: &str = "OPENAI_API_KEY";

#[derive(Copy, Clone, Debug, ValueEnum)]
pub(crate) enum EmbeddingProviderKind {
    Openai,
    Noop,
}

#[derive(Args, Debug, Clone)]
pub(crate) struct EmbeddingArgs {
    #[arg(long, value_enum)]
    pub embedding_provider: Option<EmbeddingProviderKind>,

    #[arg(long, default_value = DEFAULT_OPENAI_MODEL)]
    pub embedding_model: String,

    #[arg(long, default_value_t = CODE_INDEX_VECTOR_DIM)]
    pub embedding_dimensions: usize,

    #[arg(long)]
    pub embedding_base_url: Option<String>,

    #[arg(long, default_value = DEFAULT_OPENAI_API_KEY_ENV)]
    pub embedding_api_key_env: String,
}

impl Default for EmbeddingArgs {
    fn default() -> Self {
        Self {
            embedding_provider: None,
            embedding_model: DEFAULT_OPENAI_MODEL.to_string(),
            embedding_dimensions: CODE_INDEX_VECTOR_DIM,
            embedding_base_url: None,
            embedding_api_key_env: DEFAULT_OPENAI_API_KEY_ENV.to_string(),
        }
    }
}

pub(crate) fn build_embedding_provider(
    args: &EmbeddingArgs,
) -> Result<Option<Arc<dyn CodeEmbeddingProvider>>> {
    let provider =
        match args.embedding_provider {
            None => None,
            Some(kind) => Some(Arc::new(SdkCodeEmbeddingProvider::new(kind, args)?)
                as Arc<dyn CodeEmbeddingProvider>),
        };
    Ok(provider)
}

pub(crate) fn build_options(args: &EmbeddingArgs) -> Result<CodeIndexBuildOptions> {
    Ok(CodeIndexBuildOptions {
        embedding_provider: build_embedding_provider(args)?,
    })
}

struct SdkCodeEmbeddingProvider {
    provider: Arc<dyn SdkEmbeddingProvider>,
    config_key: String,
    model: String,
    dimensions: usize,
}

impl SdkCodeEmbeddingProvider {
    fn new(kind: EmbeddingProviderKind, args: &EmbeddingArgs) -> Result<Self> {
        let driver = match kind {
            EmbeddingProviderKind::Openai => "openai",
            EmbeddingProviderKind::Noop => "noop",
        };
        let api_key = match kind {
            EmbeddingProviderKind::Noop => String::new(),
            EmbeddingProviderKind::Openai => match std::env::var(&args.embedding_api_key_env) {
                Ok(value) => value,
                Err(_) if args.embedding_base_url.is_some() => String::new(),
                Err(_) => {
                    anyhow::bail!(
                        "embedding provider openai requires env var '{}' unless --embedding-base-url is set for a local OpenAI-compatible endpoint",
                        args.embedding_api_key_env
                    );
                }
            },
        };

        let mut init = ProviderInit::new(api_key);
        if let Some(base_url) = &args.embedding_base_url {
            init = init.with_base_url(base_url.clone());
        }
        let provider = create_sdk_provider(driver, &init)
            .with_context(|| format!("failed to initialize embedding provider '{}'", driver))?;

        Ok(Self {
            provider,
            config_key: format!(
                "{}:{}:{}:{}",
                driver,
                args.embedding_base_url.as_deref().unwrap_or("default"),
                args.embedding_model,
                args.embedding_dimensions
            ),
            model: args.embedding_model.clone(),
            dimensions: args.embedding_dimensions,
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
