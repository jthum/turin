use anyhow::{Context, Result};
use async_trait::async_trait;
use clap::{Args, ValueEnum};
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

    #[arg(long, default_value = DEFAULT_OPENAI_API_KEY_ENV)]
    pub embedding_api_key_env: String,
}

impl Default for EmbeddingArgs {
    fn default() -> Self {
        Self {
            embedding_provider: None,
            embedding_model: DEFAULT_OPENAI_MODEL.to_string(),
            embedding_api_key_env: DEFAULT_OPENAI_API_KEY_ENV.to_string(),
        }
    }
}

pub(crate) fn build_embedding_provider(
    args: &EmbeddingArgs,
) -> Result<Option<Arc<dyn CodeEmbeddingProvider>>> {
    let provider = match args.embedding_provider {
        None => None,
        Some(EmbeddingProviderKind::Noop) => {
            Some(Arc::new(NoOpCodeEmbeddingProvider) as Arc<dyn CodeEmbeddingProvider>)
        }
        Some(EmbeddingProviderKind::Openai) => {
            let api_key = std::env::var(&args.embedding_api_key_env).with_context(|| {
                format!(
                    "embedding provider openai requires env var '{}'",
                    args.embedding_api_key_env
                )
            })?;
            let provider = OpenAIEmbeddingProvider::new(api_key, args.embedding_model.clone())?;
            Some(Arc::new(provider) as Arc<dyn CodeEmbeddingProvider>)
        }
    };
    Ok(provider)
}

pub(crate) fn build_options(args: &EmbeddingArgs) -> Result<CodeIndexBuildOptions> {
    Ok(CodeIndexBuildOptions {
        embedding_provider: build_embedding_provider(args)?,
    })
}

struct OpenAIEmbeddingProvider {
    client: openai_sdk::Client,
    model: String,
}

impl OpenAIEmbeddingProvider {
    fn new(api_key: String, model: String) -> Result<Self> {
        Ok(Self {
            client: openai_sdk::Client::new(api_key)?,
            model,
        })
    }
}

#[async_trait]
impl CodeEmbeddingProvider for OpenAIEmbeddingProvider {
    fn config_key(&self) -> String {
        format!("openai:{}", self.model)
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let request = openai_sdk::EmbeddingRequest::builder()
            .input(text.to_string())
            .model(self.model.clone())
            .build();

        let response = self.client.embeddings().create(request).await?;
        let embedding = response
            .data
            .first()
            .map(|row| row.embedding.clone())
            .context("no embedding data returned from OpenAI")?;
        if embedding.len() != CODE_INDEX_VECTOR_DIM {
            anyhow::bail!(
                "openai embedding model '{}' returned {} dimensions; expected {}",
                self.model,
                embedding.len(),
                CODE_INDEX_VECTOR_DIM
            );
        }
        Ok(embedding)
    }
}

struct NoOpCodeEmbeddingProvider;

#[async_trait]
impl CodeEmbeddingProvider for NoOpCodeEmbeddingProvider {
    fn config_key(&self) -> String {
        "noop:1536".to_string()
    }

    async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
        Ok(vec![0.001; CODE_INDEX_VECTOR_DIM])
    }
}
