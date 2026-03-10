use anyhow::Result;
use async_trait::async_trait;

pub const DEFAULT_CODE_INDEX_VECTOR_DIM: usize = 1536;
pub const CODE_INDEX_VECTOR_DIM: usize = DEFAULT_CODE_INDEX_VECTOR_DIM;

#[async_trait]
pub trait CodeEmbeddingProvider: Send + Sync {
    fn config_key(&self) -> String;

    fn dimensions(&self) -> usize;

    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for text in texts {
            out.push(self.embed(text).await?);
        }
        Ok(out)
    }
}
