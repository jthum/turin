use anyhow::Result;
use async_trait::async_trait;

pub const CODE_INDEX_VECTOR_DIM: usize = 1536;

#[async_trait]
pub trait CodeEmbeddingProvider: Send + Sync {
    fn config_key(&self) -> String;

    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for text in texts {
            out.push(self.embed(text).await?);
        }
        Ok(out)
    }
}
