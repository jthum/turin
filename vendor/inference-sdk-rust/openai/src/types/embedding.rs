use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, bon::Builder)]
pub struct EmbeddingRequest {
    /// Input text to get embeddings for, encoded as a string or array of tokens.
    pub input: Vec<String>,

    /// ID of the model to use.
    pub model: String,

    /// Optional output dimensions for providers/models that support down-projection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,

    /// The format to return the embeddings in. Can be either `float` or `base64`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,

    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}
