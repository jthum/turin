use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct CodeIndexSemanticStatus {
    pub embedded_chunks: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_dimensions: Option<usize>,
}

impl CodeIndexSemanticStatus {
    pub fn disabled() -> Self {
        Self::default()
    }

    pub fn enabled(
        embedded_chunks: u64,
        embedding_key: String,
        embedding_dimensions: usize,
    ) -> Self {
        Self {
            embedded_chunks,
            embedding_key: Some(embedding_key),
            embedding_dimensions: Some(embedding_dimensions),
        }
    }
}
