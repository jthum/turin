use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CodeIndexVectorFormat {
    #[serde(rename = "float32")]
    Float32,
    #[serde(rename = "float8")]
    Float8,
}

impl CodeIndexVectorFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Float8 => "float8",
        }
    }

    pub fn from_db(value: &str) -> Result<Self> {
        match value {
            "float32" => Ok(Self::Float32),
            "float8" => Ok(Self::Float8),
            other => bail!("unsupported code index vector format '{other}'"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct CodeIndexSemanticStatus {
    pub embedded_chunks: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_dimensions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_format: Option<CodeIndexVectorFormat>,
}

impl CodeIndexSemanticStatus {
    pub fn disabled() -> Self {
        Self::default()
    }

    pub fn enabled(
        embedded_chunks: u64,
        embedding_key: String,
        embedding_dimensions: usize,
        vector_format: CodeIndexVectorFormat,
    ) -> Self {
        Self {
            embedded_chunks,
            embedding_key: Some(embedding_key),
            embedding_dimensions: Some(embedding_dimensions),
            vector_format: Some(vector_format),
        }
    }
}
