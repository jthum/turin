use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ContextPersistenceParams;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MemoryListParams {
    #[serde(default)]
    pub persistence: Option<ContextPersistenceParams>,
    #[serde(default)]
    pub scope_kind: Option<String>,
    #[serde(default)]
    pub scope_key: Option<String>,
    #[serde(default)]
    pub include_superseded: bool,
    #[serde(default)]
    pub limit: Option<u32>,
    #[serde(default)]
    pub offset: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MemoryList {
    pub memories: Vec<MemoryDetail>,
    pub scopes: Vec<MemoryScopeDetail>,
    pub total: u64,
    pub offset: u32,
    pub limit: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MemoryDetail {
    pub public_id: String,
    pub scope_kind: String,
    pub scope_key: String,
    pub content: String,
    #[serde(default)]
    pub metadata: Option<Value>,
    pub storage: String,
    #[serde(default)]
    pub embedding_key: Option<String>,
    #[serde(default)]
    pub embedding_dimensions: Option<u32>,
    pub weight: f64,
    pub retrieval_count: u64,
    #[serde(default)]
    pub last_retrieved_at: Option<String>,
    #[serde(default)]
    pub superseded_at: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MemoryScopeDetail {
    pub scope_kind: String,
    pub scope_key: String,
    pub count: u64,
}
