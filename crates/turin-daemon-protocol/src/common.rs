use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct NoParams {}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EntityIdParams {
    pub id: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct StoreTargetParams {
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub alias: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct ContextPersistenceParams {
    #[serde(default)]
    pub state: Option<StoreTargetParams>,
    #[serde(default)]
    pub store: Option<StoreTargetParams>,
}
