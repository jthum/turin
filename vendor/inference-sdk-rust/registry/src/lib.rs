use std::collections::HashMap;
use std::sync::Arc;

use futures_util::future::BoxFuture;
use inference_sdk_core::{
    EmbeddingData, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
    InferenceProvider, SdkError,
};
use thiserror::Error;

type FactoryFn =
    dyn Fn(&ProviderInit) -> Result<Arc<dyn InferenceProvider>, RegistryError> + Send + Sync;
type EmbeddingFactoryFn =
    dyn Fn(&ProviderInit) -> Result<Arc<dyn EmbeddingProvider>, RegistryError> + Send + Sync;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProviderInit {
    pub api_key: String,
    pub base_url: Option<String>,
}

impl ProviderInit {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: None,
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }
}

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("unknown provider driver '{driver}' (available: {available:?})")]
    UnknownDriver {
        driver: String,
        available: Vec<String>,
    },
    #[error("failed to initialize provider for driver '{driver}': {source}")]
    Init {
        driver: String,
        #[source]
        source: SdkError,
    },
}

#[derive(Clone, Default)]
pub struct ProviderRegistry {
    factories: HashMap<String, Arc<FactoryFn>>,
}

#[derive(Clone, Default)]
pub struct EmbeddingRegistry {
    factories: HashMap<String, Arc<EmbeddingFactoryFn>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_builtin_drivers() -> Self {
        let mut registry = Self::new();
        registry.register("openai", Arc::new(openai_factory));
        registry.register("anthropic", Arc::new(anthropic_factory));
        registry
    }

    pub fn register(
        &mut self,
        driver: impl Into<String>,
        factory: Arc<FactoryFn>,
    ) -> Option<Arc<FactoryFn>> {
        self.factories
            .insert(normalize_driver(driver.into()), factory)
    }

    pub fn drivers(&self) -> Vec<String> {
        let mut drivers = self.factories.keys().cloned().collect::<Vec<_>>();
        drivers.sort();
        drivers
    }

    pub fn create(
        &self,
        driver: &str,
        init: &ProviderInit,
    ) -> Result<Arc<dyn InferenceProvider>, RegistryError> {
        let key = normalize_driver(driver.to_string());
        let factory = self
            .factories
            .get(&key)
            .ok_or_else(|| RegistryError::UnknownDriver {
                driver: driver.to_string(),
                available: self.drivers(),
            })?;
        factory(init)
    }
}

pub fn create_provider(
    driver: &str,
    init: &ProviderInit,
) -> Result<Arc<dyn InferenceProvider>, RegistryError> {
    ProviderRegistry::with_builtin_drivers().create(driver, init)
}

impl EmbeddingRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_builtin_drivers() -> Self {
        let mut registry = Self::new();
        registry.register("openai", Arc::new(openai_embedding_factory));
        registry.register("noop", Arc::new(noop_embedding_factory));
        registry
    }

    pub fn register(
        &mut self,
        driver: impl Into<String>,
        factory: Arc<EmbeddingFactoryFn>,
    ) -> Option<Arc<EmbeddingFactoryFn>> {
        self.factories
            .insert(normalize_driver(driver.into()), factory)
    }

    pub fn drivers(&self) -> Vec<String> {
        let mut drivers = self.factories.keys().cloned().collect::<Vec<_>>();
        drivers.sort();
        drivers
    }

    pub fn create(
        &self,
        driver: &str,
        init: &ProviderInit,
    ) -> Result<Arc<dyn EmbeddingProvider>, RegistryError> {
        let key = normalize_driver(driver.to_string());
        let factory = self
            .factories
            .get(&key)
            .ok_or_else(|| RegistryError::UnknownDriver {
                driver: driver.to_string(),
                available: self.drivers(),
            })?;
        factory(init)
    }
}

pub fn create_embedding_provider(
    driver: &str,
    init: &ProviderInit,
) -> Result<Arc<dyn EmbeddingProvider>, RegistryError> {
    EmbeddingRegistry::with_builtin_drivers().create(driver, init)
}

fn normalize_driver(driver: String) -> String {
    driver.trim().to_ascii_lowercase()
}

fn openai_factory(init: &ProviderInit) -> Result<Arc<dyn InferenceProvider>, RegistryError> {
    let mut config = openai_sdk::ClientConfig::new(init.api_key.clone()).map_err(|source| {
        RegistryError::Init {
            driver: "openai".to_string(),
            source,
        }
    })?;

    if let Some(base_url) = &init.base_url {
        config = config.with_base_url(base_url.clone());
    }

    let client = openai_sdk::Client::from_config(config).map_err(|source| RegistryError::Init {
        driver: "openai".to_string(),
        source,
    })?;
    Ok(Arc::new(client))
}

fn anthropic_factory(init: &ProviderInit) -> Result<Arc<dyn InferenceProvider>, RegistryError> {
    let mut config = anthropic_sdk::ClientConfig::new(init.api_key.clone()).map_err(|source| {
        RegistryError::Init {
            driver: "anthropic".to_string(),
            source,
        }
    })?;

    if let Some(base_url) = &init.base_url {
        config = config.with_base_url(base_url.clone());
    }

    let client =
        anthropic_sdk::Client::from_config(config).map_err(|source| RegistryError::Init {
            driver: "anthropic".to_string(),
            source,
        })?;
    Ok(Arc::new(client))
}

fn openai_embedding_factory(
    init: &ProviderInit,
) -> Result<Arc<dyn EmbeddingProvider>, RegistryError> {
    let mut config = openai_sdk::ClientConfig::new(init.api_key.clone()).map_err(|source| {
        RegistryError::Init {
            driver: "openai".to_string(),
            source,
        }
    })?;

    if let Some(base_url) = &init.base_url {
        config = config.with_base_url(base_url.clone());
    }

    let client = openai_sdk::Client::from_config(config).map_err(|source| RegistryError::Init {
        driver: "openai".to_string(),
        source,
    })?;
    Ok(Arc::new(client))
}

fn noop_embedding_factory(
    _init: &ProviderInit,
) -> Result<Arc<dyn EmbeddingProvider>, RegistryError> {
    Ok(Arc::new(NoOpEmbeddingProvider))
}

struct NoOpEmbeddingProvider;

impl EmbeddingProvider for NoOpEmbeddingProvider {
    fn embed<'a>(
        &'a self,
        request: EmbeddingRequest,
        _options: Option<inference_sdk_core::RequestOptions>,
    ) -> BoxFuture<'a, Result<EmbeddingResponse, SdkError>> {
        Box::pin(async move {
            let dims = request.dimensions.unwrap_or(1536) as usize;
            let data = request
                .input
                .into_iter()
                .enumerate()
                .map(|(index, _)| EmbeddingData {
                    embedding: vec![0.001; dims],
                    index,
                })
                .collect();

            Ok(EmbeddingResponse {
                data,
                model: request.model,
                usage: Some(EmbeddingUsage {
                    input_tokens: 0,
                    total_tokens: 0,
                }),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_registry_contains_openai_and_anthropic() {
        let drivers = ProviderRegistry::with_builtin_drivers().drivers();
        assert_eq!(drivers, vec!["anthropic".to_string(), "openai".to_string()]);
    }

    #[test]
    fn unknown_driver_error_lists_available_drivers() {
        let registry = ProviderRegistry::with_builtin_drivers();
        let err = match registry.create("unknown", &ProviderInit::new("test-key")) {
            Ok(_) => panic!("unknown driver should fail"),
            Err(err) => err,
        };

        match err {
            RegistryError::UnknownDriver { available, .. } => {
                assert_eq!(
                    available,
                    vec!["anthropic".to_string(), "openai".to_string()]
                );
            }
            other => panic!("unexpected error variant: {other}"),
        }
    }

    #[test]
    fn create_openai_provider_succeeds() {
        let provider = create_provider("openai", &ProviderInit::new("test-key"));
        assert!(provider.is_ok());
    }

    #[test]
    fn builtin_embedding_registry_contains_noop_and_openai() {
        let drivers = EmbeddingRegistry::with_builtin_drivers().drivers();
        assert_eq!(drivers, vec!["noop".to_string(), "openai".to_string()]);
    }

    #[test]
    fn create_noop_embedding_provider_succeeds() {
        let provider = create_embedding_provider("noop", &ProviderInit::new(""));
        assert!(provider.is_ok());
    }
}
