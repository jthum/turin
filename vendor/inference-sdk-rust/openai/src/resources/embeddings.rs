use crate::client::Client;
use crate::types::embedding::{EmbeddingRequest, EmbeddingResponse};
use inference_sdk_core::http::{RetryConfig, send_with_retry};
use inference_sdk_core::{RequestOptions, SdkError};

#[derive(Clone, Debug)]
pub struct Embeddings {
    pub(crate) client: Client,
}

impl Embeddings {
    pub fn new(client: Client) -> Self {
        Self { client }
    }

    /// Creates an embedding vector representing the input text.
    pub async fn create(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, SdkError> {
        self.create_with_options(request, RequestOptions::default())
            .await
    }

    /// Creates an embedding vector with custom request options.
    pub async fn create_with_options(
        &self,
        request: EmbeddingRequest,
        options: RequestOptions,
    ) -> Result<EmbeddingResponse, SdkError> {
        let config = RetryConfig {
            base_url: self.client.config.base_url.clone(),
            endpoint: "/embeddings".to_string(), // Note: base_url is typically "v1", so this becomes "v1/embeddings"
            retry_policy: self.client.config.retry_policy.clone(),
            timeout_policy: self.client.config.timeout_policy.clone(),
        };

        // Note: ChatResource sets endpoint to "/chat/completions".
        // Base URL is "https://api.openai.com/v1".
        // So endpoint should be "/embeddings".

        let response =
            send_with_retry(&self.client.http_client, &config, &request, &options).await?;

        response
            .json::<EmbeddingResponse>()
            .await
            .map_err(SdkError::from)
    }
}
