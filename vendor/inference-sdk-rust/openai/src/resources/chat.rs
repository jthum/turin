use crate::client::Client;
use crate::types::chat::{
    ChatCompletion, ChatCompletionChunk, ChatCompletionRequest, StreamOptions,
};
use eventsource_stream::Eventsource;
use futures_core::Stream;
use futures_util::StreamExt;
use inference_sdk_core::http::{RetryConfig, send_with_retry};
use inference_sdk_core::{RequestOptions, SdkError};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};

static REQUEST_DUMP_SEQ: AtomicU64 = AtomicU64::new(1);

fn maybe_dump_request(kind: &str, base_url: &str, request: &ChatCompletionRequest) {
    if std::env::var_os("OPENAI_SDK_DEBUG_REQUESTS").is_none() {
        return;
    }
    let seq = REQUEST_DUMP_SEQ.fetch_add(1, Ordering::Relaxed);
    match serde_json::to_string_pretty(request) {
        Ok(json) => eprintln!(
            "\n=== openai-sdk request #{seq} ({kind}) {base_url}/chat/completions ===\n{json}\n"
        ),
        Err(err) => eprintln!(
            "\n=== openai-sdk request #{seq} ({kind}) {base_url}/chat/completions ===\n<serialize error: {err}>\n"
        ),
    }
}

fn maybe_dump_sse_event(base_url: &str, event_name: &str, data: &str) {
    if std::env::var_os("OPENAI_SDK_DEBUG_REQUESTS").is_none() {
        return;
    }
    eprintln!(
        "\n=== openai-sdk sse event {base_url}/chat/completions event={event_name:?} ===\n{data}\n"
    );
}

#[derive(Clone, Debug)]
pub struct ChatResource {
    pub(crate) client: Client,
}

impl ChatResource {
    pub fn new(client: Client) -> Self {
        Self { client }
    }

    /// Create a Chat Completion (non-streaming)
    ///
    /// POST /v1/chat/completions
    pub async fn create(&self, request: ChatCompletionRequest) -> Result<ChatCompletion, SdkError> {
        self.create_with_options(request, RequestOptions::default())
            .await
    }

    /// Create a Chat Completion with custom options
    pub async fn create_with_options(
        &self,
        request: ChatCompletionRequest,
        options: RequestOptions,
    ) -> Result<ChatCompletion, SdkError> {
        let config = RetryConfig {
            base_url: self.client.config.base_url.clone(),
            endpoint: "/chat/completions".to_string(),
            retry_policy: self.client.config.retry_policy.clone(),
            timeout_policy: self.client.config.timeout_policy.clone(),
        };
        maybe_dump_request("create", &self.client.config.base_url, &request);
        maybe_dump_request("create_stream", &self.client.config.base_url, &request);
        let response =
            send_with_retry(&self.client.http_client, &config, &request, &options).await?;
        response
            .json::<ChatCompletion>()
            .await
            .map_err(SdkError::from)
    }

    /// Create a Chat Completion Stream
    ///
    /// POST /v1/chat/completions (returning an SSE stream)
    pub async fn create_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, SdkError>> + Send + 'static>>,
        SdkError,
    > {
        self.create_stream_with_options(request, RequestOptions::default())
            .await
    }

    /// Create a Chat Completion Stream with custom options
    pub async fn create_stream_with_options(
        &self,
        mut request: ChatCompletionRequest,
        options: RequestOptions,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, SdkError>> + Send + 'static>>,
        SdkError,
    > {
        request.stream = Some(true);
        request.stream_options = Some(StreamOptions {
            include_usage: true,
        });

        let config = RetryConfig {
            base_url: self.client.config.base_url.clone(),
            endpoint: "/chat/completions".to_string(),
            retry_policy: self.client.config.retry_policy.clone(),
            timeout_policy: self.client.config.timeout_policy.clone(),
        };
        let response =
            send_with_retry(&self.client.http_client, &config, &request, &options).await?;
        let stream = response.bytes_stream().eventsource();
        let debug_base_url = self.client.config.base_url.clone();

        let mapped_stream = stream.filter_map(move |event_result| {
            let debug_base_url = debug_base_url.clone();
            async move {
                match event_result {
                    Ok(event) => {
                        maybe_dump_sse_event(&debug_base_url, &event.event, &event.data);
                        // OpenAI signals end of stream with `data: [DONE]`
                        if event.data == "[DONE]" {
                            return None;
                        }
                        Some(
                            serde_json::from_str::<ChatCompletionChunk>(&event.data)
                                .map_err(SdkError::SerializationError),
                        )
                    }
                    Err(e) => Some(Err(SdkError::StreamError(e.to_string()))),
                }
            }
        });

        Ok(Box::pin(mapped_stream))
    }
}
