use crate::client::Client;
use crate::types::message::{MessageRequest, MessageResponse, StreamEvent};
use eventsource_stream::Eventsource;
use futures_core::Stream;
use futures_util::StreamExt;
use inference_sdk_core::RequestOptions;
use inference_sdk_core::SdkError;
use inference_sdk_core::http::{RetryConfig, send_with_retry};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};

static REQUEST_DUMP_SEQ: AtomicU64 = AtomicU64::new(1);

fn maybe_dump_request(kind: &str, base_url: &str, request: &MessageRequest) {
    if std::env::var_os("ANTHROPIC_SDK_DEBUG_REQUESTS").is_none() {
        return;
    }
    let seq = REQUEST_DUMP_SEQ.fetch_add(1, Ordering::Relaxed);
    match serde_json::to_string_pretty(request) {
        Ok(json) => eprintln!(
            "\n=== anthropic-sdk request #{seq} ({kind}) {base_url}/messages ===\n{json}\n"
        ),
        Err(err) => eprintln!(
            "\n=== anthropic-sdk request #{seq} ({kind}) {base_url}/messages ===\n<serialize error: {err}>\n"
        ),
    }
}

#[derive(Clone, Debug)]
pub struct MessagesResource {
    pub(crate) client: Client,
}

impl MessagesResource {
    pub fn new(client: Client) -> Self {
        Self { client }
    }

    /// Create a Message (non-streaming)
    ///
    /// POST /v1/messages
    pub async fn create(&self, request: MessageRequest) -> Result<MessageResponse, SdkError> {
        self.create_with_options(request, RequestOptions::default())
            .await
    }

    /// Create a Message with custom options
    pub async fn create_with_options(
        &self,
        request: MessageRequest,
        options: RequestOptions,
    ) -> Result<MessageResponse, SdkError> {
        let config = RetryConfig {
            base_url: self.client.config.base_url.clone(),
            endpoint: "/messages".to_string(),
            retry_policy: self.client.config.retry_policy.clone(),
            timeout_policy: self.client.config.timeout_policy.clone(),
        };
        maybe_dump_request("create", &self.client.config.base_url, &request);
        let response =
            send_with_retry(&self.client.http_client, &config, &request, &options).await?;
        response
            .json::<MessageResponse>()
            .await
            .map_err(SdkError::from)
    }

    /// Create a Message Stream
    ///
    /// POST /v1/messages (returning an SSE stream)
    pub async fn create_stream(
        &self,
        request: MessageRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, SdkError>> + Send + 'static>>, SdkError>
    {
        self.create_stream_with_options(request, RequestOptions::default())
            .await
    }

    /// Create a Message Stream with custom options
    pub async fn create_stream_with_options(
        &self,
        mut request: MessageRequest,
        options: RequestOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, SdkError>> + Send + 'static>>, SdkError>
    {
        request.stream = Some(true);

        let config = RetryConfig {
            base_url: self.client.config.base_url.clone(),
            endpoint: "/messages".to_string(),
            retry_policy: self.client.config.retry_policy.clone(),
            timeout_policy: self.client.config.timeout_policy.clone(),
        };
        maybe_dump_request("create_stream", &self.client.config.base_url, &request);
        let response =
            send_with_retry(&self.client.http_client, &config, &request, &options).await?;

        let stream = response.bytes_stream().eventsource();

        let mapped_stream = stream.map(|event_result| match event_result {
            Ok(event) => {
                if event.event == "ping" {
                    return Ok(StreamEvent::Ping);
                }
                serde_json::from_str::<StreamEvent>(&event.data)
                    .map_err(SdkError::SerializationError)
            }
            Err(e) => Err(SdkError::StreamError(e.to_string())),
        });

        Ok(Box::pin(mapped_stream))
    }
}
