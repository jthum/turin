use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast, oneshot};
use tokio_util::sync::CancellationToken;

use crate::kernel::identity::RuntimeIdentity;

/// A tool call that requires an external authorization decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolAuthorizationRequest {
    pub id: String,
    pub identity: RuntimeIdentity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_slot_id: Option<String>,
    pub tool_call_id: String,
    pub tool_name: String,
    pub arguments: serde_json::Value,
    pub reason: String,
    pub requested_at_unix_ms: u64,
}

impl ToolAuthorizationRequest {
    pub(crate) fn new(
        identity: RuntimeIdentity,
        runtime_slot_id: Option<String>,
        tool_call_id: String,
        tool_name: String,
        arguments: serde_json::Value,
        reason: String,
    ) -> Self {
        Self {
            id: uuid::Uuid::now_v7().simple().to_string(),
            identity,
            runtime_slot_id,
            tool_call_id,
            tool_name,
            arguments,
            reason,
            requested_at_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }
}

/// The externally supplied outcome of a tool authorization request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum ToolAuthorizationDecision {
    Approve,
    Deny {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
}

impl ToolAuthorizationDecision {
    pub fn deny(reason: Option<String>) -> Self {
        Self::Deny {
            reason: reason.filter(|reason| !reason.trim().is_empty()),
        }
    }

    pub(crate) fn normalized(self) -> Self {
        match self {
            Self::Approve => Self::Approve,
            Self::Deny { reason } => Self::deny(reason),
        }
    }
}

pub type ToolAuthorizationFuture =
    Pin<Box<dyn Future<Output = ToolAuthorizationDecision> + Send + 'static>>;

/// Resolves harness-requested tool authorization without prescribing a client or UI.
pub trait ToolAuthorizer: Send + Sync {
    fn authorize(
        &self,
        request: ToolAuthorizationRequest,
        cancellation: CancellationToken,
    ) -> ToolAuthorizationFuture;
}

/// Fails closed when an embedding has not installed an authorization handler.
#[derive(Debug, Default)]
pub struct DenyUnavailableToolAuthorizer;

impl ToolAuthorizer for DenyUnavailableToolAuthorizer {
    fn authorize(
        &self,
        _request: ToolAuthorizationRequest,
        _cancellation: CancellationToken,
    ) -> ToolAuthorizationFuture {
        Box::pin(async {
            ToolAuthorizationDecision::deny(Some(
                "No tool authorization handler is configured".to_string(),
            ))
        })
    }
}

struct PendingAuthorization {
    request: ToolAuthorizationRequest,
    decision_tx: oneshot::Sender<ToolAuthorizationDecision>,
}

/// In-process broker used by daemon and embedded clients to resolve requests asynchronously.
pub struct ToolAuthorizationBroker {
    pending: Arc<RwLock<HashMap<String, PendingAuthorization>>>,
    request_tx: broadcast::Sender<ToolAuthorizationRequest>,
}

impl Default for ToolAuthorizationBroker {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolAuthorizationBroker {
    pub fn new() -> Self {
        let (request_tx, _) = broadcast::channel(64);
        Self {
            pending: Arc::new(RwLock::new(HashMap::new())),
            request_tx,
        }
    }

    /// Subscribes to newly pending requests. Consumers should list pending requests first.
    pub fn subscribe_requests(&self) -> broadcast::Receiver<ToolAuthorizationRequest> {
        self.request_tx.subscribe()
    }

    pub async fn list_pending(&self) -> Vec<ToolAuthorizationRequest> {
        let pending = self.pending.read().await;
        let mut requests: Vec<_> = pending
            .values()
            .map(|entry| entry.request.clone())
            .collect();
        requests.sort_by_key(|request| (request.requested_at_unix_ms, request.id.clone()));
        requests
    }

    /// Resolves a pending request. Returns `false` if it no longer exists.
    pub async fn resolve(&self, id: &str, decision: ToolAuthorizationDecision) -> bool {
        let pending = self.pending.write().await.remove(id);
        pending.is_some_and(|entry| entry.decision_tx.send(decision.normalized()).is_ok())
    }
}

impl ToolAuthorizer for ToolAuthorizationBroker {
    fn authorize(
        &self,
        request: ToolAuthorizationRequest,
        cancellation: CancellationToken,
    ) -> ToolAuthorizationFuture {
        let pending = Arc::clone(&self.pending);
        let request_tx = self.request_tx.clone();
        Box::pin(async move {
            let request_id = request.id.clone();
            let (decision_tx, decision_rx) = oneshot::channel();
            pending.write().await.insert(
                request_id.clone(),
                PendingAuthorization {
                    request: request.clone(),
                    decision_tx,
                },
            );

            // Notification is best-effort; the pending map remains authoritative.
            let _ = request_tx.send(request);

            let decision = tokio::select! {
                _ = cancellation.cancelled() => ToolAuthorizationDecision::deny(Some(
                    "Tool authorization was cancelled".to_string(),
                )),
                decision = decision_rx => decision.unwrap_or_else(|_| {
                    ToolAuthorizationDecision::deny(Some(
                        "Tool authorization handler became unavailable".to_string(),
                    ))
                }),
            };
            pending.write().await.remove(&request_id);
            decision
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> ToolAuthorizationRequest {
        ToolAuthorizationRequest::new(
            RuntimeIdentity::new("session", "agent"),
            Some("slot".to_string()),
            "call".to_string(),
            "shell_exec".to_string(),
            serde_json::json!({ "command": "cargo test" }),
            "Run tests?".to_string(),
        )
    }

    #[tokio::test]
    async fn broker_lists_and_resolves_pending_request() {
        let broker = Arc::new(ToolAuthorizationBroker::new());
        let mut requests = broker.subscribe_requests();
        let authorize = tokio::spawn({
            let broker = Arc::clone(&broker);
            async move { broker.authorize(request(), CancellationToken::new()).await }
        });

        let requested = requests.recv().await.unwrap();
        assert_eq!(
            broker.list_pending().await,
            std::slice::from_ref(&requested)
        );
        assert!(
            broker
                .resolve(&requested.id, ToolAuthorizationDecision::Approve)
                .await
        );
        assert_eq!(authorize.await.unwrap(), ToolAuthorizationDecision::Approve);
        assert!(broker.list_pending().await.is_empty());
    }

    #[tokio::test]
    async fn broker_cancellation_removes_pending_request() {
        let broker = Arc::new(ToolAuthorizationBroker::new());
        let mut requests = broker.subscribe_requests();
        let cancellation = CancellationToken::new();
        let authorize = tokio::spawn({
            let broker = Arc::clone(&broker);
            let cancellation = cancellation.clone();
            async move { broker.authorize(request(), cancellation).await }
        });

        requests.recv().await.unwrap();
        cancellation.cancel();
        assert!(matches!(
            authorize.await.unwrap(),
            ToolAuthorizationDecision::Deny { .. }
        ));
        assert!(broker.list_pending().await.is_empty());
    }

    #[test]
    fn empty_denial_reason_is_omitted() {
        assert_eq!(
            ToolAuthorizationDecision::deny(Some("  ".to_string())),
            ToolAuthorizationDecision::Deny { reason: None }
        );
    }

    #[tokio::test]
    async fn unavailable_authorizer_fails_closed() {
        assert_eq!(
            DenyUnavailableToolAuthorizer
                .authorize(request(), CancellationToken::new())
                .await,
            ToolAuthorizationDecision::Deny {
                reason: Some("No tool authorization handler is configured".to_string())
            }
        );
    }
}
