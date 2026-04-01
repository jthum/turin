use serde_json::json;
use tempfile::tempdir;
use uuid::Uuid;

use super::{
    MemoryFeedbackRequest, MemoryFeedbackSignal, MemoryPurgeRequest, MemorySearchMode,
    MemorySearchRequest, MemoryStoreMode, MemoryStoreRequest,
    memory_correct_backend_with_request, memory_feedback_backend_with_request,
    memory_purge_backend_with_request, memory_search_backend_with_request,
    memory_store_backend_with_request,
};
use super::memory::{memory_search_backend, memory_store_backend};
use crate::kernel::identity::ContextSelector;
use crate::persistence::manager::{StoreManager, StorePathScope};

fn test_selector() -> ContextSelector {
    ContextSelector {
        tags: vec!["agent:test".to_string()],
        namespace: "default".to_string(),
        visibility: "private".to_string(),
    }
}

#[tokio::test]
async fn memory_backend_works_without_embedding_provider() {
    let tmp = tempdir().expect("tempdir");
    let manager = StoreManager::new(tmp.path());
    let selector = test_selector();

    memory_store_backend(
        &manager,
        None,
        &selector,
        "alpha beta lexical memory",
        &json!({ "kind": "note" }),
        StorePathScope::WorkspaceOnly,
    )
    .await
    .expect("lexical-only memory store should succeed");

    let rows = memory_search_backend(
        &manager,
        None,
        &selector,
        "alpha",
        5,
        StorePathScope::WorkspaceOnly,
    )
    .await
    .expect("lexical-only memory search should succeed");

    assert_eq!(rows.len(), 1);
    assert!(rows[0].content.contains("alpha beta lexical memory"));
}

#[tokio::test]
async fn memory_store_embedded_mode_requires_embedding_provider() {
    let tmp = tempdir().expect("tempdir");
    let manager = StoreManager::new(tmp.path());
    let selector = test_selector();

    let err = memory_store_backend_with_request(
        &manager,
        None,
        &selector,
        "alpha beta lexical memory",
        &json!({ "kind": "note" }),
        &MemoryStoreRequest {
            storage: MemoryStoreMode::Embedded,
            ..MemoryStoreRequest::default()
        },
        StorePathScope::WorkspaceOnly,
    )
    .await
    .expect_err("embedded-only store should fail without provider");

    assert!(
        err.to_string()
            .contains("storage='embedded' requires an embedding provider")
    );
}

#[tokio::test]
async fn memory_search_semantic_mode_falls_back_or_errors_without_embeddings() {
    let tmp = tempdir().expect("tempdir");
    let manager = StoreManager::new(tmp.path());
    let selector = test_selector();

    memory_store_backend(
        &manager,
        None,
        &selector,
        "alpha beta lexical memory",
        &json!({ "kind": "note" }),
        StorePathScope::WorkspaceOnly,
    )
    .await
    .expect("lexical-only memory store should succeed");

    let relaxed = memory_search_backend_with_request(
        &manager,
        None,
        &selector,
        "alpha",
        &MemorySearchRequest {
            mode: MemorySearchMode::Semantic,
            ..MemorySearchRequest::default()
        },
        StorePathScope::WorkspaceOnly,
    )
    .await
    .expect("semantic search should fall back to lexical when strict is false");
    assert_eq!(relaxed.len(), 1);

    let err = memory_search_backend_with_request(
        &manager,
        None,
        &selector,
        "alpha",
        &MemorySearchRequest {
            mode: MemorySearchMode::Semantic,
            strict: true,
            ..MemorySearchRequest::default()
        },
        StorePathScope::WorkspaceOnly,
    )
    .await
    .expect_err("strict semantic search should fail without embeddings");
    assert!(
        err.to_string()
            .contains("semantic mode requires an embedding provider")
    );
}

#[tokio::test]
async fn memory_lifecycle_feedback_correct_and_purge_work() {
    let tmp = tempdir().expect("tempdir");
    let manager = StoreManager::new(tmp.path());
    let selector = test_selector();

    let stored = memory_store_backend(
        &manager,
        None,
        &selector,
        "stale alpha memory",
        &json!({ "kind": "note", "source": "initial" }),
        StorePathScope::WorkspaceOnly,
    )
    .await
    .expect("initial memory store should succeed");
    let memory_id = Uuid::from_slice(&stored.public_id)
        .expect("uuid bytes")
        .simple()
        .to_string();

    let feedback = memory_feedback_backend_with_request(
        &manager,
        &selector,
        &memory_id,
        MemoryFeedbackSignal::Up,
        &MemoryFeedbackRequest {
            step: 0.25,
            ..MemoryFeedbackRequest::default()
        },
        StorePathScope::WorkspaceOnly,
    )
    .await
    .expect("feedback should succeed");
    assert!(feedback.weight > 1.0);

    let correction = memory_correct_backend_with_request(
        &manager,
        None,
        &selector,
        &memory_id,
        "fresh beta memory",
        &json!({ "kind": "note", "source": "corrected" }),
        &MemoryStoreRequest {
            storage: MemoryStoreMode::LexicalOnly,
            ..MemoryStoreRequest::default()
        },
        StorePathScope::WorkspaceOnly,
    )
    .await
    .expect("correction should succeed");

    let visible = memory_search_backend_with_request(
        &manager,
        None,
        &selector,
        "fresh",
        &MemorySearchRequest {
            include_metadata: true,
            ..MemorySearchRequest::default()
        },
        StorePathScope::WorkspaceOnly,
    )
    .await
    .expect("corrected memory should be searchable");
    assert_eq!(visible.len(), 1);
    assert_eq!(
        Uuid::from_slice(&visible[0].public_id)
            .expect("uuid bytes")
            .simple()
            .to_string(),
        Uuid::from_slice(&correction.replacement_public_id)
            .expect("uuid bytes")
            .simple()
            .to_string()
    );

    let hidden_old = memory_search_backend_with_request(
        &manager,
        None,
        &selector,
        "stale",
        &MemorySearchRequest::default(),
        StorePathScope::WorkspaceOnly,
    )
    .await
    .expect("superseded search should succeed");
    assert!(
        hidden_old.is_empty(),
        "superseded memory should be hidden by default"
    );

    let old_visible = memory_search_backend_with_request(
        &manager,
        None,
        &selector,
        "stale",
        &MemorySearchRequest {
            include_superseded: true,
            ..MemorySearchRequest::default()
        },
        StorePathScope::WorkspaceOnly,
    )
    .await
    .expect("superseded-inclusive search should succeed");
    assert_eq!(old_visible.len(), 1);

    let dry_run = memory_purge_backend_with_request(
        &manager,
        &selector,
        &MemoryPurgeRequest {
            only_superseded: true,
            ..MemoryPurgeRequest::default()
        },
        StorePathScope::WorkspaceOnly,
    )
    .await
    .expect("purge dry-run should succeed");
    assert_eq!(dry_run.matched, 1);
    assert_eq!(dry_run.deleted, 0);
    assert!(dry_run.dry_run);

    let purge = memory_purge_backend_with_request(
        &manager,
        &selector,
        &MemoryPurgeRequest {
            only_superseded: true,
            dry_run: false,
            ..MemoryPurgeRequest::default()
        },
        StorePathScope::WorkspaceOnly,
    )
    .await
    .expect("purge should succeed");
    assert_eq!(purge.deleted, 1);

    let after_purge = memory_search_backend_with_request(
        &manager,
        None,
        &selector,
        "stale",
        &MemorySearchRequest {
            include_superseded: true,
            ..MemorySearchRequest::default()
        },
        StorePathScope::WorkspaceOnly,
    )
    .await
    .expect("post-purge search should succeed");
    assert!(after_purge.is_empty(), "purged memory should be gone");
}
