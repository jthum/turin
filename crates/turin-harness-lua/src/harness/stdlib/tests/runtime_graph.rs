use serde_json::json;

use super::*;
use crate::persistence::state::TurnWriteTarget;

fn turn(turn_index: u32) -> TurnWriteTarget {
    TurnWriteTarget::active_branch(turn_index)
}

fn branch_turn(branch_head_id: Option<i64>, turn_index: u32) -> TurnWriteTarget {
    TurnWriteTarget::branch_head(branch_head_id, turn_index)
}

#[tokio::test]
async fn selected_path_from_graph_refs_preserves_explicit_order() {
    let store = StateStore::open_memory().await.unwrap();
    let session_id = store
        .create_session(uuid::Uuid::now_v7(), "default", None)
        .await
        .unwrap();

    store
        .insert_message(
            session_id,
            turn(0),
            "user",
            &json!([{"type": "text", "text": "root"}]),
            None,
        )
        .await
        .unwrap();
    let alt_branch = store
        .create_branch_head_from_turn_index(session_id, "alt", Some(0), false)
        .await
        .unwrap();

    store
        .insert_message(
            session_id,
            turn(1),
            "assistant",
            &json!([{"type": "text", "text": "main second"}]),
            None,
        )
        .await
        .unwrap();
    store
        .insert_message(
            session_id,
            branch_turn(Some(alt_branch.id), 1),
            "assistant",
            &json!([{"type": "text", "text": "alt second"}]),
            None,
        )
        .await
        .unwrap();

    let main_branch = store
        .get_active_branch_head(session_id)
        .await
        .unwrap()
        .expect("main branch");
    let main_turn_id = main_branch.head_turn_id.expect("main head turn");
    let alt_turn_id = store
        .get_branch_head(session_id, alt_branch.id)
        .await
        .unwrap()
        .expect("alt branch")
        .head_turn_id
        .expect("alt head turn");

    let refs = vec![
        GraphRef::new("branch_head", bytes_to_simple_uuid(&alt_branch.public_id)),
        GraphRef::new("turn", main_turn_id.to_string()),
    ];

    let selected = selected_path_from_graph_refs(&store, session_id, refs)
        .await
        .unwrap();

    assert_eq!(selected, vec![alt_turn_id, main_turn_id]);
}

#[tokio::test]
async fn selected_path_from_graph_refs_rejects_duplicate_materialized_turns() {
    let store = StateStore::open_memory().await.unwrap();
    let session_id = store
        .create_session(uuid::Uuid::now_v7(), "default", None)
        .await
        .unwrap();

    store
        .insert_message(
            session_id,
            turn(0),
            "user",
            &json!([{"type": "text", "text": "root"}]),
            None,
        )
        .await
        .unwrap();

    let main_branch = store
        .get_active_branch_head(session_id)
        .await
        .unwrap()
        .expect("main branch");
    let turn_id = main_branch.head_turn_id.expect("main head turn");

    let refs = vec![
        GraphRef::new("branch_head", bytes_to_simple_uuid(&main_branch.public_id)),
        GraphRef::new("turn", turn_id.to_string()),
    ];

    let err = selected_path_from_graph_refs(&store, session_id, refs)
        .await
        .unwrap_err()
        .to_string();

    assert!(err.contains("duplicate turn"));
}

#[tokio::test]
async fn selected_path_from_graph_edges_supports_newest_first_and_limit() {
    let store = StateStore::open_memory().await.unwrap();
    let session_id = store
        .create_session(uuid::Uuid::now_v7(), "default", None)
        .await
        .unwrap();

    store
        .insert_message(
            session_id,
            turn(0),
            "user",
            &json!([{"type": "text", "text": "root"}]),
            None,
        )
        .await
        .unwrap();
    let candidate_a = store
        .create_branch_head_from_turn_index(session_id, "candidate-a", Some(0), false)
        .await
        .unwrap();
    let candidate_b = store
        .create_branch_head_from_turn_index(session_id, "candidate-b", Some(0), false)
        .await
        .unwrap();

    store
        .insert_message(
            session_id,
            branch_turn(Some(candidate_a.id), 1),
            "assistant",
            &json!([{"type": "text", "text": "candidate a"}]),
            None,
        )
        .await
        .unwrap();
    store
        .insert_message(
            session_id,
            branch_turn(Some(candidate_b.id), 1),
            "assistant",
            &json!([{"type": "text", "text": "candidate b"}]),
            None,
        )
        .await
        .unwrap();

    let group = store
        .create_graph_node(
            Some(session_id),
            "experiment",
            Some("compare candidates"),
            GraphProvenance::default(),
            None,
        )
        .await
        .unwrap();

    for branch in [&candidate_a, &candidate_b] {
        let edge = GraphEdgeCreate::new(
            GraphRef::new("graph_node", bytes_to_simple_uuid(&group.public_id)),
            GraphRef::new("branch_head", bytes_to_simple_uuid(&branch.public_id)),
            "contains",
        );
        store.create_graph_edge(edge).await.unwrap();
    }

    let selected = selected_path_from_graph_edges(
        &store,
        session_id,
        GraphRef::new("graph_node", bytes_to_simple_uuid(&group.public_id)),
        SelectedPathSourceOptions {
            relation_kind: Some("contains".to_string()),
            target_kind: Some("branch_head".to_string()),
            target_role: None,
            order: SelectedPathOrder::NewestFirst,
            limit: Some(1),
        },
    )
    .await
    .unwrap();

    let branch_b_turn = store
        .get_branch_head(session_id, candidate_b.id)
        .await
        .unwrap()
        .expect("candidate b branch")
        .head_turn_id
        .expect("candidate b head turn");

    assert_eq!(selected, vec![branch_b_turn]);
}
