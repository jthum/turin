use serde_json::json;
use turin_daemon_protocol::{SessionSearchHitKind, SessionSearchScope};

use super::*;

fn active_branch() -> SessionReadTarget {
    SessionReadTarget::ActiveBranch
}

fn turn(turn_index: u32) -> TurnWriteTarget {
    TurnWriteTarget::active_branch(turn_index)
}

fn branch_turn(branch_head_id: Option<i64>, turn_index: u32) -> TurnWriteTarget {
    TurnWriteTarget::branch_head(branch_head_id, turn_index)
}

#[tokio::test]
async fn test_schema_initialization() {
    let store = StateStore::open_memory().await.unwrap();

    let conn = store.get_connection().await.unwrap();
    let mut rows = conn
        .query("SELECT value FROM schema_info WHERE key = 'version'", ())
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    let version: String = row.get(0).unwrap();
    assert_eq!(version, SCHEMA_VERSION.to_string());
}

#[tokio::test]
async fn test_insert_and_get_events() {
    let store = StateStore::open_memory().await.unwrap();
    let session = store
        .create_session(uuid::Uuid::now_v7(), "default", None)
        .await
        .unwrap();

    store
        .insert_event(
            session,
            None,
            "session_start",
            &json!({"session_id": session}),
        )
        .await
        .unwrap();
    store
        .insert_event(session, None, "turn_start", &json!({"turn_index": 0}))
        .await
        .unwrap();

    let events = store.get_events(session, &active_branch()).await.unwrap();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].event_type, "session_start");
    assert_eq!(events[1].event_type, "turn_start");
}

#[tokio::test]
async fn test_events_isolated_by_session() {
    let store = StateStore::open_memory().await.unwrap();
    let session_a = store
        .create_session(uuid::Uuid::now_v7(), "default", None)
        .await
        .unwrap();
    let session_b = store
        .create_session(uuid::Uuid::now_v7(), "default", None)
        .await
        .unwrap();

    store
        .insert_event(session_a, None, "session_start", &json!({}))
        .await
        .unwrap();
    store
        .insert_event(session_b, None, "session_start", &json!({}))
        .await
        .unwrap();

    let events_a = store.get_events(session_a, &active_branch()).await.unwrap();
    let events_b = store.get_events(session_b, &active_branch()).await.unwrap();
    assert_eq!(events_a.len(), 1);
    assert_eq!(events_b.len(), 1);
}

#[tokio::test]
async fn test_insert_and_get_messages() {
    let store = StateStore::open_memory().await.unwrap();
    let session = store
        .create_session(uuid::Uuid::now_v7(), "default", None)
        .await
        .unwrap();

    store
        .insert_message(
            session,
            turn(0),
            "user",
            &json!([{"type": "text", "text": "hello"}]),
            None,
        )
        .await
        .unwrap();
    store
        .insert_message(
            session,
            turn(0),
            "assistant",
            &json!([{"type": "text", "text": "hi!"}]),
            Some(10),
        )
        .await
        .unwrap();

    let msgs = store.get_messages(session, &active_branch()).await.unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0].role, "user");
    assert_eq!(msgs[1].role, "assistant");
    assert_eq!(msgs[1].token_count, Some(10));
}

#[tokio::test]
async fn test_insert_and_get_tool_executions() {
    let store = StateStore::open_memory().await.unwrap();
    let session = store
        .create_session(uuid::Uuid::now_v7(), "default", None)
        .await
        .unwrap();

    store
        .insert_tool_execution(
            session,
            turn(0),
            "call_1",
            "read_file",
            &json!({"path": "main.rs"}),
            Some("fn main() {}"),
            false,
            Some(15),
            "allow",
        )
        .await
        .unwrap();

    let execs = store
        .get_tool_executions(session, &active_branch())
        .await
        .unwrap();
    assert_eq!(execs.len(), 1);
    assert_eq!(execs[0].tool_name, "read_file");
    assert_eq!(execs[0].output, Some("fn main() {}".to_string()));
    assert!(!execs[0].is_error);
    assert_eq!(execs[0].duration_ms, Some(15));
    assert_eq!(execs[0].verdict, "allow");
}

#[tokio::test]
async fn test_tool_execution_with_error() {
    let store = StateStore::open_memory().await.unwrap();
    let session = store
        .create_session(uuid::Uuid::now_v7(), "default", None)
        .await
        .unwrap();

    store
        .insert_tool_execution(
            session,
            turn(0),
            "call_2",
            "shell_exec",
            &json!({"command": "rm -rf /"}),
            Some("Permission denied"),
            true,
            Some(5),
            "reject",
        )
        .await
        .unwrap();

    let execs = store
        .get_tool_executions(session, &active_branch())
        .await
        .unwrap();
    assert_eq!(execs.len(), 1);
    assert!(execs[0].is_error);
    assert_eq!(execs[0].verdict, "reject");
}

#[tokio::test]
async fn test_update_session_title_preserves_other_metadata() {
    let store = StateStore::open_memory().await.unwrap();
    let public_id = uuid::Uuid::now_v7();
    let metadata = json!({
        "source": "test",
        "title": "Original title",
    });

    store
        .create_session(public_id, "default", Some(&metadata.to_string()))
        .await
        .unwrap();

    let updated = store
        .update_session_title(public_id, Some("Renamed session"))
        .await
        .unwrap()
        .expect("session exists");

    let parsed: serde_json::Value =
        serde_json::from_str(updated.metadata.as_deref().unwrap()).unwrap();
    assert_eq!(
        parsed.get("title").and_then(|value| value.as_str()),
        Some("Renamed session")
    );
    assert_eq!(
        parsed.get("source").and_then(|value| value.as_str()),
        Some("test")
    );

    let cleared = store
        .update_session_title(public_id, None)
        .await
        .unwrap()
        .expect("session exists");
    let parsed: serde_json::Value =
        serde_json::from_str(cleared.metadata.as_deref().unwrap()).unwrap();
    assert!(parsed.get("title").is_none());
    assert_eq!(
        parsed.get("source").and_then(|value| value.as_str()),
        Some("test")
    );
}

#[tokio::test]
async fn test_search_session_history_queries_messages_tools_events_and_titles() {
    let store = StateStore::open_memory().await.unwrap();
    let public_id = uuid::Uuid::now_v7();
    let metadata = json!({ "title": "Compiler investigations" });
    let session_id = store
        .create_session(public_id, "default", Some(&metadata.to_string()))
        .await
        .unwrap();

    store
        .insert_message(
            session_id,
            turn(0),
            "user",
            &json!([{"type": "text", "text": "Investigate the compiler panic in src/main.rs"}]),
            None,
        )
        .await
        .unwrap();
    store
        .insert_tool_execution(
            session_id,
            turn(0),
            "call_1",
            "read_file",
            &json!({"path": "src/main.rs"}),
            Some("panic!(\"boom\")"),
            false,
            Some(12),
            "allow",
        )
        .await
        .unwrap();
    store
        .insert_event(
            session_id,
            None,
            "tool_call",
            &json!({"tool_name": "read_file", "path": "src/main.rs"}),
        )
        .await
        .unwrap();

    let title_hits = store
        .search_session_history("compiler", SessionSearchScope::Sessions, 16, 0)
        .await
        .unwrap();
    assert!(
        title_hits
            .iter()
            .any(|hit| hit.kind == SessionSearchHitKind::Session)
    );

    let ranked_hits = store
        .search_session_history("compiler", SessionSearchScope::All, 16, 0)
        .await
        .unwrap();
    assert_eq!(
        ranked_hits.first().map(|hit| hit.kind),
        Some(SessionSearchHitKind::Session)
    );

    let message_hits = store
        .search_session_history("panic", SessionSearchScope::Messages, 16, 0)
        .await
        .unwrap();
    assert!(
        message_hits
            .iter()
            .any(|hit| hit.kind == SessionSearchHitKind::Message)
    );

    let tool_hits = store
        .search_session_history("read_file", SessionSearchScope::ToolExecutions, 16, 0)
        .await
        .unwrap();
    assert!(
        tool_hits
            .iter()
            .any(|hit| hit.kind == SessionSearchHitKind::ToolExecution)
    );

    let event_hits = store
        .search_session_history("tool_call", SessionSearchScope::Events, 16, 0)
        .await
        .unwrap();
    assert!(
        event_hits
            .iter()
            .any(|hit| hit.kind == SessionSearchHitKind::Event)
    );
}

#[tokio::test]
async fn test_search_session_history_follows_active_branch_path_for_messages_and_tools() {
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
            &json!([{"type": "text", "text": "shared root"}]),
            None,
        )
        .await
        .unwrap();
    store
        .insert_message(
            session_id,
            turn(1),
            "assistant",
            &json!([{"type": "text", "text": "main branch only message"}]),
            None,
        )
        .await
        .unwrap();
    store
        .insert_tool_execution(
            session_id,
            turn(1),
            "call_main",
            "main_branch_tool",
            &json!({"path": "main.rs"}),
            Some("main branch output"),
            false,
            Some(8),
            "allow",
        )
        .await
        .unwrap();

    store
        .create_branch_head_from_turn_index(session_id, "alt", Some(0), true)
        .await
        .unwrap();
    store
        .insert_message(
            session_id,
            turn(1),
            "assistant",
            &json!([{"type": "text", "text": "alt branch only message"}]),
            None,
        )
        .await
        .unwrap();
    store
        .insert_tool_execution(
            session_id,
            turn(1),
            "call_alt",
            "alt_branch_tool",
            &json!({"path": "alt.rs"}),
            Some("alt branch output"),
            false,
            Some(9),
            "allow",
        )
        .await
        .unwrap();

    let main_message_hits = store
        .search_session_history("main branch only", SessionSearchScope::Messages, 16, 0)
        .await
        .unwrap();
    assert!(main_message_hits.is_empty());

    let alt_message_hits = store
        .search_session_history("alt branch only", SessionSearchScope::Messages, 16, 0)
        .await
        .unwrap();
    assert_eq!(alt_message_hits.len(), 1);
    assert_eq!(alt_message_hits[0].kind, SessionSearchHitKind::Message);
    assert_eq!(alt_message_hits[0].turn_index, Some(1));

    let main_tool_hits = store
        .search_session_history(
            "main_branch_tool",
            SessionSearchScope::ToolExecutions,
            16,
            0,
        )
        .await
        .unwrap();
    assert!(main_tool_hits.is_empty());

    let alt_tool_hits = store
        .search_session_history("alt_branch_tool", SessionSearchScope::ToolExecutions, 16, 0)
        .await
        .unwrap();
    assert_eq!(alt_tool_hits.len(), 1);
    assert_eq!(alt_tool_hits[0].kind, SessionSearchHitKind::ToolExecution);
    assert_eq!(alt_tool_hits[0].turn_index, Some(1));
}

#[tokio::test]
async fn test_get_events_uses_hybrid_branch_filtering() {
    let store = StateStore::open_memory().await.unwrap();
    let session = store
        .create_session(uuid::Uuid::now_v7(), "default", None)
        .await
        .unwrap();

    store
        .insert_message(
            session,
            turn(0),
            "user",
            &json!([{"type": "text", "text": "root"}]),
            None,
        )
        .await
        .unwrap();
    store
        .insert_event(session, None, "session_start", &json!({"scope": "session"}))
        .await
        .unwrap();
    store
        .insert_event(
            session,
            Some(turn(1)),
            "turn_end",
            &json!({"scope": "main-branch"}),
        )
        .await
        .unwrap();

    store
        .create_branch_head_from_turn_index(session, "alt", Some(0), true)
        .await
        .unwrap();
    store
        .insert_event(
            session,
            Some(turn(1)),
            "turn_end",
            &json!({"scope": "alt-branch"}),
        )
        .await
        .unwrap();
    store
        .insert_event(
            session,
            None,
            "all_tasks_complete",
            &json!({"scope": "session"}),
        )
        .await
        .unwrap();

    let events = store.get_events(session, &active_branch()).await.unwrap();
    assert_eq!(events.len(), 3);
    assert_eq!(events[0].event_type, "session_start");
    assert!(events[0].turn_id.is_none());
    assert_eq!(events[1].event_type, "turn_end");
    assert_eq!(events[1].turn_index, Some(1));
    assert!(events[1].payload.contains("alt-branch"));
    assert_eq!(events[2].event_type, "all_tasks_complete");
    assert!(events[2].turn_id.is_none());

    let all_events = store.get_all_events(session).await.unwrap();
    assert_eq!(all_events.len(), 4);
}

#[tokio::test]
async fn test_search_session_history_follows_active_branch_path_for_events() {
    let store = StateStore::open_memory().await.unwrap();
    let session = store
        .create_session(uuid::Uuid::now_v7(), "default", None)
        .await
        .unwrap();

    store
        .insert_message(
            session,
            turn(0),
            "user",
            &json!([{"type": "text", "text": "root"}]),
            None,
        )
        .await
        .unwrap();
    store
        .insert_event(
            session,
            None,
            "all_tasks_complete",
            &json!({"marker": "shared session event"}),
        )
        .await
        .unwrap();
    store
        .insert_event(
            session,
            Some(turn(1)),
            "turn_end",
            &json!({"marker": "main branch event"}),
        )
        .await
        .unwrap();

    store
        .create_branch_head_from_turn_index(session, "alt", Some(0), true)
        .await
        .unwrap();
    store
        .insert_event(
            session,
            Some(turn(1)),
            "turn_end",
            &json!({"marker": "alt branch event"}),
        )
        .await
        .unwrap();

    let main_hits = store
        .search_session_history("main branch event", SessionSearchScope::Events, 16, 0)
        .await
        .unwrap();
    assert!(main_hits.is_empty());

    let alt_hits = store
        .search_session_history("alt branch event", SessionSearchScope::Events, 16, 0)
        .await
        .unwrap();
    assert_eq!(alt_hits.len(), 1);
    assert_eq!(alt_hits[0].kind, SessionSearchHitKind::Event);
    assert_eq!(alt_hits[0].turn_index, Some(1));

    let shared_hits = store
        .search_session_history("shared session event", SessionSearchScope::Events, 16, 0)
        .await
        .unwrap();
    assert_eq!(shared_hits.len(), 1);
    assert_eq!(shared_hits[0].kind, SessionSearchHitKind::Event);
    assert_eq!(shared_hits[0].turn_index, None);
}

#[tokio::test]
async fn test_kv_set_get_delete() {
    let store = StateStore::open_memory().await.unwrap();
    let scope_kind = "session";
    let scope_key = "test-session";

    store
        .kv_set(scope_kind, scope_key, "budget_remaining", "1000")
        .await
        .unwrap();
    let val = store
        .kv_get(scope_kind, scope_key, "budget_remaining")
        .await
        .unwrap();
    assert_eq!(val, Some("1000".to_string()));

    store
        .kv_set(scope_kind, scope_key, "budget_remaining", "500")
        .await
        .unwrap();
    let val = store
        .kv_get(scope_kind, scope_key, "budget_remaining")
        .await
        .unwrap();
    assert_eq!(val, Some("500".to_string()));

    store
        .kv_delete(scope_kind, scope_key, "budget_remaining")
        .await
        .unwrap();
    let val = store
        .kv_get(scope_kind, scope_key, "budget_remaining")
        .await
        .unwrap();
    assert_eq!(val, None);
}

#[tokio::test]
async fn test_kv_get_nonexistent() {
    let store = StateStore::open_memory().await.unwrap();
    let val = store
        .kv_get("session", "test-session", "nonexistent")
        .await
        .unwrap();
    assert_eq!(val, None);
}

#[tokio::test]
async fn test_file_based_store() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("test.db");
    let db_path_str = db_path.to_str().unwrap();

    {
        let store = StateStore::open(db_path_str).await.unwrap();
        let session = store
            .create_session(uuid::Uuid::now_v7(), "default", None)
            .await
            .unwrap();
        store
            .insert_event(session, None, "session_start", &json!({}))
            .await
            .unwrap();
        store
            .kv_set("session", "test-session", "key1", "value1")
            .await
            .unwrap();
    }

    {
        let store = StateStore::open(db_path_str).await.unwrap();
        let sessions = store.list_session_rows(4, 0).await.unwrap();
        let session = sessions.first().expect("persisted session").id;
        let events = store.get_events(session, &active_branch()).await.unwrap();
        assert_eq!(events.len(), 1);

        let val = store
            .kv_get("session", "test-session", "key1")
            .await
            .unwrap();
        assert_eq!(val, Some("value1".to_string()));
    }
}

#[tokio::test]
async fn test_create_session_initializes_main_branch() {
    let store = StateStore::open_memory().await.unwrap();
    let session = store
        .create_session(uuid::Uuid::now_v7(), "default", None)
        .await
        .unwrap();

    let head = store
        .get_active_branch_head(session)
        .await
        .unwrap()
        .expect("active branch head");
    assert_eq!(head.name, "main");
    assert_eq!(head.origin_kind, "main");
    assert!(head.is_active);
    assert!(head.head_turn_id.is_none());
}

#[tokio::test]
async fn test_get_messages_follows_active_branch_path() {
    let store = StateStore::open_memory().await.unwrap();
    let session = store
        .create_session(uuid::Uuid::now_v7(), "default", None)
        .await
        .unwrap();

    store
        .insert_message(
            session,
            turn(0),
            "user",
            &json!([{"type": "text", "text": "root"}]),
            None,
        )
        .await
        .unwrap();
    store
        .insert_message(
            session,
            turn(1),
            "assistant",
            &json!([{"type": "text", "text": "main"}]),
            None,
        )
        .await
        .unwrap();

    let branch = store
        .create_branch_head_from_turn_index(session, "alt", Some(0), true)
        .await
        .unwrap();
    assert_eq!(branch.name, "alt");
    assert_eq!(branch.origin_kind, "manual");
    assert!(branch.is_active);

    store
        .insert_message(
            session,
            turn(1),
            "assistant",
            &json!([{"type": "text", "text": "alternate"}]),
            None,
        )
        .await
        .unwrap();

    let alt_messages = store.get_messages(session, &active_branch()).await.unwrap();
    let texts = alt_messages
        .iter()
        .map(|row| row.content.clone())
        .collect::<Vec<_>>();
    assert_eq!(alt_messages.len(), 2);
    assert!(texts.iter().any(|content| content.contains("root")));
    assert!(texts.iter().any(|content| content.contains("alternate")));
    assert!(!texts.iter().any(|content| content.contains("main")));

    let main_head = store
        .checkout_branch_head_by_name(session, "main")
        .await
        .unwrap()
        .expect("main branch exists");
    assert_eq!(main_head.name, "main");
    assert!(main_head.is_active);

    let main_messages = store.get_messages(session, &active_branch()).await.unwrap();
    let texts = main_messages
        .iter()
        .map(|row| row.content.clone())
        .collect::<Vec<_>>();
    assert_eq!(main_messages.len(), 2);
    assert!(texts.iter().any(|content| content.contains("root")));
    assert!(texts.iter().any(|content| content.contains("main")));
    assert!(!texts.iter().any(|content| content.contains("alternate")));
}

#[tokio::test]
async fn test_explicit_branch_head_reads_and_writes_ignore_active_branch() {
    let store = StateStore::open_memory().await.unwrap();
    let session = store
        .create_session(uuid::Uuid::now_v7(), "default", None)
        .await
        .unwrap();

    store
        .insert_message(
            session,
            turn(0),
            "user",
            &json!([{"type": "text", "text": "root"}]),
            None,
        )
        .await
        .unwrap();

    let alt = store
        .create_branch_head_from_turn_index(session, "alt", Some(0), false)
        .await
        .unwrap();
    let active = store
        .get_active_branch_head(session)
        .await
        .unwrap()
        .expect("active branch");
    assert_eq!(active.name, "main");

    store
        .insert_message(
            session,
            branch_turn(Some(alt.id), 1),
            "assistant",
            &json!([{"type": "text", "text": "alternate"}]),
            None,
        )
        .await
        .unwrap();

    let main_messages = store.get_messages(session, &active_branch()).await.unwrap();
    assert_eq!(main_messages.len(), 1);
    assert!(main_messages[0].content.contains("root"));

    let alt_messages = store
        .get_messages(session, &SessionReadTarget::BranchHead(alt.id))
        .await
        .unwrap();
    assert_eq!(alt_messages.len(), 2);
    assert!(alt_messages.iter().any(|row| row.content.contains("root")));
    assert!(
        alt_messages
            .iter()
            .any(|row| row.content.contains("alternate"))
    );
}

#[tokio::test]
async fn test_list_branch_heads_from_source_turn_returns_siblings_only() {
    let store = StateStore::open_memory().await.unwrap();
    let session = store
        .create_session(uuid::Uuid::now_v7(), "default", None)
        .await
        .unwrap();

    store
        .insert_message(
            session,
            turn(0),
            "user",
            &json!([{"type": "text", "text": "root"}]),
            None,
        )
        .await
        .unwrap();
    store
        .insert_message(
            session,
            turn(1),
            "assistant",
            &json!([{"type": "text", "text": "main"}]),
            None,
        )
        .await
        .unwrap();

    let sibling_a = store
        .create_branch_head_from_turn_index(session, "alt-a", Some(0), false)
        .await
        .unwrap();
    let _sibling_b = store
        .create_branch_head_from_turn_index(session, "alt-b", Some(0), false)
        .await
        .unwrap();
    let _other = store
        .create_branch_head_from_turn_index(session, "alt-c", Some(1), false)
        .await
        .unwrap();

    let siblings = store
        .list_branch_heads_from_source_turn(
            session,
            sibling_a.created_from_turn_id.expect("sibling source turn"),
        )
        .await
        .unwrap();

    assert_eq!(siblings.len(), 2);
    assert!(siblings.iter().any(|branch| branch.name == "alt-a"));
    assert!(siblings.iter().any(|branch| branch.name == "alt-b"));
    assert!(!siblings.iter().any(|branch| branch.name == "alt-c"));
}

#[tokio::test]
async fn test_sparse_graph_overlay_is_empty_for_ordinary_session() {
    let store = StateStore::open_memory().await.unwrap();
    let session = store
        .create_session(uuid::Uuid::now_v7(), "default", None)
        .await
        .unwrap();

    store
        .insert_message(
            session,
            turn(0),
            "user",
            &json!([{"type": "text", "text": "ordinary"}]),
            None,
        )
        .await
        .unwrap();

    assert!(
        store
            .list_graph_nodes_for_session(session)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        store
            .list_graph_edges_for_session(session)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn test_sparse_graph_overlay_records_opt_in_branch_relationships() {
    let store = StateStore::open_memory().await.unwrap();
    let session = store
        .create_session(uuid::Uuid::now_v7(), "default", None)
        .await
        .unwrap();

    store
        .insert_message(
            session,
            turn(0),
            "user",
            &json!([{"type": "text", "text": "root"}]),
            None,
        )
        .await
        .unwrap();
    let branch = store
        .create_branch_head_from_turn_index(session, "candidate-a", Some(0), false)
        .await
        .unwrap();
    let branch_public_id = uuid::Uuid::from_slice(&branch.public_id)
        .unwrap()
        .to_string();

    let node = store
        .create_graph_node(
            Some(session),
            "experiment",
            Some("compare refactors"),
            GraphProvenance::new(Some("task-123".to_string()), Some("exec-456".to_string())),
            Some(&json!({"purpose": "speculation"})),
        )
        .await
        .unwrap();
    let node_public_id = uuid::Uuid::from_slice(&node.public_id).unwrap().to_string();
    let source = GraphRef::new("graph_node", node_public_id);
    let target = GraphRef::new("branch_head", branch_public_id);

    let mut edge_input = GraphEdgeCreate::new(source.clone(), target.clone(), "contains");
    edge_input.session_id = Some(session);
    edge_input.source_role = Some("group".to_string());
    edge_input.target_role = Some("candidate".to_string());
    edge_input.metadata = Some(json!({"rank": 1}));
    let edge = store.create_graph_edge(edge_input).await.unwrap();

    assert_eq!(edge.source, source);
    assert_eq!(edge.target, target);
    assert_eq!(edge.relation_kind, "contains");
    assert_eq!(edge.target_role.as_deref(), Some("candidate"));

    let nodes = store.list_graph_nodes_for_session(session).await.unwrap();
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].kind, "experiment");
    assert_eq!(nodes[0].origin_task_id.as_deref(), Some("task-123"));
    assert!(
        nodes[0]
            .metadata
            .as_deref()
            .unwrap()
            .contains("speculation")
    );

    let outgoing = store.list_graph_edges_from(&edge.source).await.unwrap();
    assert_eq!(outgoing.len(), 1);
    assert_eq!(outgoing[0].id, edge.id);

    let incoming = store.list_graph_edges_to(&edge.target).await.unwrap();
    assert_eq!(incoming.len(), 1);
    assert_eq!(incoming[0].id, edge.id);
}

#[tokio::test]
async fn test_prepare_turn_write_target_rejects_stale_branch_head_and_reuses_resolved_turn() {
    let store = StateStore::open_memory().await.unwrap();
    let session = store
        .create_session(uuid::Uuid::now_v7(), "default", None)
        .await
        .unwrap();

    store
        .insert_message(
            session,
            turn(0),
            "user",
            &json!([{"type": "text", "text": "root"}]),
            None,
        )
        .await
        .unwrap();

    let main = store
        .get_active_branch_head(session)
        .await
        .unwrap()
        .expect("active branch");
    let request =
        TurnWriteTarget::branch_head_with_expectation(Some(main.id), main.head_turn_id, 1);
    let resolved = store
        .prepare_turn_write_target(session, request)
        .await
        .unwrap()
        .expect("resolved turn target");

    store
        .insert_message(
            session,
            resolved,
            "assistant",
            &json!([{"type": "text", "text": "first write on resolved turn"}]),
            None,
        )
        .await
        .unwrap();
    store
        .insert_message(
            session,
            resolved,
            "assistant",
            &json!([{"type": "text", "text": "second write on resolved turn"}]),
            None,
        )
        .await
        .unwrap();

    let err = store
        .prepare_turn_write_target(session, request)
        .await
        .expect_err("stale branch advance target should be rejected");
    assert!(
        err.to_string()
            .contains("Branch head changed while preparing turn write target"),
        "unexpected stale target error: {err}"
    );

    let messages = store.get_messages(session, &active_branch()).await.unwrap();
    assert_eq!(messages.len(), 3);
    assert!(messages[1].content.contains("first write on resolved turn"));
    assert!(
        messages[2]
            .content
            .contains("second write on resolved turn")
    );
}

#[tokio::test]
async fn test_hybrid_search() {
    let store = StateStore::open_memory()
        .await
        .expect("Failed to open state store");

    let scope_kind = "session";
    let scope_key = "test-session";

    store
        .insert_memory(
            scope_kind,
            scope_key,
            "The secret code is 12345",
            Some(&[1.0, 0.0]),
            Some("test:semantic:2"),
            Some(2),
            &json!({}),
        )
        .await
        .unwrap();

    store
        .insert_memory(
            scope_kind,
            scope_key,
            "Apples are red",
            Some(&[0.0, 1.0]),
            Some("test:semantic:2"),
            Some(2),
            &json!({}),
        )
        .await
        .unwrap();

    let results = store
        .search_memories(
            scope_kind,
            scope_key,
            Some(&[1.0, 0.0]),
            Some("test:semantic:2"),
            Some(2),
            None,
            10,
            0.0,
            false,
            false,
        )
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0].content.contains("secret code"));
    assert!(results[0].semantic_score.is_some());

    let results = store
        .search_memories(
            scope_kind,
            scope_key,
            None,
            None,
            None,
            Some("12345"),
            10,
            0.0,
            false,
            false,
        )
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].content.contains("secret code"));
    assert!(results[0].lexical_score.is_some());

    let results = store
        .search_memories(
            scope_kind,
            scope_key,
            Some(&[0.0, 1.0]),
            Some("test:semantic:2"),
            Some(2),
            Some("12345"),
            10,
            0.0,
            false,
            false,
        )
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
    let found_secret = results.iter().any(|r| r.content.contains("secret code"));
    let found_apples = results.iter().any(|r| r.content.contains("Apples"));
    assert!(found_secret, "Hybrid search missing lexical result");
    assert!(found_apples, "Hybrid search missing vector result");
}

#[tokio::test]
async fn test_lexical_search_without_embeddings() {
    let store = StateStore::open_memory()
        .await
        .expect("Failed to open state store");
    let scope_kind = "session";
    let scope_key = "test-session";

    store
        .insert_memory(
            scope_kind,
            scope_key,
            "The secret code is 12345",
            None,
            None,
            None,
            &json!({}),
        )
        .await
        .unwrap();

    let results = store
        .search_memories(
            scope_kind,
            scope_key,
            None,
            None,
            None,
            Some("12345"),
            10,
            0.0,
            false,
            false,
        )
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].content.contains("secret code"));

    let hybrid_results = store
        .search_memories(
            scope_kind,
            scope_key,
            Some(&[1.0, 0.0]),
            Some("test:semantic:2"),
            Some(2),
            Some("12345"),
            10,
            0.0,
            false,
            false,
        )
        .await
        .unwrap();
    assert_eq!(hybrid_results.len(), 1);
    assert!(hybrid_results[0].content.contains("secret code"));
}

#[tokio::test]
async fn test_memory_store_returns_public_id_and_updates_retrieval_metadata() {
    let store = StateStore::open_memory()
        .await
        .expect("Failed to open state store");
    let scope_kind = "session";
    let scope_key = "test-session";

    let stored = store
        .insert_memory(
            scope_kind,
            scope_key,
            "alpha memory",
            None,
            None,
            None,
            &json!({ "kind": "note", "source": "test" }),
        )
        .await
        .expect("memory insert should succeed");

    assert_eq!(stored.public_id.len(), 16);
    assert_eq!(stored.storage.as_str(), "lexical_only");
    assert!(stored.stored_at.contains('T'));

    let rows = store
        .search_memories(
            scope_kind,
            scope_key,
            None,
            None,
            None,
            Some("alpha"),
            5,
            0.0,
            true,
            false,
        )
        .await
        .expect("memory search should succeed");

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].public_id, stored.public_id);
    assert_eq!(rows[0].retrieval_count, 1);
    assert!(rows[0].last_retrieved_at.is_some());
    let metadata = rows[0]
        .metadata
        .as_deref()
        .expect("metadata should be present");
    assert!(metadata.contains("\"kind\":\"note\""));
    assert!(metadata.contains("\"source\":\"test\""));
}

#[tokio::test]
async fn test_lexical_only_hybrid_fallback_prefers_best_text_match() {
    let store = StateStore::open_memory()
        .await
        .expect("Failed to open state store");
    let scope_kind = "session";
    let scope_key = "test-session";

    store
        .insert_memory(
            scope_kind,
            scope_key,
            "Compiler errors should stay concise and actionable",
            None,
            None,
            None,
            &json!({ "kind": "preference" }),
        )
        .await
        .unwrap();
    store
        .insert_memory(
            scope_kind,
            scope_key,
            "Cache invalidation should be explicit and session aware",
            None,
            None,
            None,
            &json!({ "kind": "note" }),
        )
        .await
        .unwrap();

    let rows = store
        .search_memories(
            scope_kind,
            scope_key,
            Some(&[1.0, 0.0]),
            Some("test:semantic:2"),
            Some(2),
            Some("compiler concise"),
            5,
            0.0,
            false,
            false,
        )
        .await
        .unwrap();

    assert!(!rows.is_empty());
    assert_eq!(
        rows[0].content,
        "Compiler errors should stay concise and actionable"
    );
    assert!(rows[0].lexical_score.is_some());
    assert!(rows[0].semantic_score.is_none());
}
