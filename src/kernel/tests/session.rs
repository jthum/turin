use super::*;

#[test]
fn tool_rate_limit_caps_reserved_calls_within_window() {
    let mut session = SessionState::new();
    assert_eq!(session.reserve_tool_calls(3, 4, Duration::from_secs(60)), 3);
    assert_eq!(session.reserve_tool_calls(3, 4, Duration::from_secs(60)), 1);
    assert_eq!(session.reserve_tool_calls(1, 4, Duration::from_secs(60)), 0);
}

#[test]
fn tool_rate_limit_resets_after_window_elapses() {
    let mut session = SessionState::new();
    assert_eq!(
        session.reserve_tool_calls(2, 2, Duration::from_millis(1)),
        2
    );
    std::thread::sleep(Duration::from_millis(5));
    assert_eq!(
        session.reserve_tool_calls(2, 2, Duration::from_millis(1)),
        2
    );
}

#[test]
fn session_defaults_to_visible_durable_execution_context() {
    let session = SessionState::new();
    assert!(session.execution_id().starts_with("ex_"));
    assert_eq!(session.selected_branch_head_id(), None);
    assert_eq!(
        session.context_target(),
        &ExecutionContextTarget::BranchHead {
            branch_head_id: None
        }
    );
    assert_eq!(session.execution.visibility, ExecutionVisibility::Visible);
    assert_eq!(session.execution.durability, ExecutionDurability::Durable);
    assert_eq!(
        session.execution.write_policy,
        ExecutionWritePolicy::AdvanceBranchHead
    );
    assert_eq!(
        session.execution.conflict_policy,
        ExecutionConflictPolicy::Reject
    );
}

#[test]
fn non_branch_targets_default_to_detached_write_policy() {
    let mut session = SessionState::new();
    session.set_selected_turn_id(7);
    assert_eq!(
        session.execution.write_policy,
        ExecutionWritePolicy::Detached
    );
    assert_eq!(session.next_turn_write_target_request(), None);
    assert_eq!(session.active_turn_write_target(), None);

    session.set_context_target(ExecutionContextTarget::ExternalReference {
        reference: "0123456789abcdef0123456789abcdef".to_string(),
    });
    assert_eq!(
        session.execution.write_policy,
        ExecutionWritePolicy::Detached
    );
    assert_eq!(session.next_turn_write_target_request(), None);
    assert_eq!(session.active_turn_write_target(), None);

    session.set_selected_branch_head_id(Some(11));
    session.set_selected_branch_head_cursor(Some(5), Some(0));
    assert_eq!(
        session.execution.write_policy,
        ExecutionWritePolicy::AdvanceBranchHead
    );
    assert_eq!(
        session.next_turn_write_target_request(),
        Some(TurnWriteTarget::branch_head_with_expectation(
            Some(11),
            Some(5),
            1
        ))
    );
    session.set_active_turn_write_target(Some(TurnWriteTarget::existing_turn(9, 0)));
    assert_eq!(
        session.active_turn_write_target(),
        Some(TurnWriteTarget::existing_turn(9, 0))
    );
}

#[test]
fn branch_write_progression_uses_persisted_head_depth() {
    let mut session = SessionState::new();
    session.set_selected_branch_head_id(Some(11));
    session.set_selected_branch_head_cursor(Some(5), Some(3));
    session.turn_index = 0;

    assert_eq!(
        session.next_turn_write_target_request(),
        Some(TurnWriteTarget::branch_head_with_expectation(
            Some(11),
            Some(5),
            4
        ))
    );
}

#[test]
fn conflict_detached_task_overrides_write_policy_temporarily() {
    let mut session = SessionState::new();
    session.execution.write_policy = ExecutionWritePolicy::AdvanceBranchHead;
    session.begin_conflict_detached_task();

    assert_eq!(
        session.effective_write_policy(),
        ExecutionWritePolicy::Detached
    );
    assert_eq!(session.next_turn_write_target_request(), None);

    session.finish_task_execution_scope();
    assert_eq!(
        session.effective_write_policy(),
        ExecutionWritePolicy::AdvanceBranchHead
    );
}

#[test]
fn task_execution_override_spawns_nested_execution_and_restores_parent() {
    let mut session = SessionState::new();
    session.set_selected_branch_head_id(Some(11));
    session.set_selected_branch_head_cursor(Some(5), Some(2));
    let original_execution = session.execution.clone();

    let refresh_needed = session
        .begin_task_execution_override(Some(&TaskExecutionOverrides {
            context_target: Some(ExecutionContextTarget::TurnId { turn_id: 5 }),
            visibility: Some(ExecutionVisibility::Hidden),
            durability: Some(ExecutionDurability::Ephemeral),
            write_policy: None,
        }))
        .expect("task override should apply");
    assert!(refresh_needed);
    assert_ne!(
        session.execution.execution_id,
        original_execution.execution_id
    );
    assert_eq!(
        session.context_target(),
        &ExecutionContextTarget::TurnId { turn_id: 5 }
    );
    assert_eq!(session.execution.visibility, ExecutionVisibility::Hidden);
    assert_eq!(session.execution.durability, ExecutionDurability::Ephemeral);
    assert_eq!(
        session.execution.write_policy,
        ExecutionWritePolicy::Detached
    );
    assert_eq!(session.selected_branch_head_cursor, None);

    let restore_refresh = session.finish_task_execution_scope();
    assert!(restore_refresh);
    assert_eq!(session.execution, original_execution);
    assert_eq!(
        session.selected_branch_head_cursor,
        Some(BranchHeadCursor {
            turn_id: 5,
            turn_index: 2
        })
    );
}

#[test]
fn task_execution_override_rejects_branch_advancing_non_branch_targets() {
    let mut session = SessionState::new();
    let error = session
        .begin_task_execution_override(Some(&TaskExecutionOverrides {
            context_target: Some(ExecutionContextTarget::TurnId { turn_id: 7 }),
            visibility: None,
            durability: None,
            write_policy: Some(ExecutionWritePolicy::AdvanceBranchHead),
        }))
        .expect_err("invalid override should fail");
    assert!(error.contains("advance_branch_head"));
    assert!(session.active_task.execution_restore.is_none());
}

#[test]
fn hot_history_prunes_old_messages_and_tracks_offset() {
    let mut session = SessionState::new();
    for idx in 0..6 {
        session.history.push(InferenceMessage {
            role: InferenceRole::User,
            content: vec![InferenceContent::Text {
                text: format!("message {idx}"),
            }],
            tool_call_id: None,
        });
    }

    let report = session
        .prune_hot_history(Some(3))
        .expect("history should be pruned");

    assert_eq!(report.dropped_messages, 3);
    assert_eq!(report.retained_messages, 3);
    assert_eq!(report.retained_offset, 3);
    assert_eq!(session.history_message_offset, 3);
    assert_eq!(session.history.len(), 3);
    assert_eq!(session.history[0].role, InferenceRole::User);
}

#[test]
fn hot_history_preserves_tool_result_adjacency_at_boundary() {
    let mut session = SessionState::new();
    session.history.push(InferenceMessage {
        role: InferenceRole::User,
        content: vec![InferenceContent::Text {
            text: "old".to_string(),
        }],
        tool_call_id: None,
    });
    session.history.push(InferenceMessage {
        role: InferenceRole::Assistant,
        content: vec![InferenceContent::ToolUse {
            id: "call-1".to_string(),
            name: "read_file".to_string(),
            input: serde_json::json!({"path": "README.md"}),
        }],
        tool_call_id: None,
    });
    session.history.push(InferenceMessage {
        role: InferenceRole::Tool,
        content: vec![InferenceContent::ToolResult {
            tool_use_id: "call-1".to_string(),
            content: "result".to_string(),
            is_error: false,
        }],
        tool_call_id: None,
    });

    let report = session
        .prune_hot_history(Some(1))
        .expect("history should be pruned");

    assert_eq!(report.dropped_messages, 1);
    assert_eq!(session.history_message_offset, 1);
    assert_eq!(session.history.len(), 2);
    assert_eq!(session.history[0].role, InferenceRole::Assistant);
    assert_eq!(session.history[1].role, InferenceRole::Tool);
}

#[test]
fn hot_history_trims_old_large_tool_results_but_keeps_recent_payloads() {
    let mut session = SessionState::new();
    for idx in 0..10 {
        session.history.push(InferenceMessage {
            role: InferenceRole::Tool,
            content: vec![InferenceContent::ToolResult {
                tool_use_id: format!("call-{idx}"),
                content: "x".repeat(256),
                is_error: false,
            }],
            tool_call_id: None,
        });
    }

    let report = session
        .trim_hot_history_payloads(Some(64))
        .expect("older large tool results should be trimmed");

    assert_eq!(report.trimmed_tool_results, 2);
    assert!(report.dropped_bytes > 0);
    match &session.history[0].content[0] {
        InferenceContent::ToolResult { content, .. } => {
            assert!(content.contains("full content remains in persisted session history"));
            assert!(content.contains("original_bytes=256"));
        }
        other => panic!("expected tool result, got {other:?}"),
    }
    match &session.history[9].content[0] {
        InferenceContent::ToolResult { content, .. } => {
            assert_eq!(content.len(), 256);
        }
        other => panic!("expected tool result, got {other:?}"),
    }
}

#[test]
fn replacing_full_history_resets_hot_window_offset() {
    let mut session = SessionState::new();
    for idx in 0..4 {
        session.history.push(InferenceMessage {
            role: InferenceRole::User,
            content: vec![InferenceContent::Text {
                text: format!("message {idx}"),
            }],
            tool_call_id: None,
        });
    }
    session.prune_hot_history(Some(2));

    session.replace_full_history(vec![InferenceMessage {
        role: InferenceRole::User,
        content: vec![InferenceContent::Text {
            text: "rehydrated".to_string(),
        }],
        tool_call_id: None,
    }]);

    assert_eq!(session.history_message_offset, 0);
    assert!(!session.history_is_pruned());
    assert_eq!(session.history.len(), 1);
}
