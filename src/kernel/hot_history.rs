use crate::inference::provider::{InferenceContent, InferenceMessage, InferenceRole};
use crate::kernel::config::HotHistoryConfig;
use crate::kernel::session::SessionState;

const RECENT_PAYLOAD_MESSAGES: usize = 8;
const TRUNCATED_TOOL_RESULT_MARKER: &str = "[older tool result omitted from hot memory; full content remains in persisted session history]";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PruneReport {
    pub dropped_messages: usize,
    pub retained_messages: usize,
    pub retained_offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PayloadTrimReport {
    pub trimmed_tool_results: usize,
    pub dropped_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct HotHistoryReport {
    pub prune: Option<PruneReport>,
    pub payload_trim: Option<PayloadTrimReport>,
}

impl HotHistoryReport {
    pub(crate) fn applied(self) -> bool {
        self.prune.is_some() || self.payload_trim.is_some()
    }
}

pub(crate) fn apply(session: &mut SessionState, config: &HotHistoryConfig) -> HotHistoryReport {
    HotHistoryReport {
        prune: prune(session, config.effective_max_messages()),
        payload_trim: trim_payloads(session, config.effective_max_tool_result_bytes()),
    }
}

pub(crate) fn prune(
    session: &mut SessionState,
    max_messages: Option<usize>,
) -> Option<PruneReport> {
    let max_messages = max_messages?;
    if session.history.len() <= max_messages {
        return None;
    }

    let mut retain_from = session.history.len().saturating_sub(max_messages);
    while retain_from > 0
        && boundary_requires_previous(
            &session.history[retain_from - 1],
            &session.history[retain_from],
        )
    {
        retain_from -= 1;
    }

    if retain_from == 0 {
        return None;
    }

    session.history.drain(0..retain_from);
    session.history.shrink_to_fit();
    session.history_message_offset = session.history_message_offset.saturating_add(retain_from);
    Some(PruneReport {
        dropped_messages: retain_from,
        retained_messages: session.history.len(),
        retained_offset: session.history_message_offset,
    })
}

pub(crate) fn trim_payloads(
    session: &mut SessionState,
    max_tool_result_bytes: Option<usize>,
) -> Option<PayloadTrimReport> {
    let max_tool_result_bytes = max_tool_result_bytes?;
    let trim_before = session
        .history
        .len()
        .saturating_sub(RECENT_PAYLOAD_MESSAGES);
    let mut trimmed_tool_results = 0usize;
    let mut dropped_bytes = 0usize;

    for message in session.history.iter_mut().take(trim_before) {
        for part in &mut message.content {
            let InferenceContent::ToolResult {
                content,
                is_error: false,
                ..
            } = part
            else {
                continue;
            };
            if content.len() <= max_tool_result_bytes
                || content.starts_with(TRUNCATED_TOOL_RESULT_MARKER)
            {
                continue;
            }

            let original_len = content.len();
            *content = format!("{TRUNCATED_TOOL_RESULT_MARKER} original_bytes={original_len}");
            content.shrink_to_fit();
            trimmed_tool_results = trimmed_tool_results.saturating_add(1);
            dropped_bytes =
                dropped_bytes.saturating_add(original_len.saturating_sub(content.len()));
        }
    }

    if trimmed_tool_results == 0 {
        return None;
    }

    Some(PayloadTrimReport {
        trimmed_tool_results,
        dropped_bytes,
    })
}

fn boundary_requires_previous(previous: &InferenceMessage, next: &InferenceMessage) -> bool {
    next.role == InferenceRole::Tool
        && (previous.role == InferenceRole::Assistant
            || previous
                .content
                .iter()
                .any(|content| matches!(content, InferenceContent::ToolUse { .. })))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_message(text: String) -> InferenceMessage {
        InferenceMessage {
            role: InferenceRole::User,
            content: vec![InferenceContent::Text { text }],
            tool_call_id: None,
        }
    }

    #[test]
    fn prunes_old_messages_and_tracks_offset() {
        let mut session = SessionState::new();
        for idx in 0..6 {
            session.history.push(text_message(format!("message {idx}")));
        }

        let report = prune(&mut session, Some(3)).expect("history should be pruned");

        assert_eq!(report.dropped_messages, 3);
        assert_eq!(report.retained_messages, 3);
        assert_eq!(report.retained_offset, 3);
        assert_eq!(session.history_message_offset, 3);
        assert_eq!(session.history.len(), 3);
        assert_eq!(session.history[0].role, InferenceRole::User);
    }

    #[test]
    fn preserves_tool_result_adjacency_at_boundary() {
        let mut session = SessionState::new();
        session.history.push(text_message("old".to_string()));
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

        let report = prune(&mut session, Some(1)).expect("history should be pruned");

        assert_eq!(report.dropped_messages, 1);
        assert_eq!(session.history_message_offset, 1);
        assert_eq!(session.history.len(), 2);
        assert_eq!(session.history[0].role, InferenceRole::Assistant);
        assert_eq!(session.history[1].role, InferenceRole::Tool);
    }

    #[test]
    fn trims_old_large_tool_results_but_keeps_recent_payloads() {
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

        let report = trim_payloads(&mut session, Some(64))
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
    fn apply_reports_pruning_and_payload_trimming() {
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

        let config = HotHistoryConfig {
            max_messages: Some(9),
            max_tool_result_bytes: Some(64),
            ..HotHistoryConfig::default()
        };
        let report = apply(&mut session, &config);

        assert!(report.applied());
        assert_eq!(report.prune.expect("prune report").dropped_messages, 1);
        assert_eq!(
            report
                .payload_trim
                .expect("payload trim report")
                .trimmed_tool_results,
            1
        );
        assert_eq!(session.history_message_offset, 1);
    }
}
