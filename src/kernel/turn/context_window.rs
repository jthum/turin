use crate::inference::provider::{InferenceContent, InferenceMessage, InferenceRole};
use crate::kernel::config::ProviderConfig;
use crate::kernel::session::ContextCompactionCheckpoint;

pub(crate) const DEFAULT_CONTEXT_WINDOW_TOKENS: u32 = 128_000;
const DEFAULT_OUTPUT_RESERVE_TOKENS: u32 = 4_096;
const MIN_OUTPUT_RESERVE_TOKENS: u32 = 1_024;
const RECENT_MESSAGES_NO_TRUNCATE: usize = 6;
const SUMMARY_RECENT_MESSAGES: usize = 8;
const SUMMARY_MAX_OUTPUT_TOKENS: u32 = 512;
const TRUNCATED_TOOL_RESULT_MARKER: &str = "[tool result omitted to fit context window]";
const CONTEXT_SUMMARY_OPEN_TAG: &str = "<turin_context_summary>";
const CONTEXT_SUMMARY_CLOSE_TAG: &str = "</turin_context_summary>";

#[derive(Debug, Clone)]
pub(crate) struct EffectiveRequestContext {
    pub system_prompt: String,
    pub messages: Vec<InferenceMessage>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompactionReport {
    pub used_tokens_before: u32,
    pub used_tokens_after: u32,
    pub context_window_tokens: u32,
    pub input_budget_tokens: u32,
    pub truncated_tool_results: usize,
    pub dropped_messages: usize,
}

impl CompactionReport {
    pub(crate) fn applied(&self) -> bool {
        self.truncated_tool_results > 0 || self.dropped_messages > 0
    }

    pub(crate) fn fits_budget(&self) -> bool {
        self.used_tokens_after <= self.input_budget_tokens
    }
}

pub(crate) fn resolve_context_window_tokens(provider: Option<&ProviderConfig>) -> u32 {
    provider
        .and_then(|provider| provider.context_window_tokens)
        .unwrap_or(DEFAULT_CONTEXT_WINDOW_TOKENS)
}

pub(crate) fn effective_input_budget_tokens(
    context_window_tokens: u32,
    max_output_tokens: Option<u32>,
    thinking_budget: Option<u32>,
) -> u32 {
    let requested_output_reserve = max_output_tokens
        .unwrap_or(DEFAULT_OUTPUT_RESERVE_TOKENS)
        .max(thinking_budget.unwrap_or(0))
        .max(MIN_OUTPUT_RESERVE_TOKENS);
    let bounded_reserve = requested_output_reserve.min(context_window_tokens / 2);
    context_window_tokens.saturating_sub(bounded_reserve)
}

pub(crate) fn estimate_history_input_tokens(
    system_prompt: &str,
    messages: &[InferenceMessage],
) -> u32 {
    estimate_support_tokens(system_prompt, &[]) + estimate_messages_tokens(messages)
}

pub(crate) fn estimate_request_input_tokens(
    system_prompt: &str,
    messages: &[InferenceMessage],
    tools: &[serde_json::Value],
) -> u32 {
    estimate_support_tokens(system_prompt, tools) + estimate_messages_tokens(messages)
}

pub(crate) fn effective_request_context(
    system_prompt: &str,
    messages: &[InferenceMessage],
    checkpoint: Option<&ContextCompactionCheckpoint>,
) -> EffectiveRequestContext {
    let Some(checkpoint) = checkpoint else {
        return EffectiveRequestContext {
            system_prompt: system_prompt.to_string(),
            messages: messages.to_vec(),
        };
    };

    let covered_message_count = checkpoint.covered_message_count.min(messages.len());
    let mut effective_system_prompt = String::with_capacity(
        system_prompt.len() + checkpoint.summary.len() + CONTEXT_SUMMARY_OPEN_TAG.len() + 64,
    );
    effective_system_prompt.push_str(system_prompt);
    effective_system_prompt.push_str("\n\n");
    effective_system_prompt.push_str(CONTEXT_SUMMARY_OPEN_TAG);
    effective_system_prompt.push_str(
        "\nThis summary replaces older session history that Turin compacted to fit the context window.\n",
    );
    effective_system_prompt.push_str(checkpoint.summary.trim());
    effective_system_prompt.push('\n');
    effective_system_prompt.push_str(CONTEXT_SUMMARY_CLOSE_TAG);

    EffectiveRequestContext {
        system_prompt: effective_system_prompt,
        messages: messages[covered_message_count..].to_vec(),
    }
}

pub(crate) fn target_checkpoint_coverage(
    history_len: usize,
    checkpoint: Option<&ContextCompactionCheckpoint>,
) -> Option<usize> {
    let target_covered_message_count = history_len.saturating_sub(SUMMARY_RECENT_MESSAGES);
    if target_covered_message_count == 0 {
        return None;
    }

    match checkpoint {
        None => Some(target_covered_message_count),
        Some(existing)
            if target_covered_message_count
                > existing
                    .covered_message_count
                    .saturating_add(SUMMARY_RECENT_MESSAGES) =>
        {
            Some(target_covered_message_count)
        }
        _ => None,
    }
}

pub(crate) fn build_checkpoint_summary_request(
    history: &[InferenceMessage],
    checkpoint: Option<&ContextCompactionCheckpoint>,
    target_covered_message_count: usize,
    context_window_tokens: u32,
) -> (String, Vec<InferenceMessage>, u32) {
    let summary_system_prompt = [
        "You are compacting Turin session history for later turns.",
        "Write a concise durable summary of the covered conversation so future turns retain the important context.",
        "Keep only durable facts:",
        "- user goals, preferences, and constraints",
        "- decisions and commitments",
        "- important tool outcomes and file changes",
        "- unresolved questions or next steps",
        "Do not copy transcript wording or pleasantries.",
        "Return plain text with short bullet-like lines.",
    ]
    .join("\n");

    let start_index = checkpoint
        .map(|checkpoint| checkpoint.covered_message_count.min(history.len()))
        .unwrap_or(0);
    let end_index = target_covered_message_count.min(history.len());

    let mut summary_messages = Vec::new();
    if let Some(checkpoint) = checkpoint {
        summary_messages.push(InferenceMessage {
            role: InferenceRole::User,
            content: vec![InferenceContent::Text {
                text: format!(
                    "Existing compacted context summary to retain and update:\n{}",
                    checkpoint.summary.trim()
                ),
            }],
            tool_call_id: None,
        });
    }
    if start_index < end_index {
        summary_messages.extend_from_slice(&history[start_index..end_index]);
    }

    let summary_input_budget =
        effective_input_budget_tokens(context_window_tokens, Some(SUMMARY_MAX_OUTPUT_TOKENS), None);
    let (summary_messages, _) = compact_messages_for_input_budget(
        &summary_system_prompt,
        &summary_messages,
        &[],
        context_window_tokens,
        summary_input_budget,
    );

    (
        summary_system_prompt,
        summary_messages,
        SUMMARY_MAX_OUTPUT_TOKENS,
    )
}

pub(crate) fn compact_messages_for_input_budget(
    system_prompt: &str,
    messages: &[InferenceMessage],
    tools: &[serde_json::Value],
    context_window_tokens: u32,
    input_budget_tokens: u32,
) -> (Vec<InferenceMessage>, CompactionReport) {
    let used_tokens_before = estimate_request_input_tokens(system_prompt, messages, tools);
    if used_tokens_before <= input_budget_tokens {
        return (
            messages.to_vec(),
            CompactionReport {
                used_tokens_before,
                used_tokens_after: used_tokens_before,
                context_window_tokens,
                input_budget_tokens,
                truncated_tool_results: 0,
                dropped_messages: 0,
            },
        );
    }

    let mut compacted = messages.to_vec();
    let mut truncated_tool_results = truncate_older_tool_results(&mut compacted);
    let mut used_tokens_after = estimate_request_input_tokens(system_prompt, &compacted, tools);
    let mut dropped_messages = 0;

    if used_tokens_after > input_budget_tokens {
        let (slid_messages, slid_dropped_messages) =
            slide_window_to_budget(&compacted, system_prompt, tools, input_budget_tokens);
        dropped_messages += slid_dropped_messages;
        compacted = slid_messages;
        used_tokens_after = estimate_request_input_tokens(system_prompt, &compacted, tools);
    }

    if compacted.is_empty() && !messages.is_empty() {
        compacted.push(messages[messages.len() - 1].clone());
        used_tokens_after = estimate_request_input_tokens(system_prompt, &compacted, tools);
        truncated_tool_results = 0;
        dropped_messages = messages.len().saturating_sub(1);
    }

    (
        compacted,
        CompactionReport {
            used_tokens_before,
            used_tokens_after,
            context_window_tokens,
            input_budget_tokens,
            truncated_tool_results,
            dropped_messages,
        },
    )
}

fn estimate_support_tokens(system_prompt: &str, tools: &[serde_json::Value]) -> u32 {
    estimate_text_tokens(system_prompt)
        .saturating_add(8)
        .saturating_add(estimate_tool_tokens(tools))
}

fn estimate_tool_tokens(tools: &[serde_json::Value]) -> u32 {
    tools
        .iter()
        .map(|tool| serde_json::to_string(tool).unwrap_or_default())
        .map(|tool| estimate_text_tokens(&tool).saturating_add(8))
        .fold(0u32, u32::saturating_add)
}

fn estimate_messages_tokens(messages: &[InferenceMessage]) -> u32 {
    messages
        .iter()
        .map(estimate_message_tokens)
        .fold(0u32, u32::saturating_add)
}

fn estimate_message_tokens(message: &InferenceMessage) -> u32 {
    let role_overhead: u32 = match message.role {
        InferenceRole::User => 6,
        InferenceRole::Assistant => 8,
        InferenceRole::Tool => 8,
    };

    let content_tokens = message
        .content
        .iter()
        .map(estimate_content_tokens)
        .fold(0u32, u32::saturating_add);

    role_overhead.saturating_add(content_tokens)
}

fn estimate_content_tokens(content: &InferenceContent) -> u32 {
    match content {
        InferenceContent::Text { text } => estimate_text_tokens(text).saturating_add(4),
        InferenceContent::ToolUse { id, name, input } => estimate_text_tokens(id)
            .saturating_add(estimate_text_tokens(name))
            .saturating_add(estimate_json_tokens(input))
            .saturating_add(12),
        InferenceContent::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => estimate_text_tokens(tool_use_id)
            .saturating_add(estimate_text_tokens(content))
            .saturating_add(if *is_error { 2 } else { 0 })
            .saturating_add(12),
        InferenceContent::Thinking { content, signature } => estimate_text_tokens(content)
            .saturating_add(signature.as_deref().map(estimate_text_tokens).unwrap_or(0))
            .saturating_add(6),
    }
}

fn estimate_json_tokens(value: &serde_json::Value) -> u32 {
    estimate_text_tokens(&serde_json::to_string(value).unwrap_or_default())
}

fn estimate_text_tokens(text: &str) -> u32 {
    let chars = text.chars().count() as u32;
    chars.saturating_add(3) / 4
}

fn truncate_older_tool_results(messages: &mut [InferenceMessage]) -> usize {
    let truncate_upto = messages.len().saturating_sub(RECENT_MESSAGES_NO_TRUNCATE);
    messages
        .iter_mut()
        .take(truncate_upto)
        .map(truncate_tool_results_in_message)
        .sum()
}

fn truncate_tool_results_in_message(message: &mut InferenceMessage) -> usize {
    if message.role != InferenceRole::Tool {
        return 0;
    }

    let mut truncated = 0;
    for content in &mut message.content {
        if let InferenceContent::ToolResult { content, .. } = content
            && content != TRUNCATED_TOOL_RESULT_MARKER
        {
            *content = TRUNCATED_TOOL_RESULT_MARKER.to_string();
            truncated += 1;
        }
    }

    truncated
}

fn slide_window_to_budget(
    messages: &[InferenceMessage],
    system_prompt: &str,
    tools: &[serde_json::Value],
    input_budget_tokens: u32,
) -> (Vec<InferenceMessage>, usize) {
    if messages.is_empty() {
        return (Vec::new(), 0);
    }

    let support_tokens = estimate_support_tokens(system_prompt, tools);
    let mut kept_rev = Vec::new();
    let mut used_tokens = support_tokens;

    for message in messages.iter().rev() {
        let message_tokens = estimate_message_tokens(message);
        if !kept_rev.is_empty() && used_tokens.saturating_add(message_tokens) > input_budget_tokens
        {
            break;
        }
        used_tokens = used_tokens.saturating_add(message_tokens);
        kept_rev.push(message.clone());
    }

    if kept_rev.is_empty() {
        kept_rev.push(messages[messages.len() - 1].clone());
    }

    kept_rev.reverse();
    let mut dropped_messages = messages.len().saturating_sub(kept_rev.len());

    while kept_rev
        .first()
        .is_some_and(|message| message.role == InferenceRole::Tool)
    {
        kept_rev.remove(0);
        dropped_messages += 1;
    }

    if kept_rev.is_empty() {
        kept_rev.push(messages[messages.len() - 1].clone());
        dropped_messages = messages.len().saturating_sub(1);
    }

    (kept_rev, dropped_messages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::provider::InferenceRole;
    use crate::kernel::session::ContextCompactionCheckpoint;

    fn text_message(role: InferenceRole, text: &str) -> InferenceMessage {
        InferenceMessage {
            role,
            content: vec![InferenceContent::Text {
                text: text.to_string(),
            }],
            tool_call_id: None,
        }
    }

    #[test]
    fn estimate_history_tokens_is_non_zero_for_text() {
        let tokens = estimate_history_input_tokens(
            "System prompt",
            &[text_message(InferenceRole::User, "Hello from Turin")],
        );

        assert!(tokens > 0);
    }

    #[test]
    fn compaction_truncates_old_tool_results_before_dropping_messages() {
        let messages = vec![
            text_message(InferenceRole::User, "First prompt"),
            InferenceMessage {
                role: InferenceRole::Tool,
                content: vec![InferenceContent::ToolResult {
                    tool_use_id: "tool_1".to_string(),
                    content: "x".repeat(4_000),
                    is_error: false,
                }],
                tool_call_id: None,
            },
            text_message(InferenceRole::User, "Follow-up one"),
            text_message(InferenceRole::Assistant, "Follow-up answer one"),
            text_message(InferenceRole::User, "Follow-up two"),
            text_message(InferenceRole::Assistant, "Follow-up answer two"),
            text_message(InferenceRole::Assistant, "Recent answer"),
            text_message(InferenceRole::User, "Current prompt"),
        ];

        let (compacted, report) =
            compact_messages_for_input_budget("system", &messages, &[], 512, 384);

        assert!(report.truncated_tool_results > 0);
        assert!(report.applied());
        assert!(report.fits_budget());
        assert_eq!(compacted.len(), messages.len());
        match &compacted[1].content[0] {
            InferenceContent::ToolResult { content, .. } => {
                assert_eq!(content, TRUNCATED_TOOL_RESULT_MARKER);
            }
            other => panic!("expected tool result, got {other:?}"),
        }
    }

    #[test]
    fn compaction_slides_message_window_when_needed() {
        let messages = vec![
            text_message(InferenceRole::User, &"a".repeat(1_000)),
            text_message(InferenceRole::Assistant, &"b".repeat(1_000)),
            text_message(InferenceRole::User, "current prompt"),
        ];

        let (compacted, report) =
            compact_messages_for_input_budget("system", &messages, &[], 512, 192);

        assert!(report.dropped_messages > 0);
        assert!(report.applied());
        let last = compacted.last().expect("expected last compacted message");
        assert_eq!(last.role, InferenceRole::User);
        match &last.content[0] {
            InferenceContent::Text { text } => assert_eq!(text, "current prompt"),
            other => panic!("expected trailing text message, got {other:?}"),
        }
    }

    #[test]
    fn effective_request_context_injects_checkpoint_and_drops_covered_history() {
        let messages = vec![
            text_message(InferenceRole::User, "old 1"),
            text_message(InferenceRole::Assistant, "old 2"),
            text_message(InferenceRole::User, "recent"),
        ];
        let checkpoint = ContextCompactionCheckpoint {
            summary: "important durable context".to_string(),
            covered_message_count: 2,
            generated_at_turn_index: 4,
            provider_name: "mock".to_string(),
            model: "mock-model".to_string(),
        };

        let effective = effective_request_context("base system", &messages, Some(&checkpoint));

        assert!(
            effective
                .system_prompt
                .contains("important durable context")
        );
        assert_eq!(effective.messages.len(), 1);
        match &effective.messages[0].content[0] {
            InferenceContent::Text { text } => assert_eq!(text, "recent"),
            other => panic!("expected recent text message, got {other:?}"),
        }
    }
}
