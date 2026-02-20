use serde::{Deserialize, Serialize};

use crate::kernel::identity::RuntimeIdentity;

/// Terminal status for a task.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskTerminalStatus {
    Success,
    Rejected,
    MaxTurns,
    Error,
    Cancelled,
}

/// Events related to the overall lifecycle of an agent session or turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LifecycleEvent {
    /// Session begins
    SessionStart { identity: RuntimeIdentity },
    /// Session completes
    SessionEnd {
        identity: RuntimeIdentity,
        turn_count: u32,
        total_input_tokens: u64,
        total_output_tokens: u64,
    },
    /// Task begins
    TaskStart {
        identity: RuntimeIdentity,
        task_id: String,
        plan_id: Option<String>,
        title: Option<String>,
        prompt: String,
        queue_depth: usize,
    },
    /// Task reaches a terminal status
    TaskComplete {
        identity: RuntimeIdentity,
        task_id: String,
        plan_id: Option<String>,
        status: TaskTerminalStatus,
        task_turn_count: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Plan reaches completion
    PlanComplete {
        identity: RuntimeIdentity,
        plan_id: String,
        title: String,
        total_tasks: usize,
        completed_tasks: usize,
    },
    /// No queued tasks remain
    AllTasksComplete { identity: RuntimeIdentity },
    /// New LLM call begins
    TurnStart {
        identity: RuntimeIdentity,
        turn_index: u32,
        task_id: String,
        task_turn_index: u32,
    },
    /// Context assembled and mutable just before provider call
    TurnPrepare {
        identity: RuntimeIdentity,
        turn_index: u32,
        task_id: String,
        task_turn_index: u32,
    },
    /// LLM call completes
    TurnEnd {
        identity: RuntimeIdentity,
        turn_index: u32,
        task_id: String,
        task_turn_index: u32,
        has_tool_calls: bool,
    },
}

/// Ephemeral high-frequency events from the LLM provider's stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Streaming message begins
    MessageStart { role: String, model: String },
    /// Streaming text chunk received
    MessageDelta { content_delta: String },
    /// Streaming thinking chunk received
    ThinkingDelta { thinking: String },
    /// Complete message assembled
    MessageEnd {
        role: String,
        input_tokens: u64,
        output_tokens: u64,
    },
    /// LLM requests a tool execution (produced by stream)
    ToolCall {
        id: String,
        name: String,
        args: serde_json::Value,
    },
}

/// Durable events for auditing, logging, and metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AuditEvent {
    /// Tool execution completed
    ToolResult {
        id: String,
        output: String,
        is_error: bool,
    },
    /// Tool execution begins (for logging/timing)
    ToolExecStart { id: String, name: String },
    /// Tool execution completes
    ToolExecEnd { id: String, success: bool },
    /// Token/cost accounting update
    TokenUsage {
        input_tokens: u64,
        output_tokens: u64,
        cost_usd: f64,
    },
    /// Harness engine rejected an action
    HarnessRejection {
        /// Which event type was rejected (e.g., "tool_call")
        event: String,
        /// Human-readable reason from the harness script
        reason: String,
    },
}

/// Every action in Turin produces a typed `KernelEvent`.
///
/// Refactored to separate events by purpose:
/// 1. **Lifecycle** — Session/Turn boundaries
/// 2. **Stream** — Ephemeral LLM output
/// 3. **Audit** — Durable execution logs
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum KernelEvent {
    Lifecycle(LifecycleEvent),
    Stream(StreamEvent),
    Audit(AuditEvent),
}

impl KernelEvent {
    /// Get the event type name as a string.
    pub fn event_type(&self) -> &'static str {
        match self {
            KernelEvent::Lifecycle(e) => match e {
                LifecycleEvent::SessionStart { .. } => "session_start",
                LifecycleEvent::SessionEnd { .. } => "session_end",
                LifecycleEvent::TaskStart { .. } => "task_start",
                LifecycleEvent::TaskComplete { .. } => "task_complete",
                LifecycleEvent::PlanComplete { .. } => "plan_complete",
                LifecycleEvent::AllTasksComplete { .. } => "all_tasks_complete",
                LifecycleEvent::TurnStart { .. } => "turn_start",
                LifecycleEvent::TurnPrepare { .. } => "turn_prepare",
                LifecycleEvent::TurnEnd { .. } => "turn_end",
            },
            KernelEvent::Stream(e) => match e {
                StreamEvent::MessageStart { .. } => "message_start",
                StreamEvent::MessageDelta { .. } => "message_delta",
                StreamEvent::ThinkingDelta { .. } => "thinking_delta",
                StreamEvent::MessageEnd { .. } => "message_end",
                StreamEvent::ToolCall { .. } => "tool_call",
            },
            KernelEvent::Audit(e) => match e {
                AuditEvent::ToolResult { .. } => "tool_result",
                AuditEvent::ToolExecStart { .. } => "tool_exec_start",
                AuditEvent::ToolExecEnd { .. } => "tool_exec_end",
                AuditEvent::TokenUsage { .. } => "token_usage",
                AuditEvent::HarnessRejection { .. } => "harness_rejection",
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_serialization() {
        let event = KernelEvent::Lifecycle(LifecycleEvent::SessionStart {
            identity: RuntimeIdentity::new("test-123"),
        });
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"session_start\""));
        assert!(json.contains("\"session_id\":\"test-123\""));
    }

    #[test]
    fn test_event_type_names() {
        assert_eq!(
            KernelEvent::Lifecycle(LifecycleEvent::SessionStart {
                identity: RuntimeIdentity::new("x")
            })
            .event_type(),
            "session_start"
        );
        assert_eq!(
            KernelEvent::Audit(AuditEvent::HarnessRejection {
                event: "tool_call".into(),
                reason: "blocked".into()
            })
            .event_type(),
            "harness_rejection"
        );
    }

    #[test]
    fn test_tool_call_event_serialization() {
        let event = KernelEvent::Stream(StreamEvent::ToolCall {
            id: "call_1".to_string(),
            name: "read_file".to_string(),
            args: serde_json::json!({ "path": "main.rs" }),
        });
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"tool_call\""));
        assert!(json.contains("\"name\":\"read_file\""));
        assert!(json.contains("\"path\":\"main.rs\""));
    }
}
