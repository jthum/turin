use serde::{Deserialize, Serialize};

use crate::kernel::governance::CapabilityDecision;
use crate::kernel::governance::GovernanceGrantSnapshot;
use crate::kernel::governance::GovernanceSnapshot;
use crate::kernel::identity::RuntimeIdentity;
use crate::kernel::session::ContextCompactionCheckpoint;

/// Terminal status for a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskTerminalStatus {
    Success,
    Rejected,
    Conflict,
    MaxTurns,
    Error,
    Cancelled,
    Killed,
}

/// Events related to the overall lifecycle of an agent session or turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LifecycleEvent {
    /// Session begins
    SessionStart { identity: RuntimeIdentity },
    /// Persisted session is resumed into a live runtime
    SessionResume { identity: RuntimeIdentity },
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
        trace_id: String,
        plan_id: Option<String>,
        title: Option<String>,
        prompt: String,
        queue_depth: usize,
    },
    /// Task reaches a terminal status
    TaskComplete {
        identity: RuntimeIdentity,
        task_id: String,
        trace_id: String,
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
        trace_id: String,
        task_turn_index: u32,
    },
    /// Context assembled and mutable just before provider call
    TurnPrepare {
        identity: RuntimeIdentity,
        turn_index: u32,
        task_id: String,
        trace_id: String,
        task_turn_index: u32,
    },
    /// LLM call completes
    TurnEnd {
        identity: RuntimeIdentity,
        turn_index: u32,
        task_id: String,
        trace_id: String,
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
    /// Streaming thinking signature chunk received (provider-agnostic pass-through)
    ThinkingSignatureDelta { signature: String },
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
    /// Governance/capability snapshot emitted for observability (G1)
    GovernanceSnapshot { snapshot: GovernanceSnapshot },
    /// Governance/capability use denied
    GovernanceDenial { decision: CapabilityDecision },
    /// Temporary governance grant issued
    GovernanceGrantIssue { grant: GovernanceGrantSnapshot },
    /// Temporary governance grant used (entered)
    GovernanceGrantUse { grant: GovernanceGrantSnapshot },
    /// Temporary governance grant revoked
    GovernanceGrantRevoke { grant: GovernanceGrantSnapshot },
    /// Durable semantic context checkpoint generated from older session history
    ContextCompaction {
        checkpoint: ContextCompactionCheckpoint,
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
                LifecycleEvent::SessionResume { .. } => "session_resume",
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
                StreamEvent::ThinkingSignatureDelta { .. } => "thinking_signature_delta",
                StreamEvent::MessageEnd { .. } => "message_end",
                StreamEvent::ToolCall { .. } => "tool_call",
            },
            KernelEvent::Audit(e) => match e {
                AuditEvent::ToolResult { .. } => "tool_result",
                AuditEvent::ToolExecStart { .. } => "tool_exec_start",
                AuditEvent::ToolExecEnd { .. } => "tool_exec_end",
                AuditEvent::TokenUsage { .. } => "token_usage",
                AuditEvent::HarnessRejection { .. } => "harness_rejection",
                AuditEvent::GovernanceSnapshot { .. } => "governance_snapshot",
                AuditEvent::GovernanceDenial { .. } => "governance_denial",
                AuditEvent::GovernanceGrantIssue { .. } => "governance_grant_issue",
                AuditEvent::GovernanceGrantUse { .. } => "governance_grant_use",
                AuditEvent::GovernanceGrantRevoke { .. } => "governance_grant_revoke",
                AuditEvent::ContextCompaction { .. } => "context_compaction",
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
            identity: RuntimeIdentity::new("test-123", "default"),
        });
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"session_start\""));
        assert!(json.contains("\"session_id\":\"test-123\""));
    }

    #[test]
    fn test_event_type_names() {
        assert_eq!(
            KernelEvent::Lifecycle(LifecycleEvent::SessionStart {
                identity: RuntimeIdentity::new("x", "default")
            })
            .event_type(),
            "session_start"
        );
        assert_eq!(
            KernelEvent::Lifecycle(LifecycleEvent::SessionResume {
                identity: RuntimeIdentity::new("x", "default")
            })
            .event_type(),
            "session_resume"
        );
        assert_eq!(
            KernelEvent::Audit(AuditEvent::HarnessRejection {
                event: "tool_call".into(),
                reason: "blocked".into()
            })
            .event_type(),
            "harness_rejection"
        );
        assert_eq!(
            KernelEvent::Audit(AuditEvent::GovernanceSnapshot {
                snapshot: GovernanceSnapshot {
                    profile: crate::kernel::config::GovernanceProfile::Open,
                    enforcement_enabled: false,
                    audit_mode: crate::kernel::config::GovernanceAuditMode::Off,
                    audit_persist_before_hooks: false,
                    audit_include_capability_context: false,
                    import_mode: crate::kernel::config::GovernanceImportMode::Legacy,
                    import_allow_unscoped_in_open: true,
                    capabilities_observability_only: true,
                    subject_agent_id: None,
                    roots: vec![],
                    agents: vec![],
                    preset_capabilities: Default::default(),
                    grants_enabled: false,
                    grants_max_ttl_ms: None,
                }
            })
            .event_type(),
            "governance_snapshot"
        );
        assert_eq!(
            KernelEvent::Audit(AuditEvent::GovernanceDenial {
                decision: CapabilityDecision {
                    capability: "fs.write".into(),
                    subject_agent_id: Some("default".into()),
                    subject_module_name: None,
                    subject_root_name: None,
                    subject_grant_id: None,
                    profile: crate::kernel::config::GovernanceProfile::Balanced,
                    enforcement_enabled: true,
                    matched_rule: Some("fs.*".into()),
                    matched_via_wildcard: true,
                    baseline_allowed: false,
                    allowed: false,
                    reason: Some("denied".into()),
                }
            })
            .event_type(),
            "governance_denial"
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
