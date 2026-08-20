use tracing::{info, warn};

use crate::harness::verdict::Verdict;
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::harness_contract::HarnessHook;
use crate::kernel::policy::PolicyScope;
use crate::kernel::session::SessionState;

#[derive(Debug, Clone)]
pub(crate) enum TokenUsageHookAction {
    Continue,
    RejectTask { reason: String },
    RejectSession { reason: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenUsageRejectMode {
    Informational,
    EnforceTask,
    EnforceSession,
}

impl ExecutionHost {
    /// Evaluate harness `on_tool_call` hook.
    ///
    /// Returns the composed verdict. If no harness is loaded, returns `Allow`.
    pub(crate) fn evaluate_tool_call(
        &self,
        session: &SessionState,
        name: &str,
        id: &str,
        args: &serde_json::Value,
    ) -> Verdict {
        if let Some(harness) = self.session_harness_engine(session) {
            let engine = harness.lock().expect("session harness mutex poisoned");
            match engine.evaluate_hook(HarnessHook::ToolCall { name, id, args }) {
                Ok(verdict) => {
                    if !verdict.is_allowed() {
                        info!(tool = %name, verdict = %verdict, "Harness verdict");
                    }
                    verdict
                }
                Err(e) => {
                    // Harness evaluation errors are non-fatal; default to ALLOW.
                    warn!(error = %e, "Harness on_tool_call error");
                    Verdict::Allow
                }
            }
        } else {
            Verdict::Allow
        }
    }

    /// Evaluate harness `on_token_usage` hook.
    ///
    /// This fires after each turn. If a harness rejects, it logs a warning.
    ///
    /// The current default behavior is informational (REJECT logs but does not halt the loop).
    /// This is intentional for flexibility-first operation; future enforcement modes should be
    /// exposed as explicit runtime/governance knobs rather than hard-coded kernel policy.
    pub(crate) async fn evaluate_token_usage(
        &self,
        session: &SessionState,
        task_turn_count: u32,
    ) -> TokenUsageHookAction {
        let verdict_result = {
            if let Some(harness) = self.session_harness_engine(session) {
                let engine = harness.lock().expect("session harness mutex poisoned");
                let task_budget = session.active_task_budget_snapshot(task_turn_count);
                Some(engine.evaluate_hook(HarnessHook::TokenUsage {
                    input_tokens: session.total_input_tokens,
                    output_tokens: session.total_output_tokens,
                    task_started_at_unix_ms: task_budget.task_started_at_unix_ms,
                    task_elapsed_ms: task_budget.task_elapsed_ms,
                    task_input_tokens: task_budget.task_input_tokens,
                    task_output_tokens: task_budget.task_output_tokens,
                    task_turn_count: task_budget.task_turn_count,
                }))
            } else {
                None
            }
        };

        let Some(result) = verdict_result else {
            return TokenUsageHookAction::Continue;
        };

        match result {
            Ok(verdict) => {
                if !verdict.is_rejected() {
                    return TokenUsageHookAction::Continue;
                }

                let reason = verdict.reason().unwrap_or("budget exceeded").to_string();
                match self.token_usage_reject_mode(session).await {
                    TokenUsageRejectMode::Informational => {
                        warn!(reason = %reason, "Token usage harness rejection");
                        TokenUsageHookAction::Continue
                    }
                    TokenUsageRejectMode::EnforceTask => {
                        warn!(reason = %reason, "Token usage harness rejection (enforcing task)");
                        TokenUsageHookAction::RejectTask { reason }
                    }
                    TokenUsageRejectMode::EnforceSession => {
                        warn!(reason = %reason, "Token usage harness rejection (enforcing session)");
                        TokenUsageHookAction::RejectSession { reason }
                    }
                }
            }
            Err(e) => {
                warn!(error = %e, "Harness on_token_usage error");
                TokenUsageHookAction::Continue
            }
        }
    }

    async fn token_usage_reject_mode(&self, session: &SessionState) -> TokenUsageRejectMode {
        let scope = PolicyScope {
            agent_id: Some(session.identity.agent_id().to_string()),
            session_id: Some(session.identity.session_id().to_string()),
            ..PolicyScope::default()
        };
        let snapshot = self.policy_manager.snapshot(&scope).await;
        match snapshot
            .get("hook.token_usage.reject_mode")
            .and_then(|v| v.as_str())
        {
            Some("enforce_task") => TokenUsageRejectMode::EnforceTask,
            Some("enforce_session") => TokenUsageRejectMode::EnforceSession,
            Some("informational") | None => TokenUsageRejectMode::Informational,
            Some(other) => {
                warn!(
                    mode = %other,
                    "Unknown hook.token_usage.reject_mode; falling back to informational"
                );
                TokenUsageRejectMode::Informational
            }
        }
    }
}
