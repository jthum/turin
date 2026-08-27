use std::collections::BTreeSet;
use std::time::Duration;

use tracing::warn;

use super::FinalToolRecord;
use crate::display;
use crate::harness::verdict::Verdict;
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::session::SessionState;

use super::super::super::PendingToolCall;

impl ExecutionHost {
    pub(super) async fn evaluate_pending_tool_calls(
        &self,
        session: &SessionState,
        pending_tool_calls: &[PendingToolCall],
        exposed_tool_names: Option<&BTreeSet<String>>,
    ) -> (Vec<FinalToolRecord>, Vec<(PendingToolCall, Verdict)>) {
        let mut immediate_records: Vec<FinalToolRecord> = Vec::new();
        let mut validated_calls: Vec<(PendingToolCall, Verdict)> = Vec::new();
        let ansi_stdout = display::stdout_ansi();

        let tool_exec_enabled = self
            .policy_manager
            .typed_snapshot(&crate::kernel::policy::PolicyScope {
                agent_id: Some(session.identity.agent_id().to_string()),
                session_id: Some(session.identity.session_id().to_string()),
                ..crate::kernel::policy::PolicyScope::default()
            })
            .await
            .tool_exec_enabled;
        for tc in pending_tool_calls {
            if !tool_exec_enabled {
                warn!(tool = %tc.name, "Tool call rejected because tool.exec_enabled is false");
                immediate_records.push(FinalToolRecord {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    args: tc.args.clone(),
                    verdict: "exec_disabled".to_string(),
                    duration_ms: 0,
                    content: "[POLICY DENIED] tool.exec_enabled is false".to_string(),
                    is_error: true,
                    emit_exec_start: true,
                    governance_denial: None,
                });
                continue;
            }
            if exposed_tool_names.is_some_and(|names| !names.contains(&tc.name)) {
                warn!(tool = %tc.name, "Tool call rejected because it was not exposed for this inference");
                immediate_records.push(FinalToolRecord {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    args: tc.args.clone(),
                    verdict: "not_exposed".to_string(),
                    duration_ms: 0,
                    content: format!(
                        "[NOT EXPOSED] Tool '{}' was not available for this inference",
                        tc.name
                    ),
                    is_error: true,
                    emit_exec_start: true,
                    governance_denial: None,
                });
                continue;
            }
            let verdict = self.evaluate_tool_call(session, &tc.name, &tc.id, &tc.args);
            match &verdict {
                Verdict::Reject(reason) => {
                    if self.paints_cli_text() {
                        println!(
                            "{}",
                            display::rejection_line("✗ Rejected by harness:", reason, ansi_stdout,)
                        );
                    }
                    warn!(tool = %tc.name, reason = %reason, "Tool rejected by on_tool_call");
                    let msg = format!("[HARNESS REJECTED] Tool '{}' blocked: {}", tc.name, reason);
                    immediate_records.push(FinalToolRecord {
                        id: tc.id.clone(),
                        name: tc.name.clone(),
                        args: tc.args.clone(),
                        verdict: verdict.to_string(),
                        duration_ms: 0,
                        content: msg,
                        is_error: true,
                        emit_exec_start: true,
                        governance_denial: None,
                    });
                }
                Verdict::Escalate(reason) => {
                    warn!(tool = %tc.name, reason = %reason, "Tool requires escalation");
                    let native_tool = session
                        .session_tools
                        .get(&tc.name)
                        .or_else(|| self.tool_registry.get(&tc.name));
                    if let Some(tool) = native_tool
                        && let Some(decision) = self.native_tool_governance_decision(
                            session.identity.agent_id(),
                            &self.session_reference(session),
                            tool.as_ref(),
                        )
                        && !decision.allowed
                    {
                        let detail = decision
                            .reason
                            .clone()
                            .unwrap_or_else(|| format!("Governance denied tool '{}'", tc.name));
                        immediate_records.push(FinalToolRecord {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            args: tc.args.clone(),
                            verdict: "governance_denied".to_string(),
                            duration_ms: 0,
                            content: format!(
                                "[GOVERNANCE DENIED] Tool '{}' blocked: {}",
                                tc.name, detail
                            ),
                            is_error: true,
                            emit_exec_start: true,
                            governance_denial: Some(decision),
                        });
                        continue;
                    }
                    let decision = self.authorize_tool_call(session, tc, reason.clone()).await;
                    if let crate::kernel::tool_authorization::ToolAuthorizationDecision::Deny {
                        reason: denial_reason,
                    } = decision
                    {
                        let detail = denial_reason
                            .as_deref()
                            .map(|reason| format!(": {reason}"))
                            .unwrap_or_default();
                        let msg =
                            format!("[AUTHORIZATION DENIED] Tool '{}' denied{}", tc.name, detail);
                        immediate_records.push(FinalToolRecord {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            args: tc.args.clone(),
                            verdict: "authorization_denied".to_string(),
                            duration_ms: 0,
                            content: msg,
                            is_error: true,
                            emit_exec_start: true,
                            governance_denial: None,
                        });
                    } else {
                        validated_calls.push((tc.clone(), Verdict::Allow));
                    }
                }
                Verdict::Allow | Verdict::Modify(_) => {
                    validated_calls.push((tc.clone(), verdict));
                }
            }
        }

        (immediate_records, validated_calls)
    }

    pub(super) fn apply_tool_rate_limit(
        &self,
        session: &mut SessionState,
        mut immediate_records: Vec<FinalToolRecord>,
        validated_calls: Vec<(PendingToolCall, Verdict)>,
    ) -> (Vec<FinalToolRecord>, Vec<(PendingToolCall, Verdict)>) {
        if validated_calls.is_empty() {
            return (immediate_records, validated_calls);
        }

        let max_calls = self.config.runtime.max_tool_calls_per_window.max(1);
        let window = Duration::from_secs(self.config.runtime.tool_call_window_seconds.max(1));
        let allowed = session.reserve_tool_calls(validated_calls.len(), max_calls, window);
        let allowed = allowed.min(session.reserve_delegation_tool_calls(allowed));
        if allowed >= validated_calls.len() {
            return (immediate_records, validated_calls);
        }

        let blocked = validated_calls.len() - allowed;
        warn!(
            session_id = %session.identity.session_id(),
            allowed,
            blocked,
            limit = max_calls,
            window_seconds = window.as_secs(),
            "Tool rate limit exceeded; blocking excess tool calls",
        );

        let mut permitted = Vec::with_capacity(allowed);
        for (index, (tc, verdict)) in validated_calls.into_iter().enumerate() {
            if index < allowed {
                permitted.push((tc, verdict));
                continue;
            }

            let msg = format!(
                "[RATE LIMITED] Tool '{}' blocked: exceeded built-in safety limit of {} tool calls per {}s session window",
                tc.name,
                max_calls,
                window.as_secs()
            );
            immediate_records.push(FinalToolRecord {
                id: tc.id,
                name: tc.name,
                args: tc.args,
                verdict: "rate_limited".to_string(),
                duration_ms: 0,
                content: msg,
                is_error: true,
                emit_exec_start: true,
                governance_denial: None,
            });
        }

        (immediate_records, permitted)
    }
}
