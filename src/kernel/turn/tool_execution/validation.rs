use std::collections::BTreeSet;
use std::time::Duration;

use tracing::warn;

use super::FinalToolRecord;
use crate::display;
use crate::harness::verdict::Verdict;
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::session::SessionState;

use super::super::super::PendingToolCall;

const MAX_TOOL_CALLS_PER_WINDOW: usize = 32;
const TOOL_CALL_WINDOW: Duration = Duration::from_secs(10);

impl ExecutionHost {
    pub(super) fn evaluate_pending_tool_calls(
        &self,
        session: &SessionState,
        pending_tool_calls: &[PendingToolCall],
        exposed_tool_names: Option<&BTreeSet<String>>,
    ) -> (Vec<FinalToolRecord>, Vec<(PendingToolCall, Verdict)>) {
        let mut immediate_records: Vec<FinalToolRecord> = Vec::new();
        let mut validated_calls: Vec<(PendingToolCall, Verdict)> = Vec::new();
        let ansi_stdout = display::stdout_ansi();

        for tc in pending_tool_calls {
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
                    if !self.json {
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
                    if !self.prompt_for_approval(reason) {
                        if !self.json {
                            println!("{}", display::approval_line(false, ansi_stdout));
                        }
                        let msg =
                            format!("[ESCALATION DENIED] Tool '{}' denied: {}", tc.name, reason);
                        immediate_records.push(FinalToolRecord {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            args: tc.args.clone(),
                            verdict: "escalate_denied".to_string(),
                            duration_ms: 0,
                            content: msg,
                            is_error: true,
                            emit_exec_start: true,
                            governance_denial: None,
                        });
                    } else {
                        if !self.json {
                            println!("{}", display::approval_line(true, ansi_stdout));
                        }
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

        let allowed = session.reserve_tool_calls(
            validated_calls.len(),
            MAX_TOOL_CALLS_PER_WINDOW,
            TOOL_CALL_WINDOW,
        );
        let allowed = allowed.min(session.reserve_delegation_tool_calls(allowed));
        if allowed >= validated_calls.len() {
            return (immediate_records, validated_calls);
        }

        let blocked = validated_calls.len() - allowed;
        warn!(
            session_id = %session.identity.session_id(),
            allowed,
            blocked,
            limit = MAX_TOOL_CALLS_PER_WINDOW,
            window_seconds = TOOL_CALL_WINDOW.as_secs(),
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
                MAX_TOOL_CALLS_PER_WINDOW,
                TOOL_CALL_WINDOW.as_secs()
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
