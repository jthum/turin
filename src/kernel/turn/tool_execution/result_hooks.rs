use std::io::{self, BufRead, Write};

use tracing::warn;

use crate::display;
use crate::harness::verdict::Verdict;
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::harness_contract::HarnessHook;
use crate::kernel::session::SessionState;

impl ExecutionHost {
    pub(super) fn prompt_for_approval(&self, reason: &str) -> bool {
        warn!(reason = %reason, "Escalation requires user approval");
        let ansi_stderr = display::stderr_ansi();
        eprint!(
            "{} {} Allow? (y/n): ",
            display::approval_prompt_prefix(ansi_stderr),
            reason
        );
        io::stderr().flush().ok();

        tokio::task::block_in_place(|| {
            let mut input = String::new();
            io::stdin().lock().read_line(&mut input).is_ok()
                && input.trim().eq_ignore_ascii_case("y")
        })
    }

    pub(super) fn apply_tool_result_hook(
        &self,
        session: &SessionState,
        id: &str,
        name: &str,
        args: &serde_json::Value,
        content: String,
        is_error: bool,
    ) -> (String, bool) {
        let Some(harness) = self.session_harness_engine(session) else {
            return (content, is_error);
        };
        let engine = harness.lock().expect("session harness mutex poisoned");

        match engine.evaluate_hook(HarnessHook::ToolResult {
            id,
            name,
            args,
            output: &content,
            is_error,
        }) {
            Ok(Verdict::Allow) => (content, is_error),
            Ok(Verdict::Reject(reason)) => (
                format!(
                    "[HARNESS REJECTED RESULT] Tool '{}' result blocked: {}",
                    name, reason
                ),
                true,
            ),
            Ok(Verdict::Escalate(reason)) => {
                if self.prompt_for_approval(&reason) {
                    (content, is_error)
                } else {
                    (
                        format!(
                            "[ESCALATION DENIED] Tool '{}' result denied by user: {}",
                            name, reason
                        ),
                        true,
                    )
                }
            }
            Ok(Verdict::Modify(val)) => {
                if let Some(s) = val.as_str() {
                    return (s.to_string(), is_error);
                }
                if let Some(obj) = val.as_object() {
                    let new_content = obj
                        .get("output")
                        .and_then(|v| v.as_str())
                        .or_else(|| obj.get("content").and_then(|v| v.as_str()))
                        .map(ToString::to_string)
                        .unwrap_or(content);
                    let new_is_error = obj
                        .get("is_error")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(is_error);
                    return (new_content, new_is_error);
                }
                warn!(tool = %name, "on_tool_result returned unsupported MODIFY payload; ignoring");
                (content, is_error)
            }
            Err(e) => {
                warn!(error = %e, "Harness on_tool_result error");
                (content, is_error)
            }
        }
    }
}
