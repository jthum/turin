use std::io::{self, BufRead, Write};

use tracing::warn;

use crate::harness::verdict::Verdict;
use crate::kernel::Kernel;

impl Kernel {
    pub(super) fn prompt_for_approval(&self, reason: &str) -> bool {
        warn!(reason = %reason, "Escalation requires user approval");
        eprint!(
            "\x1b[33m\x1b[1m! Approval Required:\x1b[0m {} Allow? (y/n): ",
            reason
        );
        io::stderr().flush().ok();

        let mut input = String::new();
        io::stdin().lock().read_line(&mut input).is_ok() && input.trim().eq_ignore_ascii_case("y")
    }

    pub(super) fn apply_tool_result_hook(
        &self,
        id: &str,
        name: &str,
        args: &serde_json::Value,
        content: String,
        is_error: bool,
    ) -> (String, bool) {
        let harness = self.lock_harness();
        let Some(engine) = &*harness else {
            return (content, is_error);
        };

        let payload = serde_json::json!({
            "id": id,
            "name": name,
            "args": args,
            "output": content,
            "is_error": is_error,
        });

        match engine.evaluate("on_tool_result", payload) {
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
