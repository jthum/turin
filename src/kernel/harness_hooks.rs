use tracing::{info, warn};

use crate::harness::verdict::Verdict;
use crate::kernel::Kernel;

impl Kernel {
    /// Evaluate harness `on_tool_call` hook.
    ///
    /// Returns the composed verdict. If no harness is loaded, returns `Allow`.
    pub(crate) fn evaluate_tool_call(
        &self,
        name: &str,
        id: &str,
        args: &serde_json::Value,
    ) -> Verdict {
        let harness = self.lock_harness();
        if let Some(ref engine) = *harness {
            let payload = serde_json::json!({
                "name": name,
                "id": id,
                "args": args,
            });
            match engine.evaluate("on_tool_call", payload) {
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
    pub fn evaluate_token_usage(&self, input_tokens: u64, output_tokens: u64) {
        let harness = self.lock_harness();
        if let Some(ref engine) = *harness {
            let payload = serde_json::json!({
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            });
            match engine.evaluate("on_token_usage", payload) {
                Ok(verdict) => {
                    if verdict.is_rejected() {
                        warn!(
                            reason = %verdict.reason().unwrap_or("budget exceeded"),
                            "Token usage harness rejection"
                        );
                    }
                }
                Err(e) => {
                    warn!(error = %e, "Harness on_token_usage error");
                }
            }
        }
    }
}
