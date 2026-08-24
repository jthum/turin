use tracing::warn;

use crate::harness::verdict::Verdict;
use crate::kernel::PendingToolCall;
use crate::kernel::event::{AuditEvent, KernelEvent};
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::harness_contract::HarnessHook;
use crate::kernel::session::SessionState;
use crate::kernel::tool_authorization::{ToolAuthorizationDecision, ToolAuthorizationRequest};

impl ExecutionHost {
    pub(super) async fn authorize_tool_call(
        &self,
        session: &SessionState,
        tool_call: &PendingToolCall,
        reason: String,
    ) -> ToolAuthorizationDecision {
        let request = ToolAuthorizationRequest::new(
            session.identity.clone(),
            session.runtime_slot_id.clone(),
            tool_call.id.clone(),
            tool_call.name.clone(),
            tool_call.args.clone(),
            reason,
        );
        self.persist_event(
            session,
            &KernelEvent::Audit(AuditEvent::ToolAuthorizationRequested {
                request: request.clone(),
            }),
        )
        .await;
        let request_id = request.id.clone();
        let decision = self
            .tool_authorizer
            .authorize(request, session.cancel_token.clone())
            .await;
        let decision = decision.normalized();
        self.persist_event(
            session,
            &KernelEvent::Audit(AuditEvent::ToolAuthorizationResolved {
                request_id,
                decision: decision.clone(),
            }),
        )
        .await;
        decision
    }

    pub(super) async fn apply_tool_result_hook(
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
        let verdict = {
            let engine = harness.lock().expect("session harness mutex poisoned");
            engine.evaluate_hook(HarnessHook::ToolResult {
                id,
                name,
                args,
                output: &content,
                is_error,
            })
        };

        match verdict {
            Ok(Verdict::Allow) => (content, is_error),
            Ok(Verdict::Reject(reason)) => (
                format!(
                    "[HARNESS REJECTED RESULT] Tool '{}' result blocked: {}",
                    name, reason
                ),
                true,
            ),
            Ok(Verdict::Escalate(reason)) => {
                let call = PendingToolCall {
                    id: id.to_string(),
                    name: name.to_string(),
                    args: args.clone(),
                };
                if matches!(
                    self.authorize_tool_call(session, &call, reason).await,
                    ToolAuthorizationDecision::Approve
                ) {
                    (content, is_error)
                } else {
                    (
                        format!("[AUTHORIZATION DENIED] Tool '{}' result denied", name),
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
