use tracing::warn;

use crate::inference::provider::{InferenceContent, InferenceMessage, InferenceRole};
use crate::kernel::session::SessionState;

use super::super::event::{KernelEvent, LifecycleEvent};
use super::super::{Kernel, PendingToolCall};
use super::TurnContext;

impl Kernel {
    pub(super) async fn finalize_assistant_turn_output(
        &mut self,
        session: &mut SessionState,
        turn_ctx: &TurnContext,
        response_thinking: &str,
        response_thinking_signature: Option<&str>,
        response_text: &str,
        pending_tool_calls: &[PendingToolCall],
    ) -> bool {
        let has_tool_calls = !pending_tool_calls.is_empty();

        self.persist_event(
            session,
            &KernelEvent::Lifecycle(LifecycleEvent::TurnEnd {
                identity: session.identity.clone(),
                turn_index: session.turn_index,
                task_id: turn_ctx.task_id.clone(),
                task_turn_index: turn_ctx.task_turn_index,
                has_tool_calls,
            }),
        );

        {
            let runtime = self.runtime_for_session(session);
            let harness = runtime.lock_engine();
            if let Some(ref engine) = *harness
                && let Err(e) = engine.evaluate(
                    "on_turn_end",
                    serde_json::json!({
                        "identity": session.identity.clone(),
                        "session_id": session.identity.session_id(),
                        "task_id": turn_ctx.task_id.clone(),
                        "plan_id": turn_ctx.plan_id.clone(),
                        "turn_index": session.turn_index,
                        "task_turn_index": turn_ctx.task_turn_index,
                        "has_tool_calls": has_tool_calls,
                    }),
                )
            {
                warn!(error = %e, "Harness on_turn_end error");
            }
        }

        if let Ok(store) = self.store_manager.get_default().await {
            let content: Vec<serde_json::Value> = {
                let mut parts = Vec::new();
                if !response_text.is_empty() {
                    parts.push(serde_json::json!({"type": "text", "text": response_text}));
                }
                for tc in pending_tool_calls {
                    parts.push(serde_json::json!({
                        "type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args,
                    }));
                }
                parts
            };
            if let Some(iid) = session.internal_id {
                let _ = store
                    .insert_message(
                        iid,
                        session.turn_index,
                        "assistant",
                        &serde_json::Value::Array(content),
                        None,
                    )
                    .await;
            }
        }

        let mut assistant_content: Vec<InferenceContent> = Vec::new();
        if !response_thinking.is_empty() {
            // Preserve in-memory thinking content for provider roundtrips (e.g. Anthropic-compatible
            // tool/result matching) without persisting it to the transcript store.
            assistant_content.push(InferenceContent::Thinking {
                content: response_thinking.to_string(),
                signature: response_thinking_signature.map(str::to_owned),
            });
        }
        if !response_text.is_empty() {
            assistant_content.push(InferenceContent::Text {
                text: response_text.to_string(),
            });
        }
        for tc in pending_tool_calls {
            assistant_content.push(InferenceContent::ToolUse {
                id: tc.id.clone(),
                name: tc.name.clone(),
                input: tc.args.clone(),
            });
        }
        session.history.push(InferenceMessage {
            role: InferenceRole::Assistant,
            content: assistant_content,
            tool_call_id: None,
        });

        has_tool_calls
    }
}
