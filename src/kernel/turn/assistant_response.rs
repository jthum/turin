use anyhow::Result;
use tracing::warn;

use crate::inference::content::encode_content_json;
use crate::inference::provider::{InferenceContent, InferenceMessage, InferenceRole};
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::harness_contract::HarnessHook;
use crate::kernel::session::SessionState;

use super::super::PendingToolCall;
use super::super::event::{KernelEvent, LifecycleEvent};
use super::TurnContext;

impl ExecutionHost {
    pub(super) async fn finalize_assistant_turn_output(
        &mut self,
        session: &mut SessionState,
        turn_ctx: &TurnContext,
        response_thinking: &str,
        response_thinking_signature: Option<&str>,
        response_text: &str,
        pending_tool_calls: &[PendingToolCall],
    ) -> Result<bool> {
        let has_tool_calls = !pending_tool_calls.is_empty();

        let mut persisted_content: Vec<InferenceContent> = Vec::new();
        if !response_text.is_empty() {
            persisted_content.push(InferenceContent::Text {
                text: response_text.to_string(),
            });
        }
        for tc in pending_tool_calls {
            persisted_content.push(InferenceContent::ToolUse {
                id: tc.id.clone(),
                name: tc.name.clone(),
                input: tc.args.clone(),
            });
        }
        self.persist_turn_message(
            session,
            "assistant",
            &encode_content_json(&persisted_content),
        )
        .await?;

        self.persist_event(
            session,
            &KernelEvent::Lifecycle(LifecycleEvent::TurnEnd {
                identity: session.identity.clone(),
                turn_index: session.turn_index,
                task_id: turn_ctx.task_id.clone(),
                trace_id: turn_ctx.trace_id.clone(),
                task_turn_index: turn_ctx.task_turn_index,
                has_tool_calls,
            }),
        );

        if let Some(harness) = self.session_harness_engine(session)
            && let Ok(engine) = harness.lock()
        {
            let session_id = self.session_reference(session);
            if let Err(e) = engine.evaluate_hook(HarnessHook::TurnEnd {
                identity: &session.identity,
                session_id: &session_id,
                task_id: &turn_ctx.task_id,
                trace_id: &turn_ctx.trace_id,
                plan_id: turn_ctx.plan_id.as_deref(),
                turn_index: session.turn_index,
                task_turn_index: turn_ctx.task_turn_index,
                has_tool_calls,
            }) {
                warn!(error = %e, "Harness on_turn_end error");
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
        let origin = session.active_history_origin();
        session.history.push_with_origin(
            InferenceMessage {
                role: InferenceRole::Assistant,
                content: assistant_content,
                tool_call_id: None,
            },
            origin,
        );

        Ok(has_tool_calls)
    }
}
