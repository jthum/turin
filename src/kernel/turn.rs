//! Agent turn execution — streaming, tool dispatch, and result collection.
//!
//! This module contains turn-level execution for the agent loop: LLM inference,
//! stream processing, hook evaluation, parallel tool execution, and side effects.

mod assistant_response;
pub(crate) mod context_window;
mod preflight;
mod streaming;
mod tool_execution;

use anyhow::Result;
use std::collections::BTreeSet;
use std::sync::Arc;

use crate::kernel::session::SessionState;
use crate::tools::ToolContext;

use super::execution_host::ExecutionHost;
use preflight::TurnPreflight;

#[derive(Debug, Clone)]
pub(crate) struct TurnContext {
    pub task_id: String,
    pub trace_id: String,
    pub plan_id: Option<String>,
    pub task_turn_index: u32,
    pub allowed_native_tools: Arc<BTreeSet<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TurnOutcome {
    Continue,
    Complete,
    Rejected,
    Cancelled,
}

impl ExecutionHost {
    /// Execute a single turn of the agent loop.
    pub(crate) async fn execute_turn(
        &mut self,
        session: &mut SessionState,
        tool_ctx: &ToolContext,
        turn_ctx: &TurnContext,
    ) -> Result<TurnOutcome> {
        let prepared = match self.prepare_turn_stream(session, turn_ctx).await? {
            TurnPreflight::Rejected => return Ok(TurnOutcome::Rejected),
            TurnPreflight::Ready(prepared) => prepared,
        };
        let provider_name = prepared.provider_name;
        let model = prepared.model;
        let stream = prepared.stream;

        let stream_output = self
            .collect_turn_stream_output(session, &provider_name, &model, stream)
            .await?;
        if stream_output.cancelled {
            return Ok(TurnOutcome::Cancelled);
        }
        let response_thinking = stream_output.response_thinking;
        let response_thinking_signature = stream_output.response_thinking_signature;
        let response_text = stream_output.response_text;
        let pending_tool_calls = stream_output.pending_tool_calls;

        let has_tool_calls = self
            .finalize_assistant_turn_output(
                session,
                turn_ctx,
                &response_thinking,
                response_thinking_signature.as_deref(),
                &response_text,
                &pending_tool_calls,
            )
            .await;

        if !has_tool_calls {
            return Ok(TurnOutcome::Complete);
        }

        // Execute tools.
        self.execute_tool_calls(session, tool_ctx, pending_tool_calls)
            .await
    }
}
