//! Agent turn execution — streaming, tool dispatch, and result collection.
//!
//! This module contains turn-level execution for the agent loop: LLM inference,
//! stream processing, hook evaluation, parallel tool execution, and side effects.

mod assistant_response;
mod preflight;
mod streaming;
mod tool_execution;

use anyhow::{Context, Result};
use std::time::Duration;

use crate::harness::context::RequestOptionsOverride;
use crate::inference::provider::{self};
use crate::kernel::session::SessionState;
use crate::tools::ToolContext;

use super::Kernel;
use preflight::TurnPreflight;

#[derive(Debug, Clone)]
pub(crate) struct TurnContext {
    pub task_id: String,
    pub plan_id: Option<String>,
    pub task_turn_index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TurnOutcome {
    Continue,
    Complete,
    Rejected,
}

fn merge_request_option_overrides(
    mut options: provider::RequestOptions,
    overrides: &RequestOptionsOverride,
) -> Result<provider::RequestOptions> {
    for (header_name, header_value) in &overrides.headers {
        options = options
            .with_header(header_name, header_value)
            .with_context(|| format!("invalid request header '{}'", header_name))?;
    }

    if let Some(max_retries) = overrides.max_retries {
        options = options.with_max_retries(max_retries);
    }

    if overrides.request_timeout_secs.is_some() || overrides.total_timeout_secs.is_some() {
        let mut timeout_policy = options.timeout_policy.clone().unwrap_or_default();
        if let Some(request_timeout_secs) = overrides.request_timeout_secs {
            timeout_policy.request_timeout = Some(Duration::from_secs(request_timeout_secs));
        }
        if let Some(total_timeout_secs) = overrides.total_timeout_secs {
            timeout_policy.total_timeout = Some(Duration::from_secs(total_timeout_secs));
        }
        options = options.with_timeout_policy(timeout_policy);
    }

    Ok(options)
}

impl Kernel {
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
        let response_text = stream_output.response_text;
        let pending_tool_calls = stream_output.pending_tool_calls;

        let has_tool_calls = self
            .finalize_assistant_turn_output(session, turn_ctx, &response_text, &pending_tool_calls)
            .await;

        if !has_tool_calls {
            return Ok(TurnOutcome::Complete);
        }

        // Execute tools.
        self.execute_tool_calls(session, tool_ctx, pending_tool_calls)
            .await
    }
}
