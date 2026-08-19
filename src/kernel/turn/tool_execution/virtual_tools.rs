use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;

use anyhow::Result;

use crate::harness::virtual_tools::{
    VirtualToolNestedResult, VirtualToolPlan, VirtualToolResultOutput, VirtualToolResultResolution,
};
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::session::SessionState;
use crate::tools::ToolContext;

use super::{FinalToolRecord, MAX_VIRTUAL_TOOL_DEPTH};
use crate::kernel::PendingToolCall;

impl ExecutionHost {
    pub(super) fn build_virtual_pending_tool_calls(
        &self,
        session: &SessionState,
        parent_tool_call_id: &str,
        plan: VirtualToolPlan,
        current_virtual_stack: &[String],
    ) -> Result<Vec<PendingToolCall>> {
        let declared_tool_names: BTreeSet<String> = {
            if let Some(harness) = self.session_harness_engine(session) {
                let engine = harness.lock().expect("session harness mutex poisoned");
                engine
                    .declared_virtual_tools()?
                    .into_iter()
                    .map(|tool| tool.name)
                    .collect()
            } else {
                BTreeSet::new()
            }
        };

        let mut out = Vec::with_capacity(plan.calls.len());
        for (index, call) in plan.calls.into_iter().enumerate() {
            if declared_tool_names.contains(&call.name) {
                if current_virtual_stack.iter().any(|name| name == &call.name) {
                    let mut chain = current_virtual_stack.to_vec();
                    chain.push(call.name.clone());
                    anyhow::bail!("virtual tool recursion detected: {}", chain.join(" -> "));
                }
                if current_virtual_stack.len() >= MAX_VIRTUAL_TOOL_DEPTH {
                    let mut chain = current_virtual_stack.to_vec();
                    chain.push(call.name.clone());
                    anyhow::bail!(
                        "virtual tool nesting depth exceeded (max {}): {}",
                        MAX_VIRTUAL_TOOL_DEPTH,
                        chain.join(" -> ")
                    );
                }
            }

            out.push(PendingToolCall {
                id: format!("{}::vt{}", parent_tool_call_id, index + 1),
                name: call.name,
                args: call.args,
            });
        }
        Ok(out)
    }

    pub(super) fn aggregate_virtual_tool_records(
        &self,
        records: &[FinalToolRecord],
    ) -> (String, bool) {
        if records.is_empty() {
            return (
                "Virtual tool completed without producing any nested tool output.".to_string(),
                false,
            );
        }

        let any_error = records.iter().any(|record| record.is_error);
        if records.len() == 1 {
            return (records[0].content.clone(), any_error);
        }

        let combined = records
            .iter()
            .enumerate()
            .map(|(index, record)| {
                let status = if record.is_error { "error" } else { "ok" };
                format!(
                    "Call {}: {} [{}]\n{}",
                    index + 1,
                    record.name,
                    status,
                    record.content
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        (combined, any_error)
    }

    pub(super) fn virtual_tool_nested_results(
        &self,
        records: &[FinalToolRecord],
    ) -> Vec<VirtualToolNestedResult> {
        records
            .iter()
            .map(|record| VirtualToolNestedResult {
                id: record.id.clone(),
                name: record.name.clone(),
                args: record.args.clone(),
                verdict: record.verdict.clone(),
                duration_ms: record.duration_ms,
                content: record.content.clone(),
                is_error: record.is_error,
            })
            .collect()
    }

    pub(super) fn invoke_virtual_result_handler(
        &self,
        session: &SessionState,
        key: &str,
        nested_results: &[VirtualToolNestedResult],
        nested_error: bool,
    ) -> Result<VirtualToolResultResolution> {
        let payload = if nested_results.len() == 1 {
            serde_json::to_value(&nested_results[0])?
        } else {
            serde_json::to_value(nested_results)?
        };
        if let Some(harness) = self.session_harness_engine(session) {
            let engine = harness.lock().expect("session harness mutex poisoned");
            engine.invoke_virtual_tool_result_handler(key, payload, nested_error)
        } else {
            anyhow::bail!(
                "virtual tool result handler '{}' could not access harness engine",
                key
            );
        }
    }

    pub(super) fn execute_virtual_plan_hidden<'a>(
        &'a mut self,
        session: &'a mut SessionState,
        tool_ctx: &'a ToolContext,
        tool_call_path: String,
        current_virtual_stack: Vec<String>,
        plan: VirtualToolPlan,
    ) -> Pin<Box<dyn Future<Output = (String, bool)> + Send + 'a>> {
        Box::pin(async move {
            let result_handler_key = plan.result_handler_key.clone();
            match self.build_virtual_pending_tool_calls(
                session,
                &tool_call_path,
                plan,
                &current_virtual_stack,
            ) {
                Ok(pending_virtual_calls) => {
                    self.execute_expanded_virtual_calls(
                        session,
                        tool_ctx,
                        tool_call_path,
                        pending_virtual_calls,
                        current_virtual_stack,
                        result_handler_key,
                    )
                    .await
                }
                Err(err) => {
                    if let Some(ref key) = result_handler_key {
                        self.discard_virtual_result_handler(session, key);
                    }
                    (format!("Error: {}", err), true)
                }
            }
        })
    }

    pub(super) fn execute_expanded_virtual_calls<'a>(
        &'a mut self,
        session: &'a mut SessionState,
        tool_ctx: &'a ToolContext,
        tool_call_path: String,
        pending_virtual_calls: Vec<PendingToolCall>,
        current_virtual_stack: Vec<String>,
        result_handler_key: Option<String>,
    ) -> Pin<Box<dyn Future<Output = (String, bool)> + Send + 'a>> {
        Box::pin(async move {
            let nested_records = match self
                .execute_tool_calls_hidden(
                    session,
                    tool_ctx,
                    pending_virtual_calls,
                    current_virtual_stack.clone(),
                )
                .await
            {
                Ok(records) => records,
                Err(error) => {
                    if let Some(ref key) = result_handler_key {
                        self.discard_virtual_result_handler(session, key);
                    }
                    return (format!("Virtual tool persistence failed: {error}"), true);
                }
            };
            if session.cancel_token.is_cancelled() {
                if let Some(ref key) = result_handler_key {
                    self.discard_virtual_result_handler(session, key);
                }
                return ("Virtual tool execution cancelled".to_string(), true);
            }

            let nested_results = self.virtual_tool_nested_results(&nested_records);
            let nested_error = nested_results.iter().any(|record| record.is_error);
            let output = if let Some(ref key) = result_handler_key {
                match self.invoke_virtual_result_handler(
                    session,
                    key,
                    &nested_results,
                    nested_error,
                ) {
                    Ok(VirtualToolResultResolution::Output(output)) => Ok(output),
                    Ok(VirtualToolResultResolution::Plan(next_plan)) => Err(next_plan),
                    Err(err) => Ok(VirtualToolResultOutput {
                        content: format!("Error: {}", err),
                        is_error: true,
                    }),
                }
            } else {
                let (content, is_error) = self.aggregate_virtual_tool_records(&nested_records);
                Ok(VirtualToolResultOutput { content, is_error })
            };

            match output {
                Ok(output) => (output.content, output.is_error),
                Err(next_plan) => {
                    self.execute_virtual_plan_hidden(
                        session,
                        tool_ctx,
                        format!("{}::cb", tool_call_path),
                        current_virtual_stack,
                        next_plan,
                    )
                    .await
                }
            }
        })
    }

    pub(super) fn discard_virtual_result_handler(&self, session: &SessionState, key: &str) {
        if let Some(harness) = self.session_harness_engine(session) {
            let engine = harness.lock().expect("session harness mutex poisoned");
            let _ = engine.discard_virtual_tool_result_handler(key);
        }
    }
}
