use std::collections::HashMap;

use anyhow::{Context, Result};

use super::FinalToolRecord;
use crate::display;
use crate::inference::content::encode_content_json;
use crate::inference::provider::{InferenceContent, InferenceMessage, InferenceRole};
use crate::kernel::event::{AuditEvent, KernelEvent};
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::session::SessionState;

use super::super::super::PendingToolCall;

impl ExecutionHost {
    pub(super) async fn finalize_tool_records(
        &mut self,
        session: &mut SessionState,
        pending_tool_calls: &[PendingToolCall],
        mut final_by_id: HashMap<String, FinalToolRecord>,
        publish_to_history: bool,
    ) -> Result<Vec<FinalToolRecord>> {
        let mut tool_results: Vec<InferenceContent> = Vec::new();
        let mut finalized_records = Vec::new();
        let ansi_stdout = display::stdout_ansi();

        for tc in pending_tool_calls {
            let Some(mut record) = final_by_id.remove(&tc.id) else {
                continue;
            };

            if record.emit_exec_start {
                self.persist_event(
                    session,
                    &KernelEvent::Audit(AuditEvent::ToolExecStart {
                        id: record.id.clone(),
                        name: record.name.clone(),
                    }),
                )
                .await;
            }

            if let Some(decision) = record.governance_denial.take() {
                self.persist_event(
                    session,
                    &KernelEvent::Audit(AuditEvent::GovernanceDenial { decision }),
                )
                .await;
            }

            let (content, is_error) = self.apply_tool_result_hook(
                session,
                &record.id,
                &record.name,
                &record.args,
                record.content,
                record.is_error,
            );
            record.content = content;
            record.is_error = is_error;

            if let (Some(iid), Some(target)) =
                (session.internal_id, session.active_turn_write_target())
            {
                let store = self
                    .store_manager
                    .open(&session.store_selector)
                    .await
                    .context("Failed to open state store for tool execution persistence")?;
                let _guard = session.persistence_lock.lock().await;
                store
                    .insert_tool_execution(
                        iid,
                        target,
                        &record.id,
                        &record.name,
                        &record.args,
                        Some(&record.content),
                        record.is_error,
                        Some(record.duration_ms),
                        &record.verdict,
                    )
                    .await
                    .with_context(|| {
                        format!("Failed to persist tool execution '{}'", record.name)
                    })?;
            }

            self.persist_event(
                session,
                &KernelEvent::Audit(AuditEvent::ToolResult {
                    id: record.id.clone(),
                    output: record.content.clone(),
                    is_error: record.is_error,
                }),
            )
            .await;
            self.persist_event(
                session,
                &KernelEvent::Audit(AuditEvent::ToolExecEnd {
                    id: record.id.clone(),
                    success: !record.is_error,
                }),
            )
            .await;

            if !self.json {
                println!(
                    "{}",
                    display::tool_status_line(&record.name, !record.is_error, ansi_stdout)
                );
            }

            if publish_to_history {
                tool_results.push(InferenceContent::ToolResult {
                    tool_use_id: record.id.clone(),
                    content: record.content.clone(),
                    is_error: record.is_error,
                });
            }

            finalized_records.push(record);
        }

        if publish_to_history && !tool_results.is_empty() {
            self.persist_turn_message(session, "tool_result", &encode_content_json(&tool_results))
                .await?;
            let origin = session.active_history_origin();
            session.history.push_with_origin(
                InferenceMessage {
                    role: InferenceRole::Tool,
                    content: tool_results.clone(),
                    tool_call_id: None,
                },
                origin,
            );
        }

        Ok(finalized_records)
    }
}
