use std::collections::HashSet;

use anyhow::{Context, Result};

use super::{SessionReadTarget, StateStore, ToolExecutionRow, TurnWriteTarget};
use crate::perf_diagnostics::{perf_stage, perf_stage_finish};

impl StateStore {
    #[allow(clippy::too_many_arguments)]
    pub async fn insert_tool_execution(
        &self,
        session_id: i64,
        target: TurnWriteTarget,
        tool_call_id: &str,
        tool_name: &str,
        args: &serde_json::Value,
        output: Option<&str>,
        is_error: bool,
        duration_ms: Option<u64>,
        verdict: &str,
    ) -> Result<()> {
        let conn = self.connect().await?;
        let args_str = serde_json::to_string(args)?;
        let turn = self
            .resolve_turn_for_write_target(session_id, target)
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!("No active branch head available for session {}", session_id)
            })?;
        conn.execute(
            "INSERT INTO tool_executions (turn_id, tool_call_id, tool_name, args, output, is_error, duration_ms, verdict) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            turso::params![
                turn.id,
                tool_call_id,
                tool_name,
                args_str,
                output,
                is_error as i64,
                duration_ms.map(|d| d as i64),
                verdict,
            ],
        )
        .await
        .with_context(|| format!("Failed to insert turn tool execution for session: {}", session_id))?;
        Ok(())
    }

    pub async fn get_tool_executions(
        &self,
        session_id: i64,
        target: &SessionReadTarget,
    ) -> Result<Vec<ToolExecutionRow>> {
        self.get_tool_executions_for_turn_indexes(session_id, target, None)
            .await
    }

    pub async fn get_tool_executions_for_turn_indexes(
        &self,
        session_id: i64,
        target: &SessionReadTarget,
        turn_indexes: Option<&HashSet<u32>>,
    ) -> Result<Vec<ToolExecutionRow>> {
        perf_stage!(
            tool_query_stage,
            "persistence.tools.query",
            None,
            serde_json::json!({
                "internal_session_id": session_id,
                "visible_turns": turn_indexes.map(HashSet::len),
            })
        );
        let conn = self.connect().await?;
        let mut execs = Vec::new();
        let turns = match target {
            SessionReadTarget::ActiveBranch => self.branch_path_turns(session_id, None).await?,
            SessionReadTarget::BranchHead(branch_head_id) => {
                self.branch_path_turns(session_id, Some(*branch_head_id))
                    .await?
            }
            SessionReadTarget::TurnId(turn_id) => {
                self.turn_path_to_turn_id(session_id, *turn_id).await?
            }
            SessionReadTarget::SelectedPath(turn_ids) => {
                self.turn_rows_for_selected_path(session_id, turn_ids)
                    .await?
            }
        };
        let _path_turns = turns.len();
        let selected_turns = turns
            .into_iter()
            .filter(|turn| turn_indexes.is_none_or(|indexes| indexes.contains(&turn.branch_depth)))
            .collect::<Vec<_>>();
        let _tool_queries = selected_turns.len();
        for turn in selected_turns {
            let mut rows = conn
                .query(
                    "SELECT id, tool_call_id, tool_name, args, output, is_error, duration_ms, verdict, created_at FROM tool_executions WHERE turn_id = ?1 ORDER BY id",
                    [turn.id],
                )
                .await?;
            while let Some(row) = rows.next().await? {
                execs.push(ToolExecutionRow {
                    id: row.get::<i64>(0)?,
                    session_id,
                    turn_index: turn.branch_depth,
                    tool_call_id: row.get::<String>(1)?,
                    tool_name: row.get::<String>(2)?,
                    args: row.get::<String>(3)?,
                    output: row.get::<Option<String>>(4)?,
                    is_error: row.get::<i64>(5)? != 0,
                    duration_ms: row.get::<Option<i64>>(6)?.map(|d| d as u64),
                    verdict: row.get::<String>(7)?,
                    created_at: row.get::<String>(8)?,
                });
            }
        }
        perf_stage_finish!(
            tool_query_stage,
            "ok",
            serde_json::json!({
                "path_turns": _path_turns,
                "turn_queries": _tool_queries,
                "rows": execs.len(),
            })
        );
        Ok(execs)
    }
}
