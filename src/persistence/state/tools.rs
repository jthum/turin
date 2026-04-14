use anyhow::{Context, Result};

use super::{SessionReadTarget, StateStore, ToolExecutionRow, TurnWriteTarget};

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
            "INSERT INTO tool_executions (session_id, turn_index, tool_call_id, tool_name, args, output, is_error, duration_ms, verdict) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            turso::params![
                session_id,
                target.turn_index() as i64,
                tool_call_id,
                tool_name,
                args_str.clone(),
                output,
                is_error as i64,
                duration_ms.map(|d| d as i64),
                verdict,
            ],
        )
        .await
        .with_context(|| format!("Failed to insert tool execution for session: {}", session_id))?;
        conn.execute(
            "INSERT INTO turn_tool_executions (turn_id, tool_call_id, tool_name, args, output, is_error, duration_ms, verdict) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
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
        for turn in turns {
            let mut rows = conn
                .query(
                    "SELECT id, tool_call_id, tool_name, args, output, is_error, duration_ms, verdict, created_at FROM turn_tool_executions WHERE turn_id = ?1 ORDER BY id",
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
        Ok(execs)
    }
}
