use anyhow::{Context, Result};

use super::{StateStore, ToolExecutionRow};

impl StateStore {
    #[allow(clippy::too_many_arguments)]
    pub async fn insert_tool_execution(
        &self,
        session_id: i64,
        turn_index: u32,
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
        conn.execute(
            "INSERT INTO tool_executions (session_id, turn_index, tool_call_id, tool_name, args, output, is_error, duration_ms, verdict) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            turso::params![
                session_id,
                turn_index as i64,
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
        .with_context(|| format!("Failed to insert tool execution for session: {}", session_id))?;
        Ok(())
    }

    pub async fn get_tool_executions(&self, session_id: i64) -> Result<Vec<ToolExecutionRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT id, session_id, turn_index, tool_call_id, tool_name, args, output, is_error, duration_ms, verdict, created_at FROM tool_executions WHERE session_id = ?1 ORDER BY id",
                [session_id],
            )
            .await?;

        let mut execs = Vec::new();
        while let Some(row) = rows.next().await? {
            execs.push(ToolExecutionRow {
                id: row.get::<i64>(0)?,
                session_id: row.get::<i64>(1)?,
                turn_index: row.get::<i64>(2)? as u32,
                tool_call_id: row.get::<String>(3)?,
                tool_name: row.get::<String>(4)?,
                args: row.get::<String>(5)?,
                output: row.get::<Option<String>>(6)?,
                is_error: row.get::<i64>(7)? != 0,
                duration_ms: row.get::<Option<i64>>(8)?.map(|d| d as u64),
                verdict: row.get::<String>(9)?,
                created_at: row.get::<String>(10)?,
            });
        }
        Ok(execs)
    }
}
