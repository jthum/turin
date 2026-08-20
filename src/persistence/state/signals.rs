use anyhow::Result;
use turso::Value as SqlValue;

use super::StateStore;
use crate::persistence::schema::SignalRow;
use crate::signal_topics::signal_topic_subscription_candidates;

const SIGNAL_COLUMNS: &str = "id, public_id, topic, source_agent_id, target_agent_id, source_session_id, target_session_id, payload, attempt_count, last_attempted_at, last_error, created_at";

#[derive(Debug, Clone)]
pub struct SignalInsert {
    pub public_id: Vec<u8>,
    pub topic: String,
    pub source_agent_id: String,
    pub target_agent_id: String,
    pub source_session_id: Option<String>,
    pub target_session_id: Option<String>,
    pub payload: String,
}

impl StateStore {
    pub async fn replace_signal_subscriptions_for_agents(
        &self,
        agent_ids: &[String],
        subscriptions: &[(String, String)],
    ) -> Result<()> {
        if agent_ids.is_empty() {
            return Ok(());
        }
        let conn = self.connect().await?;
        for agent_id in agent_ids {
            conn.execute(
                "DELETE FROM subscriptions WHERE agent_id = ?1",
                [agent_id.as_str()],
            )
            .await?;
        }
        for (agent_id, topic) in subscriptions {
            conn.execute(
                "INSERT INTO subscriptions (agent_id, topic) VALUES (?1, ?2)",
                (agent_id.as_str(), topic.as_str()),
            )
            .await?;
        }
        Ok(())
    }

    pub async fn list_signal_subscriber_agent_ids(&self, topic: &str) -> Result<Vec<String>> {
        let conn = self.connect().await?;
        let candidates = signal_topic_subscription_candidates(topic);
        let placeholders = (1..=candidates.len())
            .map(|index| format!("?{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT DISTINCT agent_id
             FROM subscriptions
             WHERE topic IN ({placeholders})
             ORDER BY agent_id ASC"
        );
        let params = candidates
            .into_iter()
            .map(SqlValue::Text)
            .collect::<Vec<_>>();
        let mut stmt = conn.prepare(&sql).await?;
        let mut rows = stmt.query(params).await?;

        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(row.get(0)?);
        }
        Ok(out)
    }

    pub async fn insert_signal(&self, insert: SignalInsert) -> Result<()> {
        let conn = self.connect().await?;
        conn.execute(
            "INSERT INTO signals (public_id, topic, source_agent_id, target_agent_id, source_session_id, target_session_id, payload) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            (
                insert.public_id,
                insert.topic,
                insert.source_agent_id,
                insert.target_agent_id,
                insert.source_session_id,
                insert.target_session_id,
                insert.payload,
            ),
        )
        .await?;
        Ok(())
    }

    pub async fn list_signals_for_agent(
        &self,
        agent_id: &str,
        session_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<SignalRow>> {
        let conn = self.connect().await?;
        let sql = format!(
            "SELECT {SIGNAL_COLUMNS}
             FROM signals
             WHERE target_agent_id = ?1
               AND (target_session_id IS NULL OR target_session_id = ?2)
             ORDER BY id ASC
             LIMIT ?3"
        );
        let mut rows = conn
            .query(&sql, (agent_id, session_id, limit as i64))
            .await?;

        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(map_signal_row(&row)?);
        }

        Ok(out)
    }

    pub async fn list_signals(
        &self,
        topic: Option<&str>,
        source_agent_id: Option<&str>,
        target_agent_id: Option<&str>,
        source_session_id: Option<&str>,
        target_session_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<SignalRow>> {
        let conn = self.connect().await?;
        let mut sql = format!("SELECT {SIGNAL_COLUMNS} FROM signals");
        let mut clauses = Vec::new();
        let mut params = Vec::new();
        push_signal_filter(&mut clauses, &mut params, "topic", topic);
        push_signal_filter(
            &mut clauses,
            &mut params,
            "source_agent_id",
            source_agent_id,
        );
        push_signal_filter(
            &mut clauses,
            &mut params,
            "source_session_id",
            source_session_id,
        );
        push_signal_filter(
            &mut clauses,
            &mut params,
            "target_session_id",
            target_session_id,
        );
        push_signal_filter(
            &mut clauses,
            &mut params,
            "target_agent_id",
            target_agent_id,
        );
        if !clauses.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&clauses.join(" AND "));
        }
        sql.push_str(&format!(" ORDER BY id ASC LIMIT ?{}", params.len() + 1));
        params.push(SqlValue::Integer(limit as i64));

        let mut stmt = conn.prepare(&sql).await?;
        let mut rows = stmt.query(params).await?;

        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(map_signal_row(&row)?);
        }
        Ok(out)
    }

    pub async fn record_signal_attempt(&self, id: i64) -> Result<()> {
        let conn = self.connect().await?;
        conn.execute(
            "UPDATE signals
             SET attempt_count = attempt_count + 1,
                 last_attempted_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                 last_error = NULL
             WHERE id = ?1",
            [id],
        )
        .await?;
        Ok(())
    }

    pub async fn set_signal_error(&self, id: i64, last_error: &str) -> Result<()> {
        let conn = self.connect().await?;
        conn.execute(
            "UPDATE signals
             SET last_error = ?2
             WHERE id = ?1",
            (id, last_error),
        )
        .await?;
        Ok(())
    }

    pub async fn delete_signal(&self, id: i64) -> Result<()> {
        let conn = self.connect().await?;
        conn.execute("DELETE FROM signals WHERE id = ?1", [id])
            .await?;
        Ok(())
    }
}

fn push_signal_filter(
    clauses: &mut Vec<String>,
    params: &mut Vec<SqlValue>,
    column: &'static str,
    value: Option<&str>,
) {
    if let Some(value) = value {
        params.push(SqlValue::Text(value.to_string()));
        clauses.push(format!("{column} = ?{}", params.len()));
    }
}

fn map_signal_row(row: &turso::Row) -> Result<SignalRow> {
    let id = row.get::<i64>(0)?;
    Ok(SignalRow {
        id,
        public_id: row.get(1)?,
        topic: row.get(2)?,
        source_agent_id: row.get(3)?,
        target_agent_id: row.get(4)?,
        source_session_id: row.get(5)?,
        target_session_id: row.get(6)?,
        payload: row.get(7)?,
        attempt_count: super::persisted_u64(
            &format!("signal {id}"),
            "attempt count",
            row.get::<i64>(8)?,
        )?,
        last_attempted_at: row.get(9)?,
        last_error: row.get(10)?,
        created_at: row.get(11)?,
    })
}
