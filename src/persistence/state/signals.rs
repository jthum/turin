use anyhow::Result;

use super::StateStore;
use crate::persistence::schema::SignalRow;

#[derive(Debug, Clone)]
pub struct SignalInsert {
    pub public_id: Vec<u8>,
    pub topic: String,
    pub source_agent_id: String,
    pub target_agent_id: String,
    pub payload: String,
}

impl StateStore {
    pub async fn replace_signal_subscriptions(
        &self,
        subscriptions: &[(String, String)],
    ) -> Result<()> {
        let conn = self.connect().await?;
        conn.execute("DELETE FROM signal_subscriptions", ()).await?;
        for (agent_id, topic) in subscriptions {
            conn.execute(
                "INSERT INTO signal_subscriptions (agent_id, topic) VALUES (?1, ?2)",
                (agent_id.as_str(), topic.as_str()),
            )
            .await?;
        }
        Ok(())
    }

    pub async fn list_signal_subscriber_agent_ids(&self, topic: &str) -> Result<Vec<String>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT agent_id
                 FROM signal_subscriptions
                 WHERE topic = ?1
                 ORDER BY agent_id ASC",
                [topic],
            )
            .await?;

        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(row.get(0)?);
        }
        Ok(out)
    }

    pub async fn insert_signal(&self, insert: SignalInsert) -> Result<()> {
        let conn = self.connect().await?;
        conn.execute(
            "INSERT INTO signals (public_id, topic, source_agent_id, target_agent_id, payload) VALUES (?1, ?2, ?3, ?4, ?5)",
            (
                insert.public_id,
                insert.topic,
                insert.source_agent_id,
                insert.target_agent_id,
                insert.payload,
            ),
        )
        .await?;
        Ok(())
    }

    pub async fn list_signals_for_agent(
        &self,
        agent_id: &str,
        limit: usize,
    ) -> Result<Vec<SignalRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT id, public_id, topic, source_agent_id, target_agent_id, payload, attempt_count, last_attempted_at, last_error, created_at
                 FROM signals
                 WHERE target_agent_id = ?1
                 ORDER BY id ASC
                 LIMIT ?2",
                (agent_id, limit as i64),
            )
            .await?;

        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(SignalRow {
                id: row.get(0)?,
                public_id: row.get(1)?,
                topic: row.get(2)?,
                source_agent_id: row.get(3)?,
                target_agent_id: row.get(4)?,
                payload: row.get(5)?,
                attempt_count: row.get::<i64>(6)? as u64,
                last_attempted_at: row.get(7)?,
                last_error: row.get(8)?,
                created_at: row.get(9)?,
            });
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
