use anyhow::{Context, Result};

use super::StateStore;

impl StateStore {
    pub async fn kv_set(
        &self,
        scope_kind: &str,
        scope_key: &str,
        key: &str,
        value: &str,
    ) -> Result<()> {
        const MAX_KV_VALUE_SIZE: usize = 1_048_576;

        if value.len() > MAX_KV_VALUE_SIZE {
            anyhow::bail!(
                "KV value exceeds maximum size of {} bytes (got {})",
                MAX_KV_VALUE_SIZE,
                value.len()
            );
        }

        let conn = self.connect().await?;
        conn.execute(
            "INSERT OR REPLACE INTO kv (scope_kind, scope_key, key, value, updated_at) VALUES (?1, ?2, ?3, ?4, datetime('now'))",
            turso::params![scope_kind, scope_key, key, value],
        )
        .await
        .with_context(|| {
            format!(
                "Failed to set KV pair for scope {}:{} key {}",
                scope_kind, scope_key, key
            )
        })?;
        Ok(())
    }

    pub async fn kv_get(
        &self,
        scope_kind: &str,
        scope_key: &str,
        key: &str,
    ) -> Result<Option<String>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT value FROM kv WHERE scope_kind = ?1 AND scope_key = ?2 AND key = ?3 AND (expires_at IS NULL OR expires_at > datetime('now'))",
                turso::params![scope_kind, scope_key, key],
            )
            .await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(row.get::<String>(0)?))
        } else {
            Ok(None)
        }
    }

    pub async fn kv_delete(&self, scope_kind: &str, scope_key: &str, key: &str) -> Result<()> {
        let conn = self.connect().await?;
        conn.execute(
            "DELETE FROM kv WHERE scope_kind = ?1 AND scope_key = ?2 AND key = ?3",
            turso::params![scope_kind, scope_key, key],
        )
        .await?;
        Ok(())
    }
}
