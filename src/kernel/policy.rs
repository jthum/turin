use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use serde_json::Value;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PolicyScope {
    pub scope: Option<String>,
    pub agent_id: Option<String>,
    pub session_id: Option<String>,
    pub run_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RuntimePolicy {
    pub spawn_enabled: bool,
    pub spawn_max_depth: u32,
    pub spawn_max_fan_out: usize,
    pub spawn_max_concurrent_children: usize,
    pub spawn_root_max_total_tokens: Option<u64>,
    pub spawn_root_max_duration_ms: Option<u64>,
    pub spawn_root_max_tool_calls: Option<u64>,
    pub db_allow_dynamic_open: bool,
    pub db_path_scope: String,
    pub db_max_open_handles: usize,
    pub db_idle_close_seconds: u64,
    pub queue_max_depth: usize,
    pub tool_exec_enabled: bool,
    pub hook_token_usage_reject_mode: String,
}

impl Default for RuntimePolicy {
    fn default() -> Self {
        Self {
            spawn_enabled: true,
            spawn_max_depth: 3,
            spawn_max_fan_out: 64,
            spawn_max_concurrent_children: 16,
            spawn_root_max_total_tokens: None,
            spawn_root_max_duration_ms: None,
            spawn_root_max_tool_calls: None,
            db_allow_dynamic_open: true,
            db_path_scope: "workspace_only".to_string(),
            db_max_open_handles: 128,
            db_idle_close_seconds: 300,
            queue_max_depth: 1024,
            tool_exec_enabled: true,
            hook_token_usage_reject_mode: "informational".to_string(),
        }
    }
}

impl RuntimePolicy {
    pub fn to_map(&self) -> HashMap<String, Value> {
        let mut map = HashMap::new();
        map.insert("spawn.enabled".to_string(), Value::Bool(self.spawn_enabled));
        map.insert(
            "spawn.max_depth".to_string(),
            Value::from(self.spawn_max_depth as u64),
        );
        map.insert(
            "spawn.max_fan_out".to_string(),
            Value::from(self.spawn_max_fan_out as u64),
        );
        map.insert(
            "spawn.max_concurrent_children".to_string(),
            Value::from(self.spawn_max_concurrent_children as u64),
        );
        map.insert(
            "spawn.root_max_total_tokens".to_string(),
            self.spawn_root_max_total_tokens
                .map_or(Value::Null, Value::from),
        );
        map.insert(
            "spawn.root_max_duration_ms".to_string(),
            self.spawn_root_max_duration_ms
                .map_or(Value::Null, Value::from),
        );
        map.insert(
            "spawn.root_max_tool_calls".to_string(),
            self.spawn_root_max_tool_calls
                .map_or(Value::Null, Value::from),
        );
        map.insert(
            "db.allow_dynamic_open".to_string(),
            Value::Bool(self.db_allow_dynamic_open),
        );
        map.insert(
            "db.path_scope".to_string(),
            Value::String(self.db_path_scope.clone()),
        );
        map.insert(
            "db.max_open_handles".to_string(),
            Value::from(self.db_max_open_handles as u64),
        );
        map.insert(
            "db.idle_close_seconds".to_string(),
            Value::from(self.db_idle_close_seconds),
        );
        map.insert(
            "queue.max_depth".to_string(),
            Value::from(self.queue_max_depth as u64),
        );
        map.insert(
            "tool.exec_enabled".to_string(),
            Value::Bool(self.tool_exec_enabled),
        );
        map.insert(
            "hook.token_usage.reject_mode".to_string(),
            Value::String(self.hook_token_usage_reject_mode.clone()),
        );
        map
    }
}

#[derive(Debug, Default)]
struct PolicyState {
    global: HashMap<String, Value>,
    per_agent: HashMap<String, HashMap<String, Value>>,
    per_session: HashMap<String, HashMap<String, Value>>,
    per_run: HashMap<String, HashMap<String, Value>>,
}

#[derive(Clone, Debug)]
pub struct RuntimePolicyManager {
    defaults: RuntimePolicy,
    state: Arc<RwLock<PolicyState>>,
}

impl RuntimePolicyManager {
    pub fn new() -> Self {
        Self {
            defaults: RuntimePolicy::default(),
            state: Arc::new(RwLock::new(PolicyState::default())),
        }
    }

    pub async fn snapshot(&self, scope: &PolicyScope) -> HashMap<String, Value> {
        let mut merged = self.defaults.to_map();
        let state = self.state.read().await;

        for (k, v) in &state.global {
            merged.insert(k.clone(), v.clone());
        }
        if let Some(agent_id) = scope.agent_id.as_ref()
            && let Some(overrides) = state.per_agent.get(agent_id)
        {
            for (k, v) in overrides {
                merged.insert(k.clone(), v.clone());
            }
        }
        if let Some(session_id) = scope.session_id.as_ref()
            && let Some(overrides) = state.per_session.get(session_id)
        {
            for (k, v) in overrides {
                merged.insert(k.clone(), v.clone());
            }
        }
        if let Some(run_id) = scope.run_id.as_ref()
            && let Some(overrides) = state.per_run.get(run_id)
        {
            for (k, v) in overrides {
                merged.insert(k.clone(), v.clone());
            }
        }
        merged
    }

    pub async fn get(&self, key: &str, scope: &PolicyScope) -> Result<Option<Value>> {
        validate_key(key)?;
        Ok(self.snapshot(scope).await.get(key).cloned())
    }

    pub async fn set(&self, key: &str, value: Value, scope: &PolicyScope) -> Result<()> {
        validate_key(key)?;
        validate_value(key, &value)?;

        let mut state = self.state.write().await;
        match scope.scope.as_deref().unwrap_or("global") {
            "global" => {
                state.global.insert(key.to_string(), value);
            }
            "agent" => {
                let agent_id = scope
                    .agent_id
                    .as_deref()
                    .ok_or_else(|| anyhow!("agent scope requires agent_id"))?;
                state
                    .per_agent
                    .entry(agent_id.to_string())
                    .or_default()
                    .insert(key.to_string(), value);
            }
            "session" => {
                let session_id = scope
                    .session_id
                    .as_deref()
                    .ok_or_else(|| anyhow!("session scope requires session_id"))?;
                state
                    .per_session
                    .entry(session_id.to_string())
                    .or_default()
                    .insert(key.to_string(), value);
            }
            "run" => {
                let run_id = scope
                    .run_id
                    .as_deref()
                    .ok_or_else(|| anyhow!("run scope requires run_id"))?;
                state
                    .per_run
                    .entry(run_id.to_string())
                    .or_default()
                    .insert(key.to_string(), value);
            }
            other => return Err(anyhow!("Unsupported policy scope '{}'", other)),
        }

        Ok(())
    }
}

impl Default for RuntimePolicyManager {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_key(key: &str) -> Result<()> {
    match key {
        "spawn.enabled"
        | "spawn.max_depth"
        | "spawn.max_fan_out"
        | "spawn.max_concurrent_children"
        | "spawn.root_max_total_tokens"
        | "spawn.root_max_duration_ms"
        | "spawn.root_max_tool_calls"
        | "runtime.idle_timeout_seconds"
        | "db.allow_dynamic_open"
        | "db.path_scope"
        | "db.max_open_handles"
        | "db.idle_close_seconds"
        | "queue.max_depth"
        | "tool.exec_enabled"
        | "hook.token_usage.reject_mode" => Ok(()),
        _ => Err(anyhow!("Unknown policy key '{}'", key)),
    }
}

fn validate_value(key: &str, value: &Value) -> Result<()> {
    match key {
        "spawn.enabled" | "db.allow_dynamic_open" | "tool.exec_enabled" => {
            if value.is_boolean() {
                Ok(())
            } else {
                Err(anyhow!("Policy '{}' expects boolean", key))
            }
        }
        "spawn.max_depth"
        | "spawn.max_fan_out"
        | "spawn.max_concurrent_children"
        | "db.max_open_handles"
        | "db.idle_close_seconds"
        | "queue.max_depth" => {
            if value.as_u64().is_some() {
                Ok(())
            } else {
                Err(anyhow!("Policy '{}' expects non-negative integer", key))
            }
        }
        "spawn.root_max_total_tokens"
        | "spawn.root_max_duration_ms"
        | "spawn.root_max_tool_calls" => {
            if value.is_null() || value.as_u64().is_some() {
                Ok(())
            } else {
                Err(anyhow!(
                    "Policy '{}' expects null or non-negative integer",
                    key
                ))
            }
        }
        "runtime.idle_timeout_seconds" => {
            if value.is_null() || value.as_u64().is_some() {
                Ok(())
            } else {
                Err(anyhow!(
                    "Policy '{}' expects null or non-negative integer",
                    key
                ))
            }
        }
        "db.path_scope" => match value.as_str() {
            Some("workspace_only" | "allow_any") => Ok(()),
            _ => Err(anyhow!(
                "Policy '{}' expects one of: workspace_only, allow_any",
                key
            )),
        },
        "hook.token_usage.reject_mode" => match value.as_str() {
            Some("informational" | "enforce_task" | "enforce_session") => Ok(()),
            _ => Err(anyhow!(
                "Policy '{}' expects one of: informational, enforce_task, enforce_session",
                key
            )),
        },
        _ => Err(anyhow!("Unknown policy key '{}'", key)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn policy_defaults_and_overrides_merge() {
        let mgr = RuntimePolicyManager::new();
        let base_scope = PolicyScope::default();
        let default = mgr.get("spawn.enabled", &base_scope).await.unwrap();
        assert_eq!(default, Some(Value::Bool(true)));
        assert_eq!(
            mgr.get("spawn.max_fan_out", &base_scope).await.unwrap(),
            Some(Value::from(64u64))
        );
        assert_eq!(
            mgr.get("spawn.max_concurrent_children", &base_scope)
                .await
                .unwrap(),
            Some(Value::from(16u64))
        );
        assert_eq!(
            mgr.get("spawn.root_max_total_tokens", &base_scope)
                .await
                .unwrap(),
            Some(Value::Null)
        );

        mgr.set(
            "spawn.enabled",
            Value::Bool(false),
            &PolicyScope {
                scope: Some("global".to_string()),
                ..PolicyScope::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(
            mgr.get("spawn.enabled", &base_scope).await.unwrap(),
            Some(Value::Bool(false))
        );

        let agent_scope = PolicyScope {
            scope: Some("agent".to_string()),
            agent_id: Some("coder".to_string()),
            ..PolicyScope::default()
        };
        mgr.set("spawn.max_depth", Value::from(1u64), &agent_scope)
            .await
            .unwrap();

        let snapshot = mgr
            .snapshot(&PolicyScope {
                agent_id: Some("coder".to_string()),
                ..PolicyScope::default()
            })
            .await;
        assert_eq!(snapshot.get("spawn.max_depth"), Some(&Value::from(1u64)));
    }

    #[tokio::test]
    async fn policy_validates_keys_and_values() {
        let mgr = RuntimePolicyManager::new();
        let err = mgr
            .set(
                "unknown.key",
                Value::Bool(true),
                &PolicyScope {
                    scope: Some("global".to_string()),
                    ..PolicyScope::default()
                },
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("Unknown policy key"));

        mgr.set(
            "spawn.root_max_tool_calls",
            Value::from(100u64),
            &PolicyScope::default(),
        )
        .await
        .unwrap();
        mgr.set(
            "spawn.root_max_tool_calls",
            Value::Null,
            &PolicyScope::default(),
        )
        .await
        .unwrap();

        let err = mgr
            .set(
                "spawn.max_depth",
                Value::String("bad".to_string()),
                &PolicyScope {
                    scope: Some("global".to_string()),
                    ..PolicyScope::default()
                },
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("expects non-negative integer"));

        let err = mgr
            .set(
                "hook.token_usage.reject_mode",
                Value::String("bad".to_string()),
                &PolicyScope {
                    scope: Some("global".to_string()),
                    ..PolicyScope::default()
                },
            )
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("informational, enforce_task, enforce_session")
        );
    }
}
