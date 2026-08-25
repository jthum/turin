use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use serde_json::Value;
use tokio::sync::RwLock;

use crate::persistence::manager::StorePathScope;

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
    pub db_path_scope: StorePathScope,
    pub db_max_open_handles: usize,
    pub db_idle_close_seconds: u64,
    pub queue_max_depth: usize,
    pub tool_exec_enabled: bool,
    pub hook_token_usage_reject_mode: String,
    pub runtime_idle_timeout_seconds: Option<u64>,
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
            db_path_scope: StorePathScope::WorkspaceOnly,
            db_max_open_handles: 128,
            db_idle_close_seconds: 300,
            queue_max_depth: 1024,
            tool_exec_enabled: true,
            hook_token_usage_reject_mode: "informational".to_string(),
            runtime_idle_timeout_seconds: None,
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
            Value::String(self.db_path_scope.as_str().to_string()),
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
        map.insert(
            "runtime.idle_timeout_seconds".to_string(),
            self.runtime_idle_timeout_seconds
                .map_or(Value::Null, Value::from),
        );
        map
    }

    pub fn from_map(map: &HashMap<String, Value>) -> Self {
        let defaults = Self::default();
        fn bool_val(map: &HashMap<String, Value>, key: &str, default: bool) -> bool {
            map.get(key).and_then(Value::as_bool).unwrap_or(default)
        }
        fn usize_val(map: &HashMap<String, Value>, key: &str, default: usize) -> usize {
            map.get(key)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(default)
        }
        fn u32_val(map: &HashMap<String, Value>, key: &str, default: u32) -> u32 {
            map.get(key)
                .and_then(Value::as_u64)
                .map(|v| v as u32)
                .unwrap_or(default)
        }
        fn opt_u64(map: &HashMap<String, Value>, key: &str) -> Option<u64> {
            match map.get(key) {
                None | Some(Value::Null) => None,
                Some(v) => v.as_u64(),
            }
        }
        fn string_val(map: &HashMap<String, Value>, key: &str, default: &str) -> String {
            map.get(key)
                .and_then(Value::as_str)
                .unwrap_or(default)
                .to_string()
        }
        Self {
            spawn_enabled: bool_val(map, "spawn.enabled", defaults.spawn_enabled),
            spawn_max_depth: u32_val(map, "spawn.max_depth", defaults.spawn_max_depth),
            spawn_max_fan_out: usize_val(map, "spawn.max_fan_out", defaults.spawn_max_fan_out),
            spawn_max_concurrent_children: usize_val(
                map,
                "spawn.max_concurrent_children",
                defaults.spawn_max_concurrent_children,
            ),
            spawn_root_max_total_tokens: opt_u64(map, "spawn.root_max_total_tokens"),
            spawn_root_max_duration_ms: opt_u64(map, "spawn.root_max_duration_ms"),
            spawn_root_max_tool_calls: opt_u64(map, "spawn.root_max_tool_calls"),
            db_allow_dynamic_open: bool_val(
                map,
                "db.allow_dynamic_open",
                defaults.db_allow_dynamic_open,
            ),
            db_path_scope: StorePathScope::from_policy(
                map.get("db.path_scope")
                    .and_then(Value::as_str)
                    .unwrap_or(defaults.db_path_scope.as_str()),
            ),
            db_max_open_handles: usize_val(
                map,
                "db.max_open_handles",
                defaults.db_max_open_handles,
            ),
            db_idle_close_seconds: map
                .get("db.idle_close_seconds")
                .and_then(Value::as_u64)
                .unwrap_or(defaults.db_idle_close_seconds),
            queue_max_depth: usize_val(map, "queue.max_depth", defaults.queue_max_depth),
            tool_exec_enabled: bool_val(map, "tool.exec_enabled", defaults.tool_exec_enabled),
            hook_token_usage_reject_mode: string_val(
                map,
                "hook.token_usage.reject_mode",
                &defaults.hook_token_usage_reject_mode,
            ),
            runtime_idle_timeout_seconds: opt_u64(map, "runtime.idle_timeout_seconds"),
        }
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

    pub async fn typed_snapshot(&self, scope: &PolicyScope) -> RuntimePolicy {
        RuntimePolicy::from_map(&self.snapshot(scope).await)
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

    pub(crate) async fn clear_transient_scopes(&self, session_id: &str, run_id: Option<&str>) {
        let mut state = self.state.write().await;
        state.per_session.remove(session_id);
        if let Some(run_id) = run_id {
            state.per_run.remove(run_id);
        }
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

        let typed = mgr.typed_snapshot(&base_scope).await;
        assert_eq!(typed.db_path_scope, StorePathScope::WorkspaceOnly);
        assert!(typed.tool_exec_enabled);
        mgr.set(
            "db.path_scope",
            Value::String("allow_any".to_string()),
            &PolicyScope {
                scope: Some("global".to_string()),
                ..PolicyScope::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(
            mgr.typed_snapshot(&base_scope).await.db_path_scope,
            StorePathScope::AllowAny
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

    #[tokio::test]
    async fn clearing_transient_policy_scopes_preserves_agent_policy() {
        let mgr = RuntimePolicyManager::new();
        let agent_scope = PolicyScope {
            scope: Some("agent".to_string()),
            agent_id: Some("coder".to_string()),
            ..PolicyScope::default()
        };
        let session_scope = PolicyScope {
            scope: Some("session".to_string()),
            session_id: Some("session-1".to_string()),
            ..PolicyScope::default()
        };
        let run_scope = PolicyScope {
            scope: Some("run".to_string()),
            run_id: Some("run-1".to_string()),
            ..PolicyScope::default()
        };

        mgr.set("spawn.max_depth", Value::from(7), &agent_scope)
            .await
            .unwrap();
        mgr.set("spawn.max_depth", Value::from(2), &session_scope)
            .await
            .unwrap();
        mgr.set("spawn.max_depth", Value::from(1), &run_scope)
            .await
            .unwrap();

        mgr.clear_transient_scopes("session-1", Some("run-1")).await;

        assert_eq!(
            mgr.snapshot(&PolicyScope {
                agent_id: Some("coder".to_string()),
                session_id: Some("session-1".to_string()),
                run_id: Some("run-1".to_string()),
                ..PolicyScope::default()
            })
            .await
            .get("spawn.max_depth"),
            Some(&Value::from(7))
        );
    }
}
