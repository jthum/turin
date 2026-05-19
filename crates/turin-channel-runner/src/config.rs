use anyhow::{Context, Result};
use serde_json::Value;
use std::path::PathBuf;
use std::time::Duration;
use turin_types::ToolsConfig;

use crate::ChannelAccessPolicy;

#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub channel_id: String,
    pub state_path: PathBuf,
    pub access_state_path: PathBuf,
    pub idle_ttl: Option<Duration>,
    pub access_policy: ChannelAccessPolicy,
    pub tools: ToolsConfig,
}

pub fn task_timeout_ms_from_settings(settings: &Value) -> Result<Option<u64>> {
    let map = settings
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Channel settings must be a JSON object"))?;
    read_task_timeout_ms(map.get("task_timeout_ms"))
}

pub fn tools_config_from_settings(settings: &Value) -> Result<ToolsConfig> {
    let map = settings
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Channel settings must be a JSON object"))?;
    let Some(tools) = map.get("tools") else {
        return Ok(ToolsConfig::default());
    };
    serde_json::from_value(tools.clone()).context("failed to parse 'tools' settings")
}

fn read_task_timeout_ms(value: Option<&Value>) -> Result<Option<u64>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let timeout_ms = value.as_u64().ok_or_else(|| {
        anyhow::anyhow!("channel setting 'task_timeout_ms' must be a non-negative integer")
    })?;
    if timeout_ms == 0 {
        Ok(None)
    } else {
        Ok(Some(timeout_ms))
    }
}
