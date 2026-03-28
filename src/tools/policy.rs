use std::collections::BTreeSet;

use anyhow::{Result, anyhow, bail};
use turin_types::ToolSelectionConfig;

use crate::kernel::config::{AgentConfig, TurinConfig};
use crate::tools::builtins::{
    BUILTIN_TOOL_NAMES, DEFAULT_EXPOSED_TOOL_NAMES, expand_builtin_group,
};

fn expand_selector(selector: &str) -> Result<BTreeSet<String>> {
    if let Some(group_name) = selector.strip_prefix("group:") {
        let members = expand_builtin_group(group_name)
            .ok_or_else(|| anyhow!("Unknown tool group '{}'", selector))?;
        return Ok(members.iter().map(|name| (*name).to_string()).collect());
    }

    if BUILTIN_TOOL_NAMES.contains(&selector) {
        return Ok(BTreeSet::from([selector.to_string()]));
    }

    bail!("Unknown native tool selector '{}'", selector)
}

pub fn expand_tool_selectors(selectors: &[String]) -> Result<BTreeSet<String>> {
    let mut expanded = BTreeSet::new();
    for selector in selectors {
        expanded.extend(expand_selector(selector)?);
    }
    Ok(expanded)
}

pub fn default_native_tool_set() -> BTreeSet<String> {
    DEFAULT_EXPOSED_TOOL_NAMES
        .iter()
        .map(|name| (*name).to_string())
        .collect()
}

pub fn full_native_tool_set() -> BTreeSet<String> {
    BUILTIN_TOOL_NAMES
        .iter()
        .map(|name| (*name).to_string())
        .collect()
}

pub fn resolve_root_tool_selection(selection: &ToolSelectionConfig) -> Result<BTreeSet<String>> {
    let mut current = if let Some(allow) = selection.allow.as_ref() {
        expand_tool_selectors(allow)?
    } else {
        default_native_tool_set()
    };
    current.retain(|name| full_native_tool_set().contains(name));
    if !selection.exclude.is_empty() {
        let excluded = expand_tool_selectors(&selection.exclude)?;
        current.retain(|name| !excluded.contains(name));
    }
    Ok(current)
}

pub fn resolve_child_tool_selection(
    parent: &BTreeSet<String>,
    selection: &ToolSelectionConfig,
    scope: &str,
) -> Result<BTreeSet<String>> {
    let mut current = if let Some(allow) = selection.allow.as_ref() {
        let requested = expand_tool_selectors(allow)?;
        let disallowed = requested.difference(parent).cloned().collect::<Vec<_>>();
        if !disallowed.is_empty() {
            bail!(
                "{} requests tools not granted by its parent: {}",
                scope,
                disallowed.join(", ")
            );
        }
        requested
    } else {
        parent.clone()
    };

    if !selection.exclude.is_empty() {
        let excluded = expand_tool_selectors(&selection.exclude)?;
        current.retain(|name| !excluded.contains(name));
    }

    Ok(current)
}

fn agent_config<'a>(config: &'a TurinConfig, agent_id: &str) -> Result<&'a AgentConfig> {
    if agent_id == config.agent.id {
        Ok(&config.agent)
    } else {
        config
            .agents
            .get(agent_id)
            .ok_or_else(|| anyhow!("Unknown agent profile '{}'", agent_id))
    }
}

pub fn resolve_effective_native_tools(
    config: &TurinConfig,
    agent_id: &str,
    channel_override: Option<&ToolSelectionConfig>,
) -> Result<BTreeSet<String>> {
    let root = resolve_root_tool_selection(&config.tools.selection)?;
    let agent = agent_config(config, agent_id)?;
    let agent_tools =
        resolve_child_tool_selection(&root, &agent.tools, &format!("agent '{agent_id}'"))?;
    match channel_override {
        Some(selection) if !selection.is_empty() => {
            resolve_child_tool_selection(&agent_tools, selection, "channel/tool override")
        }
        _ => Ok(agent_tools),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_tools(root: ToolSelectionConfig, agent: ToolSelectionConfig) -> TurinConfig {
        let mut config = TurinConfig::default();
        config.agent.model = "mock-model".into();
        config.agent.provider = "mock".into();
        config.providers.insert(
            "mock".into(),
            crate::kernel::config::ProviderConfig {
                kind: "mock".into(),
                ..crate::kernel::config::ProviderConfig::default()
            },
        );
        config.tools.selection = root;
        config.agent.tools = agent;
        config
    }

    #[test]
    fn root_default_set_excludes_only_future_opt_in_tools() {
        let tools = resolve_root_tool_selection(&ToolSelectionConfig::default()).unwrap();
        assert!(tools.contains("read_file"));
        assert!(tools.contains("web_fetch"));
    }

    #[test]
    fn child_cannot_expand_beyond_parent() {
        let config = config_with_tools(
            ToolSelectionConfig {
                allow: Some(vec!["group:web".into()]),
                exclude: Vec::new(),
            },
            ToolSelectionConfig {
                allow: Some(vec!["shell_exec".into()]),
                exclude: Vec::new(),
            },
        );
        let err = resolve_effective_native_tools(&config, "default", None).unwrap_err();
        assert!(err.to_string().contains("not granted"));
    }

    #[test]
    fn channel_override_can_subset_agent_tools() {
        let config = config_with_tools(
            ToolSelectionConfig {
                allow: Some(vec!["group:web".into(), "read_file".into()]),
                exclude: Vec::new(),
            },
            ToolSelectionConfig::default(),
        );
        let resolved = resolve_effective_native_tools(
            &config,
            "default",
            Some(&ToolSelectionConfig {
                allow: Some(vec!["web_fetch".into()]),
                exclude: Vec::new(),
            }),
        )
        .unwrap();
        assert_eq!(resolved, BTreeSet::from(["web_fetch".to_string()]));
    }
}
