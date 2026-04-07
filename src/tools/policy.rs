use std::collections::BTreeSet;

use anyhow::{Result, anyhow, bail};
use turin_types::{
    BraveSearchToolSettings, SearxngSearchToolSettings, TavilySearchToolSettings,
    ToolSelectionConfig, ToolsConfig, WebFetchToolSettings, WebSearchToolSettings,
};

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
    let agent_tools = resolve_child_tool_selection(
        &root,
        &agent.tools.selection,
        &format!("agent '{agent_id}'"),
    )?;
    match channel_override {
        Some(selection) if !selection.is_empty() => {
            resolve_child_tool_selection(&agent_tools, selection, "channel/tool override")
        }
        _ => Ok(agent_tools),
    }
}

fn merge_web_fetch_tools(
    parent: &WebFetchToolSettings,
    child: &WebFetchToolSettings,
) -> WebFetchToolSettings {
    WebFetchToolSettings {
        user_agent: child
            .user_agent
            .clone()
            .or_else(|| parent.user_agent.clone()),
        accept: child.accept.clone().or_else(|| parent.accept.clone()),
        accept_language: child
            .accept_language
            .clone()
            .or_else(|| parent.accept_language.clone()),
        accept_encoding: child
            .accept_encoding
            .clone()
            .or_else(|| parent.accept_encoding.clone()),
    }
}

fn merge_brave_search_tools(
    parent: &BraveSearchToolSettings,
    child: &BraveSearchToolSettings,
) -> BraveSearchToolSettings {
    BraveSearchToolSettings {
        api_key_env: child
            .api_key_env
            .clone()
            .or_else(|| parent.api_key_env.clone()),
        base_url: child.base_url.clone().or_else(|| parent.base_url.clone()),
    }
}

fn merge_tavily_search_tools(
    parent: &TavilySearchToolSettings,
    child: &TavilySearchToolSettings,
) -> TavilySearchToolSettings {
    TavilySearchToolSettings {
        api_key_env: child
            .api_key_env
            .clone()
            .or_else(|| parent.api_key_env.clone()),
        base_url: child.base_url.clone().or_else(|| parent.base_url.clone()),
    }
}

fn merge_searxng_search_tools(
    parent: &SearxngSearchToolSettings,
    child: &SearxngSearchToolSettings,
) -> SearxngSearchToolSettings {
    SearxngSearchToolSettings {
        base_url: child.base_url.clone().or_else(|| parent.base_url.clone()),
    }
}

fn merge_web_search_tools(
    parent: &WebSearchToolSettings,
    child: &WebSearchToolSettings,
) -> WebSearchToolSettings {
    WebSearchToolSettings {
        providers: child.providers.clone().or_else(|| parent.providers.clone()),
        user_agent: child
            .user_agent
            .clone()
            .or_else(|| parent.user_agent.clone()),
        brave: merge_brave_search_tools(&parent.brave, &child.brave),
        tavily: merge_tavily_search_tools(&parent.tavily, &child.tavily),
        searxng: merge_searxng_search_tools(&parent.searxng, &child.searxng),
    }
}

pub fn merge_tools_config(parent: &ToolsConfig, child: &ToolsConfig) -> ToolsConfig {
    ToolsConfig {
        selection: child.selection.clone(),
        web_fetch: merge_web_fetch_tools(&parent.web_fetch, &child.web_fetch),
        web_search: merge_web_search_tools(&parent.web_search, &child.web_search),
    }
}

pub fn resolve_effective_tools_config(
    config: &TurinConfig,
    agent_id: &str,
    channel_override: Option<&ToolsConfig>,
) -> Result<ToolsConfig> {
    let root = resolve_root_tool_selection(&config.tools.selection)?;
    let agent = agent_config(config, agent_id)?;
    let agent_tools = resolve_child_tool_selection(
        &root,
        &agent.tools.selection,
        &format!("agent '{agent_id}'"),
    )?;
    let resolved = match channel_override {
        Some(selection) if !selection.selection.is_empty() => resolve_child_tool_selection(
            &agent_tools,
            &selection.selection,
            "channel/tool override",
        )?,
        _ => agent_tools,
    };

    let mut effective = merge_tools_config(&config.tools, &agent.tools);
    if let Some(channel_override) = channel_override {
        effective = merge_tools_config(&effective, channel_override);
    }
    effective.selection = ToolSelectionConfig {
        allow: Some(resolved.iter().cloned().collect()),
        exclude: Vec::new(),
    };
    crate::tools::builtins::validate_tools_config(&effective)?;
    Ok(effective)
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
        config.agent.tools.selection = agent;
        config
    }

    #[test]
    fn root_default_set_excludes_only_future_opt_in_tools() {
        let tools = resolve_root_tool_selection(&ToolSelectionConfig::default()).unwrap();
        assert!(tools.contains("read_file"));
        assert!(tools.contains("web_fetch"));
        assert!(!tools.contains("bridge_mcp"));
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

    #[test]
    fn effective_tools_merge_root_agent_and_channel_behavior() {
        let mut config = config_with_tools(
            ToolSelectionConfig::default(),
            ToolSelectionConfig::default(),
        );
        config.tools.web_fetch.user_agent = Some("root-agent".into());
        config.tools.web_search.brave.api_key_env = Some("BRAVE_KEY".into());
        config.agent.tools.web_fetch.user_agent = Some("agent-agent".into());
        config.agent.tools.web_search.providers = Some(vec!["brave".into()]);

        let mut channel = ToolsConfig::default();
        channel.web_fetch.accept_language = Some("fr-FR,fr;q=0.9".into());

        let resolved = resolve_effective_tools_config(&config, "default", Some(&channel)).unwrap();
        assert_eq!(
            resolved.web_fetch.user_agent.as_deref(),
            Some("agent-agent")
        );
        assert_eq!(
            resolved.web_fetch.accept_language.as_deref(),
            Some("fr-FR,fr;q=0.9")
        );
        assert_eq!(
            resolved.web_search.brave.api_key_env.as_deref(),
            Some("BRAVE_KEY")
        );
        assert_eq!(
            resolved.web_search.providers,
            Some(vec!["brave".to_string()])
        );
    }

    #[test]
    fn effective_tools_validation_uses_merged_provider_settings() {
        let mut config = config_with_tools(
            ToolSelectionConfig::default(),
            ToolSelectionConfig::default(),
        );
        config.tools.web_search.tavily.api_key_env = Some("TAVILY_KEY".into());
        config.agent.tools.web_search.providers = Some(vec!["tavily".into()]);

        let resolved = resolve_effective_tools_config(&config, "default", None).unwrap();
        assert_eq!(
            resolved.web_search.tavily.api_key_env.as_deref(),
            Some("TAVILY_KEY")
        );
        assert_eq!(
            resolved.web_search.providers,
            Some(vec!["tavily".to_string()])
        );
    }
}
