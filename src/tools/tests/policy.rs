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
fn request_override_can_subset_agent_tools() {
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
fn registered_custom_tools_participate_in_selection() {
    let config = config_with_tools(
        ToolSelectionConfig {
            allow: Some(vec!["records_write".into()]),
            exclude: Vec::new(),
        },
        ToolSelectionConfig::default(),
    );
    let available = BTreeSet::from(["records_write".to_string()]);

    let resolved =
        resolve_effective_tools_config_for_registry(&config, "default", None, &available).unwrap();
    assert_eq!(resolved.selection.allow, Some(vec!["records_write".into()]));
}

#[test]
fn registered_dynamic_tools_are_default_on_but_request_restrictable() {
    let config = config_with_tools(
        ToolSelectionConfig::default(),
        ToolSelectionConfig::default(),
    );
    let available = BTreeSet::from(["read_file".to_string(), "mcp_dynamic".to_string()]);

    let default =
        resolve_effective_tools_config_for_registry(&config, "default", None, &available).unwrap();
    assert_eq!(
        default.selection.allow,
        Some(vec!["mcp_dynamic".to_string(), "read_file".to_string()])
    );

    let mut request = ToolsConfig::default();
    request.selection.allow = Some(vec!["read_file".to_string()]);
    let restricted =
        resolve_effective_tools_config_for_registry(&config, "default", Some(&request), &available)
            .unwrap();
    assert_eq!(
        restricted.selection.allow,
        Some(vec!["read_file".to_string()])
    );
}

#[test]
fn registry_validation_rejects_unknown_exact_tool_names() {
    let config = config_with_tools(
        ToolSelectionConfig {
            allow: Some(vec!["misspelled_tool".into()]),
            exclude: Vec::new(),
        },
        ToolSelectionConfig::default(),
    );
    let error =
        resolve_effective_tools_config_for_registry(&config, "default", None, &BTreeSet::new())
            .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Unknown registered tool selector")
    );
}

#[test]
fn effective_tools_merge_root_agent_and_request_behavior() {
    let mut config = config_with_tools(
        ToolSelectionConfig::default(),
        ToolSelectionConfig::default(),
    );
    config.tools.web_fetch.user_agent = Some("root-agent".into());
    config.tools.web_fetch.max_response_bytes = Some(32 * 1024 * 1024);
    config.tools.web_search.brave.api_key_env = Some("BRAVE_KEY".into());
    config.agent.tools.web_fetch.user_agent = Some("agent-agent".into());
    config.agent.tools.web_search.providers = Some(vec!["brave".into()]);

    let mut request = ToolsConfig::default();
    request.web_fetch.accept_language = Some("fr-FR,fr;q=0.9".into());

    let resolved = resolve_effective_tools_config(&config, "default", Some(&request)).unwrap();
    assert_eq!(
        resolved.web_fetch.user_agent.as_deref(),
        Some("agent-agent")
    );
    assert_eq!(
        resolved.web_fetch.accept_language.as_deref(),
        Some("fr-FR,fr;q=0.9")
    );
    assert_eq!(
        resolved.web_fetch.max_response_bytes,
        Some(32 * 1024 * 1024)
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
