use std::collections::BTreeSet;

use turin::tools::{builtins, policy};
use turin_types::ToolSelectionConfig;

fn set(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| (*name).to_string()).collect()
}

#[test]
fn default_native_tool_exposure_matches_current_capability_contract() {
    let expected = set(&[
        "read_file",
        "write_file",
        "edit_file",
        "shell_exec",
        "web_fetch",
        "web_search",
        "remember",
        "recall",
        "submit_plan",
    ]);

    let default_tools = policy::default_native_tool_set();
    assert_eq!(default_tools, expected);

    assert!(
        default_tools.contains("shell_exec"),
        "shell_exec is currently default-exposed; update this test and docs together if that changes"
    );
    assert!(
        !default_tools.contains("apply_patch"),
        "apply_patch should remain opt-in unless the capability contract changes"
    );
    assert!(
        !default_tools.contains("bridge_mcp"),
        "bridge_mcp should remain opt-in unless the capability contract changes"
    );
}

#[test]
fn default_registry_registers_every_builtin_tool_once() {
    let registry = builtins::create_default_registry();
    let declared = set(builtins::BUILTIN_TOOL_NAMES);
    let registered = registry
        .tool_definitions()
        .into_iter()
        .map(|definition| {
            definition["name"]
                .as_str()
                .expect("tool definition name should be a string")
                .to_string()
        })
        .collect::<BTreeSet<_>>();

    assert_eq!(registered, declared);
    assert_eq!(policy::full_native_tool_set(), declared);
}

#[test]
fn group_all_is_broader_than_default_exposure() {
    let all = policy::expand_tool_selectors(&["group:all".to_string()]).unwrap();
    let default = policy::default_native_tool_set();

    assert!(all.is_superset(&default));
    assert!(all.contains("apply_patch"));
    assert!(all.contains("bridge_mcp"));
}

#[test]
fn delegated_tool_selection_cannot_expand_parent_scope() {
    let parent = policy::expand_tool_selectors(&["group:web".to_string()]).unwrap();
    let child_request = ToolSelectionConfig {
        allow: Some(vec!["shell_exec".to_string()]),
        exclude: Vec::new(),
    };

    let err = policy::resolve_child_tool_selection(&parent, &child_request, "characterization")
        .expect_err("child selection must not expand beyond parent tools");

    assert!(err.to_string().contains("not granted"));
}
