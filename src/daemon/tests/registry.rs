use super::*;
use anyhow::Result;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

use crate::kernel::config::TurinConfig;

fn scan_registry_with_default(config: &TurinConfig, root: &Path) -> Result<RegistryLoad> {
    let adapter = crate::kernel::harness_runtime::test_script_adapter_factory();
    scan_registry(config, root, Some(&adapter))
}

fn bootstrap_config(root: &Path) -> TurinConfig {
    let mut config = TurinConfig::default();
    config.agent.model = "mock-model".to_string();
    config.agent.provider = "mock".to_string();
    config.kernel.workspace_root = root.to_string_lossy().to_string();
    config.layout.root = Some(".turin".to_string());
    config.harness.directory = root.join("default-harness").to_string_lossy().to_string();
    config.providers.insert(
        "mock".to_string(),
        crate::kernel::config::ProviderConfig {
            kind: "mock".to_string(),
            ..crate::kernel::config::ProviderConfig::default()
        },
    );
    config
}

#[test]
fn scans_local_agent_and_builds_effective_config() -> Result<()> {
    let tmp = tempdir()?;
    let root = tmp.path();
    fs::create_dir_all(root.join("default-harness"))?;
    fs::create_dir_all(root.join(".turin/runtime/agents/docs-reviewer/harness"))?;
    fs::write(
        root.join(".turin/runtime/agents/docs-reviewer/config.toml"),
        r#"
model = "mock-model"
provider = "mock"
system_prompt = "Docs reviewer"
"#,
    )?;

    let bootstrap = bootstrap_config(root);
    let load = scan_registry_with_default(&bootstrap, root)?;
    assert_eq!(load.agents.len(), 1);
    assert_eq!(load.issues.len(), 0);
    assert_eq!(load.agents[0].harness_kind, HarnessKind::Local);
    assert_eq!(load.agents[0].harness_id, "agent::docs-reviewer");

    let effective = build_effective_config(&bootstrap, &load)?;
    assert!(effective.agents.contains_key("docs-reviewer"));
    assert!(effective.harnesses.contains_key("agent::docs-reviewer"));
    Ok(())
}

#[test]
fn isolates_invalid_agent_toml() -> Result<()> {
    let tmp = tempdir()?;
    let root = tmp.path();
    fs::create_dir_all(root.join("default-harness"))?;
    fs::create_dir_all(root.join(".turin/runtime/agents/good/harness"))?;
    fs::create_dir_all(root.join(".turin/runtime/agents/bad/harness"))?;
    fs::write(
        root.join(".turin/runtime/agents/good/config.toml"),
        r#"
model = "mock-model"
provider = "mock"
"#,
    )?;
    fs::write(
        root.join(".turin/runtime/agents/bad/config.toml"),
        "not = [valid",
    )?;

    let bootstrap = bootstrap_config(root);
    let load = scan_registry_with_default(&bootstrap, root)?;
    assert_eq!(load.agents.len(), 1);
    assert_eq!(load.agents[0].id, "good");
    assert_eq!(load.issues.len(), 1);
    Ok(())
}

#[test]
fn isolates_invalid_local_harness() -> Result<()> {
    let tmp = tempdir()?;
    let root = tmp.path();
    fs::create_dir_all(root.join("default-harness"))?;
    fs::create_dir_all(root.join(".turin/runtime/agents/good/harness"))?;
    fs::create_dir_all(root.join(".turin/runtime/agents/bad/harness"))?;
    fs::write(
        root.join(".turin/runtime/agents/good/config.toml"),
        r#"
model = "mock-model"
provider = "mock"
"#,
    )?;
    fs::write(
        root.join(".turin/runtime/agents/good/harness/main.lua"),
        "function on_turn_prepare(ctx)\n  return ALLOW\nend\n",
    )?;
    fs::write(
        root.join(".turin/runtime/agents/bad/config.toml"),
        r#"
model = "mock-model"
provider = "mock"
"#,
    )?;
    fs::write(
        root.join(".turin/runtime/agents/bad/harness/main.lua"),
        "INVALID TEST SOURCE",
    )?;

    let bootstrap = bootstrap_config(root);
    let load = scan_registry_with_default(&bootstrap, root)?;
    assert_eq!(load.agents.len(), 1);
    assert_eq!(load.agents[0].id, "good");
    assert_eq!(load.issues.len(), 1);
    assert!(load.issues[0].path.contains(".turin/runtime/agents/bad"));
    Ok(())
}

#[test]
fn supports_shared_harness_reference() -> Result<()> {
    let tmp = tempdir()?;
    let root = tmp.path();
    fs::create_dir_all(root.join("default-harness"))?;
    fs::create_dir_all(root.join(".turin/harnesses/reviewer"))?;
    fs::create_dir_all(root.join(".turin/runtime/agents/docs-reviewer"))?;
    fs::write(
        root.join(".turin/runtime/agents/docs-reviewer/config.toml"),
        r#"
model = "mock-model"
provider = "mock"
harness = "reviewer"
"#,
    )?;

    let bootstrap = bootstrap_config(root);
    let load = scan_registry_with_default(&bootstrap, root)?;
    assert_eq!(load.shared_harnesses.len(), 1);
    assert_eq!(load.agents[0].harness_kind, HarnessKind::Shared);
    assert_eq!(load.agents[0].harness_id, "reviewer");
    Ok(())
}
