use anyhow::{Context, Result};
use clap::ValueEnum;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum InitProvider {
    Anthropic,
    Openai,
    Mock,
}

impl InitProvider {
    pub fn alias(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::Openai => "openai",
            Self::Mock => "mock",
        }
    }

    pub fn default_model(self) -> &'static str {
        match self {
            Self::Anthropic => "claude-sonnet-4-20250514",
            Self::Openai => "gpt-4o",
            Self::Mock => "mock-model",
        }
    }

    pub fn default_api_key_env(self) -> Option<&'static str> {
        match self {
            Self::Anthropic => Some("ANTHROPIC_API_KEY"),
            Self::Openai => Some("OPENAI_API_KEY"),
            Self::Mock => None,
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Anthropic => "Anthropic API",
            Self::Openai => "OpenAI or another OpenAI-compatible endpoint",
            Self::Mock => "Local mock provider for no-key quickstarts and harness tests",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum GovernancePreset {
    Open,
    Balanced,
    Governed,
}

impl GovernancePreset {
    pub fn profile(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Balanced => "balanced",
            Self::Governed => "governed",
        }
    }

    pub fn enforcement_enabled(self) -> bool {
        !matches!(self, Self::Open)
    }

    pub fn audit_mode(self) -> &'static str {
        match self {
            Self::Open => "off",
            Self::Balanced => "observational",
            Self::Governed => "immutable",
        }
    }

    pub fn import_mode(self) -> &'static str {
        match self {
            Self::Open => "legacy",
            Self::Balanced => "mixed",
            Self::Governed => "scoped",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Open => "Open experimentation with minimal governance friction",
            Self::Balanced => "Safer defaults with observability and capability enforcement",
            Self::Governed => "Tighter audit and scoped-import defaults for regulated setups",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum HarnessTemplate {
    Starter,
    Safety,
    CodingAssistant,
    Reviewer,
}

impl HarnessTemplate {
    pub fn name(self) -> &'static str {
        match self {
            Self::Starter => "starter",
            Self::Safety => "safety",
            Self::CodingAssistant => "coding-assistant",
            Self::Reviewer => "reviewer",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Starter => "Readable starter harness with light workspace context",
            Self::Safety => "Safety-first harness that blocks destructive shell commands",
            Self::CodingAssistant => {
                "Coding-oriented harness that folds checked-in project briefs into prompts"
            }
            Self::Reviewer => "Review-focused harness that pushes findings-first output",
        }
    }

    pub fn files(self) -> &'static [HarnessTemplateFile] {
        match self {
            Self::Starter => STARTER_TEMPLATE_FILES,
            Self::Safety => SAFETY_TEMPLATE_FILES,
            Self::CodingAssistant => CODING_ASSISTANT_TEMPLATE_FILES,
            Self::Reviewer => REVIEWER_TEMPLATE_FILES,
        }
    }
}

#[derive(Clone, Debug)]
pub struct InitOptions {
    pub provider: InitProvider,
    pub model: String,
    pub harness_template: HarnessTemplate,
    pub governance: GovernancePreset,
    pub force: bool,
}

#[derive(Clone, Debug)]
pub struct ScaffoldResult {
    pub created_paths: Vec<PathBuf>,
    pub updated_paths: Vec<PathBuf>,
    pub provider: InitProvider,
    pub model: String,
    pub harness_template: HarnessTemplate,
    pub governance: GovernancePreset,
}

#[derive(Clone, Copy)]
pub struct HarnessTemplateFile {
    pub path: &'static str,
    pub contents: &'static str,
}

const STARTER_TEMPLATE_FILES: &[HarnessTemplateFile] = &[HarnessTemplateFile {
    path: "main.lua",
    contents: r#"-- Starter harness: keep policy close to the job.
-- Add stricter guardrails only where they make the workflow clearer.

function on_turn_prepare(ctx)
  local brief = cache.file("TURIN.md", { include_content = true })
  if brief and brief.content then
    ctx.system_prompt = ctx.system_prompt .. "\n\nWorkspace brief:\n" .. brief.content
  end

  return ALLOW
end

function on_tool_call(call)
  return ALLOW
end
"#,
}];

const SAFETY_TEMPLATE_FILES: &[HarnessTemplateFile] = &[HarnessTemplateFile {
    path: "00_safety.lua",
    contents: r#"-- Safety harness: block obviously destructive shell commands.

function on_tool_call(call)
  if call.name ~= "shell_exec" then
    return ALLOW
  end

  local command = call.args.command or ""
  local blocked = {
    "rm %-rf",
    "mkfs",
    "dd if=",
    "shred",
  }

  for _, pattern in ipairs(blocked) do
    if command:find(pattern) then
      return REJECT, "Blocked destructive shell command: " .. pattern
    end
  end

  return ALLOW
end
"#,
}];

const CODING_ASSISTANT_TEMPLATE_FILES: &[HarnessTemplateFile] = &[
    HarnessTemplateFile {
        path: "00_safety.lua",
        contents: SAFETY_TEMPLATE_FILES[0].contents,
    },
    HarnessTemplateFile {
        path: "10_coding_assistant.lua",
        contents: r#"-- Coding assistant harness: keep the agent grounded in checked-in context.

function on_turn_prepare(ctx)
  local brief = cache.file("TURIN.md", { include_content = true })
  if brief and brief.content then
    ctx.system_prompt = ctx.system_prompt .. "\n\nProject brief:\n" .. brief.content
  end

  local constraints = cache.file("CONSTRAINTS.md", { include_content = true })
  if constraints and constraints.content then
    ctx.system_prompt = ctx.system_prompt .. "\n\nDelivery constraints:\n" .. constraints.content
  end

  return ALLOW
end
"#,
    },
];

const REVIEWER_TEMPLATE_FILES: &[HarnessTemplateFile] = &[HarnessTemplateFile {
    path: "main.lua",
    contents: r#"-- Reviewer harness: bias the agent toward concrete findings and risk calls.

function on_turn_prepare(ctx)
  ctx.system_prompt = ctx.system_prompt
    .. "\n\nReview style:\n"
    .. "- Findings first\n"
    .. "- Focus on bugs, regressions, missing tests, and unclear assumptions\n"
    .. "- Keep summaries short and only after concrete findings\n"

  return ALLOW
end
"#,
}];

pub fn scaffold_project(root: &Path, options: &InitOptions) -> Result<ScaffoldResult> {
    let config_path = root.join("turin.toml");
    if config_path.exists() && !options.force {
        anyhow::bail!(
            "turin.toml already exists at '{}'; rerun with --force to overwrite it",
            config_path.display()
        );
    }

    let harness_dir = root.join(".turin").join("harnesses");
    fs::create_dir_all(&harness_dir)
        .with_context(|| format!("failed to create '{}'", harness_dir.display()))?;
    let mut created_paths = vec![harness_dir.clone()];
    let mut updated_paths = Vec::new();

    let template_paths =
        scaffold_harness_template(&harness_dir, options.harness_template, options.force)?;
    created_paths.extend(template_paths);

    let state_db_path = root.join(".turin").join("state.db");
    if !state_db_path.exists() {
        if let Some(parent) = state_db_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create '{}'", parent.display()))?;
        }
        fs::File::create(&state_db_path)
            .with_context(|| format!("failed to create '{}'", state_db_path.display()))?;
        created_paths.push(state_db_path);
    }

    fs::write(&config_path, render_turin_toml(options))
        .with_context(|| format!("failed to write '{}'", config_path.display()))?;
    if config_path.exists() {
        updated_paths.push(config_path);
    }

    let gitignore_path = root.join(".gitignore");
    if ensure_gitignore_entry(&gitignore_path, ".turin/")? {
        updated_paths.push(gitignore_path);
    }

    Ok(ScaffoldResult {
        created_paths,
        updated_paths,
        provider: options.provider,
        model: options.model.clone(),
        harness_template: options.harness_template,
        governance: options.governance,
    })
}

pub fn scaffold_harness_template(
    dir: &Path,
    template: HarnessTemplate,
    force: bool,
) -> Result<Vec<PathBuf>> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create '{}'", dir.display()))?;
    let mut written = Vec::new();

    for file in template.files() {
        let path = dir.join(file.path);
        if path.exists() && !force {
            anyhow::bail!(
                "harness file '{}' already exists; rerun with --force to overwrite it",
                path.display()
            );
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create '{}'", parent.display()))?;
        }
        fs::write(&path, file.contents)
            .with_context(|| format!("failed to write '{}'", path.display()))?;
        written.push(path);
    }

    Ok(written)
}

fn ensure_gitignore_entry(path: &Path, entry: &str) -> Result<bool> {
    let normalized = entry.trim();
    if normalized.is_empty() {
        return Ok(false);
    }

    let mut contents = if path.exists() {
        fs::read_to_string(path).with_context(|| format!("failed to read '{}'", path.display()))?
    } else {
        String::new()
    };

    if contents
        .lines()
        .any(|line| line.trim_end_matches('/') == normalized.trim_end_matches('/'))
    {
        return Ok(false);
    }

    if !contents.is_empty() && !contents.ends_with('\n') {
        contents.push('\n');
    }
    contents.push_str(normalized);
    contents.push('\n');
    fs::write(path, contents).with_context(|| format!("failed to write '{}'", path.display()))?;
    Ok(true)
}

pub fn render_turin_toml(options: &InitOptions) -> String {
    let mut toml = String::new();
    toml.push_str("[agent]\n");
    toml.push_str("id = \"default\"\n");
    toml.push_str(&format!(
        "system_prompt = \"{}\"\n",
        default_system_prompt(options.harness_template)
    ));
    toml.push_str(&format!("model = \"{}\"\n", options.model));
    toml.push_str(&format!("provider = \"{}\"\n", options.provider.alias()));
    toml.push_str("mode = \"auto\"\n\n");

    toml.push_str("[kernel]\n");
    toml.push_str("workspace_root = \".\"\n");
    toml.push_str("max_turns = 50\n");
    toml.push_str("heartbeat_interval_secs = 30\n");
    toml.push_str("initial_spawn_depth = 0\n\n");

    toml.push_str("[persistence.state]\n");
    toml.push_str("path = \".turin/state.db\"\n\n");

    toml.push_str("[harness]\n");
    toml.push_str("directory = \".turin/harnesses\"\n");
    toml.push_str("fs_root = \".\"\n\n");

    render_provider_block(&mut toml, options.provider);
    if options.provider != InitProvider::Mock {
        toml.push('\n');
    }
    if options.provider != InitProvider::Mock {
        render_mock_provider_block(&mut toml);
    } else {
        toml.push_str("# Replace [providers.mock] with Anthropic or OpenAI when you are ready for real inference.\n");
    }

    toml.push_str("\n# Optional embeddings for semantic memory and code search.\n");
    toml.push_str(
        "# Reuse an existing provider alias or point at a local OpenAI-compatible endpoint.\n",
    );
    toml.push_str("# [providers.local_embeddings]\n");
    toml.push_str("# type = \"openai\"\n");
    toml.push_str("# base_url = \"http://127.0.0.1:11434/v1\"\n");
    toml.push_str("#\n");
    toml.push_str("# [embeddings]\n");
    toml.push_str("# provider = \"local_embeddings\"\n");
    toml.push_str("# model = \"your-small-embedding-model\"\n");
    toml.push_str("# dimensions = 384\n");
    toml.push_str("#\n");
    toml.push_str("# Then run `turin-map index` from the project root.\n");

    toml.push_str("\n[governance]\n");
    toml.push_str(&format!("profile = \"{}\"\n", options.governance.profile()));
    toml.push_str(&format!(
        "enforcement_enabled = {}\n\n",
        options.governance.enforcement_enabled()
    ));

    toml.push_str("[governance.audit]\n");
    toml.push_str(&format!("mode = \"{}\"\n", options.governance.audit_mode()));
    toml.push_str("include_capability_context = false\n\n");

    toml.push_str("[governance.import]\n");
    toml.push_str(&format!(
        "mode = \"{}\"\n",
        options.governance.import_mode()
    ));
    toml.push_str(&format!(
        "allow_unscoped_in_open = {}\n",
        matches!(options.governance, GovernancePreset::Open)
    ));

    toml
}

fn render_provider_block(toml: &mut String, provider: InitProvider) {
    match provider {
        InitProvider::Anthropic => {
            toml.push_str("[providers.anthropic]\n");
            toml.push_str("type = \"anthropic\"\n");
            toml.push_str("api_key_env = \"ANTHROPIC_API_KEY\"\n");
            toml.push_str("# base_url = \"https://api.anthropic.com/v1\"\n");
        }
        InitProvider::Openai => {
            toml.push_str("[providers.openai]\n");
            toml.push_str("type = \"openai\"\n");
            toml.push_str("api_key_env = \"OPENAI_API_KEY\"\n");
            toml.push_str("# base_url = \"https://api.openai.com/v1\"\n");
        }
        InitProvider::Mock => render_mock_provider_block(toml),
    }
}

fn render_mock_provider_block(toml: &mut String) {
    toml.push_str("[providers.mock]\n");
    toml.push_str("type = \"mock\"\n");
    toml.push_str(
        "# Change this text to shape the quickstart response without calling a real model.\n",
    );
    toml.push_str("base_url = \"Turin quickstart is wired correctly. Replace providers.mock when you are ready.\"\n");
}

fn default_system_prompt(template: HarnessTemplate) -> &'static str {
    match template {
        HarnessTemplate::Starter | HarnessTemplate::Safety => {
            "You are a helpful assistant. Use the harness to stay inside project rules."
        }
        HarnessTemplate::CodingAssistant => {
            "You are a careful coding assistant. Prefer concrete edits, tests, and explicit tradeoffs."
        }
        HarnessTemplate::Reviewer => {
            "You are a strict code reviewer. Focus on correctness, regressions, and missing tests."
        }
    }
}
