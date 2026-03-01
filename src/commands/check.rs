use anyhow::Result;
use std::path::Path;
use turin::display;
use turin::kernel::Kernel;
use turin::kernel::config::TurinConfig;

pub async fn run_check(config_path: &Path) -> Result<()> {
    let ansi = display::stdout_ansi();
    println!(
        "{}",
        display::header("Checking Turin project configuration...", ansi)
    );

    // 1. Load turin.toml
    let config = match TurinConfig::from_file(config_path) {
        Ok(c) => {
            println!(
                "{} Configuration file is valid TOML.",
                display::ok_mark(ansi)
            );
            c
        }
        Err(e) => {
            println!("{} Configuration error: {}", display::err_mark(ansi), e);
            return Ok(());
        }
    };

    // 2. Check API keys
    let provider = &config.agent.provider;
    if let Some(provider_config) = config.providers.get(provider) {
        if let Some(ref env_var) = provider_config.api_key_env {
            if std::env::var(env_var).is_err() {
                println!(
                    "{} Warning: API key for provider '{}' ({}) is not set in environment.",
                    display::warn_mark(ansi),
                    provider,
                    env_var
                );
            } else {
                println!(
                    "{} API key for provider '{}' is set.",
                    display::ok_mark(ansi),
                    provider
                );
            }
        }
    } else {
        println!(
            "{} Provider '{}' not found in [providers].",
            display::err_mark(ansi),
            provider
        );
    }

    // 3. Validate Harnesses
    let mut harness_entries = Vec::new();
    harness_entries.push(("default".to_string(), config.harness.directory.clone()));
    for (harness_id, harness_cfg) in &config.harnesses {
        harness_entries.push((harness_id.clone(), harness_cfg.directory.clone()));
    }

    for (harness_id, harness_dir) in &harness_entries {
        let harness_dir = Path::new(harness_dir);
        if !harness_dir.exists() {
            println!(
                "{} Warning: Harness '{}' directory '{}' does not exist.",
                display::warn_mark(ansi),
                harness_id,
                harness_dir.display()
            );
        } else {
            println!(
                "{} Harness '{}' directory exists.",
                display::ok_mark(ansi),
                harness_id
            );
        }
    }

    println!("  Validating harness scripts...");
    let mut kernel = match Kernel::builder(config.clone()).build() {
        Ok(k) => k,
        Err(e) => {
            println!("{} Failed to build Kernel: {}", display::err_mark(ansi), e);
            return Ok(());
        }
    };

    match kernel.init_harness().await {
        Ok(_) => {
            let loaded = kernel.loaded_scripts();
            if loaded.is_empty() {
                println!(
                    "    {}",
                    display::paint(
                        "(No .lua scripts found in default harness directory)",
                        "33",
                        ansi
                    )
                );
            } else {
                for script in loaded {
                    println!(
                        "    {} Loaded and parsed: {}",
                        display::ok_mark(ansi),
                        script
                    );
                }
            }
        }
        Err(e) => {
            println!(
                "\n{} Harness validation failed:\n{}",
                display::err_mark(ansi),
                e
            );
        }
    }

    // 4. Check DB
    let db_path = Path::new(&config.persistence.database_path);
    if db_path.exists() {
        println!(
            "{} State database found at '{}'.",
            display::ok_mark(ansi),
            db_path.display()
        );
    } else {
        println!(
            "{} State database will be created at '{}' on first run.",
            display::info_mark(ansi),
            db_path.display()
        );
    }

    println!("\n{} Validation complete!", display::ok_mark(ansi));
    Ok(())
}
