use anyhow::Result;
use std::path::Path;
use turin::kernel::Kernel;
use turin::kernel::config::TurinConfig;

pub async fn run_check(config_path: &Path) -> Result<()> {
    println!("\x1b[36m\x1b[1mChecking Turin project configuration...\x1b[0m");

    // 1. Load turin.toml
    let config = match TurinConfig::from_file(config_path) {
        Ok(c) => {
            println!("\x1b[32m\x1b[1m✓\x1b[0m Configuration file is valid TOML.");
            c
        }
        Err(e) => {
            println!("\x1b[31m\x1b[1m✗\x1b[0m Configuration error: {}", e);
            return Ok(());
        }
    };

    // 2. Check API keys
    let provider = &config.agent.provider;
    if let Some(provider_config) = config.providers.get(provider) {
        if let Some(ref env_var) = provider_config.api_key_env {
            if std::env::var(env_var).is_err() {
                println!(
                    "\x1b[33m\x1b[1m! Warning:\x1b[0m API key for provider '{}' ({}) is not set in environment.",
                    provider, env_var
                );
            } else {
                println!(
                    "\x1b[32m\x1b[1m✓\x1b[0m API key for provider '{}' is set.",
                    provider
                );
            }
        }
    } else {
        println!(
            "\x1b[31m\x1b[1m✗\x1b[0m Provider '{}' not found in [providers].",
            provider
        );
    }

    // 3. Validate Harness
    let harness_dir = Path::new(&config.harness.directory);
    if !harness_dir.exists() {
        println!(
            "\x1b[33m\x1b[1m! Warning:\x1b[0m Harness directory '{}' does not exist.",
            harness_dir.display()
        );
    } else {
        println!("\x1b[32m\x1b[1m✓\x1b[0m Harness directory exists.");

        println!("  Validating harness scripts...");
        let mut kernel = match Kernel::builder(config.clone()).build() {
            Ok(k) => k,
            Err(e) => {
                println!("\x1b[31m\x1b[1m✗\x1b[0m Failed to build Kernel: {}", e);
                return Ok(());
            }
        };

        match kernel.init_harness().await {
            Ok(_) => {
                let loaded = kernel.loaded_scripts();
                if loaded.is_empty() {
                    println!("    \x1b[33m(No .lua scripts found in harness directory)\x1b[0m");
                } else {
                    for script in loaded {
                        println!("    \x1b[32m\x1b[1m✓\x1b[0m Loaded and parsed: {}", script);
                    }
                }
            }
            Err(e) => {
                println!(
                    "\n\x1b[31m\x1b[1m✗ Harness validation failed:\x1b[0m\n{}",
                    e
                );
            }
        }
    }

    // 4. Check DB
    let db_path = Path::new(&config.persistence.database_path);
    if db_path.exists() {
        println!(
            "\x1b[32m\x1b[1m✓\x1b[0m State database found at '{}'.",
            db_path.display()
        );
    } else {
        println!(
            "\x1b[34mℹ\x1b[0m State database will be created at '{}' on first run.",
            db_path.display()
        );
    }

    println!("\n\x1b[32m\x1b[1mValidation complete!\x1b[0m");
    Ok(())
}
