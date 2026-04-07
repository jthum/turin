use anyhow::Result;
use turin_code_index::code_index_reader::CodeIndexStatus;
use turin_code_index::metadata::CodeIndexSemanticStatus;
use turin_code_index_writer::{CodeIndexBuildReport, CodeIndexRemoveReport};
use turin_types::layout::DEFAULT_BOOTSTRAP_CONFIG_PATH;

pub(crate) fn print_build_report(json: bool, report: &CodeIndexBuildReport) -> Result<()> {
    if json {
        print_json(report)
    } else {
        println!("Indexed {}", report.root);
        println!("Index: {}", report.index_path);
        if let Some(codebase_id) = &report.codebase_id {
            println!("Codebase ID: {}", codebase_id);
        }
        println!("Files: {}", report.files_indexed);
        println!("Chunks: {}", report.chunks_indexed);
        println!(
            "Capabilities: {}",
            capabilities_summary(
                report.capabilities.lexical,
                report.capabilities.semantic,
                report.capabilities.hybrid,
                &report.capabilities.languages
            )
        );
        println!("Semantic: {}", semantic_summary(&report.semantic));
        println!("Updated: {}", report.updated_at);
        if report.semantic.embedded_chunks == 0 {
            println!(
                "Hint: add [providers.local_embeddings] plus [embeddings] to {}, rerun `turin-map index`, then confirm `turin-map status` shows `Semantic: enabled`.",
                DEFAULT_BOOTSTRAP_CONFIG_PATH
            );
        }
        Ok(())
    }
}

pub(crate) fn print_remove_report(json: bool, report: &CodeIndexRemoveReport) -> Result<()> {
    if json {
        print_json(report)
    } else {
        println!("Removed {}", report.path);
        println!("Index: {}", report.index_path);
        println!("Removed chunks: {}", report.removed_chunks);
        println!("Updated: {}", report.updated_at);
        Ok(())
    }
}

pub(crate) fn print_status(json: bool, status: &CodeIndexStatus) -> Result<()> {
    if json {
        print_json(status)
    } else {
        println!("Index ready for {}", status.root);
        println!("Index: {}", status.index_path);
        if let Some(codebase_id) = &status.codebase_id {
            println!("Codebase ID: {}", codebase_id);
        }
        println!(
            "Capabilities: {}",
            capabilities_summary(
                status.capabilities.lexical,
                status.capabilities.semantic,
                status.capabilities.hybrid,
                &status.capabilities.languages
            )
        );
        println!("Semantic: {}", semantic_summary(&status.semantic));
        println!("Updated: {}", status.updated_at);
        println!("Age: {}s", status.index_age_seconds);
        if status.semantic.embedded_chunks == 0 {
            println!(
                "Hint: add [providers.local_embeddings] plus [embeddings] to {}, rerun `turin-map index`, then confirm this command shows `Semantic: enabled`.",
                DEFAULT_BOOTSTRAP_CONFIG_PATH
            );
        }
        Ok(())
    }
}

fn print_json(value: &impl serde::Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn capabilities_summary(
    lexical: bool,
    semantic: bool,
    hybrid: bool,
    languages: &[String],
) -> String {
    let mut parts = Vec::new();
    if lexical {
        parts.push("lexical");
    }
    if semantic {
        parts.push("semantic");
    }
    if hybrid {
        parts.push("hybrid");
    }
    let mut summary = parts.join(", ");
    if !languages.is_empty() {
        summary.push_str(" [");
        summary.push_str(&languages.join(", "));
        summary.push(']');
    }
    summary
}

fn semantic_summary(semantic: &CodeIndexSemanticStatus) -> String {
    if semantic.embedded_chunks == 0 {
        return "disabled".to_string();
    }

    let key = semantic.embedding_key.as_deref().unwrap_or("unknown");
    let dims = semantic
        .embedding_dimensions
        .map(|value| value.to_string())
        .unwrap_or_else(|| "?".to_string());
    let vector_format = semantic
        .vector_format
        .as_ref()
        .map(|format| format.as_str())
        .unwrap_or("unknown");

    format!(
        "enabled ({} chunks, key {}, {} dims, {})",
        semantic.embedded_chunks, key, dims, vector_format
    )
}
