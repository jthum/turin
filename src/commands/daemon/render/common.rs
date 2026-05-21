use anyhow::Result;
use serde_json::Value;

use turin::daemon::protocol::{ErrorCode, ErrorEnvelope, ResponseEnvelope};

use super::super::IssueView;

pub(in crate::commands::daemon) fn print_response(
    response: ResponseEnvelope,
    json_output: bool,
) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string_pretty(&response)?);
        return Ok(());
    }

    if response.ok {
        if let Some(result) = response.result {
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            println!("ok");
        }
        Ok(())
    } else {
        let error = response.error.unwrap_or(ErrorEnvelope {
            code: ErrorCode::InternalError,
            message: "Unknown daemon error".to_string(),
            details: None,
        });
        anyhow::bail!("{}: {}", error.code, error.message);
    }
}

pub(in crate::commands::daemon) fn decode_result<T: serde::de::DeserializeOwned>(
    response: ResponseEnvelope,
) -> Result<T> {
    if response.ok {
        let value = response
            .result
            .ok_or_else(|| anyhow::anyhow!("Daemon response did not include a result payload"))?;
        Ok(serde_json::from_value(value)?)
    } else {
        let error = response.error.unwrap_or(ErrorEnvelope {
            code: ErrorCode::InternalError,
            message: "Unknown daemon error".to_string(),
            details: None,
        });
        anyhow::bail!("{}: {}", error.code, error.message);
    }
}

pub(in crate::commands::daemon) fn print_issue_list(title: &str, issues: &[IssueView]) {
    println!("{}", title);
    if issues.is_empty() {
        println!("  none");
        return;
    }
    for issue in issues {
        println!("- {}", issue.path);
        println!("  {}", issue.message);
    }
}

pub(in crate::commands::daemon) fn yes_no(value: bool) -> String {
    if value { "yes" } else { "no" }.to_string()
}

pub(super) fn json_snippet(value: &Value, max_chars: usize) -> String {
    let mut rendered = match value {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| "<unserializable>".to_string()),
    };
    rendered = rendered.replace('\n', "\\n");
    let char_count = rendered.chars().count();
    if char_count > max_chars {
        let truncated: String = rendered.chars().take(max_chars.saturating_sub(1)).collect();
        format!("{}…", truncated)
    } else {
        rendered
    }
}

pub(super) fn print_indented(text: &str) {
    for line in text.lines() {
        println!("    {}", line);
    }
}

pub(super) fn print_table(rows: &[Vec<String>]) {
    if rows.is_empty() {
        return;
    }
    let cols = rows[0].len();
    let mut widths = vec![0usize; cols];
    for row in rows {
        for (idx, cell) in row.iter().enumerate() {
            widths[idx] = widths[idx].max(cell.len());
        }
    }

    for (row_idx, row) in rows.iter().enumerate() {
        let line = row
            .iter()
            .enumerate()
            .map(|(idx, cell)| format!("{:width$}", cell, width = widths[idx]))
            .collect::<Vec<_>>()
            .join("  ");
        println!("{}", line);
        if row_idx == 0 {
            let sep = widths
                .iter()
                .map(|width| "-".repeat(*width))
                .collect::<Vec<_>>()
                .join("  ");
            println!("{}", sep);
        }
    }
}

pub(super) fn format_context_target(target: &Value) -> String {
    let Some(kind) = target.get("kind").and_then(|value| value.as_str()) else {
        return "-".to_string();
    };
    match kind {
        "branch_head" => match target
            .get("branch_head_id")
            .and_then(|value| value.as_i64())
        {
            Some(branch_head_id) => format!("branch_head:{branch_head_id}"),
            None => "branch_head:active".to_string(),
        },
        "turn_id" => target
            .get("turn_id")
            .and_then(|value| value.as_i64())
            .map(|turn_id| format!("turn:{turn_id}"))
            .unwrap_or_else(|| "turn".to_string()),
        "selected_path" => target
            .get("turn_ids")
            .and_then(|value| value.as_array())
            .map(|turn_ids| format!("selected_path:{}", turn_ids.len()))
            .unwrap_or_else(|| "selected_path".to_string()),
        "external_reference" => target
            .get("reference")
            .and_then(|value| value.as_str())
            .map(|reference| format!("external:{reference}"))
            .unwrap_or_else(|| "external".to_string()),
        "summary_source" => target
            .get("source_turn_id")
            .and_then(|value| value.as_i64())
            .map(|turn_id| format!("summary:{turn_id}"))
            .unwrap_or_else(|| "summary".to_string()),
        other => other.to_string(),
    }
}
