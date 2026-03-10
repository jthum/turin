use anyhow::{Result, bail};
use std::path::Path;
use turso::Value;

use super::{CodeIndexCapabilities, CodeSearchMode, CodeSearchRequest};

pub(super) fn negotiated_search_mode(
    requested_mode: CodeSearchMode,
    capabilities: &CodeIndexCapabilities,
    root: &Path,
    strict: bool,
) -> Result<CodeSearchMode> {
    match requested_mode {
        CodeSearchMode::Lexical => Ok(CodeSearchMode::Lexical),
        CodeSearchMode::Semantic if capabilities.semantic => Ok(CodeSearchMode::Semantic),
        CodeSearchMode::Semantic if !strict => Ok(CodeSearchMode::Lexical),
        CodeSearchMode::Semantic => bail!(
            "semantic capability not available for root '{}'",
            root.display()
        ),
        CodeSearchMode::Hybrid if capabilities.hybrid => Ok(CodeSearchMode::Hybrid),
        CodeSearchMode::Hybrid if capabilities.semantic && !strict => Ok(CodeSearchMode::Semantic),
        CodeSearchMode::Hybrid if !strict => Ok(CodeSearchMode::Lexical),
        CodeSearchMode::Hybrid => bail!(
            "hybrid capability not available for root '{}'",
            root.display()
        ),
    }
}

pub(super) fn build_search_sql(
    view_name: &str,
    query: &str,
    request: &CodeSearchRequest,
    has_search_text: bool,
) -> (String, Vec<Value>) {
    let like_value = escape_like_pattern(query);
    let mut params = vec![
        Value::Text(like_value.clone()),
        Value::Text(query.to_string()),
    ];
    let pattern_slot = "?1";
    let exact_slot = "?2";

    let lexical_score_expr = if view_name == CodeSearchMode::Lexical.view_name() {
        Some(format!(
            "CASE \
                WHEN LOWER(name) = LOWER({exact_slot}) THEN 120.0 \
                WHEN LOWER(COALESCE(signature, '')) = LOWER({exact_slot}) THEN 90.0 \
                WHEN LOWER(name) LIKE LOWER({pattern_slot}) ESCAPE '\\' THEN 70.0 \
                WHEN LOWER(COALESCE(signature, '')) LIKE LOWER({pattern_slot}) ESCAPE '\\' THEN 45.0 \
                WHEN LOWER(snippet) LIKE LOWER({pattern_slot}) ESCAPE '\\' THEN 20.0 \
                WHEN LOWER(path) LIKE LOWER({pattern_slot}) ESCAPE '\\' THEN 10.0 \
                ELSE 0.0 \
            END"
        ))
    } else {
        None
    };

    let (mut sql, mut clauses) = if let Some(lexical_score) = lexical_score_expr.as_deref() {
        (
            format!(
                "SELECT chunk_key, path, language, kind, name, signature, snippet, start_line, end_line, {lexical_score} AS score, {lexical_score} AS lexical_score, NULL AS semantic_score FROM {view_name}"
            ),
            vec![lexical_match_clause(pattern_slot, has_search_text)],
        )
    } else {
        (
            format!(
                "SELECT chunk_key, path, language, kind, name, signature, snippet, start_line, end_line, score, lexical_score, semantic_score FROM {view_name}"
            ),
            vec![lexical_match_clause(pattern_slot, has_search_text)],
        )
    };

    if request.min_score > 0.0 {
        params.push(Value::Real(request.min_score));
        if let Some(lexical_score) = lexical_score_expr.as_deref() {
            clauses.push(format!("({lexical_score}) >= ?{}", params.len()));
        } else {
            clauses.push(format!("score >= ?{}", params.len()));
        }
    }

    if !request.languages.is_empty() {
        let slots = push_in_params(&mut params, &request.languages);
        clauses.push(format!("language IN ({})", slots.join(", ")));
    }

    if !request.kinds.is_empty() {
        let slots = push_in_params(&mut params, &request.kinds);
        clauses.push(format!("kind IN ({})", slots.join(", ")));
    }

    if !clauses.is_empty() {
        sql.push_str(" WHERE ");
        sql.push_str(&clauses.join(" AND "));
    }

    sql.push_str(" ORDER BY score DESC, path ASC, start_line ASC");
    let limit = request.limit.max(1);
    params.push(Value::Integer(limit as i64));
    sql.push_str(&format!(" LIMIT ?{}", params.len()));
    (sql, params)
}

fn lexical_match_clause(pattern_slot: &str, has_search_text: bool) -> String {
    if has_search_text {
        format!("search_text LIKE {pattern_slot} ESCAPE '\\'")
    } else {
        format!(
            "(path LIKE {pattern_slot} ESCAPE '\\' OR name LIKE {pattern_slot} ESCAPE '\\' OR COALESCE(signature, '') LIKE {pattern_slot} ESCAPE '\\' OR snippet LIKE {pattern_slot} ESCAPE '\\')"
        )
    }
}

fn push_in_params(params: &mut Vec<Value>, values: &[String]) -> Vec<String> {
    let mut slots = Vec::with_capacity(values.len());
    for value in values {
        params.push(Value::Text(value.clone()));
        slots.push(format!("?{}", params.len()));
    }
    slots
}

fn escape_like_pattern(query: &str) -> String {
    let mut out = String::with_capacity(query.len() + 2);
    out.push('%');
    for ch in query.chars() {
        match ch {
            '\\' | '%' | '_' => {
                out.push('\\');
                out.push(ch);
            }
            _ => out.push(ch),
        }
    }
    out.push('%');
    out
}
