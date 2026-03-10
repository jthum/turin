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

pub(super) fn build_lexical_search_sql(
    source_name: &str,
    use_computed_lexical_score: bool,
    query: &str,
    request: &CodeSearchRequest,
    has_search_text: bool,
    limit: usize,
) -> (String, Vec<Value>) {
    if has_search_text && should_use_fts_lexical(query) {
        return build_fts_lexical_search_sql(
            source_name,
            use_computed_lexical_score,
            query,
            request,
            limit,
        );
    }

    let like_value = escape_like_pattern(query);
    let mut params = vec![
        Value::Text(like_value.clone()),
        Value::Text(query.to_string()),
    ];
    let pattern_slot = "?1";
    let exact_slot = "?2";
    let path_bonus_expr = build_path_bonus_expr(&mut params, query, exact_slot, pattern_slot);
    let use_token_match = has_search_text && query.chars().any(char::is_whitespace);
    let token_query = if use_token_match {
        build_token_query(&mut params, query)
    } else {
        None
    };
    let token_bonus_expr = token_query
        .as_ref()
        .map(|query| query.bonus_expr.as_str())
        .unwrap_or("0.0");

    let lexical_score_expr = if use_computed_lexical_score {
        Some(format!(
            "(CASE \
                WHEN LOWER(name) = LOWER({exact_slot}) THEN 120.0 \
                WHEN LOWER(COALESCE(signature, '')) = LOWER({exact_slot}) THEN 90.0 \
                WHEN LOWER(name) LIKE LOWER({pattern_slot}) ESCAPE '\\' THEN 70.0 \
                WHEN LOWER(COALESCE(signature, '')) LIKE LOWER({pattern_slot}) ESCAPE '\\' THEN 45.0 \
                WHEN LOWER(snippet) LIKE LOWER({pattern_slot}) ESCAPE '\\' THEN 20.0 \
                ELSE 0.0 \
            END + {path_bonus_expr} + \
            {token_bonus_expr})"
        ))
    } else {
        None
    };

    let (mut sql, mut clauses) = if let Some(lexical_score) = lexical_score_expr.as_deref() {
        (
            format!(
                "SELECT chunk_key, path, language, kind, name, signature, snippet, start_line, end_line, {lexical_score} AS score, {lexical_score} AS lexical_score, NULL AS semantic_score FROM {source_name}"
            ),
            vec![lexical_match_clause(
                pattern_slot,
                token_query
                    .as_ref()
                    .map(|query| query.match_clause.as_str()),
            )],
        )
    } else {
        (
            format!(
                "SELECT chunk_key, path, language, kind, name, signature, snippet, start_line, end_line, score, lexical_score, semantic_score FROM {source_name}"
            ),
            vec![lexical_match_clause(
                pattern_slot,
                token_query
                    .as_ref()
                    .map(|query| query.match_clause.as_str()),
            )],
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
    params.push(Value::Integer(limit as i64));
    sql.push_str(&format!(" LIMIT ?{}", params.len()));
    (sql, params)
}

fn build_fts_lexical_search_sql(
    source_name: &str,
    use_computed_lexical_score: bool,
    query: &str,
    request: &CodeSearchRequest,
    limit: usize,
) -> (String, Vec<Value>) {
    let fts_query = build_fts_query(query);
    let like_value = escape_like_pattern(query);
    let mut params = vec![
        Value::Text(fts_query),
        Value::Text(query.to_string()),
        Value::Text(like_value.clone()),
    ];
    let fts_slot = "?1";
    let exact_slot = "?2";
    let pattern_slot = "?3";
    let path_bonus_expr = build_path_bonus_expr(&mut params, query, exact_slot, pattern_slot);
    let token_query = if query.chars().any(char::is_whitespace) {
        build_token_query(&mut params, query)
    } else {
        None
    };
    let token_bonus_expr = token_query
        .as_ref()
        .map(|query| query.bonus_expr.as_str())
        .unwrap_or("0.0");
    let fts_score_expr = format!(
        "CASE WHEN fts_match(search_text, {fts_slot}) THEN fts_score(search_text, {fts_slot}) ELSE 0.0 END"
    );
    let fallback_match_clause = lexical_match_clause(
        pattern_slot,
        token_query
            .as_ref()
            .map(|query| query.match_clause.as_str()),
    );

    let lexical_score_expr = if use_computed_lexical_score {
        Some(format!(
            "({fts_score_expr} + \
              CASE \
                WHEN LOWER(name) = LOWER({exact_slot}) THEN 120.0 \
                WHEN LOWER(COALESCE(signature, '')) = LOWER({exact_slot}) THEN 90.0 \
                WHEN LOWER(name) LIKE LOWER({pattern_slot}) ESCAPE '\\' THEN 70.0 \
                WHEN LOWER(COALESCE(signature, '')) LIKE LOWER({pattern_slot}) ESCAPE '\\' THEN 45.0 \
                WHEN LOWER(snippet) LIKE LOWER({pattern_slot}) ESCAPE '\\' THEN 20.0 \
                ELSE 0.0 \
              END + {path_bonus_expr} + \
              {token_bonus_expr})"
        ))
    } else {
        None
    };

    let (mut sql, mut clauses) = if let Some(lexical_score) = lexical_score_expr.as_deref() {
        (
            format!(
                "SELECT chunk_key, path, language, kind, name, signature, snippet, start_line, end_line, {lexical_score} AS score, {lexical_score} AS lexical_score, NULL AS semantic_score FROM {source_name}"
            ),
            vec![format!(
                "(fts_match(search_text, {fts_slot}) OR {fallback_match_clause})"
            )],
        )
    } else {
        (
            format!(
                "SELECT chunk_key, path, language, kind, name, signature, snippet, start_line, end_line, score, lexical_score, semantic_score FROM {source_name}"
            ),
            vec![format!(
                "(fts_match(search_text, {fts_slot}) OR {fallback_match_clause})"
            )],
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
    params.push(Value::Integer(limit as i64));
    sql.push_str(&format!(" LIMIT ?{}", params.len()));
    (sql, params)
}

pub(super) fn build_semantic_search_sql(
    view_name: &str,
    request: &CodeSearchRequest,
    limit: usize,
) -> (String, Vec<Value>) {
    let mut params = vec![Value::Blob(Vec::new())];
    let semantic_score_expr = "1.0 - vector_distance_cos(embedding, vector8(?1))";
    let mut clauses = vec!["embedding IS NOT NULL".to_string()];

    if request.min_score > 0.0 {
        params.push(Value::Real(request.min_score));
        clauses.push(format!("({semantic_score_expr}) >= ?{}", params.len()));
    }

    if !request.languages.is_empty() {
        let slots = push_in_params(&mut params, &request.languages);
        clauses.push(format!("language IN ({})", slots.join(", ")));
    }

    if !request.kinds.is_empty() {
        let slots = push_in_params(&mut params, &request.kinds);
        clauses.push(format!("kind IN ({})", slots.join(", ")));
    }

    let mut sql = format!(
        "SELECT chunk_key, path, language, kind, name, signature, snippet, start_line, end_line, {semantic_score_expr} AS score, NULL AS lexical_score, {semantic_score_expr} AS semantic_score FROM {view_name}"
    );
    sql.push_str(" WHERE ");
    sql.push_str(&clauses.join(" AND "));
    sql.push_str(" ORDER BY score DESC, path ASC, start_line ASC");
    params.push(Value::Integer(limit.max(1) as i64));
    sql.push_str(&format!(" LIMIT ?{}", params.len()));
    (sql, params)
}

pub(super) fn hybrid_candidate_limit(limit: usize) -> usize {
    (limit.max(1) * 5).max(20)
}

pub(super) fn reciprocal_rank(rank: usize) -> f64 {
    100.0 / (60.0 + rank as f64)
}

fn lexical_match_clause(pattern_slot: &str, token_match_clause: Option<&str>) -> String {
    let fallback = format!(
        "(path LIKE {pattern_slot} ESCAPE '\\' OR name LIKE {pattern_slot} ESCAPE '\\' OR COALESCE(signature, '') LIKE {pattern_slot} ESCAPE '\\' OR snippet LIKE {pattern_slot} ESCAPE '\\')"
    );
    if let Some(token_match_clause) = token_match_clause {
        format!("({fallback} OR ({token_match_clause}))")
    } else {
        fallback
    }
}

struct TokenQuery {
    bonus_expr: String,
    match_clause: String,
}

fn build_path_bonus_expr(
    params: &mut Vec<Value>,
    query: &str,
    exact_slot: &str,
    pattern_slot: &str,
) -> String {
    if looks_like_path_query(query) {
        let basename = basename_query(query);
        params.push(Value::Text(escape_like_pattern(&basename)));
        let basename_slot = format!("?{}", params.len());
        format!(
            "CASE \
                WHEN LOWER(path) = LOWER({exact_slot}) THEN 220.0 \
                WHEN LOWER(path) LIKE LOWER({pattern_slot}) ESCAPE '\\' THEN 150.0 \
                WHEN LOWER(path) LIKE LOWER({basename_slot}) ESCAPE '\\' THEN 110.0 \
                ELSE 0.0 \
            END"
        )
    } else {
        format!(
            "CASE \
                WHEN LOWER(path) = LOWER({exact_slot}) THEN 35.0 \
                WHEN LOWER(path) LIKE LOWER({pattern_slot}) ESCAPE '\\' THEN 10.0 \
                ELSE 0.0 \
            END"
        )
    }
}

fn build_token_query(params: &mut Vec<Value>, query: &str) -> Option<TokenQuery> {
    let mut parts = Vec::new();
    let mut matches = Vec::new();
    let tokens = lexical_tokens(query);
    for token in tokens {
        params.push(Value::Text(escape_like_pattern(&token)));
        let slot = format!("?{}", params.len());
        matches.push(format!("LOWER(search_text) LIKE LOWER({slot}) ESCAPE '\\'"));
        parts.push(format!(
            "CASE WHEN LOWER(search_text) LIKE LOWER({slot}) ESCAPE '\\' THEN 12.0 ELSE 0.0 END"
        ));
    }

    if parts.is_empty() {
        None
    } else {
        Some(TokenQuery {
            bonus_expr: parts.join(" + "),
            match_clause: matches.join(" AND "),
        })
    }
}

fn lexical_tokens(query: &str) -> Vec<String> {
    let mut out = Vec::new();
    for token in query
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .map(str::trim)
        .filter(|token| token.len() >= 2)
    {
        let lowered = token.to_ascii_lowercase();
        if !out.iter().any(|existing| existing == &lowered) {
            out.push(lowered);
        }
    }
    out
}

fn build_fts_query(query: &str) -> String {
    let tokens = lexical_tokens(query);
    if tokens.is_empty() {
        format!("\"{}\"", escape_fts_term(query.trim()))
    } else {
        tokens
            .into_iter()
            .map(|token| format!("\"{}\"", escape_fts_term(&token)))
            .collect::<Vec<_>>()
            .join(" AND ")
    }
}

fn should_use_fts_lexical(query: &str) -> bool {
    let tokens = lexical_tokens(query);
    query.chars().any(char::is_whitespace) || tokens.len() > 1
}

fn looks_like_path_query(query: &str) -> bool {
    let query = query.trim();
    query.contains('/') || query.contains('\\') || query.contains('.') || query.contains("::")
}

fn basename_query(query: &str) -> String {
    query
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(query)
        .trim()
        .to_string()
}

fn escape_fts_term(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
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
