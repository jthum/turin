use anyhow::Result;
use tracing::{debug, instrument};
use turin_daemon_protocol::SessionSearchScope;

use super::session_summary_from_row_and_selector;
use crate::daemon::state::{DaemonState, SessionSearchHit, SessionSummary};
use crate::kernel::session_refs::{describe_store_selector, format_session_reference};
use crate::persistence::manager::StoreSelector;

impl DaemonState {
    #[instrument(skip(self), fields(store = %store_selector_label(store_selector.as_ref())))]
    pub async fn list_sessions(
        &self,
        limit: usize,
        offset: usize,
        store_selector: Option<StoreSelector>,
    ) -> Result<Vec<SessionSummary>> {
        let store = match &store_selector {
            Some(selector) => self.kernel.store_manager().open(selector).await?,
            None => self.kernel.store_manager().get_default().await?,
        };
        let rows = store.list_session_rows(limit, offset).await?;
        debug!(count = rows.len(), "Listed persisted sessions");
        Ok(rows
            .iter()
            .map(|row| match &store_selector {
                Some(selector) => session_summary_from_row_and_selector(row, selector),
                None => super::super::helpers::session_summary_from_row(row),
            })
            .collect())
    }

    pub async fn list_linked_sessions(
        &self,
        parent_session_id: &str,
        limit: usize,
        offset: usize,
    ) -> Result<Option<Vec<SessionSummary>>> {
        let Some((store_selector, store, parent)) =
            self.resolve_persisted_session(parent_session_id).await?
        else {
            return Ok(None);
        };
        let rows = store
            .list_linked_session_rows(parent.id, limit, offset)
            .await?;
        Ok(Some(
            rows.iter()
                .map(|row| session_summary_from_row_and_selector(row, &store_selector))
                .collect(),
        ))
    }

    #[instrument(
        skip(self, query),
        fields(scope = ?scope, store = %store_selector_label(store_selector.as_ref()))
    )]
    pub async fn search_sessions(
        &self,
        query: &str,
        scope: SessionSearchScope,
        limit: usize,
        offset: usize,
        store_selector: Option<StoreSelector>,
    ) -> Result<Vec<SessionSearchHit>> {
        let store = match &store_selector {
            Some(selector) => self.kernel.store_manager().open(selector).await?,
            None => self.kernel.store_manager().get_default().await?,
        };
        let rows = store
            .search_session_history(query, scope, limit, offset)
            .await?;
        debug!(count = rows.len(), "Searched persisted session history");
        Ok(rows
            .into_iter()
            .map(|row| {
                let title =
                    super::super::helpers::session_title_from_metadata(row.metadata.as_deref());
                let session_id = super::super::helpers::format_uuid_bytes_simple(&row.public_id);
                SessionSearchHit {
                    kind: row.kind,
                    score: row.score,
                    session_id: match &store_selector {
                        Some(selector) => format_session_reference(&session_id, selector),
                        None => session_id,
                    },
                    agent_id: row.agent_id,
                    title,
                    created_at: row.created_at,
                    turn_index: row.turn_index,
                    role: row.role,
                    tool_name: row.tool_name,
                    event_type: row.event_type,
                    summary: excerpt_search_text(&row.match_text, query, 72),
                    snippet: excerpt_search_text(&row.match_text, query, 220),
                }
            })
            .collect())
    }
}

fn store_selector_label(selector: Option<&StoreSelector>) -> String {
    selector
        .map(describe_store_selector)
        .unwrap_or_else(|| "state".to_string())
}

fn excerpt_search_text(text: &str, query: &str, max_chars: usize) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = collapsed.trim();
    if trimmed.chars().count() <= max_chars {
        return trimmed.to_string();
    }

    let normalized_query = query.trim().to_ascii_lowercase();
    if normalized_query.is_empty() {
        let excerpt = trimmed.chars().take(max_chars).collect::<String>();
        return format!("{excerpt}…");
    }

    let lower_trimmed = trimmed.to_ascii_lowercase();
    if let Some(byte_index) = lower_trimmed.find(&normalized_query) {
        let match_start = trimmed[..byte_index].chars().count();
        let match_len = normalized_query.chars().count();
        let context_before = max_chars / 3;
        let context_after = max_chars.saturating_sub(context_before + match_len);
        let start = match_start.saturating_sub(context_before);
        let end = (match_start + match_len + context_after).min(trimmed.chars().count());
        let mut excerpt = slice_chars(trimmed, start, end).trim().to_string();
        if start > 0 {
            excerpt = format!("…{excerpt}");
        }
        if end < trimmed.chars().count() {
            excerpt.push('…');
        }
        return excerpt;
    }

    let excerpt = trimmed.chars().take(max_chars).collect::<String>();
    format!("{excerpt}…")
}

fn slice_chars(value: &str, start: usize, end: usize) -> String {
    value
        .chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::excerpt_search_text;

    #[test]
    fn search_excerpt_centers_query_when_possible() {
        let text = "prefix context before the actual match lands on compiler panic in src/main.rs";
        let excerpt = excerpt_search_text(text, "compiler", 28);
        assert!(excerpt.contains("compiler"));
        assert!(excerpt.starts_with('…'));
        assert!(excerpt.ends_with('…'));
    }

    #[test]
    fn search_excerpt_falls_back_to_prefix_without_query() {
        let text = "alpha beta gamma delta epsilon";
        let excerpt = excerpt_search_text(text, "", 12);
        assert_eq!(excerpt, "alpha beta g…");
    }
}
