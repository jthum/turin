use anyhow::{Context, Result};

use crate::persistence::manager::StoreSelector;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionReference {
    pub public_id: String,
    pub store_selector: Option<StoreSelector>,
}

impl SessionReference {
    pub fn bare(public_id: impl Into<String>) -> Self {
        Self {
            public_id: public_id.into(),
            store_selector: None,
        }
    }
}

pub fn format_session_reference(public_id: &str, store_selector: &StoreSelector) -> String {
    match store_selector {
        StoreSelector::Alias(alias) if alias == "state" => public_id.to_string(),
        StoreSelector::Alias(alias) => format!("{public_id}@{alias}"),
        StoreSelector::Path(path) => format!("{public_id}@{path}"),
        StoreSelector::Handle(_) => public_id.to_string(),
    }
}

pub fn parse_session_reference(raw: &str) -> Result<SessionReference> {
    let trimmed = raw.trim();
    anyhow::ensure!(!trimmed.is_empty(), "Session reference must not be empty");

    let (public_id, qualifier) = match trimmed.split_once('@') {
        Some((public_id, qualifier)) => (public_id.trim(), Some(qualifier.trim())),
        None => (trimmed, None),
    };

    uuid::Uuid::parse_str(public_id)
        .with_context(|| format!("Invalid session id '{}'", public_id))?;

    let store_selector = qualifier
        .filter(|value| !value.is_empty())
        .map(parse_store_selector_qualifier)
        .transpose()?;

    // A bare session id intentionally does not carry any store qualifier. Callers that resolve
    // persisted sessions decide what the default context means; today that is the primary
    // `state` store unless a surrounding runtime context already pins a different store.
    Ok(SessionReference {
        public_id: public_id.to_string(),
        store_selector,
    })
}

fn parse_store_selector_qualifier(raw: &str) -> Result<StoreSelector> {
    anyhow::ensure!(
        !raw.eq_ignore_ascii_case("handle"),
        "Session references cannot target db handles"
    );
    if looks_like_path(raw) {
        Ok(StoreSelector::Path(raw.to_string()))
    } else {
        Ok(StoreSelector::Alias(raw.to_string()))
    }
}

fn looks_like_path(raw: &str) -> bool {
    raw.contains('/')
        || raw.contains('\\')
        || raw.starts_with('.')
        || raw.starts_with('~')
        || raw.ends_with(".db")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_alias_formats_as_bare_id() {
        assert_eq!(
            format_session_reference(
                "0123456789abcdef0123456789abcdef",
                &StoreSelector::Alias("state".to_string())
            ),
            "0123456789abcdef0123456789abcdef"
        );
    }

    #[test]
    fn non_default_alias_formats_with_qualifier() {
        assert_eq!(
            format_session_reference(
                "0123456789abcdef0123456789abcdef",
                &StoreSelector::Alias("telegram".to_string())
            ),
            "0123456789abcdef0123456789abcdef@telegram"
        );
    }

    #[test]
    fn parse_path_qualified_reference() {
        let parsed =
            parse_session_reference("018f1f4f1f4f4f4f8f8f8f8f8f8f8f8f@.turin/channels/telegram.db")
                .expect("session ref parses");
        assert_eq!(parsed.public_id, "018f1f4f1f4f4f4f8f8f8f8f8f8f8f8f");
        assert_eq!(
            parsed.store_selector,
            Some(StoreSelector::Path(
                ".turin/channels/telegram.db".to_string()
            ))
        );
    }
}
