use turin_daemon_protocol::UiBadgeIntent;

pub fn ui_badge_text(badge: Option<&UiBadgeIntent>, fallback: Option<&str>) -> Option<String> {
    let label = badge
        .and_then(|badge| badge.label.as_deref())
        .or(fallback)
        .filter(|label| !label.is_empty());
    let count = badge.and_then(|badge| badge.count);
    match (label, count) {
        (Some(label), Some(count)) => Some(format!("{label} {count}")),
        (Some(label), None) => Some(label.to_string()),
        (None, Some(count)) => Some(count.to_string()),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use turin_daemon_protocol::{UiBadgeIntent, UiNoticeLevel};

    use crate::ui_badge_text;

    fn badge(label: Option<&str>, count: Option<u64>) -> UiBadgeIntent {
        UiBadgeIntent {
            app_id: "release".to_string(),
            target: "approvals".to_string(),
            count,
            label: label.map(str::to_string),
            level: Some(UiNoticeLevel::Info),
            data: Default::default(),
        }
    }

    #[test]
    fn badge_text_combines_label_and_count() {
        let badge = badge(Some("approvals"), Some(3));

        assert_eq!(
            ui_badge_text(Some(&badge), Some("fallback")).as_deref(),
            Some("approvals 3")
        );
    }

    #[test]
    fn badge_text_uses_fallback_label_without_dynamic_label() {
        let badge = badge(None, Some(3));

        assert_eq!(
            ui_badge_text(Some(&badge), Some("dashboard")).as_deref(),
            Some("dashboard 3")
        );
    }

    #[test]
    fn badge_text_handles_label_only_count_only_and_empty() {
        let label_only = badge(Some("live"), None);
        let count_only = badge(None, Some(7));

        assert_eq!(
            ui_badge_text(Some(&label_only), None).as_deref(),
            Some("live")
        );
        assert_eq!(ui_badge_text(Some(&count_only), None).as_deref(), Some("7"));
        assert_eq!(
            ui_badge_text(None, Some("fallback")).as_deref(),
            Some("fallback")
        );
        assert_eq!(ui_badge_text(None, Some("")).as_deref(), None);
        assert_eq!(ui_badge_text(None, None), None);
    }
}
