use turin_daemon_protocol::UiScreenIntent;

use crate::UiAppRecord;

pub fn ui_default_screen_index(app: &UiAppRecord) -> usize {
    app.opens_with
        .as_deref()
        .and_then(|target| ui_screen_index_for_target(app, target))
        .unwrap_or_default()
}

pub fn ui_screen_index_for_target(app: &UiAppRecord, target: &str) -> Option<usize> {
    app.screens
        .values()
        .position(|screen| screen_matches_target(screen, target))
}

fn screen_matches_target(screen: &UiScreenIntent, target: &str) -> bool {
    screen.id == target || screen.title == target
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use turin_daemon_protocol::{UiIntentSource, UiScreenIntent};

    use crate::{UiAppRecord, ui_default_screen_index, ui_screen_index_for_target};

    fn test_app(opens_with: Option<&str>) -> UiAppRecord {
        UiAppRecord {
            id: "release".to_string(),
            source: UiIntentSource::default(),
            definition: None,
            screens: BTreeMap::from([
                ("approvals".to_string(), screen("approvals", "Approvals")),
                ("home".to_string(), screen("home", "Release Desk")),
                ("intake".to_string(), screen("intake", "Intake")),
            ]),
            panes: BTreeMap::new(),
            menus: Vec::new(),
            opens_with: opens_with.map(str::to_string),
            badges: BTreeMap::new(),
        }
    }

    fn screen(id: &str, title: &str) -> UiScreenIntent {
        UiScreenIntent {
            app_id: "release".to_string(),
            id: id.to_string(),
            title: title.to_string(),
            presentation: None,
            nodes: Vec::new(),
        }
    }

    #[test]
    fn default_screen_uses_opens_with_target_id() {
        let app = test_app(Some("intake"));

        assert_eq!(ui_default_screen_index(&app), 2);
    }

    #[test]
    fn default_screen_uses_opens_with_target_title() {
        let app = test_app(Some("Release Desk"));

        assert_eq!(ui_default_screen_index(&app), 1);
    }

    #[test]
    fn default_screen_falls_back_to_first_screen_for_missing_target() {
        let app = test_app(Some("missing"));

        assert_eq!(ui_default_screen_index(&app), 0);
    }

    #[test]
    fn screen_index_matches_id_or_title() {
        let app = test_app(None);

        assert_eq!(ui_screen_index_for_target(&app, "home"), Some(1));
        assert_eq!(ui_screen_index_for_target(&app, "Approvals"), Some(0));
        assert_eq!(ui_screen_index_for_target(&app, "missing"), None);
    }
}
