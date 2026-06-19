use turin_daemon_protocol::HarnessActionRunResult;

use crate::{controller::HarnessActionFailure, intents::UiAppRecord};

pub fn ui_harness_action_result_matches_app(
    result: &HarnessActionRunResult,
    app: &UiAppRecord,
) -> bool {
    if let Some(harness_id) = result.harness_id.as_deref()
        && app.source.harness_id.as_deref() != Some(harness_id)
    {
        return false;
    }
    if let Some(agent_id) = app.source.agent_id.as_deref()
        && result.agent_id != agent_id
    {
        return false;
    }
    true
}

pub fn ui_harness_action_failure_matches_app(
    failure: &HarnessActionFailure,
    app: &UiAppRecord,
) -> bool {
    if let Some(harness_id) = failure.harness_id.as_deref()
        && app.source.harness_id.as_deref() != Some(harness_id)
    {
        return false;
    }
    if let Some(agent_id) = app.source.agent_id.as_deref()
        && failure
            .agent_id
            .as_deref()
            .is_some_and(|value| value != agent_id)
    {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;
    use turin_daemon_protocol::{HarnessActionRunResult, UiIntentSource};

    use crate::{
        HarnessActionFailure, UiAppRecord, ui_harness_action_failure_matches_app,
        ui_harness_action_result_matches_app,
    };

    fn test_app() -> UiAppRecord {
        UiAppRecord {
            id: "release".to_string(),
            source: UiIntentSource {
                harness_id: Some("release-harness".to_string()),
                app_id: Some("release".to_string()),
                agent_id: Some("release-agent".to_string()),
                package_id: None,
            },
            definition: None,
            screens: BTreeMap::new(),
            panes: BTreeMap::new(),
            menus: Vec::new(),
            opens_with: None,
            badges: BTreeMap::new(),
        }
    }

    #[test]
    fn action_result_matches_selected_harness_and_agent() {
        let app = test_app();
        let matching = HarnessActionRunResult {
            action: "release.seed".to_string(),
            agent_id: "release-agent".to_string(),
            harness_id: Some("release-harness".to_string()),
            result: json!({ "status": "ok" }),
            ui_intents: Vec::new(),
        };
        let other_harness = HarnessActionRunResult {
            harness_id: Some("qa-harness".to_string()),
            ..matching.clone()
        };
        let other_agent = HarnessActionRunResult {
            agent_id: "qa-agent".to_string(),
            ..matching.clone()
        };

        assert!(ui_harness_action_result_matches_app(&matching, &app));
        assert!(!ui_harness_action_result_matches_app(&other_harness, &app));
        assert!(!ui_harness_action_result_matches_app(&other_agent, &app));
    }

    #[test]
    fn action_result_without_harness_matches_selected_agent() {
        let app = test_app();
        let result = HarnessActionRunResult {
            action: "release.seed".to_string(),
            agent_id: "release-agent".to_string(),
            harness_id: None,
            result: json!({ "status": "ok" }),
            ui_intents: Vec::new(),
        };

        assert!(ui_harness_action_result_matches_app(&result, &app));
    }

    #[test]
    fn action_failure_matches_selected_harness_and_known_agent() {
        let app = test_app();
        let matching = HarnessActionFailure {
            action: "release.fail_diagnostic".to_string(),
            agent_id: Some("release-agent".to_string()),
            harness_id: Some("release-harness".to_string()),
            message: "Release Operator diagnostic failure".to_string(),
        };
        let other_harness = HarnessActionFailure {
            harness_id: Some("qa-harness".to_string()),
            ..matching.clone()
        };
        let other_agent = HarnessActionFailure {
            agent_id: Some("qa-agent".to_string()),
            ..matching.clone()
        };

        assert!(ui_harness_action_failure_matches_app(&matching, &app));
        assert!(!ui_harness_action_failure_matches_app(&other_harness, &app));
        assert!(!ui_harness_action_failure_matches_app(&other_agent, &app));
    }

    #[test]
    fn action_failure_without_agent_matches_selected_harness() {
        let app = test_app();
        let failure = HarnessActionFailure {
            action: "release.fail_diagnostic".to_string(),
            agent_id: None,
            harness_id: Some("release-harness".to_string()),
            message: "Release Operator diagnostic failure".to_string(),
        };

        assert!(ui_harness_action_failure_matches_app(&failure, &app));
    }
}
