use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CapabilityRuleMatch {
    pub rule: Option<String>,
    pub via_wildcard: bool,
    pub allowed: Option<bool>,
}

pub(super) fn match_capability_rule(
    caps: &BTreeMap<String, serde_json::Value>,
    capability: &str,
) -> CapabilityRuleMatch {
    match_capability_rule_bool_iter(
        caps.iter()
            .filter_map(|(rule, value)| value.as_bool().map(|b| (rule.as_str(), b))),
        capability,
    )
}

fn match_capability_rule_json_map(
    caps: &HashMap<String, serde_json::Value>,
    capability: &str,
) -> CapabilityRuleMatch {
    match_capability_rule_bool_iter(
        caps.iter()
            .filter_map(|(rule, value)| value.as_bool().map(|b| (rule.as_str(), b))),
        capability,
    )
}

fn match_capability_rule_bool_map(
    caps: &BTreeMap<String, bool>,
    capability: &str,
) -> CapabilityRuleMatch {
    match_capability_rule_bool_iter(
        caps.iter().map(|(rule, value)| (rule.as_str(), *value)),
        capability,
    )
}

fn match_capability_rule_bool_iter<'a, I>(iter: I, capability: &str) -> CapabilityRuleMatch
where
    I: IntoIterator<Item = (&'a str, bool)>,
{
    let mut best: Option<(&str, bool)> = None;
    for (rule, allowed) in iter {
        if rule == capability {
            return CapabilityRuleMatch {
                rule: Some(capability.to_string()),
                via_wildcard: false,
                allowed: Some(allowed),
            };
        }
        let Some(prefix) = rule.strip_suffix(".*") else {
            continue;
        };
        let Some(suffix) = capability.strip_prefix(prefix) else {
            continue;
        };
        if suffix.is_empty() || suffix.starts_with('.') {
            match best {
                Some((best_rule, _)) if best_rule.len() >= rule.len() => {}
                _ => best = Some((rule, allowed)),
            }
        }
    }

    match best {
        Some((rule, allowed)) => CapabilityRuleMatch {
            rule: Some(rule.to_string()),
            via_wildcard: true,
            allowed: Some(allowed),
        },
        None => CapabilityRuleMatch {
            rule: None,
            via_wildcard: false,
            allowed: None,
        },
    }
}

pub(crate) fn capability_allowed_by_bool_rules(
    caps: &BTreeMap<String, bool>,
    capability: &str,
) -> bool {
    match_capability_rule_bool_map(caps, capability)
        .allowed
        .unwrap_or(false)
}

pub(super) fn capability_ceiling_denial_reason_json_map(
    caps: &HashMap<String, serde_json::Value>,
    capability: &str,
    source_kind: &str,
    source_name: &str,
    default_deny_on_no_match: bool,
) -> Option<String> {
    if caps.is_empty() {
        return None;
    }
    let matched = match_capability_rule_json_map(caps, capability);
    let allowed = match matched.allowed {
        Some(v) => v,
        None => !default_deny_on_no_match,
    };
    if allowed {
        None
    } else {
        Some(match matched.rule {
            Some(rule) => format!(
                "Governance denial: capability '{}' denied by {} '{}' (rule '{}')",
                capability, source_kind, source_name, rule
            ),
            None => format!(
                "Governance denial: capability '{}' denied by {} '{}' (no matching allow rule)",
                capability, source_kind, source_name
            ),
        })
    }
}

pub(super) fn capability_ceiling_denial_reason_bool_map(
    caps: &BTreeMap<String, bool>,
    capability: &str,
    source_kind: &str,
    source_name: &str,
    default_deny_on_no_match: bool,
) -> Option<String> {
    if caps.is_empty() {
        return if default_deny_on_no_match {
            Some(format!(
                "Governance denial: capability '{}' denied by {} '{}' (empty allowlist)",
                capability, source_kind, source_name
            ))
        } else {
            None
        };
    }
    let matched = match_capability_rule_bool_map(caps, capability);
    let allowed = match matched.allowed {
        Some(v) => v,
        None => !default_deny_on_no_match,
    };
    if allowed {
        None
    } else {
        Some(match matched.rule {
            Some(rule) => format!(
                "Governance denial: capability '{}' denied by {} '{}' (rule '{}')",
                capability, source_kind, source_name, rule
            ),
            None => format!(
                "Governance denial: capability '{}' denied by {} '{}' (no matching allow rule)",
                capability, source_kind, source_name
            ),
        })
    }
}

pub(crate) fn tool_capability_name(tool_name: &str) -> Option<&'static str> {
    match tool_name {
        "read_file" => Some("fs.read"),
        "write_file" | "edit_file" | "apply_patch" => Some("fs.write"),
        "shell_exec" => Some("shell.exec"),
        "bridge_mcp" => Some("integration.mcp.bridge"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_rule_overrides_wildcard_rule() {
        let caps = BTreeMap::from([
            ("runtime.db.*".to_string(), false),
            ("runtime.db.query".to_string(), true),
        ]);

        let matched = match_capability_rule_bool_map(&caps, "runtime.db.query");
        assert_eq!(matched.rule.as_deref(), Some("runtime.db.query"));
        assert!(!matched.via_wildcard);
        assert_eq!(matched.allowed, Some(true));
        assert!(capability_allowed_by_bool_rules(&caps, "runtime.db.query"));
    }

    #[test]
    fn longest_wildcard_rule_wins() {
        let caps = BTreeMap::from([
            ("runtime.*".to_string(), true),
            ("runtime.db.*".to_string(), false),
        ]);

        let matched = match_capability_rule_bool_map(&caps, "runtime.db.exec");
        assert_eq!(matched.rule.as_deref(), Some("runtime.db.*"));
        assert!(matched.via_wildcard);
        assert_eq!(matched.allowed, Some(false));
        assert!(!capability_allowed_by_bool_rules(&caps, "runtime.db.exec"));
    }

    #[test]
    fn wildcard_rule_matches_prefix_capability_itself() {
        let caps = BTreeMap::from([("runtime.db.*".to_string(), true)]);

        let matched = match_capability_rule_bool_map(&caps, "runtime.db");
        assert_eq!(matched.rule.as_deref(), Some("runtime.db.*"));
        assert!(matched.via_wildcard);
        assert_eq!(matched.allowed, Some(true));
    }
}
