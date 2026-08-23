pub fn validate_signal_topic_pattern(topic: &str) -> Result<(), String> {
    if topic.trim().is_empty() {
        return Err("signal topic must not be empty".to_string());
    }
    if !topic.contains('*') {
        return Ok(());
    }
    if topic == "*" {
        return Ok(());
    }
    if let Some(prefix) = topic.strip_suffix(".*")
        && !prefix.is_empty()
        && !prefix.contains('*')
    {
        return Ok(());
    }
    Err(
        "signal topic wildcards must be '*' or terminal prefix patterns like 'deploy.*'"
            .to_string(),
    )
}

pub fn signal_topic_subscription_candidates(topic: &str) -> Vec<String> {
    let mut out = vec![topic.to_string()];
    let mut cursor = topic;
    while let Some(index) = cursor.rfind('.') {
        let prefix = &cursor[..index];
        if prefix.is_empty() {
            break;
        }
        out.push(format!("{prefix}.*"));
        cursor = prefix;
    }
    if topic != "*" {
        out.push("*".to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subscription_candidates_include_exact_specific_wildcards_and_global() {
        assert_eq!(
            signal_topic_subscription_candidates("deploy.api.complete"),
            vec!["deploy.api.complete", "deploy.api.*", "deploy.*", "*"]
        );
    }

    #[test]
    fn validate_signal_topic_pattern_accepts_exact_and_terminal_wildcards() {
        assert!(validate_signal_topic_pattern("deploy.complete").is_ok());
        assert!(validate_signal_topic_pattern("deploy.*").is_ok());
        assert!(validate_signal_topic_pattern("*").is_ok());
    }

    #[test]
    fn validate_signal_topic_pattern_rejects_non_terminal_wildcards() {
        assert!(validate_signal_topic_pattern("").is_err());
        assert!(validate_signal_topic_pattern("deploy*").is_err());
        assert!(validate_signal_topic_pattern("*.complete").is_err());
        assert!(validate_signal_topic_pattern("deploy.*.complete").is_err());
    }
}
