use std::ffi::OsStr;

const TRIM_ON_PEER_IDLE_ENV: &str = "TURIN_TRIM_ALLOCATOR_ON_PEER_IDLE";

pub(super) fn trim_after_peer_idle_if_enabled() {
    if !trim_on_peer_idle_enabled() {
        return;
    }

    if trim_allocator() {
        tracing::debug!("Allocator trim requested after peer runtime idle shutdown");
    } else {
        tracing::debug!(
            "Allocator trim requested after peer runtime idle shutdown but no pages were released"
        );
    }
}

fn trim_on_peer_idle_enabled() -> bool {
    trim_on_peer_idle_enabled_from(std::env::var_os(TRIM_ON_PEER_IDLE_ENV).as_deref())
}

fn trim_on_peer_idle_enabled_from(value: Option<&OsStr>) -> bool {
    let Some(value) = value.and_then(OsStr::to_str) else {
        return false;
    };
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "no" | "off"
    )
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn trim_allocator() -> bool {
    // SAFETY: `malloc_trim` is a process-wide glibc allocator hint with no pointer
    // arguments. Calling it is safe; whether pages can be released is its return value.
    unsafe { libc::malloc_trim(0) != 0 }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn trim_allocator() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_env_parser_accepts_truthy_values() {
        assert!(trim_on_peer_idle_enabled_from(Some(OsStr::new("1"))));
        assert!(trim_on_peer_idle_enabled_from(Some(OsStr::new("true"))));
        assert!(trim_on_peer_idle_enabled_from(Some(OsStr::new("yes"))));
        assert!(trim_on_peer_idle_enabled_from(Some(OsStr::new("anything"))));
    }

    #[test]
    fn trim_env_parser_rejects_absent_and_falsey_values() {
        assert!(!trim_on_peer_idle_enabled_from(None));
        assert!(!trim_on_peer_idle_enabled_from(Some(OsStr::new(""))));
        assert!(!trim_on_peer_idle_enabled_from(Some(OsStr::new("0"))));
        assert!(!trim_on_peer_idle_enabled_from(Some(OsStr::new("false"))));
        assert!(!trim_on_peer_idle_enabled_from(Some(OsStr::new("OFF"))));
    }
}
