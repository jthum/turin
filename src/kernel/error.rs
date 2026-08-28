use std::fmt;

/// Stable, operation-level categories for failures crossing the Kernel API boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelErrorKind {
    Configuration,
    Client,
    Persistence,
    Harness,
    Session,
    Task,
    Agent,
    Runtime,
}

impl fmt::Display for KernelErrorKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Configuration => "configuration",
            Self::Client => "client",
            Self::Persistence => "persistence",
            Self::Harness => "harness",
            Self::Session => "session",
            Self::Task => "task",
            Self::Agent => "agent",
            Self::Runtime => "runtime",
        })
    }
}

/// A classified failure returned by public Kernel and AgentManager operations.
#[derive(Debug, thiserror::Error)]
#[error("{kind} error: {source}")]
pub struct KernelError {
    kind: KernelErrorKind,
    #[source]
    source: anyhow::Error,
}

impl KernelError {
    pub fn kind(&self) -> KernelErrorKind {
        self.kind
    }

    pub(crate) fn new(kind: KernelErrorKind, source: impl Into<anyhow::Error>) -> Self {
        Self {
            kind,
            source: source.into(),
        }
    }
}

pub type KernelResult<T> = Result<T, KernelError>;

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    #[test]
    fn classified_error_retains_kind_and_source() {
        let error = KernelError::new(
            KernelErrorKind::Persistence,
            anyhow::anyhow!("database unavailable"),
        );

        assert_eq!(error.kind(), KernelErrorKind::Persistence);
        assert_eq!(error.to_string(), "persistence error: database unavailable");
        assert_eq!(
            error.source().map(ToString::to_string).as_deref(),
            Some("database unavailable")
        );
    }
}
