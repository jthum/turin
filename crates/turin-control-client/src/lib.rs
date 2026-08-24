mod authorizations;
mod client;
mod harnesses;
mod health;
mod memories;
mod models;
mod schedules;
mod sessions;
mod tasks;
mod worklists;

pub use client::{
    ConnectionKind, ConnectionSpec, ControlClient, ManagedEventStream, ManagedSubscribeOptions,
};
pub use health::ControlHealth;
pub use models::*;
pub use turin_daemon_protocol::{
    HarnessSourceEntry, HarnessSourceFile, HarnessSourceListResult, HarnessSourceOverlay,
    HarnessSourceSaveChange, HarnessSourceSaveResult, HarnessSourceValidationResult,
    ToolAuthorizationRequestDetail, ToolAuthorizationResolution, ToolAuthorizationResolveResult,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn remote_env_requires_set_variable() {
        let spec = ConnectionSpec::RemoteEnv {
            base_url: "http://127.0.0.1:9324".into(),
            auth_token_env: "TURIN_CONTROL_CLIENT_TEST_TOKEN_MISSING".into(),
        };
        let err = ControlClient::connect(&spec)
            .await
            .expect_err("missing env rejected");
        assert!(err.to_string().contains("is not set"));
    }
}
