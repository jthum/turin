use anyhow::Result;
use turin_daemon_protocol::{DaemonRequest, EntityIdParams, NoParams, UiIntentMessage};

use crate::client::ControlClient;
use crate::models::{HarnessDetail, HarnessRuntime, HarnessRuntimeList};

impl ControlClient {
    pub async fn list_harnesses(&self) -> Result<Vec<HarnessRuntime>> {
        let response: HarnessRuntimeList = self
            .request_ok(None, DaemonRequest::HarnessList(NoParams::default()))
            .await?;
        Ok(response.harnesses)
    }

    pub async fn get_harness(&self, harness_id: impl Into<String>) -> Result<HarnessDetail> {
        self.request_ok(
            None,
            DaemonRequest::HarnessGet(EntityIdParams {
                id: harness_id.into(),
            }),
        )
        .await
    }

    pub async fn list_harness_ui_intents(
        &self,
        harness_id: impl Into<String>,
    ) -> Result<Vec<UiIntentMessage>> {
        Ok(self.get_harness(harness_id).await?.ui_intents)
    }
}
