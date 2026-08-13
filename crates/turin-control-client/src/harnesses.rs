use anyhow::Result;
use turin_daemon_protocol::{
    DaemonRequest, EntityIdParams, HarnessActionRunParams, HarnessActionRunResult,
    HarnessSourceFile, HarnessSourceGetParams, HarnessSourceListResult, HarnessSourceOverlay,
    HarnessSourceSaveChange, HarnessSourceSaveParams, HarnessSourceSaveResult,
    HarnessSourceValidateParams, HarnessSourceValidationResult, NoParams, UiIntentMessage,
};

use crate::client::ControlClient;
use crate::models::{
    HarnessDetail, HarnessRuntime, HarnessRuntimeList, HarnessValidation, Issue, IssueList,
};

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

    pub async fn create_harness(&self, harness_id: impl Into<String>) -> Result<HarnessDetail> {
        self.request_ok(
            None,
            DaemonRequest::HarnessCreate(EntityIdParams {
                id: harness_id.into(),
            }),
        )
        .await
    }

    pub async fn list_harness_issues(&self, harness_id: impl Into<String>) -> Result<Vec<Issue>> {
        let response: IssueList = self
            .request_ok(
                None,
                DaemonRequest::HarnessIssues(EntityIdParams {
                    id: harness_id.into(),
                }),
            )
            .await?;
        Ok(response.issues)
    }

    pub async fn reload_harness(&self, harness_id: impl Into<String>) -> Result<HarnessDetail> {
        self.request_ok(
            None,
            DaemonRequest::HarnessReload(EntityIdParams {
                id: harness_id.into(),
            }),
        )
        .await
    }

    pub async fn validate_harness(
        &self,
        harness_id: impl Into<String>,
    ) -> Result<HarnessValidation> {
        self.request_ok(
            None,
            DaemonRequest::HarnessValidate(EntityIdParams {
                id: harness_id.into(),
            }),
        )
        .await
    }

    pub async fn list_harness_sources(
        &self,
        harness_id: impl Into<String>,
    ) -> Result<HarnessSourceListResult> {
        self.request_ok(
            None,
            DaemonRequest::HarnessSourceList(EntityIdParams {
                id: harness_id.into(),
            }),
        )
        .await
    }

    pub async fn get_harness_source(
        &self,
        harness_id: impl Into<String>,
        path: impl Into<String>,
    ) -> Result<HarnessSourceFile> {
        self.request_ok(
            None,
            DaemonRequest::HarnessSourceGet(HarnessSourceGetParams {
                id: harness_id.into(),
                path: path.into(),
            }),
        )
        .await
    }

    pub async fn validate_harness_sources(
        &self,
        harness_id: impl Into<String>,
        changes: Vec<HarnessSourceOverlay>,
    ) -> Result<HarnessSourceValidationResult> {
        self.request_ok(
            None,
            DaemonRequest::HarnessSourceValidate(HarnessSourceValidateParams {
                id: harness_id.into(),
                changes,
            }),
        )
        .await
    }

    pub async fn save_harness_sources(
        &self,
        harness_id: impl Into<String>,
        changes: Vec<HarnessSourceSaveChange>,
    ) -> Result<HarnessSourceSaveResult> {
        self.request_ok(
            None,
            DaemonRequest::HarnessSourceSave(HarnessSourceSaveParams {
                id: harness_id.into(),
                changes,
            }),
        )
        .await
    }

    pub async fn delete_harness(&self, harness_id: impl Into<String>) -> Result<()> {
        let _: serde_json::Value = self
            .request_ok(
                None,
                DaemonRequest::HarnessDelete(EntityIdParams {
                    id: harness_id.into(),
                }),
            )
            .await?;
        Ok(())
    }

    pub async fn list_harness_ui_intents(
        &self,
        harness_id: impl Into<String>,
    ) -> Result<Vec<UiIntentMessage>> {
        Ok(self.get_harness(harness_id).await?.ui_intents)
    }

    pub async fn run_harness_action(
        &self,
        params: HarnessActionRunParams,
    ) -> Result<HarnessActionRunResult> {
        self.request_ok(None, DaemonRequest::HarnessActionRun(params))
            .await
    }
}
