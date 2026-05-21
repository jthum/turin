use anyhow::Result;
use turin_daemon_protocol::{
    ContextPersistenceParams, DaemonRequest, WorkItemDetail, WorkItemList, WorkItemTargetParams,
    WorklistDetail, WorklistItemsParams, WorklistList, WorklistListParams, WorklistTargetParams,
};

use crate::client::ControlClient;

impl ControlClient {
    pub async fn list_worklists(&self, params: WorklistListParams) -> Result<Vec<WorklistDetail>> {
        let response: WorklistList = self
            .request_ok(None, DaemonRequest::WorklistList(params))
            .await?;
        Ok(response.worklists)
    }

    pub async fn get_worklist(
        &self,
        id: impl Into<String>,
        persistence: Option<ContextPersistenceParams>,
    ) -> Result<WorklistDetail> {
        self.request_ok(
            None,
            DaemonRequest::WorklistGet(WorklistTargetParams {
                id: id.into(),
                persistence,
            }),
        )
        .await
    }

    pub async fn list_worklist_items(&self, params: WorklistItemsParams) -> Result<WorkItemList> {
        self.request_ok(None, DaemonRequest::WorklistItems(params))
            .await
    }

    pub async fn get_workitem(
        &self,
        id: impl Into<String>,
        persistence: Option<ContextPersistenceParams>,
    ) -> Result<WorkItemDetail> {
        self.request_ok(
            None,
            DaemonRequest::WorkItemGet(WorkItemTargetParams {
                id: id.into(),
                persistence,
            }),
        )
        .await
    }
}
