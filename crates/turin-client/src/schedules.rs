use anyhow::Result;
use turin_daemon_protocol::{
    DaemonRequest, EntityIdParams, NoParams, ScheduleCreateParams, ScheduleJobDetail,
    ScheduleJobList, ScheduleJobRunList, ScheduleRunsParams, ScheduleUpdateParams,
};

use crate::client::Client;

impl Client {
    pub async fn create_schedule(&self, params: ScheduleCreateParams) -> Result<ScheduleJobDetail> {
        self.request_ok(None, DaemonRequest::ScheduleCreate(params))
            .await
    }

    pub async fn get_schedule(&self, id: impl Into<String>) -> Result<ScheduleJobDetail> {
        self.request_ok(None, DaemonRequest::ScheduleGet(entity_id(id)))
            .await
    }

    pub async fn update_schedule(&self, params: ScheduleUpdateParams) -> Result<ScheduleJobDetail> {
        self.request_ok(None, DaemonRequest::ScheduleUpdate(params))
            .await
    }

    pub async fn list_schedules(&self) -> Result<Vec<ScheduleJobDetail>> {
        let response: ScheduleJobList = self
            .request_ok(None, DaemonRequest::ScheduleList(NoParams::default()))
            .await?;
        Ok(response.jobs)
    }

    pub async fn list_schedule_runs(
        &self,
        id: impl Into<String>,
        active_only: bool,
        limit: Option<u32>,
    ) -> Result<ScheduleJobRunList> {
        self.request_ok(
            None,
            DaemonRequest::ScheduleRuns(ScheduleRunsParams {
                id: id.into(),
                active_only,
                limit,
            }),
        )
        .await
    }

    pub async fn enable_schedule(&self, id: impl Into<String>) -> Result<ScheduleJobDetail> {
        self.request_ok(None, DaemonRequest::ScheduleEnable(entity_id(id)))
            .await
    }

    pub async fn disable_schedule(&self, id: impl Into<String>) -> Result<ScheduleJobDetail> {
        self.request_ok(None, DaemonRequest::ScheduleDisable(entity_id(id)))
            .await
    }

    pub async fn delete_schedule(&self, id: impl Into<String>) -> Result<ScheduleJobDetail> {
        self.request_ok(None, DaemonRequest::ScheduleDelete(entity_id(id)))
            .await
    }
}

fn entity_id(id: impl Into<String>) -> EntityIdParams {
    EntityIdParams { id: id.into() }
}
