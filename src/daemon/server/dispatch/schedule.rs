use crate::daemon::protocol::{NoParams, ResponseEnvelope, ScheduleCreateParams, ScheduleJobList};

use super::{DispatchContext, serialize_response_with_event, validation_error};
use crate::daemon::state::{CreateScheduledJobInput, ScheduledJobOverlapPolicy};

pub(super) async fn create(
    id: Option<String>,
    params: ScheduleCreateParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let overlap_policy = match params
        .overlap_policy
        .as_deref()
        .unwrap_or("skip")
        .parse::<ScheduledJobOverlapPolicy>()
    {
        Ok(policy) => policy,
        Err(err) => return validation_error(id, err),
    };

    let guard = ctx.state.read().await;
    match guard
        .create_scheduled_job(CreateScheduledJobInput {
            agent_id: params.agent_id,
            prompt: params.prompt,
            persistence: params.persistence,
            next_run_unix_ms: params.next_run_unix_ms,
            interval_seconds: params.interval_seconds,
            overlap_policy,
            enabled: params.enabled,
        })
        .await
    {
        Ok(job) => serialize_response_with_event(
            id,
            job,
            "created scheduled job",
            &ctx.event_tx,
            "schedule.created",
        ),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn list(
    id: Option<String>,
    _params: NoParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.list_scheduled_jobs().await {
        Ok(jobs) => match serde_json::to_value(ScheduleJobList { jobs }) {
            Ok(value) => ResponseEnvelope::ok(id, value),
            Err(err) => validation_error(id, err),
        },
        Err(err) => validation_error(id, err),
    }
}
