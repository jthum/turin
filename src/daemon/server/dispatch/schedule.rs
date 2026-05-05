use crate::daemon::protocol::{
    EntityIdParams, ErrorCode, NoParams, ResponseEnvelope, ScheduleCreateParams, ScheduleJobList,
    ScheduleUpdateParams,
};

use super::{
    DispatchContext, not_found_error, serialize_response, serialize_response_with_event,
    validation_error,
};
use crate::daemon::state::{
    CreateScheduledJobInput, ScheduledJobOverlapPolicy, UpdateScheduledJobInput,
};

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

pub(super) async fn update(
    id: Option<String>,
    params: ScheduleUpdateParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let overlap_policy = match params
        .overlap_policy
        .as_deref()
        .map(str::parse::<ScheduledJobOverlapPolicy>)
        .transpose()
    {
        Ok(policy) => policy,
        Err(err) => return validation_error(id, err),
    };

    let guard = ctx.state.read().await;
    match guard
        .update_scheduled_job(
            &params.id,
            UpdateScheduledJobInput {
                agent_id: params.agent_id,
                prompt: params.prompt,
                persistence: params.persistence,
                next_run_unix_ms: params.next_run_unix_ms,
                interval_seconds: params.interval_seconds,
                overlap_policy,
                enabled: params.enabled,
            },
        )
        .await
    {
        Ok(Some(job)) => serialize_response_with_event(
            id,
            job,
            "updated scheduled job",
            &ctx.event_tx,
            "schedule.updated",
        ),
        Ok(None) => not_found_error(
            id,
            ErrorCode::ScheduleNotFound,
            format!("Scheduled job '{}' not found", params.id),
        ),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn get(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.scheduled_job_detail(&params.id).await {
        Ok(Some(job)) => serialize_response(id, job, "scheduled job detail"),
        Ok(None) => not_found_error(
            id,
            ErrorCode::ScheduleNotFound,
            format!("Scheduled job '{}' not found", params.id),
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

pub(super) async fn enable(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    set_enabled(id, params.id, true, ctx).await
}

pub(super) async fn disable(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    set_enabled(id, params.id, false, ctx).await
}

async fn set_enabled(
    id: Option<String>,
    public_id: String,
    enabled: bool,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.set_scheduled_job_enabled(&public_id, enabled).await {
        Ok(Some(job)) => serialize_response_with_event(
            id,
            job,
            "scheduled job toggle result",
            &ctx.event_tx,
            if enabled {
                "schedule.enabled"
            } else {
                "schedule.disabled"
            },
        ),
        Ok(None) => not_found_error(
            id,
            ErrorCode::ScheduleNotFound,
            format!("Scheduled job '{}' not found", public_id),
        ),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn delete(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.delete_scheduled_job(&params.id).await {
        Ok(Some(job)) => serialize_response_with_event(
            id,
            job,
            "deleted scheduled job",
            &ctx.event_tx,
            "schedule.deleted",
        ),
        Ok(None) => not_found_error(
            id,
            ErrorCode::ScheduleNotFound,
            format!("Scheduled job '{}' not found", params.id),
        ),
        Err(err) => validation_error(id, err),
    }
}
