use crate::daemon::protocol::{
    EntityIdParams, ErrorCode, NoParams, ResponseEnvelope, ScheduleCreateParams, ScheduleJobList,
    ScheduleRunsParams, ScheduleUpdateParams,
};

use super::{
    DispatchContext, optional_response, optional_response_with_event,
    serialize_response_with_event, validation_error,
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
            content: params.content,
            tools: params.tools,
            conflict_policy: params.conflict_policy,
            action: params.action,
            persistence: params.persistence,
            next_run_unix_ms: params.next_run_unix_ms,
            interval_seconds: params.interval_seconds,
            recurring_pattern: params.recurring_pattern,
            overlap_policy,
            work_key: params.work_key,
            max_concurrency: params.max_concurrency,
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
    let result = guard
        .update_scheduled_job(
            &params.id,
            UpdateScheduledJobInput {
                agent_id: params.agent_id,
                prompt: params.prompt,
                content: params.content,
                tools: params.tools,
                conflict_policy: params.conflict_policy,
                action: params.action,
                persistence: params.persistence,
                next_run_unix_ms: params.next_run_unix_ms,
                interval_seconds: params.interval_seconds,
                recurring_pattern: params.recurring_pattern,
                overlap_policy,
                work_key: params.work_key,
                max_concurrency: params.max_concurrency,
                enabled: params.enabled,
            },
        )
        .await;
    optional_response_with_event(
        id,
        result,
        "updated scheduled job",
        &ctx.event_tx,
        "schedule.updated",
        ErrorCode::ScheduleNotFound,
        || format!("Scheduled job '{}' not found", params.id),
    )
}

pub(super) async fn get(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    let result = guard.scheduled_job_detail(&params.id).await;
    optional_response(
        id,
        result,
        "scheduled job detail",
        ErrorCode::ScheduleNotFound,
        || format!("Scheduled job '{}' not found", params.id),
    )
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

pub(super) async fn runs(
    id: Option<String>,
    params: ScheduleRunsParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    let result = guard
        .scheduled_job_runs(&params.id, params.active_only, params.limit)
        .await;
    optional_response(
        id,
        result,
        "scheduled job runs",
        ErrorCode::ScheduleNotFound,
        || format!("Scheduled job '{}' not found", params.id),
    )
}

async fn set_enabled(
    id: Option<String>,
    public_id: String,
    enabled: bool,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    let result = guard.set_scheduled_job_enabled(&public_id, enabled).await;
    optional_response_with_event(
        id,
        result,
        "scheduled job toggle result",
        &ctx.event_tx,
        if enabled {
            "schedule.enabled"
        } else {
            "schedule.disabled"
        },
        ErrorCode::ScheduleNotFound,
        || format!("Scheduled job '{}' not found", public_id),
    )
}

pub(super) async fn delete(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    let result = guard.delete_scheduled_job(&params.id).await;
    optional_response_with_event(
        id,
        result,
        "deleted scheduled job",
        &ctx.event_tx,
        "schedule.deleted",
        ErrorCode::ScheduleNotFound,
        || format!("Scheduled job '{}' not found", params.id),
    )
}
