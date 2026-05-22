use mlua::{Lua, Result as LuaResult, Value};

use crate::harness::globals::{ActiveHarnessExecutionContext, HarnessAppData};
use crate::harness::stdlib::binding_common::{nil_err, string_ok};
use crate::harness::stdlib::policy_support::policy_u64;
use crate::kernel::session::QueuedTask;

fn ensure_local_task_id(task: &mut QueuedTask) -> String {
    if task.task_id.is_empty() {
        task.task_id = format!("t_{}", uuid::Uuid::now_v7().simple());
    }
    task.task_id.clone()
}

pub(crate) fn active_trace_id(app_data: &HarnessAppData) -> Option<String> {
    app_data
        .execution_ctx
        .lock()
        .ok()
        .and_then(|ctx| ctx.trace_id.clone())
}

pub(crate) fn queue_max(snapshot: &std::collections::HashMap<String, serde_json::Value>) -> usize {
    policy_u64(snapshot, "queue.max_depth", 1024) as usize
}

pub(super) fn lua_string_result(
    lua: &Lua,
    result: Result<String, String>,
) -> LuaResult<(Value, Value)> {
    match result {
        Ok(value) => string_ok(lua, &value),
        Err(err) => nil_err(lua, &err),
    }
}

pub(crate) async fn queue_push_one(
    execution_ctx: &ActiveHarnessExecutionContext,
    mut task: QueuedTask,
    queue_max: usize,
    push_front: bool,
) -> Result<String, String> {
    let queue = execution_ctx
        .lock()
        .ok()
        .and_then(|lock| lock.queue.clone())
        .ok_or_else(|| "No active session queue".to_string())?;
    let mut q = queue.lock().await;
    if q.len() >= queue_max {
        return Err(format!(
            "Policy denial: queue.max_depth={} reached",
            queue_max
        ));
    }
    let task_id = ensure_local_task_id(&mut task);
    if push_front {
        q.push_front(task);
    } else {
        q.push_back(task);
    }
    Ok(task_id)
}

pub(super) async fn queue_push_many(
    execution_ctx: &ActiveHarnessExecutionContext,
    mut tasks: Vec<QueuedTask>,
    queue_max: usize,
) -> Result<Vec<String>, String> {
    let queue = execution_ctx
        .lock()
        .ok()
        .and_then(|lock| lock.queue.clone())
        .ok_or_else(|| "No active session queue".to_string())?;
    let mut q = queue.lock().await;
    if q.len().saturating_add(tasks.len()) > queue_max {
        return Err(format!(
            "Policy denial: queue.max_depth={} would be exceeded",
            queue_max
        ));
    }
    let mut task_ids = Vec::with_capacity(tasks.len());
    for task in &mut tasks {
        task_ids.push(ensure_local_task_id(task));
    }
    for task in tasks {
        q.push_back(task);
    }
    Ok(task_ids)
}
