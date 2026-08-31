use crate::daemon::protocol::{
    ErrorCode, MemoryCorrectParams, MemoryListParams, MemoryTargetParams, ResponseEnvelope,
};

use super::{DispatchContext, optional_response, serialize_response, validation_error};

pub(super) async fn list(
    id: Option<String>,
    params: MemoryListParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.list_memories(&params).await {
        Ok(memories) => serialize_response(id, memories, "memory list"),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn get(
    id: Option<String>,
    params: MemoryTargetParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    let result = guard.memory_detail(&params).await;
    optional_response(
        id,
        result,
        "memory detail",
        ErrorCode::MemoryNotFound,
        || format!("Memory '{}' not found", params.id),
    )
}

pub(super) async fn correct(
    id: Option<String>,
    params: MemoryCorrectParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.correct_memory(&params).await {
        Ok(memory) => serialize_response(id, memory, "memory correction"),
        Err(error) => validation_error(id, error),
    }
}

pub(super) async fn delete(
    id: Option<String>,
    params: MemoryTargetParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.delete_memory(&params).await {
        Ok(result) if result.deleted => serialize_response(id, result, "memory deletion"),
        Ok(_) => crate::daemon::protocol::ResponseEnvelope::err(
            id,
            ErrorCode::MemoryNotFound,
            format!("Memory '{}' not found", params.id),
            None,
        ),
        Err(error) => validation_error(id, error),
    }
}
