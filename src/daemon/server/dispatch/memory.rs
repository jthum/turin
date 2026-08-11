use crate::daemon::protocol::{MemoryListParams, ResponseEnvelope};

use super::{DispatchContext, serialize_response, validation_error};

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
