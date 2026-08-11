use anyhow::Result;
use turin_daemon_protocol::{DaemonRequest, MemoryList, MemoryListParams};

use crate::client::ControlClient;

impl ControlClient {
    pub async fn list_memories(&self, params: MemoryListParams) -> Result<MemoryList> {
        self.request_ok(None, DaemonRequest::MemoryList(params))
            .await
    }
}
