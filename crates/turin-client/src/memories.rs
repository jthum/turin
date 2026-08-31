use anyhow::Result;
use turin_daemon_protocol::{
    DaemonRequest, MemoryCorrectParams, MemoryDeleteResult, MemoryDetail, MemoryList,
    MemoryListParams, MemoryTargetParams,
};

use crate::client::Client;

impl Client {
    pub async fn list_memories(&self, params: MemoryListParams) -> Result<MemoryList> {
        self.request_ok(None, DaemonRequest::MemoryList(params))
            .await
    }

    pub async fn get_memory(&self, params: MemoryTargetParams) -> Result<MemoryDetail> {
        self.request_ok(None, DaemonRequest::MemoryGet(params))
            .await
    }

    pub async fn correct_memory(&self, params: MemoryCorrectParams) -> Result<MemoryDetail> {
        self.request_ok(None, DaemonRequest::MemoryCorrect(params))
            .await
    }

    pub async fn delete_memory(&self, params: MemoryTargetParams) -> Result<MemoryDeleteResult> {
        self.request_ok(None, DaemonRequest::MemoryDelete(params))
            .await
    }
}
