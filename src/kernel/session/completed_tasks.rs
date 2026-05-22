use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use tokio::sync::RwLock;
use turin_types::TaskInputContent;

use crate::kernel::event::{TaskBranchOutcome, TaskTerminalStatus};
use crate::kernel::session::ExecutionStatusSnapshot;
use crate::kernel::task_promotion::{PromotedTaskBranch, TaskPromotionCandidate};

pub type CompletedLocalTaskResultsHandle = Arc<RwLock<CompletedLocalTaskResults>>;

const MAX_COMPLETED_LOCAL_TASK_RESULTS: usize = 128;

/// Completed current-session task result retained in runtime memory.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LocalTaskResult {
    pub task_id: String,
    pub trace_id: String,
    pub execution: ExecutionStatusSnapshot,
    pub status: TaskTerminalStatus,
    pub task_turn_count: u32,
    pub branch_outcome: Option<TaskBranchOutcome>,
    pub promotion_candidate: Option<TaskPromotionCandidate>,
    pub promoted_branch: Option<PromotedTaskBranch>,
    pub output: Option<String>,
    pub assistant_content: Option<Vec<TaskInputContent>>,
    #[serde(skip_serializing)]
    pub promotion_input_content: Option<Vec<TaskInputContent>>,
    pub error: Option<String>,
}

#[derive(Debug, Default)]
pub struct CompletedLocalTaskResults {
    order: VecDeque<String>,
    results: HashMap<String, LocalTaskResult>,
}

impl CompletedLocalTaskResults {
    pub fn insert(&mut self, result: LocalTaskResult) {
        let task_id = result.task_id.clone();
        if !self.results.contains_key(&task_id) {
            self.order.push_back(task_id.clone());
        }
        self.results.insert(task_id, result);
        while self.order.len() > MAX_COMPLETED_LOCAL_TASK_RESULTS {
            if let Some(evicted) = self.order.pop_front() {
                self.results.remove(&evicted);
            }
        }
    }

    pub fn get(&self, task_id: &str) -> Option<&LocalTaskResult> {
        self.results.get(task_id)
    }

    pub fn mark_promoted(&mut self, task_id: &str, branch: PromotedTaskBranch) {
        if let Some(result) = self.results.get_mut(task_id) {
            result.promoted_branch = Some(branch);
        }
    }
}
