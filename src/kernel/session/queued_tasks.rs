use turin_types::{TaskInputContent, ToolsConfig};

use crate::kernel::event::TaskBranchOutcome;
use crate::kernel::session::{ExecutionConflictPolicy, TaskExecutionOverrides};

/// One queued unit of work to be executed by the kernel.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QueuedTask {
    pub task_id: String,
    pub plan_id: Option<String>,
    pub title: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub content: Option<Vec<TaskInputContent>>,
    #[serde(default)]
    pub tools: Option<ToolsConfig>,
    #[serde(default)]
    pub conflict_policy: Option<ExecutionConflictPolicy>,
    #[serde(default)]
    pub execution: Option<TaskExecutionOverrides>,
    #[serde(default)]
    pub branch_outcome: Option<TaskBranchOutcome>,
    #[serde(default = "new_trace_id")]
    pub trace_id: String,
}

impl QueuedTask {
    pub fn ad_hoc(prompt: impl Into<String>) -> Self {
        Self {
            task_id: String::new(), // Assigned by SessionState
            plan_id: None,
            title: None,
            prompt: prompt.into(),
            content: None,
            tools: None,
            conflict_policy: None,
            execution: None,
            branch_outcome: None,
            trace_id: new_trace_id(),
        }
    }

    pub fn with_plan(
        prompt: impl Into<String>,
        plan_id: impl Into<String>,
        title: Option<String>,
    ) -> Self {
        Self {
            task_id: String::new(), // Assigned by SessionState
            plan_id: Some(plan_id.into()),
            title,
            prompt: prompt.into(),
            content: None,
            tools: None,
            conflict_policy: None,
            execution: None,
            branch_outcome: None,
            trace_id: new_trace_id(),
        }
    }

    pub fn with_inherited_trace(mut self, trace_id: Option<&str>) -> Self {
        if let Some(trace_id) = trace_id
            && !trace_id.is_empty()
        {
            self.trace_id = trace_id.to_string();
        }
        self
    }

    pub fn with_conflict_policy(
        mut self,
        conflict_policy: Option<ExecutionConflictPolicy>,
    ) -> Self {
        self.conflict_policy = conflict_policy;
        self
    }

    pub fn with_execution(mut self, execution: Option<TaskExecutionOverrides>) -> Self {
        self.execution = execution;
        self
    }

    pub fn with_branch_outcome(mut self, branch_outcome: Option<TaskBranchOutcome>) -> Self {
        self.branch_outcome = branch_outcome;
        self
    }
}

fn new_trace_id() -> String {
    format!("tr_{}", uuid::Uuid::now_v7().simple())
}
