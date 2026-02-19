use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;

use crate::tools::{Tool, ToolContext, ToolError, parse_args};

pub struct SubmitPlanTool;

#[derive(Deserialize)]
struct SubmitPlanArgs {
    /// Title of the proposed plan
    title: String,
    /// List of tasks to execute
    tasks: Vec<String>,
    /// Whether to clear the existing queue before adding these tasks
    #[serde(default)]
    clear_existing: bool,
}

#[async_trait]
impl Tool for SubmitPlanTool {
    fn name(&self) -> &str {
        "submit_plan"
    }

    fn description(&self) -> &str {
        "Submit a plan with one or more tasks to execute in order."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title of the overall plan"
                },
                "tasks": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of tasks to execute in order"
                },
                "clear_existing": {
                    "type": "boolean",
                    "description": "If true, clears the current queue before adding these tasks. Default is false."
                }
            },
            "required": ["title", "tasks"]
        })
    }

    #[tracing::instrument(skip(self, params, _ctx), fields(title = %params["title"].as_str().unwrap_or("unknown")))]
    async fn execute(
        &self,
        params: Value,
        _ctx: &ToolContext,
    ) -> Result<crate::tools::ToolEffect, ToolError> {
        let args: SubmitPlanArgs = parse_args(params)?;
        tracing::info!(title = %args.title, tasks = args.tasks.len(), "Submitting plan");

        Ok(crate::tools::ToolEffect::EnqueuePlan {
            title: args.title,
            tasks: args.tasks,
            clear_existing: args.clear_existing,
        })
    }
}
