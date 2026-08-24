use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;

use crate::DaemonRequest;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RequestEnvelope {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(flatten)]
    pub request: DaemonRequest,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    InvalidRequest,
    InvalidParams,
    AgentNotFound,
    TaskNotFound,
    ToolAuthorizationNotFound,
    ScheduleNotFound,
    WorklistNotFound,
    WorkItemNotFound,
    SessionNotFound,
    HarnessNotFound,
    ValidationFailed,
    Conflict,
    ResourceBusy,
    UnsupportedOperation,
    InternalError,
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            ErrorCode::InvalidRequest => "invalid_request",
            ErrorCode::InvalidParams => "invalid_params",
            ErrorCode::AgentNotFound => "agent_not_found",
            ErrorCode::TaskNotFound => "task_not_found",
            ErrorCode::ToolAuthorizationNotFound => "tool_authorization_not_found",
            ErrorCode::ScheduleNotFound => "schedule_not_found",
            ErrorCode::WorklistNotFound => "worklist_not_found",
            ErrorCode::WorkItemNotFound => "workitem_not_found",
            ErrorCode::SessionNotFound => "session_not_found",
            ErrorCode::HarnessNotFound => "harness_not_found",
            ErrorCode::ValidationFailed => "validation_failed",
            ErrorCode::Conflict => "conflict",
            ErrorCode::ResourceBusy => "resource_busy",
            ErrorCode::UnsupportedOperation => "unsupported_operation",
            ErrorCode::InternalError => "internal_error",
        };
        f.write_str(name)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ErrorEnvelope {
    pub code: ErrorCode,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponseEnvelope {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorEnvelope>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EventEnvelope {
    pub event: String,
    #[serde(default)]
    pub data: Value,
}

impl ResponseEnvelope {
    pub fn ok(id: Option<String>, result: Value) -> Self {
        Self {
            id,
            ok: true,
            result: Some(result),
            error: None,
        }
    }

    pub fn err(
        id: Option<String>,
        code: ErrorCode,
        message: impl Into<String>,
        details: Option<Value>,
    ) -> Self {
        Self {
            id,
            ok: false,
            result: None,
            error: Some(ErrorEnvelope {
                code,
                message: message.into(),
                details,
            }),
        }
    }
}

impl EventEnvelope {
    pub fn new(event: impl Into<String>, data: Value) -> Self {
        Self {
            event: event.into(),
            data,
        }
    }
}

impl RequestEnvelope {
    pub fn new(id: Option<String>, request: DaemonRequest) -> Self {
        Self { id, request }
    }
}
