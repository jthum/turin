use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeclaredVirtualTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

impl DeclaredVirtualTool {
    pub fn tool_definition(&self) -> Value {
        json!({
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VirtualToolCall {
    pub name: String,
    #[serde(default = "default_call_args")]
    pub args: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VirtualToolPlan {
    pub calls: Vec<VirtualToolCall>,
    pub result_handler_key: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VirtualToolNestedResult {
    pub id: String,
    pub name: String,
    pub args: Value,
    pub verdict: String,
    pub duration_ms: u64,
    pub content: String,
    pub is_error: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VirtualToolResultOutput {
    pub content: String,
    pub is_error: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VirtualToolResultResolution {
    Output(VirtualToolResultOutput),
    Plan(VirtualToolPlan),
}

fn default_call_args() -> Value {
    Value::Object(Map::new())
}

pub fn normalize_tool_declaration(
    name: &str,
    description: &str,
    params: Option<Value>,
    input_schema: Option<Value>,
) -> Result<DeclaredVirtualTool> {
    let name = name.trim();
    if name.is_empty() {
        bail!("tool name must not be empty");
    }

    let description = description.trim();
    if description.is_empty() {
        bail!("tool '{}' description must not be empty", name);
    }

    let input_schema = match (params, input_schema) {
        (Some(_), Some(_)) => {
            bail!(
                "tool '{}' declaration must use either params or input_schema, not both",
                name
            )
        }
        (None, None) => {
            bail!(
                "tool '{}' declaration must define params or input_schema",
                name
            )
        }
        (Some(params), None) => params_to_input_schema(name, params)?,
        (None, Some(schema)) => {
            if !schema.is_object() {
                bail!("tool '{}' input_schema must be a JSON object", name);
            }
            schema
        }
    };

    Ok(DeclaredVirtualTool {
        name: name.to_string(),
        description: description.to_string(),
        input_schema,
    })
}

fn params_to_input_schema(tool_name: &str, params: Value) -> Result<Value> {
    let params = params
        .as_object()
        .with_context(|| format!("tool '{}' params must be an object", tool_name))?;

    let mut properties = Map::new();
    let mut required: Vec<Value> = Vec::new();

    for (param_name, param_spec) in params {
        let normalized = normalize_param_spec(tool_name, param_name, param_spec)?;
        if normalized.required {
            required.push(Value::String(param_name.clone()));
        }
        properties.insert(param_name.clone(), normalized.schema);
    }

    let mut schema = Map::new();
    schema.insert("type".to_string(), Value::String("object".to_string()));
    schema.insert("properties".to_string(), Value::Object(properties));
    if !required.is_empty() {
        schema.insert("required".to_string(), Value::Array(required));
    }

    Ok(Value::Object(schema))
}

struct NormalizedParamSpec {
    schema: Value,
    required: bool,
}

fn normalize_param_spec(
    tool_name: &str,
    param_name: &str,
    spec: &Value,
) -> Result<NormalizedParamSpec> {
    let mut object = match spec {
        Value::Object(map) => map.clone(),
        Value::String(type_name) => {
            let mut map = Map::new();
            map.insert("type".to_string(), Value::String(type_name.clone()));
            map
        }
        _ => bail!(
            "tool '{}' param '{}' must be an object or type string",
            tool_name,
            param_name
        ),
    };

    let required = object
        .remove("required")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);

    let has_schema_shape = object.contains_key("type")
        || object.contains_key("oneOf")
        || object.contains_key("anyOf")
        || object.contains_key("allOf")
        || object.contains_key("$ref");
    if !has_schema_shape {
        bail!(
            "tool '{}' param '{}' must define a schema shape (for example type = \"string\")",
            tool_name,
            param_name
        );
    }

    Ok(NormalizedParamSpec {
        schema: Value::Object(object),
        required,
    })
}

pub fn parse_handler_plan(value: &Value) -> Result<VirtualToolPlan> {
    let Some(kind) = value.get("__kind").and_then(Value::as_str) else {
        bail!("virtual tool handler must return tool.call(...) or tool.sequence(...)");
    };
    let result_handler_key = value
        .get("__result_handler_key")
        .and_then(Value::as_str)
        .map(str::to_string);

    match kind {
        "tool_call" => Ok(VirtualToolPlan {
            calls: vec![parse_virtual_call(value)?],
            result_handler_key,
        }),
        "tool_sequence" => {
            let calls = value
                .get("calls")
                .and_then(Value::as_array)
                .context("tool.sequence(...) must include a calls array")?;
            if calls.is_empty() {
                bail!("tool.sequence(...) must include at least one tool call");
            }
            let mut parsed = Vec::with_capacity(calls.len());
            for call in calls {
                parsed.push(parse_virtual_call(call)?);
            }
            Ok(VirtualToolPlan {
                calls: parsed,
                result_handler_key,
            })
        }
        other => bail!("unknown virtual tool handler result kind '{}'", other),
    }
}

pub fn parse_result_handler_output(
    value: &Value,
    default_is_error: bool,
) -> Result<VirtualToolResultResolution> {
    if let Some(kind) = value.get("__kind").and_then(Value::as_str)
        && matches!(kind, "tool_call" | "tool_sequence")
    {
        return Ok(VirtualToolResultResolution::Plan(parse_handler_plan(
            value,
        )?));
    }

    match value {
        Value::String(content) => Ok(VirtualToolResultOutput {
            content: content.clone(),
            is_error: default_is_error,
        }
        .into()),
        Value::Object(map) => {
            let content = map
                .get("content")
                .and_then(Value::as_str)
                .context("virtual tool result handler object must include a string content field")?
                .to_string();
            let is_error = map
                .get("is_error")
                .and_then(Value::as_bool)
                .unwrap_or(default_is_error);
            Ok(VirtualToolResultOutput { content, is_error }.into())
        }
        _ => bail!(
            "virtual tool result handler must return tool.call(...), tool.sequence(...), a string, or {{ content = ..., is_error? = ... }}"
        ),
    }
}

impl From<VirtualToolResultOutput> for VirtualToolResultResolution {
    fn from(value: VirtualToolResultOutput) -> Self {
        Self::Output(value)
    }
}

fn parse_virtual_call(value: &Value) -> Result<VirtualToolCall> {
    let name = value
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .context("virtual tool call is missing a non-empty name")?;
    let args = value.get("args").cloned().unwrap_or_else(default_call_args);

    Ok(VirtualToolCall {
        name: name.to_string(),
        args,
    })
}

pub fn shell_quote(input: &str) -> String {
    if input.is_empty() {
        return "''".to_string();
    }

    let mut out = String::from("'");
    for ch in input.chars() {
        if ch == '\'' {
            out.push_str("'\"'\"'");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

#[cfg(test)]
#[path = "tests/virtual_tools.rs"]
mod tests;
