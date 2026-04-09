use anyhow::{Result, bail};
use serde_json::Value;

use crate::inference::provider::{InferenceJsonSchemaConfig, InferenceResponseFormat};

const DEFAULT_SCHEMA_NAME: &str = "structured_output";

pub fn response_format_for_schema(
    name: Option<&str>,
    description: Option<&str>,
    schema: &Value,
    strict: bool,
) -> InferenceResponseFormat {
    InferenceResponseFormat::JsonSchema {
        json_schema: InferenceJsonSchemaConfig {
            name: normalize_schema_name(name),
            description: description.map(str::to_string),
            schema: schema.clone(),
            strict: Some(strict),
        },
    }
}

pub fn fallback_system_prompt(
    base_system_prompt: &str,
    name: Option<&str>,
    description: Option<&str>,
    schema: &Value,
) -> String {
    let mut prompt = String::new();
    if !base_system_prompt.trim().is_empty() {
        prompt.push_str(base_system_prompt.trim());
        prompt.push_str("\n\n");
    }
    prompt.push_str("Return a single valid JSON value that matches the required schema exactly.");
    prompt.push_str(" Do not wrap the JSON in markdown fences.");
    prompt.push_str(" Do not add any explanatory text before or after the JSON.");
    prompt.push_str("\n\nSchema name: ");
    prompt.push_str(&normalize_schema_name(name));
    if let Some(description) = description.filter(|text| !text.trim().is_empty()) {
        prompt.push_str("\nSchema description: ");
        prompt.push_str(description.trim());
    }
    prompt.push_str("\nSchema:\n");
    prompt.push_str(&serde_json::to_string_pretty(schema).unwrap_or_else(|_| schema.to_string()));
    prompt
}

pub fn parse_and_validate_json_response(raw: &str, schema: &Value) -> Result<Value> {
    ensure_supported_schema(schema)?;
    let payload = strip_code_fences(raw);
    let parsed: Value = serde_json::from_str(payload)
        .map_err(|err| anyhow::anyhow!("response was not valid JSON: {err}"))?;
    validate_value_against_schema(&parsed, schema, "$")?;
    Ok(parsed)
}

fn normalize_schema_name(name: Option<&str>) -> String {
    let Some(name) = name else {
        return DEFAULT_SCHEMA_NAME.to_string();
    };
    let trimmed = name.trim();
    if trimmed.is_empty() {
        DEFAULT_SCHEMA_NAME.to_string()
    } else {
        trimmed.to_string()
    }
}

fn strip_code_fences(raw: &str) -> &str {
    let trimmed = raw.trim();
    if let Some(rest) = trimmed.strip_prefix("```") {
        let rest = rest.strip_prefix("json").unwrap_or(rest);
        let rest = rest.strip_prefix('\n').unwrap_or(rest);
        if let Some(inner) = rest.strip_suffix("```") {
            return inner.trim();
        }
    }
    trimmed
}

fn ensure_supported_schema(schema: &Value) -> Result<()> {
    let Value::Object(object) = schema else {
        bail!("structured output schema must be a JSON object");
    };

    for key in object.keys() {
        match key.as_str() {
            "type"
            | "properties"
            | "required"
            | "items"
            | "enum"
            | "additionalProperties"
            | "description"
            | "title" => {}
            _ => bail!("unsupported schema keyword '{key}'"),
        }
    }

    if let Some(schema_type) = object.get("type").and_then(Value::as_str) {
        match schema_type {
            "object" => {
                if let Some(properties) = object.get("properties") {
                    let Value::Object(properties) = properties else {
                        bail!("schema.properties must be an object");
                    };
                    for child in properties.values() {
                        ensure_supported_schema(child)?;
                    }
                }
                if let Some(required) = object.get("required") {
                    let Value::Array(required) = required else {
                        bail!("schema.required must be an array");
                    };
                    for item in required {
                        if !item.is_string() {
                            bail!("schema.required entries must be strings");
                        }
                    }
                }
                if let Some(additional) = object.get("additionalProperties")
                    && !additional.is_boolean()
                {
                    bail!("schema.additionalProperties must be a boolean");
                }
            }
            "array" => {
                let Some(items) = object.get("items") else {
                    bail!("array schema must define items");
                };
                ensure_supported_schema(items)?;
            }
            "string" | "number" | "integer" | "boolean" | "null" => {}
            other => bail!("unsupported schema type '{other}'"),
        }
    } else if object.contains_key("enum") {
        let Value::Array(_values) = object.get("enum").expect("checked contains_key") else {
            bail!("schema.enum must be an array");
        };
    } else {
        bail!("schema must define a supported type or enum");
    }

    Ok(())
}

fn validate_value_against_schema(value: &Value, schema: &Value, path: &str) -> Result<()> {
    let object = schema
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("invalid schema at {path}: expected object"))?;

    if let Some(enum_values) = object.get("enum") {
        let values = enum_values
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("invalid schema at {path}: enum must be an array"))?;
        if values.iter().any(|candidate| candidate == value) {
            return Ok(());
        }
        bail!("{path}: value does not match any allowed enum entry");
    }

    let schema_type = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("invalid schema at {path}: missing type"))?;

    match schema_type {
        "object" => {
            let map = value
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("{path}: expected object"))?;

            let properties = object
                .get("properties")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();

            if let Some(required) = object.get("required").and_then(Value::as_array) {
                for key in required.iter().filter_map(Value::as_str) {
                    if !map.contains_key(key) {
                        bail!("{path}: missing required property '{key}'");
                    }
                }
            }

            let allow_additional = object
                .get("additionalProperties")
                .and_then(Value::as_bool)
                .unwrap_or(true);

            for (key, child_value) in map {
                if let Some(child_schema) = properties.get(key) {
                    let child_path = format!("{path}.{key}");
                    validate_value_against_schema(child_value, child_schema, &child_path)?;
                } else if !allow_additional {
                    bail!("{path}: unexpected property '{key}'");
                }
            }
        }
        "array" => {
            let values = value
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("{path}: expected array"))?;
            let item_schema = object
                .get("items")
                .ok_or_else(|| anyhow::anyhow!("{path}: array schema missing items"))?;
            for (index, item) in values.iter().enumerate() {
                let child_path = format!("{path}[{index}]");
                validate_value_against_schema(item, item_schema, &child_path)?;
            }
        }
        "string" => {
            if !value.is_string() {
                bail!("{path}: expected string");
            }
        }
        "number" => {
            if !value.is_number() {
                bail!("{path}: expected number");
            }
        }
        "integer" => {
            let Some(number) = value.as_i64().or_else(|| value.as_u64().map(|v| v as i64)) else {
                bail!("{path}: expected integer");
            };
            let _ = number;
        }
        "boolean" => {
            if !value.is_boolean() {
                bail!("{path}: expected boolean");
            }
        }
        "null" => {
            if !value.is_null() {
                bail!("{path}: expected null");
            }
        }
        other => bail!("{path}: unsupported schema type '{other}'"),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_object_schema_with_required_fields() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "approved": { "type": "boolean" },
                "summary": { "type": "string" }
            },
            "required": ["approved", "summary"],
            "additionalProperties": false
        });

        let value =
            parse_and_validate_json_response(r#"{"approved":true,"summary":"ready"}"#, &schema)
                .expect("valid structured response");
        assert_eq!(value["approved"], true);
    }

    #[test]
    fn rejects_unexpected_properties_when_additional_properties_false() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "summary": { "type": "string" }
            },
            "additionalProperties": false
        });

        let err = parse_and_validate_json_response(r#"{"summary":"ok","extra":1}"#, &schema)
            .expect_err("extra field should be rejected");
        assert!(err.to_string().contains("unexpected property 'extra'"));
    }

    #[test]
    fn strips_markdown_fences_before_json_parse() {
        let schema = serde_json::json!({
            "type": "array",
            "items": { "type": "string" }
        });

        let value = parse_and_validate_json_response("```json\n[\"a\",\"b\"]\n```", &schema)
            .expect("fenced json should parse");
        assert_eq!(value, serde_json::json!(["a", "b"]));
    }
}
