use serde_json::{Number, Value};
use turin_daemon_protocol::{UiFormField, UiFormNode};

pub fn ui_form_default_value(form: &UiFormNode, field: &UiFormField) -> String {
    field
        .default
        .as_ref()
        .or_else(|| form.params.get(&field.name))
        .map(ui_form_value_string)
        .or_else(|| field.options.first().map(ui_form_value_string))
        .unwrap_or_else(|| {
            if ui_form_is_bool_field(field) {
                "false".to_string()
            } else {
                String::new()
            }
        })
}

pub fn ui_form_field_kind(field: &UiFormField) -> String {
    field.kind.as_deref().unwrap_or("text").to_ascii_lowercase()
}

pub fn ui_form_value_string(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(value) => value.clone(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

pub fn ui_form_is_bool_field(field: &UiFormField) -> bool {
    matches!(
        ui_form_field_kind(field).as_str(),
        "bool" | "boolean" | "checkbox" | "switch"
    )
}

pub fn ui_form_is_multiline_field(field: &UiFormField) -> bool {
    matches!(
        ui_form_field_kind(field).as_str(),
        "textarea" | "multiline" | "markdown"
    )
}

pub fn ui_form_is_password_field(field: &UiFormField) -> bool {
    matches!(
        ui_form_field_kind(field).as_str(),
        "password" | "secret" | "passphrase"
    )
}

pub fn parse_ui_form_value(field: &UiFormField, value: &str) -> Result<Value, String> {
    if let Some(option) = field
        .options
        .iter()
        .find(|option| ui_form_value_string(option) == value)
    {
        return Ok(option.clone());
    }

    match ui_form_field_kind(field).as_str() {
        "number" | "float" | "decimal" => {
            let parsed = value
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("Form field '{}' must be a valid number", field.label))?;
            Number::from_f64(parsed)
                .map(Value::Number)
                .ok_or_else(|| format!("Form field '{}' must be a finite number", field.label))
        }
        "int" | "integer" => value
            .trim()
            .parse::<i64>()
            .map(|value| Value::Number(value.into()))
            .map_err(|_| format!("Form field '{}' must be a valid integer", field.label)),
        "bool" | "boolean" | "checkbox" | "switch" => {
            Ok(Value::Bool(matches!(value, "true" | "1" | "yes" | "on")))
        }
        _ => Ok(Value::String(value.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use turin_daemon_protocol::{UiFormField, UiFormNode};

    use super::{
        parse_ui_form_value, ui_form_default_value, ui_form_field_kind, ui_form_is_bool_field,
        ui_form_is_multiline_field, ui_form_is_password_field,
    };

    #[test]
    fn default_value_uses_field_form_param_option_then_bool_fallback() {
        let form = UiFormNode {
            id: Some("seed".to_string()),
            title: "Seed".to_string(),
            action: "release.seed".to_string(),
            fields: Vec::new(),
            params: json!({ "count": 3 }),
        };
        let mut field = test_field("count", "integer");
        assert_eq!(ui_form_default_value(&form, &field), "3");

        field.default = Some(json!(4));
        assert_eq!(ui_form_default_value(&form, &field), "4");

        let mut option_field = test_field("lane", "text");
        option_field.options = vec![json!("qa")];
        assert_eq!(ui_form_default_value(&form, &option_field), "qa");

        let bool_field = test_field("confirmed", "boolean");
        assert_eq!(ui_form_default_value(&form, &bool_field), "false");
    }

    #[test]
    fn parse_value_preserves_typed_options_before_scalar_parsing() {
        let mut field = test_field("risk", "integer");
        field.options = vec![json!(2), json!("3")];

        assert_eq!(parse_ui_form_value(&field, "2"), Ok(json!(2)));
        assert_eq!(parse_ui_form_value(&field, "3"), Ok(json!("3")));
    }

    #[test]
    fn parse_value_coerces_numbers_booleans_and_strings() {
        assert_eq!(
            parse_ui_form_value(&test_field("threshold", "decimal"), "0.82"),
            Ok(json!(0.82))
        );
        assert_eq!(
            parse_ui_form_value(&test_field("count", "integer"), "4"),
            Ok(json!(4))
        );
        assert_eq!(
            parse_ui_form_value(&test_field("confirmed", "switch"), "yes"),
            Ok(json!(true))
        );
        assert_eq!(
            parse_ui_form_value(&test_field("title", "text"), "Release"),
            Ok(json!("Release"))
        );
    }

    #[test]
    fn field_kind_helpers_normalize_common_aliases() {
        let bool_field = test_field("confirmed", "checkbox");
        let multiline_field = test_field("notes", "markdown");
        let password_field = test_field("token", "SECRET");

        assert_eq!(ui_form_field_kind(&test_field("plain", "TEXT")), "text");
        assert!(ui_form_is_bool_field(&bool_field));
        assert!(ui_form_is_multiline_field(&multiline_field));
        assert!(ui_form_is_password_field(&password_field));
    }

    fn test_field(name: &str, kind: &str) -> UiFormField {
        UiFormField {
            name: name.to_string(),
            label: name.to_string(),
            kind: Some(kind.to_string()),
            default: None,
            required: None,
            options: Vec::new(),
        }
    }
}
