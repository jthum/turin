use super::*;

#[test]
fn params_normalize_to_json_schema() {
    let tool = normalize_tool_declaration(
        "play_song",
        "Play a song",
        Some(json!({
            "filename": { "type": "string", "required": true },
            "volume": { "type": "integer" }
        })),
        None,
    )
    .unwrap();

    assert_eq!(tool.name, "play_song");
    assert_eq!(tool.description, "Play a song");
    assert_eq!(
        tool.input_schema,
        json!({
            "type": "object",
            "properties": {
                "filename": { "type": "string" },
                "volume": { "type": "integer" }
            },
            "required": ["filename"]
        })
    );
}

#[test]
fn full_input_schema_passthrough_is_supported() {
    let tool = normalize_tool_declaration(
        "lookup",
        "Lookup nested values",
        None,
        Some(json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" },
                "opts": {
                    "type": "object",
                    "properties": {
                        "limit": { "type": "integer" }
                    }
                }
            },
            "required": ["query"]
        })),
    )
    .unwrap();

    assert_eq!(tool.input_schema["properties"]["opts"]["type"], "object");
}

#[test]
fn handler_plan_supports_single_and_sequence_shapes() {
    let single = parse_handler_plan(&json!({
        "__kind": "tool_call",
        "__result_handler_key": "cb_single",
        "name": "shell_exec",
        "args": { "command": "echo hi" }
    }))
    .unwrap();
    assert_eq!(single.calls.len(), 1);
    assert_eq!(single.calls[0].name, "shell_exec");
    assert_eq!(single.result_handler_key.as_deref(), Some("cb_single"));

    let seq = parse_handler_plan(&json!({
        "__kind": "tool_sequence",
        "__result_handler_key": "cb_seq",
        "calls": [
            { "__kind": "tool_call", "name": "read_file", "args": { "path": "a.txt" } },
            { "__kind": "tool_call", "name": "read_file", "args": { "path": "b.txt" } }
        ]
    }))
    .unwrap();
    assert_eq!(seq.calls.len(), 2);
    assert_eq!(seq.calls[1].name, "read_file");
    assert_eq!(seq.result_handler_key.as_deref(), Some("cb_seq"));
}

#[test]
fn shell_quote_handles_empty_and_embedded_quotes() {
    assert_eq!(shell_quote(""), "''");
    assert_eq!(shell_quote("plain text"), "'plain text'");
    assert_eq!(shell_quote("a'b"), "'a'\"'\"'b'");
}

#[test]
fn result_handler_string_uses_default_error() {
    let out = parse_result_handler_output(&json!("wrapped"), true).unwrap();
    assert_eq!(
        out,
        VirtualToolResultResolution::Output(VirtualToolResultOutput {
            content: "wrapped".to_string(),
            is_error: true,
        })
    );
}

#[test]
fn result_handler_object_can_override_error() {
    let out = parse_result_handler_output(
        &json!({
            "content": "ok",
            "is_error": false
        }),
        true,
    )
    .unwrap();
    assert_eq!(
        out,
        VirtualToolResultResolution::Output(VirtualToolResultOutput {
            content: "ok".to_string(),
            is_error: false,
        })
    );
}

#[test]
fn result_handler_can_return_follow_up_plan() {
    let out = parse_result_handler_output(
        &json!({
            "__kind": "tool_call",
            "name": "read_file",
            "args": { "path": "next.txt" }
        }),
        false,
    )
    .unwrap();
    assert_eq!(
        out,
        VirtualToolResultResolution::Plan(VirtualToolPlan {
            calls: vec![VirtualToolCall {
                name: "read_file".to_string(),
                args: json!({ "path": "next.txt" }),
            }],
            result_handler_key: None,
        })
    );
}
