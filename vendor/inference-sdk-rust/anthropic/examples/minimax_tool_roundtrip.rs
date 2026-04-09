use anthropic_sdk::normalization::to_anthropic_request;
use anthropic_sdk::{
    Client, ClientConfig, InferenceContent, InferenceMessage, InferenceProvider, InferenceRequest,
    InferenceRole,
};
use inference_sdk_core::Tool;
use std::env;
use std::time::{SystemTime, UNIX_EPOCH};

fn env_required(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    Ok(env::var(name).map_err(|_| format!("missing env var: {name}"))?)
}

fn normalize_anthropic_base_url(mut base_url: String) -> String {
    while base_url.ends_with('/') {
        base_url.pop();
    }
    if !base_url.ends_with("/v1") {
        format!("{base_url}/v1")
    } else {
        base_url
    }
}

fn make_nonce() -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("MINIMAX_SDK_NONCE_{ts}_{}", std::process::id())
}

fn dump_normalized_request(label: &str, req: &InferenceRequest) {
    match to_anthropic_request(req.clone()) {
        Ok(msg_req) => match serde_json::to_string_pretty(&msg_req) {
            Ok(json) => {
                println!("\\n=== {label} normalized Anthropic request ===\\n{json}\\n");
            }
            Err(err) => {
                eprintln!("failed to serialize normalized request ({label}): {err}");
            }
        },
        Err(err) => {
            eprintln!("failed to normalize request ({label}): {err}");
        }
    }
}

fn tool_def() -> Tool {
    Tool {
        name: "get_nonce".to_string(),
        description: "Returns the server-provided nonce string.".to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }),
    }
}

fn first_tool_use_id(result: &[InferenceContent]) -> Option<String> {
    result.iter().find_map(|c| match c {
        InferenceContent::ToolUse { id, .. } => Some(id.clone()),
        _ => None,
    })
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = env_required("ANTHROPIC_API_KEY")?;
    let model = env_required("ANTHROPIC_MODEL")?;
    let raw_base_url = env_required("ANTHROPIC_BASE_URL")?;
    let base_url = normalize_anthropic_base_url(raw_base_url.clone());
    let nonce = make_nonce();

    println!("Using model: {model}");
    println!("Base URL (raw): {raw_base_url}");
    println!("Base URL (normalized): {base_url}");
    println!("Nonce: {nonce}");

    let client = Client::from_config(ClientConfig::new(api_key)?.with_base_url(base_url))?;
    let tools = vec![tool_def()];

    let user_prompt = "You must call the get_nonce tool exactly once before answering. After receiving the tool result, reply with exactly the nonce string and nothing else.".to_string();

    let req1 = InferenceRequest {
        model: model.clone(),
        messages: vec![InferenceMessage {
            role: InferenceRole::User,
            content: vec![InferenceContent::Text {
                text: user_prompt.clone(),
            }],
            tool_call_id: None,
        }],
        system: Some("You are a strict tool-using assistant.".to_string()),
        tools: Some(tools.clone()),
        temperature: None,
        max_tokens: Some(512),
        thinking_budget: None,
        response_format: None,
    };

    dump_normalized_request("TURN 1", &req1);
    let res1 = client.complete(req1.clone(), None).await?;
    println!("Turn 1 stop_reason: {:?}", res1.stop_reason);
    println!(
        "Turn 1 content: {}",
        serde_json::to_string_pretty(&res1.content)?
    );

    let tool_use_id = first_tool_use_id(&res1.content).ok_or_else(|| {
        format!(
            "model did not emit a tool call; response text={}",
            res1.text()
        )
    })?;
    println!("Tool use id: {tool_use_id}");

    let req2 = InferenceRequest {
        model: model.clone(),
        messages: vec![
            InferenceMessage {
                role: InferenceRole::User,
                content: vec![InferenceContent::Text { text: user_prompt }],
                tool_call_id: None,
            },
            InferenceMessage {
                role: InferenceRole::Assistant,
                content: res1.content.clone(),
                tool_call_id: None,
            },
            InferenceMessage {
                role: InferenceRole::Tool,
                content: vec![InferenceContent::ToolResult {
                    tool_use_id: tool_use_id.clone(),
                    content: nonce.clone(),
                    is_error: false,
                }],
                tool_call_id: Some(tool_use_id.clone()),
            },
        ],
        system: Some("You are a strict tool-using assistant.".to_string()),
        tools: Some(tools),
        temperature: None,
        max_tokens: Some(256),
        thinking_budget: None,
        response_format: None,
    };

    dump_normalized_request("TURN 2", &req2);
    let res2 = client.complete(req2, None).await?;
    println!("Turn 2 stop_reason: {:?}", res2.stop_reason);
    println!("Turn 2 text: {}", res2.text());
    println!(
        "Turn 2 content: {}",
        serde_json::to_string_pretty(&res2.content)?
    );

    if res2.text().trim() == nonce {
        println!("\\nMINIMAX_TOOL_ROUNDTRIP_OK");
        Ok(())
    } else {
        Err(format!("unexpected final text (expected nonce): {:?}", res2.text()).into())
    }
}
