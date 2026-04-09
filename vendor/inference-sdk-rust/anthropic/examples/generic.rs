use anthropic_sdk::{
    Client, InferenceContent, InferenceMessage, InferenceProvider, InferenceRequest, InferenceRole,
};
use dotenvy::dotenv;
use std::env;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let api_key = env::var("ANTHROPIC_API_KEY")?;
    let client = Client::new(api_key)?;

    let request = InferenceRequest::builder()
        .model("claude-3-5-sonnet-20240620")
        .messages(vec![InferenceMessage {
            role: InferenceRole::User,
            content: vec![InferenceContent::Text {
                text: "Hello from generic trait!".to_string(),
            }],
            tool_call_id: None,
        }])
        .max_tokens(1024)
        .build();

    println!("Sending generic request via complete()...");
    // Test complete (default impl using stream)
    let result = client.complete(request, None).await?;

    println!("Model: {}", result.model);
    println!("Stop Reason: {:?}", result.stop_reason);
    println!("Usage: {:?}", result.usage);
    println!("Text: {}", result.text());

    Ok(())
}
