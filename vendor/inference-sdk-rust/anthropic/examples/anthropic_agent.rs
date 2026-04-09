use anthropic_sdk::{
    Client,
    types::message::{Content, ContentBlock, Message, MessageRequest, Role},
};
use dotenvy::dotenv;
use std::env;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let api_key = env::var("ANTHROPIC_API_KEY")?;
    let client = Client::new(api_key)?;

    // Agents often need to send a system prompt and a user message
    let request = MessageRequest::builder()
        .model("claude-3-5-sonnet-20240620")
        .system("You are a helpful AI assistant.")
        .messages(vec![Message {
            role: Role::User,
            content: Content::Text("What is the capital of France?".to_string()),
        }])
        .max_tokens(1024)
        .build();

    let response = client.messages().create(request).await?;

    for block in response.content {
        if let ContentBlock::Text { text } = block {
            println!("{}", text);
        }
    }

    Ok(())
}
