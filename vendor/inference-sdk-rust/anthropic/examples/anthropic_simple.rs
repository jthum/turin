use anthropic_sdk::{
    Client,
    types::message::{Content, Message, MessageRequest, Role},
};
use dotenvy::dotenv;
use std::env;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let api_key = env::var("ANTHROPIC_API_KEY")?;

    let client = Client::new(api_key)?;

    let request = MessageRequest::builder()
        .model("claude-3-opus-20240229")
        .messages(vec![Message {
            role: Role::User,
            content: Content::Text("Hello, Claude!".to_string()),
        }])
        .max_tokens(1024)
        .build();

    println!("Sending request...");
    let response = client.messages().create(request).await?;

    println!("Response: {:?}", response);

    Ok(())
}
