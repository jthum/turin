use anthropic_sdk::{
    Client,
    types::message::{Content, ContentBlockDelta, Message, MessageRequest, Role, StreamEvent},
};
use dotenvy::dotenv;
use futures_util::StreamExt;
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
            content: Content::Text("Write a haiku about Rust".to_string()),
        }])
        .max_tokens(1024)
        .build();

    println!("Sending streaming request...");
    let mut stream = client.messages().create_stream(request).await?;

    while let Some(event_result) = stream.next().await {
        match event_result {
            Ok(event) => match event {
                StreamEvent::ContentBlockDelta {
                    delta: ContentBlockDelta::TextDelta { text },
                    ..
                } => print!("{}", text),
                StreamEvent::Error { error } => {
                    eprintln!("\nError from stream: {}", error.message);
                }
                _ => {}
            },
            Err(e) => eprintln!("\nStream Error: {}", e),
        }
    }
    println!();

    Ok(())
}
