use futures_util::StreamExt;
use openai_sdk::{
    Client,
    types::chat::{ChatCompletionRequest, ChatContent, ChatMessage, ChatRole},
};
use std::env;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let client = Client::new(api_key)?;

    let request = ChatCompletionRequest::builder()
        .model("gpt-4o-mini")
        .messages(vec![ChatMessage {
            role: ChatRole::User,
            content: Some(ChatContent::Text(
                "Explain Rust in three sentences.".to_string(),
            )),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }])
        .max_tokens(200_u32)
        .build();

    let mut stream = client.chat().create_stream(request).await?;

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                for choice in &chunk.choices {
                    if let Some(content) = &choice.delta.content {
                        print!("{}", content);
                    }
                }
            }
            Err(e) => {
                eprintln!("\nStream Error: {}", e);
                break;
            }
        }
    }
    println!();

    Ok(())
}
