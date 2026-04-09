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
            content: Some(ChatContent::Text("Say hello in one sentence.".to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }])
        .max_tokens(100_u32)
        .build();

    let response = client.chat().create(request).await?;

    for choice in response.choices {
        if let Some(ChatContent::Text(text)) = choice.message.content {
            println!("{}", text);
        }
    }

    Ok(())
}
