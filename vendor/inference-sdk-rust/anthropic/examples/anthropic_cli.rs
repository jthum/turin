use anthropic_sdk::{
    AnthropicRequestExt, Client, ClientConfig, RequestOptions,
    types::message::{Content, ContentBlockDelta, Message, MessageRequest, Role, StreamEvent},
};
use clap::Parser;
use dotenvy::dotenv;
use futures_util::StreamExt;
use std::env;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// API Key for Anthropic (can also be set via ANTHROPIC_API_KEY env var)
    #[arg(short, long)]
    api_key: Option<String>,

    /// Base URL for the API
    #[arg(short, long, default_value = "https://api.anthropic.com/v1")]
    base_url: String,

    /// Model to use
    #[arg(short, long, default_value = "claude-3-opus-20240229")]
    model: String,

    /// Prompt to send
    #[arg(short, long)]
    prompt: String,

    /// Enable streaming
    #[arg(short, long)]
    stream: bool,

    /// Max tokens to sample
    #[arg(long, default_value_t = 1024)]
    max_tokens: u32,

    /// Max retries
    #[arg(long, default_value_t = 2)]
    max_retries: u32,

    /// Anthropic Beta header (e.g. tools-2024-04-04)
    #[arg(long)]
    beta: Option<String>,

    /// Request timeout in seconds
    #[arg(long, default_value_t = 60)]
    timeout: u64,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let args = Args::parse();

    let api_key = args
        .api_key
        .or_else(|| env::var("ANTHROPIC_API_KEY").ok())
        .expect("API Key must be provided via --api-key or ANTHROPIC_API_KEY env var");

    let config = ClientConfig::new(api_key)?
        .with_base_url(args.base_url)
        .with_timeout(Duration::from_secs(60))
        .with_max_retries(args.max_retries);

    let client = Client::from_config(config)?;

    let request = MessageRequest::builder()
        .model(args.model)
        .messages(vec![Message {
            role: Role::User,
            content: Content::Text(args.prompt),
        }])
        .max_tokens(args.max_tokens)
        .build();

    let mut options = RequestOptions::new()
        .with_timeout(Duration::from_secs(args.timeout))
        .with_retries(args.max_retries);

    if let Some(beta) = args.beta {
        options = options.beta(&beta)?;
    }

    if args.stream {
        println!("Streaming response...");
        let mut stream = client
            .messages()
            .create_stream_with_options(request, options)
            .await?;

        while let Some(event_result) = stream.next().await {
            match event_result {
                Ok(event) => match event {
                    StreamEvent::ContentBlockDelta { delta, .. } => match delta {
                        ContentBlockDelta::TextDelta { text } => print!("{}", text),
                        ContentBlockDelta::ThinkingDelta { thinking } => print!("{}", thinking),
                        _ => {}
                    },
                    StreamEvent::Error { error } => {
                        eprintln!("\nStream Error: {}", error.message);
                    }
                    _ => {}
                },
                Err(e) => eprintln!("\nError: {}", e),
            }
        }
        println!();
    } else {
        println!("Sending request...");
        let response = client
            .messages()
            .create_with_options(request, options)
            .await?;
        println!("Response:");
        for block in response.content {
            match block {
                anthropic_sdk::types::message::ContentBlock::Text { text } => {
                    println!("{}", text)
                }
                anthropic_sdk::types::message::ContentBlock::Thinking { thinking, .. } => {
                    println!("Thinking:\n{}", thinking)
                }
                _ => println!("[Non-text content]"),
            }
        }
    }

    Ok(())
}
