use std::io::{self, Write};
use std::pin::Pin;

use anyhow::{Context, Result};
use futures::{Stream, StreamExt};

use crate::kernel::session::SessionState;

use super::super::event::{KernelEvent, StreamEvent};
use super::super::{Kernel, PendingToolCall};

#[derive(Debug, Default)]
pub(super) struct TurnStreamOutput {
    pub response_thinking: String,
    pub response_thinking_signature: Option<String>,
    pub response_text: String,
    pub pending_tool_calls: Vec<PendingToolCall>,
}

impl Kernel {
    pub(super) async fn collect_turn_stream_output(
        &mut self,
        session: &mut SessionState,
        provider_name: &str,
        model: &str,
        mut stream: Pin<Box<dyn Stream<Item = Result<KernelEvent>> + Send>>,
    ) -> Result<TurnStreamOutput> {
        let mut output = TurnStreamOutput {
            response_thinking: String::with_capacity(2048),
            response_thinking_signature: None,
            response_text: String::with_capacity(4096),
            pending_tool_calls: Vec::new(),
        };
        let mut is_thinking = false;

        while let Some(event_result) = stream.next().await {
            let event = event_result.with_context(|| {
                format!(
                    "inference stream event failure (provider='{}', model='{}')",
                    provider_name, model
                )
            })?;
            match &event {
                KernelEvent::Stream(e) => match e {
                    StreamEvent::ThinkingDelta { .. } => {
                        if !self.json && !is_thinking {
                            print!("\x1b[35m💭 Thinking...\x1b[0m");
                            io::stdout().flush().ok();
                            is_thinking = true;
                        }
                        self.persist_event(session, &event);
                        if let StreamEvent::ThinkingDelta { thinking } = e {
                            output.response_thinking.push_str(thinking);
                        }
                    }
                    StreamEvent::ThinkingSignatureDelta { signature } => {
                        self.persist_event(session, &event);
                        match output.response_thinking_signature.as_mut() {
                            Some(existing) => existing.push_str(signature),
                            None => output.response_thinking_signature = Some(signature.clone()),
                        }
                    }
                    StreamEvent::MessageDelta { content_delta } => {
                        if is_thinking {
                            if !self.json {
                                println!();
                            }
                            is_thinking = false;
                        }
                        if !self.json {
                            print!("{}", content_delta);
                            io::stdout().flush().ok();
                        }
                        self.persist_event(session, &event);
                        output.response_text.push_str(content_delta);
                    }
                    StreamEvent::MessageEnd {
                        input_tokens,
                        output_tokens,
                        ..
                    } => {
                        if is_thinking {
                            if !self.json {
                                println!();
                            }
                            is_thinking = false;
                        }
                        session.total_input_tokens += *input_tokens;
                        session.total_output_tokens += *output_tokens;
                        self.persist_event(session, &event);
                    }
                    StreamEvent::ToolCall { id, name, args } => {
                        if is_thinking {
                            if !self.json {
                                println!();
                            }
                            is_thinking = false;
                        }
                        if !self.json {
                            println!(
                                "\n\x1b[33m⚒️  Tool Call:\x1b[0m \x1b[1m{}\x1b[0m({})",
                                name, args
                            );
                        }
                        self.persist_event(session, &event);
                        output.pending_tool_calls.push(PendingToolCall {
                            id: id.clone(),
                            name: name.clone(),
                            args: args.clone(),
                        });
                    }
                    _ => {
                        self.persist_event(session, &event);
                    }
                },
                _ => {
                    self.persist_event(session, &event);
                }
            }
        }

        if !self.json && !output.response_text.is_empty() && !output.response_text.ends_with('\n') {
            println!();
        }

        Ok(output)
    }
}
