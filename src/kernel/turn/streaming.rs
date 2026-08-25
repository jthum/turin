use std::io::{self, Write};
use std::pin::Pin;

use anyhow::{Context, Result};
use futures::{Stream, StreamExt};

use crate::display;
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::session::SessionState;

use super::super::PendingToolCall;
use super::super::event::{KernelEvent, StreamEvent};

#[derive(Debug, Default)]
pub(super) struct TurnStreamOutput {
    pub response_thinking: String,
    pub response_thinking_signature: Option<String>,
    pub response_text: String,
    pub pending_tool_calls: Vec<PendingToolCall>,
    pub cancelled: bool,
}

impl ExecutionHost {
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
            cancelled: false,
        };
        let mut is_thinking = false;
        let mut durable_stream_events = Vec::new();
        let mut durable_thinking = String::new();
        let mut durable_thinking_signature = String::new();
        let mut durable_text = String::new();
        let ansi_stdout = display::stdout_ansi();

        loop {
            let next_event = tokio::select! {
                _ = session.cancel_token.cancelled() => {
                    output.cancelled = true;
                    break;
                }
                event = stream.next() => event,
            };

            let Some(event_result) = next_event else {
                break;
            };
            let event = event_result.with_context(|| {
                format!(
                    "inference stream event failure (provider='{}', model='{}')",
                    provider_name, model
                )
            })?;
            match &event {
                KernelEvent::Stream(e) => match e {
                    StreamEvent::ThinkingDelta { .. } => {
                        if self.paints_cli_text() && !is_thinking {
                            print!("{}", display::thinking_label(ansi_stdout));
                            io::stdout().flush().ok();
                            is_thinking = true;
                        }
                        let published = self.publish_ephemeral_event(session, &event);
                        if let StreamEvent::ThinkingDelta { thinking } = e {
                            if published {
                                durable_thinking.push_str(thinking);
                            }
                            output.response_thinking.push_str(thinking);
                        }
                    }
                    StreamEvent::ThinkingSignatureDelta { signature } => {
                        if self.publish_ephemeral_event(session, &event) {
                            durable_thinking_signature.push_str(signature);
                        }
                        match output.response_thinking_signature.as_mut() {
                            Some(existing) => existing.push_str(signature),
                            None => output.response_thinking_signature = Some(signature.clone()),
                        }
                    }
                    StreamEvent::MessageDelta { content_delta } => {
                        if is_thinking {
                            if self.paints_cli_text() {
                                println!();
                            }
                            is_thinking = false;
                        }
                        if self.paints_cli_text() {
                            print!("{}", content_delta);
                            io::stdout().flush().ok();
                        }
                        if self.publish_ephemeral_event(session, &event) {
                            durable_text.push_str(content_delta);
                        }
                        output.response_text.push_str(content_delta);
                    }
                    StreamEvent::MessageEnd {
                        input_tokens,
                        output_tokens,
                        ..
                    } => {
                        if is_thinking {
                            if self.paints_cli_text() {
                                println!();
                            }
                            is_thinking = false;
                        }
                        session.total_input_tokens += *input_tokens;
                        session.total_output_tokens += *output_tokens;
                        session
                            .record_delegation_tokens(input_tokens.saturating_add(*output_tokens));
                        if self.publish_ephemeral_event(session, &event) {
                            durable_stream_events.push(event.clone());
                        }
                    }
                    StreamEvent::ToolCall { id, name, args } => {
                        if is_thinking {
                            if self.paints_cli_text() {
                                println!();
                            }
                            is_thinking = false;
                        }
                        if self.paints_cli_text() {
                            println!("{}", display::tool_call_line(name, args, ansi_stdout));
                        }
                        if self.publish_ephemeral_event(session, &event) {
                            durable_stream_events.push(event.clone());
                        }
                        output.pending_tool_calls.push(PendingToolCall {
                            id: id.clone(),
                            name: name.clone(),
                            args: args.clone(),
                        });
                    }
                    _ => {
                        if self.publish_ephemeral_event(session, &event) {
                            durable_stream_events.push(event.clone());
                        }
                    }
                },
                _ => {
                    self.persist_event(session, &event).await;
                }
            }
        }

        for event in coalesced_durable_stream_events(
            durable_stream_events,
            durable_thinking,
            durable_thinking_signature,
            durable_text,
        ) {
            self.persist_published_event(session, &event).await;
        }

        if self.paints_cli_text() && !output.response_text.is_empty() && !output.response_text.ends_with('\n') {
            println!();
        }

        Ok(output)
    }
}

fn coalesced_durable_stream_events(
    mut events: Vec<KernelEvent>,
    thinking: String,
    thinking_signature: String,
    text: String,
) -> Vec<KernelEvent> {
    let mut coalesced = Vec::with_capacity(events.len() + 3);
    if let Some(position) = events
        .iter()
        .position(|event| matches!(event, KernelEvent::Stream(StreamEvent::MessageStart { .. })))
    {
        coalesced.push(events.remove(position));
    }
    if !thinking.is_empty() {
        coalesced.push(KernelEvent::Stream(StreamEvent::ThinkingDelta { thinking }));
    }
    if !thinking_signature.is_empty() {
        coalesced.push(KernelEvent::Stream(StreamEvent::ThinkingSignatureDelta {
            signature: thinking_signature,
        }));
    }
    if !text.is_empty() {
        coalesced.push(KernelEvent::Stream(StreamEvent::MessageDelta {
            content_delta: text,
        }));
    }
    coalesced.extend(events);
    coalesced
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durable_stream_content_is_coalesced_without_losing_terminal_events() {
        let events = vec![
            KernelEvent::Stream(StreamEvent::MessageStart {
                role: "assistant".to_string(),
                model: "mock".to_string(),
            }),
            KernelEvent::Stream(StreamEvent::MessageEnd {
                role: "assistant".to_string(),
                input_tokens: 10,
                output_tokens: 4,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            }),
        ];

        let coalesced = coalesced_durable_stream_events(
            events,
            "reasoning".to_string(),
            "signature".to_string(),
            "complete response".to_string(),
        );

        assert_eq!(coalesced.len(), 5);
        assert!(matches!(
            &coalesced[0],
            KernelEvent::Stream(StreamEvent::MessageStart { .. })
        ));
        assert!(matches!(
            &coalesced[1],
            KernelEvent::Stream(StreamEvent::ThinkingDelta { thinking }) if thinking == "reasoning"
        ));
        assert!(matches!(
            &coalesced[3],
            KernelEvent::Stream(StreamEvent::MessageDelta { content_delta }) if content_delta == "complete response"
        ));
        assert!(matches!(
            &coalesced[4],
            KernelEvent::Stream(StreamEvent::MessageEnd { .. })
        ));
    }
}
