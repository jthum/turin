use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::{Instant, MissedTickBehavior};
use turin_channel_core::{ConversationBinding, InboundEvent, OutboundMessage};
use turin_daemon_protocol::{RuntimeEventsSubscribeParams, WaitTaskParams};

use crate::stream::{
    WorkerStreamConfig, attach_final_thinking, preview_char_count, preview_thinking,
    should_flush_preview, should_subscribe_to_session_events,
};
use crate::{
    ChannelDriver, ChannelProgressUpdate, ChannelRunner, ChannelStreamMode, EventAccessDecision,
    TaskSnapshot, serialize_binding_key, task_to_outbound,
};

#[derive(Debug, Clone)]
struct QueuedInboundEvent {
    conversation_id: String,
    event: InboundEvent,
    reset_requested: bool,
    stream: WorkerStreamConfig,
}

#[derive(Debug)]
enum WorkerAction {
    Progress {
        event: InboundEvent,
        update: ChannelProgressUpdate,
    },
    Completed {
        conversation_id: String,
        event: InboundEvent,
        outbound: OutboundMessage,
    },
}

struct DriverDispatchState {
    action_tx: mpsc::UnboundedSender<WorkerAction>,
    active_conversations: HashSet<String>,
    queued_events: HashMap<String, VecDeque<QueuedInboundEvent>>,
}

impl DriverDispatchState {
    fn new(action_tx: mpsc::UnboundedSender<WorkerAction>) -> Self {
        Self {
            action_tx,
            active_conversations: HashSet::new(),
            queued_events: HashMap::new(),
        }
    }
}

struct WorkerTaskContext<'a> {
    event: &'a InboundEvent,
    binding: &'a ConversationBinding,
    submitted: &'a TaskSnapshot,
}

impl ChannelRunner {
    pub async fn run_driver<D: ChannelDriver + Send>(
        &self,
        agent_id: &str,
        driver: &mut D,
        timeout_ms: Option<u64>,
    ) -> Result<()> {
        let (action_tx, mut action_rx) = mpsc::unbounded_channel::<WorkerAction>();
        let mut dispatch_state = DriverDispatchState::new(action_tx);
        let mut driver_closed = false;

        let run_result = async {
            loop {
                while let Ok(action) = action_rx.try_recv() {
                    self.handle_worker_action(
                        driver,
                        timeout_ms,
                        agent_id,
                        action,
                        &mut dispatch_state,
                    )
                    .await?;
                }

                if driver_closed && dispatch_state.active_conversations.is_empty() {
                    break;
                }

                if driver_closed {
                    match action_rx.recv().await {
                        Some(action) => {
                            self.handle_worker_action(
                                driver,
                                timeout_ms,
                                agent_id,
                                action,
                                &mut dispatch_state,
                            )
                            .await?;
                        }
                        None => break,
                    }
                    continue;
                }

                if dispatch_state.active_conversations.is_empty() {
                    match driver.next_event().await? {
                        Some(event) => {
                            self.handle_inbound_event(
                                agent_id,
                                driver,
                                event,
                                timeout_ms,
                                &mut dispatch_state,
                            )
                            .await?;
                        }
                        None => driver_closed = true,
                    }
                    continue;
                }

                enum DriverLoopOutcome {
                    Event(Result<Option<InboundEvent>>),
                    Action(Option<WorkerAction>),
                }

                let outcome = {
                    let next_event = driver.next_event();
                    tokio::pin!(next_event);
                    tokio::select! {
                        event_result = &mut next_event => DriverLoopOutcome::Event(event_result),
                        maybe_action = action_rx.recv() => DriverLoopOutcome::Action(maybe_action),
                    }
                };

                match outcome {
                    DriverLoopOutcome::Event(event_result) => match event_result? {
                        Some(event) => {
                            self.handle_inbound_event(
                                agent_id,
                                driver,
                                event,
                                timeout_ms,
                                &mut dispatch_state,
                            )
                            .await?;
                        }
                        None => driver_closed = true,
                    },
                    DriverLoopOutcome::Action(maybe_action) => match maybe_action {
                        Some(action) => {
                            self.handle_worker_action(
                                driver,
                                timeout_ms,
                                agent_id,
                                action,
                                &mut dispatch_state,
                            )
                            .await?;
                        }
                        None => break,
                    },
                }
            }
            Result::<()>::Ok(())
        }
        .await;

        let shutdown_result = driver.shutdown().await;
        run_result?;
        shutdown_result
    }

    async fn handle_inbound_event<D: ChannelDriver + Send>(
        &self,
        agent_id: &str,
        driver: &mut D,
        event: InboundEvent,
        timeout_ms: Option<u64>,
        dispatch_state: &mut DriverDispatchState,
    ) -> Result<()> {
        match self.authorize_event(driver, &event).await? {
            EventAccessDecision::Allow => {}
            EventAccessDecision::Ignore => return Ok(()),
            EventAccessDecision::Pending { notify } => {
                if notify {
                    driver
                        .send(&event.conversation, pending_approval_message())
                        .await?;
                }
                return Ok(());
            }
        }

        let reset_requested = event
            .metadata
            .get("reset_session")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let queued = QueuedInboundEvent {
            conversation_id: serialize_binding_key(&event.conversation)?,
            event,
            reset_requested,
            stream: WorkerStreamConfig {
                mode: driver.stream_mode(),
                stream_thinking: driver.stream_thinking(),
                persist_thinking: driver.persist_thinking(),
            },
        };

        if dispatch_state
            .active_conversations
            .contains(&queued.conversation_id)
        {
            dispatch_state
                .queued_events
                .entry(queued.conversation_id.clone())
                .or_default()
                .push_back(queued);
            return Ok(());
        }

        self.spawn_worker(
            agent_id,
            queued,
            timeout_ms,
            &dispatch_state.action_tx,
            &mut dispatch_state.active_conversations,
        );
        Ok(())
    }

    async fn handle_worker_action<D: ChannelDriver + Send>(
        &self,
        driver: &mut D,
        timeout_ms: Option<u64>,
        agent_id: &str,
        action: WorkerAction,
        dispatch_state: &mut DriverDispatchState,
    ) -> Result<()> {
        match action {
            WorkerAction::Progress { event, update } => {
                let _ = driver.send_progress(&event, update).await;
            }
            WorkerAction::Completed {
                conversation_id,
                event,
                outbound,
            } => {
                let outbound = driver.enrich_outbound_for_event(&event, outbound);
                driver.send(&event.conversation, outbound).await?;
                dispatch_state.active_conversations.remove(&conversation_id);
                if let Some(queue) = dispatch_state.queued_events.get_mut(&conversation_id) {
                    if let Some(next) = queue.pop_front() {
                        self.spawn_worker(
                            agent_id,
                            next,
                            timeout_ms,
                            &dispatch_state.action_tx,
                            &mut dispatch_state.active_conversations,
                        );
                    }
                    if queue.is_empty() {
                        dispatch_state.queued_events.remove(&conversation_id);
                    }
                }
            }
        }
        Ok(())
    }

    fn spawn_worker(
        &self,
        agent_id: &str,
        queued: QueuedInboundEvent,
        timeout_ms: Option<u64>,
        action_tx: &mpsc::UnboundedSender<WorkerAction>,
        active_conversations: &mut HashSet<String>,
    ) {
        active_conversations.insert(queued.conversation_id.clone());
        let runner = self.clone();
        let agent_id = agent_id.to_string();
        let action_tx = action_tx.clone();
        tokio::spawn(async move {
            let event = queued.event.clone();
            let outbound = match runner
                .handle_event_with_progress(
                    &agent_id,
                    &queued.event,
                    queued.reset_requested,
                    timeout_ms,
                    queued.stream,
                    &action_tx,
                )
                .await
            {
                Ok(message) => message,
                Err(err) => OutboundMessage::text(format!("Turin error: {}", err)),
            };
            let _ = action_tx.send(WorkerAction::Completed {
                conversation_id: queued.conversation_id,
                event,
                outbound,
            });
        });
    }

    async fn handle_event_with_progress(
        &self,
        agent_id: &str,
        event: &InboundEvent,
        reset_requested: bool,
        timeout_ms: Option<u64>,
        stream: WorkerStreamConfig,
        action_tx: &mpsc::UnboundedSender<WorkerAction>,
    ) -> Result<OutboundMessage> {
        let binding = self
            .ensure_session(agent_id, &event.conversation, reset_requested)
            .await?;
        let session_events = if should_subscribe_to_session_events(&stream) {
            self.daemon
                .subscribe_managed(RuntimeEventsSubscribeParams {
                    agent_id: None,
                    session_id: Some(binding.session_id.clone()),
                    slot_id: Some(binding.slot_id.clone()),
                })
                .await
                .ok()
        } else {
            None
        };
        let submitted = self.submit_with_binding(&binding, event).await?;
        let task_ctx = WorkerTaskContext {
            event,
            binding: &binding,
            submitted: &submitted,
        };
        let (task, final_thinking) = self
            .wait_for_task_with_progress(&task_ctx, session_events, timeout_ms, &stream, action_tx)
            .await?;
        let outbound = task_to_outbound(&task);
        Ok(attach_final_thinking(outbound, final_thinking))
    }

    async fn wait_for_task_with_progress(
        &self,
        task_ctx: &WorkerTaskContext<'_>,
        mut session_events: Option<turin_daemon_client::ManagedEventStream>,
        timeout_ms: Option<u64>,
        stream: &WorkerStreamConfig,
        action_tx: &mpsc::UnboundedSender<WorkerAction>,
    ) -> Result<(TaskSnapshot, Option<String>)> {
        let capture_thinking = stream.stream_thinking || stream.persist_thinking;
        if stream.mode == ChannelStreamMode::Off {
            let task = self
                .daemon
                .request_ok(
                    None,
                    turin_daemon_protocol::DaemonRequest::TaskWait(WaitTaskParams {
                        request_id: task_ctx.submitted.request_id.clone(),
                        timeout_ms,
                    }),
                )
                .await?;
            return Ok((task, None));
        }

        if stream.mode.sends_typing() {
            self.emit_worker_progress(action_tx, task_ctx.event, ChannelProgressUpdate::Typing);
        }

        let wait_task = self.daemon.request_ok(
            None,
            turin_daemon_protocol::DaemonRequest::TaskWait(WaitTaskParams {
                request_id: task_ctx.submitted.request_id.clone(),
                timeout_ms,
            }),
        );
        tokio::pin!(wait_task);

        let mut typing_tick = tokio::time::interval_at(
            Instant::now() + Duration::from_secs(4),
            Duration::from_secs(4),
        );
        typing_tick.set_missed_tick_behavior(MissedTickBehavior::Delay);

        let mut task_started = false;
        let mut text_preview = String::new();
        let mut thinking_preview = String::new();
        let mut last_flushed_chars = 0usize;
        let mut last_flush_at = Instant::now();

        loop {
            tokio::select! {
                result = &mut wait_task => {
                    if stream.mode.streams_text()
                        && preview_char_count(&text_preview, stream.stream_thinking.then_some(thinking_preview.as_str())) > last_flushed_chars
                    {
                        self.emit_worker_progress(
                            action_tx,
                            task_ctx.event,
                            ChannelProgressUpdate::StreamingPreview {
                                text: text_preview.clone(),
                                thinking: preview_thinking(stream.stream_thinking, &thinking_preview),
                            },
                        );
                    }
                    let task = result?;
                    let final_thinking = preview_thinking(capture_thinking, &thinking_preview);
                    return Ok((task, final_thinking));
                }
                _ = typing_tick.tick(), if stream.mode.sends_typing() => {
                    self.emit_worker_progress(action_tx, task_ctx.event, ChannelProgressUpdate::Typing);
                }
                event_result = next_managed_event(session_events.as_mut()), if session_events.is_some() => {
                    let Ok(kernel_event) = event_result else {
                        session_events = None;
                        continue;
                    };
                    if kernel_event.data.get("session_id").and_then(|value| value.as_str()) != Some(task_ctx.binding.session_id.as_str()) {
                        continue;
                    }

                    match kernel_event.event.as_str() {
                        "task_start" if kernel_event.data.get("trace_id").and_then(|value| value.as_str()) == Some(task_ctx.submitted.trace_id.as_str()) => {
                            task_started = true;
                        }
                        "message_delta" if task_started => {
                            if let Some(delta) = kernel_event.data.get("content_delta").and_then(|value| value.as_str()) {
                                text_preview.push_str(delta);
                            }
                            if should_flush_preview(
                                stream.mode,
                                &text_preview,
                                stream.stream_thinking.then_some(thinking_preview.as_str()),
                                last_flushed_chars,
                                last_flush_at,
                            ) {
                                self.emit_worker_progress(
                                    action_tx,
                                    task_ctx.event,
                                    ChannelProgressUpdate::StreamingPreview {
                                        text: text_preview.clone(),
                                        thinking: preview_thinking(stream.stream_thinking, &thinking_preview),
                                    },
                                );
                                last_flushed_chars = preview_char_count(
                                    &text_preview,
                                    stream.stream_thinking.then_some(thinking_preview.as_str()),
                                );
                                last_flush_at = Instant::now();
                            }
                        }
                        "thinking_delta" if task_started && capture_thinking => {
                            if let Some(delta) = kernel_event.data.get("thinking").and_then(|value| value.as_str()) {
                                thinking_preview.push_str(delta);
                            }
                            if should_flush_preview(
                                stream.mode,
                                &text_preview,
                                stream.stream_thinking.then_some(thinking_preview.as_str()),
                                last_flushed_chars,
                                last_flush_at,
                            ) {
                                self.emit_worker_progress(
                                    action_tx,
                                    task_ctx.event,
                                    ChannelProgressUpdate::StreamingPreview {
                                        text: text_preview.clone(),
                                        thinking: preview_thinking(stream.stream_thinking, &thinking_preview),
                                    },
                                );
                                last_flushed_chars = preview_char_count(
                                    &text_preview,
                                    stream.stream_thinking.then_some(thinking_preview.as_str()),
                                );
                                last_flush_at = Instant::now();
                            }
                        }
                        "message_end"
                            if task_started
                                && preview_char_count(
                                    &text_preview,
                                    stream.stream_thinking.then_some(thinking_preview.as_str()),
                                ) > last_flushed_chars =>
                        {
                            self.emit_worker_progress(
                                action_tx,
                                task_ctx.event,
                                ChannelProgressUpdate::StreamingPreview {
                                    text: text_preview.clone(),
                                    thinking: preview_thinking(
                                        stream.stream_thinking,
                                        &thinking_preview,
                                    ),
                                },
                            );
                            last_flushed_chars = preview_char_count(
                                &text_preview,
                                stream.stream_thinking.then_some(thinking_preview.as_str()),
                            );
                            last_flush_at = Instant::now();
                        }
                        "task_complete" if kernel_event.data.get("trace_id").and_then(|value| value.as_str()) == Some(task_ctx.submitted.trace_id.as_str()) => {
                            task_started = false;
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    fn emit_worker_progress(
        &self,
        action_tx: &mpsc::UnboundedSender<WorkerAction>,
        event: &InboundEvent,
        update: ChannelProgressUpdate,
    ) {
        let _ = action_tx.send(WorkerAction::Progress {
            event: event.clone(),
            update,
        });
    }
}

fn pending_approval_message() -> OutboundMessage {
    OutboundMessage::text(
        "This conversation is pending approval. Turin will not respond here until the operator approves this room.",
    )
}

async fn next_managed_event(
    stream: Option<&mut turin_daemon_client::ManagedEventStream>,
) -> Result<turin_daemon_protocol::EventEnvelope> {
    let stream = stream.context("managed event stream missing")?;
    stream.next_event().await
}
