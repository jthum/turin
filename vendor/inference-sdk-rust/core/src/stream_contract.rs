use crate::{InferenceEvent, StreamInvariantViolation};

/// Validates normalized event order for a single assistant response stream.
#[derive(Debug, Default)]
pub struct EventOrderValidator {
    message_started: bool,
    message_ended: bool,
    tool_call_started: bool,
}

impl EventOrderValidator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate one event against current stream state.
    pub fn validate_event(
        &mut self,
        event: &InferenceEvent,
    ) -> Result<(), StreamInvariantViolation> {
        if self.message_ended {
            return Err(StreamInvariantViolation::EventAfterMessageEnd);
        }

        match event {
            InferenceEvent::MessageStart { .. } => {
                if self.message_started {
                    return Err(StreamInvariantViolation::DuplicateMessageStart);
                }
                self.message_started = true;
            }
            InferenceEvent::MessageDelta { .. }
            | InferenceEvent::ThinkingDelta { .. }
            | InferenceEvent::ThinkingSignatureDelta { .. } => {
                if !self.message_started {
                    return Err(StreamInvariantViolation::MessageNotStarted);
                }
            }
            InferenceEvent::ToolCallStart { .. } => {
                if !self.message_started {
                    return Err(StreamInvariantViolation::MessageNotStarted);
                }
                self.tool_call_started = true;
            }
            InferenceEvent::ToolCallDelta { .. } => {
                if !self.message_started {
                    return Err(StreamInvariantViolation::MessageNotStarted);
                }
                if !self.tool_call_started {
                    return Err(StreamInvariantViolation::ToolCallDeltaBeforeStart);
                }
            }
            InferenceEvent::MessageEnd { .. } => {
                if !self.message_started {
                    return Err(StreamInvariantViolation::MessageEndBeforeStart);
                }
                self.message_ended = true;
                self.tool_call_started = false;
            }
        }

        Ok(())
    }

    /// Final validation after stream completion.
    pub fn finish(&self) -> Result<(), StreamInvariantViolation> {
        if !self.message_started {
            return Err(StreamInvariantViolation::MissingMessageStart);
        }
        if !self.message_ended {
            return Err(StreamInvariantViolation::MissingMessageEnd);
        }
        Ok(())
    }
}

/// Validate an entire event sequence.
pub fn validate_event_sequence(events: &[InferenceEvent]) -> Result<(), StreamInvariantViolation> {
    let mut validator = EventOrderValidator::new();
    for event in events {
        validator.validate_event(event)?;
    }
    validator.finish()
}
