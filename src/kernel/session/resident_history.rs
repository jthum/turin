use std::ops::{Deref, Index};

use crate::inference::provider::InferenceMessage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoryOrigin {
    pub turn_id: i64,
    pub turn_index: u32,
}

#[derive(Debug, Clone, Default)]
pub struct ResidentHistory {
    messages: Vec<InferenceMessage>,
    origins: Vec<Option<HistoryOrigin>>,
    has_prior_history: bool,
}

impl ResidentHistory {
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    pub fn messages(&self) -> &[InferenceMessage] {
        &self.messages
    }

    pub fn messages_mut(&mut self) -> &mut [InferenceMessage] {
        &mut self.messages
    }

    pub fn iter(&self) -> std::slice::Iter<'_, InferenceMessage> {
        self.messages.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, InferenceMessage> {
        self.messages.iter_mut()
    }

    pub fn to_messages(&self) -> Vec<InferenceMessage> {
        self.messages.clone()
    }

    pub fn into_messages(self) -> Vec<InferenceMessage> {
        self.messages
    }

    pub fn push(&mut self, message: InferenceMessage) {
        self.push_with_origin(message, None);
    }

    pub fn push_with_origin(&mut self, message: InferenceMessage, origin: Option<HistoryOrigin>) {
        self.messages.push(message);
        self.origins.push(origin);
    }

    pub fn replace(
        &mut self,
        entries: Vec<(InferenceMessage, Option<HistoryOrigin>)>,
        has_prior_history: bool,
    ) {
        let (messages, origins) = entries.into_iter().unzip();
        self.messages = messages;
        self.origins = origins;
        self.has_prior_history = has_prior_history;
    }

    pub fn replace_untracked(&mut self, messages: Vec<InferenceMessage>, has_prior_history: bool) {
        self.origins = vec![None; messages.len()];
        self.messages = messages;
        self.has_prior_history = has_prior_history;
    }

    pub fn drain_prefix(&mut self, count: usize) {
        self.messages.drain(0..count);
        self.origins.drain(0..count);
        self.messages.shrink_to_fit();
        self.origins.shrink_to_fit();
        self.has_prior_history = true;
    }

    pub fn clear(&mut self) {
        self.messages.clear();
        self.origins.clear();
        self.has_prior_history = false;
    }

    pub fn has_prior_history(&self) -> bool {
        self.has_prior_history
    }

    pub fn origin(&self, index: usize) -> Option<HistoryOrigin> {
        self.origins.get(index).copied().flatten()
    }

    pub fn untracked_suffix(&self) -> &[InferenceMessage] {
        let start = self
            .origins
            .iter()
            .rposition(Option::is_some)
            .map_or(0, |index| index + 1);
        &self.messages[start..]
    }

    pub fn suffix_after_turn(&self, turn_id: i64, turn_index: u32) -> &[InferenceMessage] {
        if let Some(index) = self.index_after_turn(turn_id) {
            return &self.messages[index..];
        }
        let starts_after_checkpoint = self
            .origins
            .iter()
            .flatten()
            .all(|origin| origin.turn_index > turn_index);
        if starts_after_checkpoint {
            &self.messages
        } else {
            &[]
        }
    }

    pub fn into_suffix_after_turn(
        mut self,
        turn_id: i64,
        turn_index: u32,
    ) -> Vec<InferenceMessage> {
        if let Some(index) = self.index_after_turn(turn_id) {
            self.messages.drain(..index);
            return self.messages;
        }
        if self
            .origins
            .iter()
            .flatten()
            .all(|origin| origin.turn_index > turn_index)
        {
            self.messages
        } else {
            Vec::new()
        }
    }

    pub fn index_after_turn(&self, turn_id: i64) -> Option<usize> {
        self.origins
            .iter()
            .rposition(|origin| origin.is_some_and(|origin| origin.turn_id == turn_id))
            .map(|index| index + 1)
    }
}

impl Index<usize> for ResidentHistory {
    type Output = InferenceMessage;

    fn index(&self, index: usize) -> &Self::Output {
        &self.messages[index]
    }
}

impl Deref for ResidentHistory {
    type Target = [InferenceMessage];

    fn deref(&self) -> &Self::Target {
        &self.messages
    }
}

impl<'a> IntoIterator for &'a ResidentHistory {
    type Item = &'a InferenceMessage;
    type IntoIter = std::slice::Iter<'a, InferenceMessage>;

    fn into_iter(self) -> Self::IntoIter {
        self.messages.iter()
    }
}

impl<'a> IntoIterator for &'a mut ResidentHistory {
    type Item = &'a mut InferenceMessage;
    type IntoIter = std::slice::IterMut<'a, InferenceMessage>;

    fn into_iter(self) -> Self::IntoIter {
        self.messages.iter_mut()
    }
}
