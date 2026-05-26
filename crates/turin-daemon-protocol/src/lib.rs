mod agents;
mod channels;
mod common;
mod envelopes;
mod handshake;
mod request;
mod schedule;
mod sessions;
mod tasks;
mod ui;
mod worklists;

pub use agents::*;
pub use channels::*;
pub use common::*;
pub use envelopes::*;
pub use handshake::*;
pub use request::*;
pub use schedule::*;
pub use sessions::*;
pub use tasks::*;
pub use ui::*;
pub use worklists::*;

pub(crate) fn default_enabled() -> bool {
    true
}

pub(crate) fn default_session_limit() -> usize {
    50
}

pub(crate) fn default_search_limit() -> usize {
    64
}

#[cfg(test)]
mod tests;
