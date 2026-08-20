use serde::{Deserialize, Serialize};

pub const DAEMON_PROTOCOL_VERSION: u32 = 1;
pub const DAEMON_TRANSPORT_UNIX: &str = "unix";
pub const DAEMON_TRANSPORT_NAMED_PIPE: &str = "named_pipe";
pub const DAEMON_WIRE_FORMAT_NDJSON: &str = "ndjson";

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct DaemonCapabilities {
    pub runtime_snapshot_v1: bool,
    pub scoped_event_snapshots: bool,
    pub lag_resnapshot: bool,
    pub watcher_rescan_failed_events: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct DaemonHandshake {
    pub pong: bool,
    pub version: String,
    pub protocol_version: u32,
    pub transport: String,
    pub wire_format: String,
    pub capabilities: DaemonCapabilities,
}
