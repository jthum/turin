mod effective;
mod files;
mod scan;
mod snapshot;
mod types;

pub(crate) use files::{read_agent_file, read_channel_file, write_agent_file, write_channel_file};
pub use scan::scan_registry;
pub use snapshot::snapshot;
pub(crate) use types::{AgentFileConfig, ChannelFileConfig};
pub use types::{
    AgentSummary, ChannelSummary, DiscoveredAgent, DiscoveredChannel, HarnessKind, RegistryIssue,
    RegistryLoad, RegistrySnapshot, SharedHarness, SharedHarnessSummary,
};

pub use effective::build_effective_config;

#[cfg(test)]
#[path = "tests/registry.rs"]
mod tests;
