use std::path::PathBuf;

mod channels;
mod doctor;
mod init;

pub(crate) use channels::{run_channels_list, run_channels_status, run_configure_channel};
pub(crate) use doctor::run_doctor;
pub(crate) use init::run_init;

#[derive(Debug, Clone)]
pub(crate) struct InitArgs {
    pub(crate) config: PathBuf,
    pub(crate) force: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct DoctorArgs {
    pub(crate) config: PathBuf,
}

#[derive(Debug, Clone)]
pub(crate) struct ChannelsListArgs {
    pub(crate) config: PathBuf,
}

#[derive(Debug, Clone)]
pub(crate) struct ConfigureChannelArgs {
    pub(crate) config: PathBuf,
    pub(crate) kind: String,
    pub(crate) channel_id: Option<String>,
    pub(crate) agent_id: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ChannelsStatusArgs {
    pub(crate) config: PathBuf,
}
