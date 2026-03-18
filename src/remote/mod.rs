mod config;
mod routes;
mod server;

pub use config::RemoteServeOptions;
pub use server::{RunningRemoteServer, serve, start};
