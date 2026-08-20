use serde::Serialize;
use turin_daemon_protocol::DaemonHandshake;

use crate::client::ConnectionKind;
use crate::models::DaemonStatus;

#[derive(Debug, Clone, Serialize)]
pub struct ControlHealth {
    pub connection_kind: ConnectionKind,
    pub target: String,
    pub ready: bool,
    pub version: String,
    pub protocol_version: u32,
    pub transport: String,
    pub wire_format: String,
    pub issue_count: usize,
    pub agent_count: usize,
    pub harness_count: usize,
    pub running_agent_count: usize,
    pub active_task_count: usize,
    pub queued_task_count: usize,
    pub awaiting_result_count: usize,
}

pub(crate) fn build_health(
    connection_kind: ConnectionKind,
    target: String,
    handshake: DaemonHandshake,
    status: &DaemonStatus,
) -> ControlHealth {
    let running_agent_count = status
        .agent_runtimes
        .iter()
        .filter(|runtime| runtime.running)
        .count();
    let active_task_count = status
        .agent_runtimes
        .iter()
        .map(|runtime| runtime.active_tasks)
        .sum();
    let queued_task_count = status
        .agent_runtimes
        .iter()
        .map(|runtime| runtime.queued_tasks)
        .sum();
    let awaiting_result_count = status
        .agent_runtimes
        .iter()
        .map(|runtime| runtime.awaiting_results)
        .sum();
    ControlHealth {
        connection_kind,
        target: match connection_kind {
            ConnectionKind::Local => status.endpoint.clone(),
            ConnectionKind::Remote => target,
        },
        ready: status.registry.issues.is_empty(),
        version: handshake.version,
        protocol_version: handshake.protocol_version,
        transport: handshake.transport,
        wire_format: handshake.wire_format,
        issue_count: status.registry.issues.len(),
        agent_count: status.agent_runtimes.len(),
        harness_count: status.registry.shared_harnesses.len(),
        running_agent_count,
        active_task_count,
        queued_task_count,
        awaiting_result_count,
    }
}
