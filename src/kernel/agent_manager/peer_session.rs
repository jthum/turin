use std::sync::Arc;

use anyhow::Result;
use tracing::{info, warn};

use crate::kernel::execution_host::ExecutionHost;
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::LinkedSessionCreate;

use super::peer_runtime::PeerRuntime;
use super::{
    AgentManager, ExecutionStatusSnapshot, LinkedSessionTarget, LiveSessionHistorySnapshot,
    RuntimeControl, SessionContextOverrides, SessionResetRequest,
};

#[derive(Debug, Clone, Default)]
pub(super) struct SessionBootstrap {
    pub(super) initial_session_id: Option<String>,
    pub(super) initial_state_selector: Option<StoreSelector>,
    pub(super) initial_default_store_selector: Option<StoreSelector>,
    pub(super) context: SessionContextOverrides,
    pub(super) link: Option<LinkedSessionCreate>,
}

impl PeerRuntime {
    pub(super) async fn start(
        manager: Arc<AgentManager>,
        agent_id: &str,
        slot_id: &str,
        control: Arc<RuntimeControl>,
        bootstrap: SessionBootstrap,
    ) -> Result<Self> {
        let mut host = fork_peer_kernel(&manager);
        if host.clients.is_empty() {
            host.init_clients()?;
        }

        let mut session = if let Some(session_id) = bootstrap.initial_session_id.as_deref() {
            host.resume_session_for_agent_with_context(
                agent_id,
                session_id,
                bootstrap.context.origin_id.clone(),
                bootstrap.context.inference.clone(),
            )
            .await?
        } else if let Some(link) = bootstrap.link {
            host.create_linked_session_for_agent_with_context(
                agent_id,
                bootstrap
                    .initial_state_selector
                    .ok_or_else(|| anyhow::anyhow!("Linked peer session requires a state store"))?,
                bootstrap.initial_default_store_selector,
                bootstrap.context.origin_id.clone(),
                bootstrap.context.inference.clone(),
                link,
            )
            .await?
        } else {
            host.create_session_for_agent_with_context(
                agent_id,
                bootstrap.initial_state_selector,
                bootstrap.initial_default_store_selector,
                bootstrap.context.origin_id.clone(),
                bootstrap.context.inference.clone(),
            )
            .await
        };
        session.runtime_slot_id = Some(slot_id.to_string());
        host.start_session(&mut session).await?;
        control.set_current_session(
            Some(host.session_reference(&session)),
            Some(session.event_tx.clone()),
            session_context_from_session(&session),
            Some(ExecutionStatusSnapshot::from_session(&session)),
            session.execution.conflict_policy,
            Some(LiveSessionHistorySnapshot::from_session(&session)),
        );

        Ok(Self {
            manager,
            control,
            host,
            session,
            agent_id: agent_id.to_string(),
            slot_id: slot_id.to_string(),
        })
    }

    pub(super) async fn activate_linked_session(
        &mut self,
        target: LinkedSessionTarget,
    ) -> Result<()> {
        let store = self
            .manager
            .store_manager
            .open(&target.state_selector)
            .await?;
        if let Some(linked) = store
            .find_linked_session(
                target.link.parent_session_id,
                &self.agent_id,
                &target.link.thread_key,
            )
            .await?
        {
            let public_id = uuid::Uuid::from_slice(&linked.public_id)?
                .simple()
                .to_string();
            let session_id = crate::kernel::session_refs::format_session_reference(
                &public_id,
                &target.state_selector,
            );
            if self
                .control
                .current_session_id()
                .as_deref()
                .is_some_and(|current| {
                    crate::kernel::session_refs::session_references_match(current, &session_id)
                })
            {
                return Ok(());
            }
            return self.restore_session(&session_id, target.context).await;
        }

        let session = self
            .host
            .create_linked_session_for_agent_with_context(
                &self.agent_id,
                target.state_selector,
                target.default_store_selector,
                target.context.origin_id,
                target.context.inference,
                target.link,
            )
            .await?;
        self.replace_session(session).await
    }

    pub(super) async fn shutdown(mut self) {
        if let Err(e) = self.host.end_session(&mut self.session).await {
            warn!(agent_id = %self.agent_id, error = %e, "Peer agent session end error");
        }
        self.control.clear_active_task();
        self.control.set_current_session(
            None,
            None,
            SessionContextOverrides::default(),
            None,
            crate::kernel::session::ExecutionConflictPolicy::Reject,
            None,
        );
        self.host.shutdown_mcp_clients().await;
        super::allocator::trim_after_peer_idle_if_enabled();
        info!(agent_id = %self.agent_id, "Peer runtime shut down");
    }

    pub(super) async fn reset_session_if_requested(&mut self) -> Result<bool> {
        let Some(request) = self.control.take_session_reset_request() else {
            return Ok(false);
        };

        match request {
            SessionResetRequest::Fresh(context) => self.reset_session(context).await?,
            SessionResetRequest::Resume {
                session_id,
                context,
            } => self.restore_session(&session_id, context).await?,
        }
        Ok(true)
    }

    async fn reset_session(&mut self, context: SessionContextOverrides) -> Result<()> {
        let context = effective_session_context(&self.session, context);
        let session = self
            .host
            .create_session_for_agent_with_context(
                &self.agent_id,
                Some(self.session.store_selector.clone()),
                self.session.default_store_selector.clone(),
                context.origin_id.clone(),
                context.inference.clone(),
            )
            .await;
        self.replace_session(session).await
    }

    async fn restore_session(
        &mut self,
        session_id: &str,
        context: SessionContextOverrides,
    ) -> Result<()> {
        let context = effective_session_context(&self.session, context);
        let session = self
            .host
            .resume_session_for_agent_with_context(
                &self.agent_id,
                session_id,
                context.origin_id.clone(),
                context.inference.clone(),
            )
            .await?;
        self.replace_session(session).await
    }

    async fn replace_session(
        &mut self,
        mut session: crate::kernel::session::SessionState,
    ) -> Result<()> {
        session.runtime_slot_id = Some(self.slot_id.clone());
        self.host.start_session(&mut session).await?;
        let previous_end = self.host.end_session(&mut self.session).await;
        self.control.set_current_session(
            Some(self.host.session_reference(&session)),
            Some(session.event_tx.clone()),
            session_context_from_session(&session),
            Some(ExecutionStatusSnapshot::from_session(&session)),
            session.execution.conflict_policy,
            Some(LiveSessionHistorySnapshot::from_session(&session)),
        );
        self.session = session;
        previous_end
    }
}

fn session_context_from_session(
    session: &crate::kernel::session::SessionState,
) -> SessionContextOverrides {
    SessionContextOverrides {
        origin_id: session.identity.origin_id().map(ToOwned::to_owned),
        inference: session.inference.clone(),
    }
}

fn effective_session_context(
    session: &crate::kernel::session::SessionState,
    requested: SessionContextOverrides,
) -> SessionContextOverrides {
    SessionContextOverrides {
        origin_id: requested
            .origin_id
            .or_else(|| session.identity.origin_id().map(ToOwned::to_owned)),
        inference: if requested.inference.is_empty() {
            session.inference.clone()
        } else {
            requested.inference
        },
    }
}

pub(super) fn fork_peer_kernel(manager: &Arc<AgentManager>) -> ExecutionHost {
    let shared = manager
        .shared_runtime()
        .expect("AgentManager shared runtime not bound");
    let inference = manager
        .shared_inference
        .lock()
        .expect("agent manager shared inference mutex poisoned")
        .clone();
    let (config, harness_manager) = manager.runtime_catalog_snapshot();

    ExecutionHost {
        config,
        json: shared.json,
        tool_registry: shared.tool_registry.clone(),
        store_manager: Arc::clone(&manager.store_manager),
        agent_manager: Arc::clone(manager),
        policy_manager: Arc::clone(&shared.policy_manager),
        governance_manager: Arc::clone(&shared.governance_manager),
        harness_manager,
        scheduler: manager.shared_scheduler(),
        persistence_locks: Arc::clone(&shared.persistence_locks),
        clients: inference.clients,
        embedding_provider: inference.embedding_provider,
        rust_harness_factories: None,
        script_harness_adapter: shared.script_harness_adapter.clone(),
        mcp_clients: Vec::new(),
    }
}
