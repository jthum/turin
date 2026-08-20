use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::Result;

use super::{AgentManager, AgentRuntimeHandle, RuntimeSlotKey, SessionContextOverrides};
use crate::kernel::session_refs::session_references_match;
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::LinkedSessionCreate;

impl AgentManager {
    pub(super) async fn ensure_runtime(
        self: &Arc<Self>,
        agent_id: &str,
    ) -> Result<Arc<AgentRuntimeHandle>> {
        self.ensure_runtime_slot(RuntimeSlotKey::default_for(agent_id))
            .await
    }

    pub(super) async fn ensure_runtime_slot(
        self: &Arc<Self>,
        runtime_key: RuntimeSlotKey,
    ) -> Result<Arc<AgentRuntimeHandle>> {
        self.ensure_runtime_slot_in_store(
            runtime_key,
            None,
            None,
            SessionContextOverrides::default(),
        )
        .await
    }

    pub(super) async fn ensure_runtime_slot_in_store(
        self: &Arc<Self>,
        runtime_key: RuntimeSlotKey,
        initial_state_selector: Option<StoreSelector>,
        initial_default_store_selector: Option<StoreSelector>,
        session_context: SessionContextOverrides,
    ) -> Result<Arc<AgentRuntimeHandle>> {
        let _catalog_guard = self.catalog_gate(&runtime_key.agent_id).read_owned().await;
        if self.shutting_down.load(Ordering::Acquire) {
            anyhow::bail!("Agent manager is shutting down");
        }
        {
            let runtimes = self.runtimes.read().await;
            if let Some(handle) = runtimes.get(&runtime_key)
                && handle.is_running()
            {
                return Ok(Arc::clone(handle));
            }
        }

        self.ensure_runtime_with_write_lock(
            runtime_key,
            initial_state_selector,
            initial_default_store_selector,
            session_context,
        )
        .await
    }

    pub(super) async fn ensure_runtime_slot_resumed(
        self: &Arc<Self>,
        runtime_key: RuntimeSlotKey,
        session_id: String,
        session_context: SessionContextOverrides,
    ) -> Result<Arc<AgentRuntimeHandle>> {
        let _catalog_guard = self.catalog_gate(&runtime_key.agent_id).read_owned().await;
        if self.shutting_down.load(Ordering::Acquire) {
            anyhow::bail!("Agent manager is shutting down");
        }
        {
            let runtimes = self.runtimes.read().await;
            if let Some(handle) = runtimes.get(&runtime_key)
                && handle.is_running()
            {
                if handle
                    .control
                    .current_session_id()
                    .as_deref()
                    .is_some_and(|current| session_references_match(current, &session_id))
                {
                    return Ok(Arc::clone(handle));
                }
                handle
                    .control
                    .request_session_resume(session_id, session_context.clone());
                handle.notify.notify_one();
                return Ok(Arc::clone(handle));
            }
        }

        self.ensure_runtime_with_write_lock_and_resume(
            runtime_key,
            Some(session_id),
            session_context,
        )
        .await
    }

    pub(super) async fn ensure_runtime_slot_for_linked(
        self: &Arc<Self>,
        runtime_key: RuntimeSlotKey,
        initial_session_id: Option<&str>,
        state_selector: StoreSelector,
        default_store_selector: Option<StoreSelector>,
        session_context: SessionContextOverrides,
        link: LinkedSessionCreate,
    ) -> Result<Arc<AgentRuntimeHandle>> {
        let _catalog_guard = self.catalog_gate(&runtime_key.agent_id).read_owned().await;
        let mut runtimes = self.runtimes.write().await;
        if self.shutting_down.load(Ordering::Acquire) {
            anyhow::bail!("Agent manager is shutting down");
        }
        if let Some(handle) = runtimes.get(&runtime_key)
            && handle.is_running()
        {
            return Ok(Arc::clone(handle));
        }

        let (initial_state_selector, initial_link) = if initial_session_id.is_some() {
            (None, None)
        } else {
            (Some(state_selector), Some(link))
        };
        let handle = Arc::new(
            self.start_agent(
                &runtime_key,
                initial_session_id,
                initial_state_selector,
                default_store_selector,
                session_context,
                initial_link,
            )
            .await?,
        );
        runtimes.insert(runtime_key, Arc::clone(&handle));
        Ok(handle)
    }

    async fn ensure_runtime_with_write_lock(
        self: &Arc<Self>,
        runtime_key: RuntimeSlotKey,
        initial_state_selector: Option<StoreSelector>,
        initial_default_store_selector: Option<StoreSelector>,
        session_context: SessionContextOverrides,
    ) -> Result<Arc<AgentRuntimeHandle>> {
        let mut runtimes = self.runtimes.write().await;
        if self.shutting_down.load(Ordering::Acquire) {
            anyhow::bail!("Agent manager is shutting down");
        }
        if let Some(handle) = runtimes.get(&runtime_key)
            && handle.is_running()
        {
            return Ok(Arc::clone(handle));
        }

        let handle = Arc::new(
            self.start_agent(
                &runtime_key,
                None,
                initial_state_selector,
                initial_default_store_selector,
                session_context,
                None,
            )
            .await?,
        );
        runtimes.insert(runtime_key, Arc::clone(&handle));
        Ok(handle)
    }

    async fn ensure_runtime_with_write_lock_and_resume(
        self: &Arc<Self>,
        runtime_key: RuntimeSlotKey,
        initial_session_id: Option<String>,
        session_context: SessionContextOverrides,
    ) -> Result<Arc<AgentRuntimeHandle>> {
        let mut runtimes = self.runtimes.write().await;
        if self.shutting_down.load(Ordering::Acquire) {
            anyhow::bail!("Agent manager is shutting down");
        }
        if let Some(handle) = runtimes.get(&runtime_key)
            && handle.is_running()
        {
            if let Some(session_id) = initial_session_id {
                handle
                    .control
                    .request_session_resume(session_id, session_context.clone());
                handle.notify.notify_one();
            }
            return Ok(Arc::clone(handle));
        }

        let handle = Arc::new(
            self.start_agent(
                &runtime_key,
                initial_session_id.as_deref(),
                None,
                None,
                session_context,
                None,
            )
            .await?,
        );
        runtimes.insert(runtime_key, Arc::clone(&handle));
        Ok(handle)
    }
}
