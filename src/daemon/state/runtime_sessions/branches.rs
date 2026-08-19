use anyhow::Result;
use tracing::{debug, info, instrument};
use uuid::Uuid;

use super::{branch_detail_from_row, graph_turn_preview, session_summary_from_row_and_selector};
use crate::daemon::state::helpers;
use crate::daemon::state::{
    DaemonState, SessionBranchDetail, SessionGraphDetail, SessionGraphTurnDetail,
};
use crate::kernel::session_refs::describe_store_selector;

impl DaemonState {
    pub async fn list_session_branches(
        &self,
        session_id: &str,
    ) -> Result<Option<Vec<SessionBranchDetail>>> {
        let Some((store_selector, store, row)) = self.resolve_persisted_session(session_id).await?
        else {
            return Ok(None);
        };
        debug!(
            store = %describe_store_selector(&store_selector),
            "Listing session branches"
        );
        let branches = store
            .list_branch_heads(row.id)
            .await?
            .into_iter()
            .map(branch_detail_from_row)
            .collect();
        Ok(Some(branches))
    }

    #[instrument(skip(self), fields(session_id = %session_id))]
    pub async fn get_session_graph(&self, session_id: &str) -> Result<Option<SessionGraphDetail>> {
        let Some((store_selector, store, row)) = self.resolve_persisted_session(session_id).await?
        else {
            return Ok(None);
        };
        let turns = store
            .list_session_graph_turns(row.id)
            .await?
            .into_iter()
            .map(|graph_turn| SessionGraphTurnDetail {
                turn_id: graph_turn.turn.id,
                turn_public_id: helpers::format_uuid_bytes_simple(&graph_turn.turn.public_id),
                parent_turn_id: graph_turn.turn.parent_turn_id,
                turn_index: graph_turn.turn.branch_depth,
                message_count: graph_turn.message_count,
                tool_execution_count: graph_turn.tool_execution_count,
                preview: graph_turn.preview.as_deref().and_then(graph_turn_preview),
                created_at: graph_turn.turn.created_at,
            })
            .collect();
        let branches = store
            .list_branch_heads(row.id)
            .await?
            .into_iter()
            .map(branch_detail_from_row)
            .collect();
        Ok(Some(SessionGraphDetail {
            session: session_summary_from_row_and_selector(&row, &store_selector),
            turns,
            branches,
        }))
    }

    #[instrument(skip(self), fields(session_id = %session_id, source_turn_id = source_turn_id))]
    pub async fn list_session_branch_siblings(
        &self,
        session_id: &str,
        source_turn_id: i64,
    ) -> Result<Option<Vec<SessionBranchDetail>>> {
        let Some((store_selector, store, row)) = self.resolve_persisted_session(session_id).await?
        else {
            return Ok(None);
        };
        debug!(
            store = %describe_store_selector(&store_selector),
            "Listing session branch siblings"
        );
        let branches = store
            .list_branch_heads_from_source_turn(row.id, source_turn_id)
            .await?
            .into_iter()
            .map(branch_detail_from_row)
            .collect();
        Ok(Some(branches))
    }

    #[instrument(
        skip(self),
        fields(
            session_id = %session_id,
            branch = %name,
            slot_id = ?slot_id,
            from_turn_index = ?from_turn_index,
            activate = activate
        )
    )]
    pub async fn create_session_branch(
        &self,
        session_id: &str,
        name: &str,
        slot_id: Option<&str>,
        from_turn_index: Option<u32>,
        activate: bool,
    ) -> Result<Option<SessionBranchDetail>> {
        let Some((store_selector, store, row)) = self.resolve_persisted_session(session_id).await?
        else {
            return Ok(None);
        };
        let live_snapshot = if activate {
            self.resolve_live_branch_target(session_id, &row.public_id, slot_id, "activate branch")
                .await?
        } else {
            None
        };
        debug!(
            store = %describe_store_selector(&store_selector),
            live_session = live_snapshot.is_some(),
            "Creating session branch"
        );
        let branch = store
            .create_branch_head_from_turn_index(row.id, name, from_turn_index, activate)
            .await?;
        if activate && let Some(live_snapshot) = live_snapshot.as_ref() {
            self.kernel
                .agent_manager()
                .reload_session(session_id, Some(&live_snapshot.slot_id))
                .await?;
        }
        info!(
            session_id = %session_id,
            store = %describe_store_selector(&store_selector),
            branch = %branch.name,
            activate = activate,
            reloaded_live_session = live_snapshot.is_some(),
            "Created session branch"
        );
        Ok(Some(branch_detail_from_row(branch)))
    }

    #[instrument(
        skip(self),
        fields(
            session_id = %session_id,
            branch = %name,
            slot_id = ?slot_id,
            from_turn_id = from_turn_id,
            activate = activate
        )
    )]
    pub async fn create_session_branch_from_turn_id(
        &self,
        session_id: &str,
        name: &str,
        slot_id: Option<&str>,
        from_turn_id: i64,
        activate: bool,
    ) -> Result<Option<SessionBranchDetail>> {
        let Some((store_selector, store, row)) = self.resolve_persisted_session(session_id).await?
        else {
            return Ok(None);
        };
        let live_snapshot = if activate {
            self.resolve_live_branch_target(session_id, &row.public_id, slot_id, "activate branch")
                .await?
        } else {
            None
        };
        let branch = store
            .create_branch_head_from_turn_id(row.id, name, from_turn_id, activate)
            .await?;
        if activate && let Some(live_snapshot) = live_snapshot.as_ref() {
            self.kernel
                .agent_manager()
                .reload_session(session_id, Some(&live_snapshot.slot_id))
                .await?;
        }
        info!(
            session_id = %session_id,
            store = %describe_store_selector(&store_selector),
            branch = %branch.name,
            from_turn_id = from_turn_id,
            activate = activate,
            reloaded_live_session = live_snapshot.is_some(),
            "Created session branch from exact turn"
        );
        Ok(Some(branch_detail_from_row(branch)))
    }

    #[instrument(skip(self), fields(session_id = %session_id, branch = %branch, slot_id = ?slot_id))]
    pub async fn checkout_session_branch(
        &self,
        session_id: &str,
        branch: &str,
        slot_id: Option<&str>,
    ) -> Result<Option<SessionBranchDetail>> {
        let Some((store_selector, store, row)) = self.resolve_persisted_session(session_id).await?
        else {
            return Ok(None);
        };
        let live_snapshot = self
            .resolve_live_branch_target(session_id, &row.public_id, slot_id, "check out branch")
            .await?;
        debug!(
            store = %describe_store_selector(&store_selector),
            live_session = live_snapshot.is_some(),
            "Checking out session branch"
        );
        let branch = if let Ok(branch_id) = Uuid::parse_str(branch) {
            store
                .checkout_branch_head_by_public_id(row.id, branch_id)
                .await?
        } else {
            store.checkout_branch_head_by_name(row.id, branch).await?
        };
        if branch.is_some()
            && let Some(live_snapshot) = live_snapshot.as_ref()
        {
            self.kernel
                .agent_manager()
                .reload_session(session_id, Some(&live_snapshot.slot_id))
                .await?;
        }
        if let Some(branch) = &branch {
            info!(
                session_id = %session_id,
                store = %describe_store_selector(&store_selector),
                branch = %branch.name,
                reloaded_live_session = live_snapshot.is_some(),
                "Checked out session branch"
            );
        }
        Ok(branch.map(branch_detail_from_row))
    }
}
