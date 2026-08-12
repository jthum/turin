use anyhow::Result;
use turin_daemon_protocol::{
    DaemonRequest, LiveSessionTargetParams, NoParams, OpenSessionParams, ResumeSessionParams,
    SessionBranchCheckoutParams, SessionBranchCreateParams, SessionBranchSiblingsParams,
    SessionGetParams, SessionIdParams, SessionListParams, SessionSearchParams, SessionSearchScope,
    SessionTitleParams,
};

use crate::client::ControlClient;
use crate::models::{
    LiveSession, LiveSessionList, SessionActionResult, SessionBranchDetail, SessionBranchList,
    SessionDetail, SessionGraphDetail, SessionList, SessionSearchHit, SessionSearchResultList,
    SessionSummary,
};

impl ControlClient {
    pub async fn get_session_graph(&self, session_id: &str) -> Result<SessionGraphDetail> {
        self.request_ok(
            None,
            DaemonRequest::SessionGraphGet(SessionIdParams {
                session_id: session_id.to_string(),
            }),
        )
        .await
    }

    pub async fn list_live_sessions(&self) -> Result<Vec<LiveSession>> {
        let response: LiveSessionList = self
            .request_ok(None, DaemonRequest::SessionListLive(NoParams::default()))
            .await?;
        Ok(response.sessions)
    }

    pub async fn list_sessions(&self, limit: usize, offset: usize) -> Result<Vec<SessionSummary>> {
        self.list_sessions_in(limit, offset, None, None).await
    }

    pub async fn list_sessions_in(
        &self,
        limit: usize,
        offset: usize,
        store: Option<&str>,
        path: Option<&str>,
    ) -> Result<Vec<SessionSummary>> {
        let response: SessionList = self
            .request_ok(
                None,
                DaemonRequest::SessionList(SessionListParams {
                    limit,
                    offset,
                    store: store.map(str::to_string),
                    path: path.map(str::to_string),
                }),
            )
            .await?;
        Ok(response.sessions)
    }

    pub async fn get_session(&self, session_id: &str) -> Result<SessionDetail> {
        self.request_ok(
            None,
            DaemonRequest::SessionGet(SessionGetParams {
                session_id: session_id.to_string(),
                message_limit: None,
                message_offset: None,
                include_events: None,
                include_efficiency: Some(true),
            }),
        )
        .await
    }

    pub async fn get_session_window(
        &self,
        session_id: &str,
        message_limit: usize,
    ) -> Result<SessionDetail> {
        self.get_session_window_at(session_id, message_limit, None)
            .await
    }

    pub async fn get_session_window_at(
        &self,
        session_id: &str,
        message_limit: usize,
        message_offset: Option<usize>,
    ) -> Result<SessionDetail> {
        self.request_ok(
            None,
            DaemonRequest::SessionGet(SessionGetParams {
                session_id: session_id.to_string(),
                message_limit: Some(message_limit),
                message_offset,
                include_events: Some(false),
                include_efficiency: Some(true),
            }),
        )
        .await
    }

    pub async fn search_sessions(
        &self,
        query: &str,
        scope: SessionSearchScope,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<SessionSearchHit>> {
        self.search_sessions_in(query, scope, limit, offset, None, None)
            .await
    }

    pub async fn search_sessions_in(
        &self,
        query: &str,
        scope: SessionSearchScope,
        limit: usize,
        offset: usize,
        store: Option<&str>,
        path: Option<&str>,
    ) -> Result<Vec<SessionSearchHit>> {
        let response: SessionSearchResultList = self
            .request_ok(
                None,
                DaemonRequest::SessionSearch(SessionSearchParams {
                    query: query.to_string(),
                    scope: Some(scope),
                    limit,
                    offset,
                    store: store.map(str::to_string),
                    path: path.map(str::to_string),
                }),
            )
            .await?;
        Ok(response.hits)
    }

    pub async fn set_session_title(
        &self,
        session_id: &str,
        title: Option<String>,
    ) -> Result<SessionSummary> {
        self.request_ok(
            None,
            DaemonRequest::SessionSetTitle(SessionTitleParams {
                session_id: session_id.to_string(),
                title,
            }),
        )
        .await
    }

    pub async fn list_session_branches(
        &self,
        session_id: &str,
    ) -> Result<Vec<SessionBranchDetail>> {
        let response: SessionBranchList = self
            .request_ok(
                None,
                DaemonRequest::SessionBranchList(SessionIdParams {
                    session_id: session_id.to_string(),
                }),
            )
            .await?;
        Ok(response.branches)
    }

    pub async fn create_session_branch(
        &self,
        session_id: &str,
        name: &str,
        from_turn_index: Option<u32>,
        activate: bool,
    ) -> Result<SessionBranchDetail> {
        self.create_session_branch_in_slot(session_id, None, name, from_turn_index, activate)
            .await
    }

    pub async fn create_session_branch_in_slot(
        &self,
        session_id: &str,
        slot_id: Option<String>,
        name: &str,
        from_turn_index: Option<u32>,
        activate: bool,
    ) -> Result<SessionBranchDetail> {
        self.request_ok(
            None,
            DaemonRequest::SessionBranchCreate(SessionBranchCreateParams {
                session_id: session_id.to_string(),
                name: name.to_string(),
                slot_id,
                from_turn_index,
                from_turn_id: None,
                activate,
            }),
        )
        .await
    }

    pub async fn create_session_branch_from_turn_id(
        &self,
        session_id: &str,
        slot_id: Option<String>,
        name: &str,
        from_turn_id: i64,
        activate: bool,
    ) -> Result<SessionBranchDetail> {
        self.request_ok(
            None,
            DaemonRequest::SessionBranchCreate(SessionBranchCreateParams {
                session_id: session_id.to_string(),
                name: name.to_string(),
                slot_id,
                from_turn_index: None,
                from_turn_id: Some(from_turn_id),
                activate,
            }),
        )
        .await
    }

    pub async fn checkout_session_branch(
        &self,
        session_id: &str,
        branch: &str,
    ) -> Result<SessionBranchDetail> {
        self.checkout_session_branch_in_slot(session_id, None, branch)
            .await
    }

    pub async fn checkout_session_branch_in_slot(
        &self,
        session_id: &str,
        slot_id: Option<String>,
        branch: &str,
    ) -> Result<SessionBranchDetail> {
        self.request_ok(
            None,
            DaemonRequest::SessionBranchCheckout(SessionBranchCheckoutParams {
                session_id: session_id.to_string(),
                branch: branch.to_string(),
                slot_id,
            }),
        )
        .await
    }

    pub async fn list_session_branch_siblings(
        &self,
        session_id: &str,
        source_turn_id: i64,
    ) -> Result<Vec<SessionBranchDetail>> {
        let response: SessionBranchList = self
            .request_ok(
                None,
                DaemonRequest::SessionBranchSiblings(SessionBranchSiblingsParams {
                    session_id: session_id.to_string(),
                    source_turn_id,
                }),
            )
            .await?;
        Ok(response.branches)
    }

    pub async fn open_session(
        &self,
        agent_id: &str,
        slot_id: Option<String>,
    ) -> Result<LiveSession> {
        self.request_ok(
            None,
            DaemonRequest::SessionOpen(OpenSessionParams {
                agent_id: agent_id.to_string(),
                slot_id,
                channel_id: None,
            }),
        )
        .await
    }

    pub async fn resume_session(
        &self,
        session_id: &str,
        slot_id: Option<String>,
    ) -> Result<LiveSession> {
        self.request_ok(
            None,
            DaemonRequest::SessionResume(ResumeSessionParams {
                session_id: session_id.to_string(),
                slot_id,
            }),
        )
        .await
    }

    pub async fn cancel_live_session(
        &self,
        session_id: &str,
        slot_id: Option<String>,
    ) -> Result<SessionActionResult> {
        self.request_ok(
            None,
            DaemonRequest::SessionCancel(LiveSessionTargetParams {
                session_id: session_id.to_string(),
                slot_id,
            }),
        )
        .await
    }

    pub async fn cancel_session(&self, session_id: &str) -> Result<SessionActionResult> {
        self.cancel_live_session(session_id, None).await
    }

    pub async fn kill_live_session(
        &self,
        session_id: &str,
        slot_id: Option<String>,
    ) -> Result<SessionActionResult> {
        self.request_ok(
            None,
            DaemonRequest::SessionKill(LiveSessionTargetParams {
                session_id: session_id.to_string(),
                slot_id,
            }),
        )
        .await
    }

    pub async fn kill_session(&self, session_id: &str) -> Result<SessionActionResult> {
        self.kill_live_session(session_id, None).await
    }
}
