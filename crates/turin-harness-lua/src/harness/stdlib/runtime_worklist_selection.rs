use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::persistence::schema::WorkItemRow;
use crate::work_items::{
    WorkItemParentId, work_item_claimable_now, work_item_dependencies_satisfied,
    work_item_is_orphaned, work_item_matches_where, work_item_next_candidates, work_item_pause_due,
    work_item_paused, work_item_status_map,
};

#[derive(Debug, Clone, Copy)]
pub(super) struct WorkItemSelection<'a> {
    pub parent_item_id: Option<i64>,
    pub where_map: Option<&'a JsonMap<String, JsonValue>>,
    pub limit: Option<usize>,
}

impl<'a> WorkItemSelection<'a> {
    pub fn new(
        parent_item_id: Option<i64>,
        where_map: Option<&'a JsonMap<String, JsonValue>>,
        limit: Option<usize>,
    ) -> Self {
        Self {
            parent_item_id,
            where_map,
            limit,
        }
    }

    fn in_scope(self, row: &WorkItemRow) -> bool {
        row.parent_item_id == self.parent_item_id
            && work_item_matches_where(row, self.where_map, WorkItemParentId::DatabaseId)
    }

    fn take_limit<T>(self, rows: impl Iterator<Item = T>) -> Vec<T> {
        rows.take(self.limit.unwrap_or(usize::MAX)).collect()
    }
}

pub(super) fn children(rows: Vec<WorkItemRow>, parent_item_id: i64) -> Vec<WorkItemRow> {
    rows.into_iter()
        .filter(|row| row.parent_item_id == Some(parent_item_id))
        .collect()
}

pub(super) fn all_rows(
    rows: Vec<WorkItemRow>,
    selection: WorkItemSelection<'_>,
) -> Vec<WorkItemRow> {
    selection.take_limit(rows.into_iter().filter(|row| selection.in_scope(row)))
}

pub(super) fn pending_rows(
    rows: Vec<WorkItemRow>,
    selection: WorkItemSelection<'_>,
) -> Vec<WorkItemRow> {
    let status_map = work_item_status_map(&rows);
    selection.take_limit(rows.into_iter().filter(|row| {
        selection.in_scope(row)
            && row.status == "pending"
            && row.claim_execution_id.is_none()
            && work_item_dependencies_satisfied(row, &status_map)
    }))
}

pub(super) fn orphaned_rows(
    rows: Vec<WorkItemRow>,
    selection: WorkItemSelection<'_>,
    stale_before_unix_ms: i64,
) -> Vec<WorkItemRow> {
    selection.take_limit(
        rows.into_iter().filter(|row| {
            selection.in_scope(row) && work_item_is_orphaned(row, stale_before_unix_ms)
        }),
    )
}

pub(super) fn paused_rows(
    rows: Vec<WorkItemRow>,
    selection: WorkItemSelection<'_>,
    due_only: bool,
    now_unix_ms: i64,
) -> Vec<WorkItemRow> {
    selection.take_limit(rows.into_iter().filter(|row| {
        selection.in_scope(row)
            && work_item_paused(row)
            && (!due_only || work_item_pause_due(row, now_unix_ms))
    }))
}

pub(super) fn find_matching(
    rows: Vec<WorkItemRow>,
    parent_item_id: Option<i64>,
    where_map: &JsonMap<String, JsonValue>,
) -> Option<WorkItemRow> {
    rows.into_iter()
        .filter(|row| row.parent_item_id == parent_item_id)
        .find(|row| work_item_matches_where(row, Some(where_map), WorkItemParentId::DatabaseId))
}

pub(super) fn active_for_current_claim(
    rows: Vec<WorkItemRow>,
    parent_item_id: Option<i64>,
    session_id: Option<&str>,
    execution_id: Option<&str>,
) -> Option<WorkItemRow> {
    rows.into_iter().find(|row| {
        row.parent_item_id == parent_item_id
            && row.status == "active"
            && (row.claim_execution_id.as_deref() == execution_id
                || row.claim_session_id.as_deref() == session_id)
    })
}

pub(super) fn next_candidates<'a>(
    rows: &'a [WorkItemRow],
    parent_item_id: Option<i64>,
    where_map: Option<&'a JsonMap<String, JsonValue>>,
    now_unix_ms: i64,
) -> Vec<&'a WorkItemRow> {
    work_item_next_candidates(rows, parent_item_id, where_map, now_unix_ms)
}

pub(super) fn has_pending_work(
    rows: &[WorkItemRow],
    parent_item_id: Option<i64>,
    now_unix_ms: i64,
) -> bool {
    let status_map = work_item_status_map(rows);
    rows.iter().any(|row| {
        row.parent_item_id == parent_item_id
            && work_item_claimable_now(row, now_unix_ms)
            && row.claim_execution_id.is_none()
            && work_item_dependencies_satisfied(row, &status_map)
    })
}

pub(super) fn progress_counts(rows: &[WorkItemRow], parent_item_id: Option<i64>) -> (usize, usize) {
    rows.iter()
        .filter(|row| row.parent_item_id == parent_item_id)
        .fold((0, 0), |(done, total), row| {
            (done + usize::from(row.status == "done"), total + 1)
        })
}
