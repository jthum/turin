use std::collections::{HashMap, HashSet};

use anyhow::{Result, anyhow};

use super::session_summary_from_row_and_selector;
use crate::daemon::state::{DaemonState, SessionFamilyDetail, SessionFamilyMember};
use crate::kernel::session_refs::{format_session_reference, session_references_match};

impl DaemonState {
    pub async fn get_session_family(
        &self,
        session_id: &str,
    ) -> Result<Option<SessionFamilyDetail>> {
        let Some((selector, store, requested)) = self.resolve_persisted_session(session_id).await?
        else {
            return Ok(None);
        };
        let stats = store
            .linked_session_family_stats(requested.id)
            .await?
            .expect("resolved session must have family statistics");
        let root_id = requested.root_session_id.unwrap_or(requested.id);
        let root = store
            .get_session_row(root_id)
            .await?
            .ok_or_else(|| anyhow!("Session family root '{}' not found", root_id))?;
        let mut rows = vec![root.clone()];
        rows.extend(store.list_linked_session_descendants(root_id).await?);
        let child_counts = rows
            .iter()
            .fold(HashMap::<i64, usize>::new(), |mut counts, row| {
                if let Some(parent_id) = row.parent_session_id {
                    *counts.entry(parent_id).or_default() += 1;
                }
                counts
            });
        let by_id = rows
            .iter()
            .map(|row| (row.id, row.parent_session_id))
            .collect::<HashMap<_, _>>();
        let live = self.list_live_sessions().await;
        let mut members = Vec::with_capacity(rows.len());
        for row in rows {
            let summary = session_summary_from_row_and_selector(&row, &selector);
            let matching = live
                .iter()
                .filter(|item| session_references_match(&item.session_id, &summary.session_id))
                .collect::<Vec<_>>();
            let mut depth = 0usize;
            let mut parent = row.parent_session_id;
            let mut visited = HashSet::new();
            while let Some(parent_id) = parent {
                anyhow::ensure!(visited.insert(parent_id), "Session family contains a cycle");
                depth += 1;
                parent = by_id.get(&parent_id).copied().flatten();
            }
            members.push(SessionFamilyMember {
                session: summary,
                depth,
                direct_children: child_counts.get(&row.id).copied().unwrap_or(0),
                live_slots: matching.iter().map(|item| item.slot_id.clone()).collect(),
                active_tasks: matching.iter().map(|item| item.active_tasks).sum(),
                queued_tasks: matching.iter().map(|item| item.queued_tasks).sum(),
            });
        }
        members.sort_by_key(|member| (member.depth, member.session.created_at.clone()));
        let root_session_id = session_summary_from_row_and_selector(&root, &selector).session_id;
        Ok(Some(SessionFamilyDetail {
            requested_session_id: format_session_reference(
                &uuid::Uuid::from_slice(&requested.public_id)?
                    .simple()
                    .to_string(),
                &selector,
            ),
            root_session_id,
            requested_depth: stats.depth,
            direct_children: stats.direct_child_count,
            descendants: stats.descendant_count,
            family_size: stats.root_family_size,
            members,
        }))
    }

    pub async fn archive_linked_session(&self, session_id: &str) -> Result<Option<usize>> {
        let Some((selector, store, session)) = self.resolve_persisted_session(session_id).await?
        else {
            return Ok(None);
        };
        anyhow::ensure!(
            session.parent_session_id.is_some(),
            "Top-level sessions cannot be archived as linked work"
        );
        let archive_root_id = session.id;
        let mut family = store.list_linked_session_descendants(session.id).await?;
        family.push(session);
        let session_ids = family
            .iter()
            .map(|row| {
                Ok(format_session_reference(
                    &uuid::Uuid::from_slice(&row.public_id)?.simple().to_string(),
                    &selector,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let persisted_ids = family
            .iter()
            .map(|row| (selector.clone(), row.id))
            .collect::<HashSet<_>>();
        let work_count = self
            .kernel
            .agent_manager()
            .session_family_work_count(&session_ids, &persisted_ids)
            .await;
        anyhow::ensure!(
            work_count == 0,
            "Session family has {work_count} active or queued task(s)"
        );
        Ok(Some(
            store.archive_linked_session_family(archive_root_id).await?,
        ))
    }
}
