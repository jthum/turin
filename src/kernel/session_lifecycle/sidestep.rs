use std::sync::Arc;

use anyhow::{Context, Result, anyhow};

use crate::kernel::event::TaskBranchOutcome;
use crate::kernel::session::{
    ExecutionConflictPolicy, ExecutionContextTarget, ExecutionDurability, ExecutionVisibility,
    ExecutionWritePolicy, PreparedSidestepExecution, SidestepMode, TaskExecutionOverrides,
};
use crate::kernel::session_refs::{format_session_reference, parse_session_reference};
use crate::persistence::manager::{StoreManager, StoreSelector};
use crate::persistence::schema::{BranchHeadRow, BranchProvenance, SessionRow};
use crate::persistence::state::StateStore;

pub(crate) async fn prepare_persisted_session_sidestep(
    store_manager: &Arc<StoreManager>,
    session_id: &str,
    default_target: &ExecutionContextTarget,
    mode: SidestepMode,
    requested_target: Option<ExecutionContextTarget>,
) -> Result<PreparedSidestepExecution> {
    let (store_selector, row) = resolve_persisted_session_row(store_manager, session_id).await?;
    let store = store_manager.open(&store_selector).await?;
    let resolved_target = requested_target.unwrap_or_else(|| default_target.clone());
    let resolved_target = normalize_sidestep_target(
        store_manager,
        &store_selector,
        &store,
        &row,
        resolved_target,
    )
    .await?;

    match mode {
        SidestepMode::Ephemeral => Ok(PreparedSidestepExecution {
            execution: TaskExecutionOverrides {
                context_target: Some(
                    snapshot_sidestep_target(&store, &row, resolved_target).await?,
                ),
                visibility: Some(ExecutionVisibility::Hidden),
                durability: Some(ExecutionDurability::Ephemeral),
                write_policy: Some(ExecutionWritePolicy::Detached),
            },
            conflict_policy: ExecutionConflictPolicy::Detached,
            branch_outcome: None,
        }),
        SidestepMode::ForkSibling => {
            let source = resolve_sidestep_branch_source(&store, &row, resolved_target).await?;
            let branch_name = format!("sidestep-{}", uuid::Uuid::now_v7().simple());
            let branch = store
                .create_branch_head_from_turn_index_with_provenance(
                    row.id,
                    &branch_name,
                    source.turn_index,
                    false,
                    BranchProvenance::sidestep(),
                )
                .await?;
            let branch_public_id = uuid::Uuid::from_slice(&branch.public_id)
                .map(|value| value.to_string())
                .map_err(anyhow::Error::from)?;

            Ok(PreparedSidestepExecution {
                execution: TaskExecutionOverrides {
                    context_target: Some(ExecutionContextTarget::BranchHead {
                        branch_head_id: Some(branch.id),
                    }),
                    visibility: Some(ExecutionVisibility::Hidden),
                    durability: Some(ExecutionDurability::Durable),
                    write_policy: Some(ExecutionWritePolicy::AdvanceBranchHead),
                },
                conflict_policy: ExecutionConflictPolicy::Reject,
                branch_outcome: Some(TaskBranchOutcome::SidestepSibling {
                    branch_id: branch.id,
                    branch_public_id,
                    branch_name: branch.name,
                    source_turn_id: branch.created_from_turn_id,
                    persisted_active_head_unchanged: !branch.is_active,
                }),
            })
        }
    }
}

async fn normalize_sidestep_target(
    store_manager: &Arc<StoreManager>,
    default_store_selector: &StoreSelector,
    store: &StateStore,
    row: &SessionRow,
    target: ExecutionContextTarget,
) -> Result<ExecutionContextTarget> {
    match target {
        ExecutionContextTarget::BranchHead { branch_head_id } => match branch_head_id {
            Some(branch_head_id) => {
                let branch = store
                    .get_branch_head(row.id, branch_head_id)
                    .await?
                    .ok_or_else(|| anyhow!("Branch head '{}' not found", branch_head_id))?;
                Ok(ExecutionContextTarget::BranchHead {
                    branch_head_id: Some(branch.id),
                })
            }
            None => Ok(ExecutionContextTarget::BranchHead {
                branch_head_id: None,
            }),
        },
        ExecutionContextTarget::TurnId { turn_id } => {
            validate_session_turn_target(store, row.id, turn_id, "sidestep target").await?;
            Ok(ExecutionContextTarget::TurnId { turn_id })
        }
        ExecutionContextTarget::SelectedPath { turn_ids } => {
            store
                .turn_rows_for_selected_path(row.id, &turn_ids)
                .await
                .context("Invalid selected sidestep path")?;
            Ok(ExecutionContextTarget::SelectedPath { turn_ids })
        }
        ExecutionContextTarget::SummarySource { source_turn_id } => {
            validate_session_turn_target(store, row.id, source_turn_id, "sidestep summary source")
                .await?;
            Ok(ExecutionContextTarget::SummarySource { source_turn_id })
        }
        ExecutionContextTarget::ExternalReference { reference } => {
            let session_ref = parse_session_reference(&reference)?;
            let resolved_selector = session_ref
                .store_selector
                .clone()
                .unwrap_or_else(|| default_store_selector.clone());
            let external_store = store_manager.open(&resolved_selector).await?;
            let public_id = uuid::Uuid::parse_str(&session_ref.public_id)
                .with_context(|| format!("Invalid session id '{}'", session_ref.public_id))?;
            let Some(_) = external_store
                .get_session_row_by_public_id(public_id)
                .await?
            else {
                anyhow::bail!("External sidestep reference '{}' not found", reference);
            };
            Ok(ExecutionContextTarget::ExternalReference {
                reference: format_session_reference(&session_ref.public_id, &resolved_selector),
            })
        }
    }
}

async fn snapshot_sidestep_target(
    store: &StateStore,
    row: &SessionRow,
    target: ExecutionContextTarget,
) -> Result<ExecutionContextTarget> {
    match target {
        ExecutionContextTarget::BranchHead { branch_head_id } => {
            let branch = match branch_head_id {
                Some(branch_head_id) => store.get_branch_head(row.id, branch_head_id).await?,
                None => store.get_active_branch_head(row.id).await?,
            };
            Ok(snapshot_target_from_branch_head(branch, branch_head_id))
        }
        other => Ok(other),
    }
}

async fn resolve_persisted_session_row(
    store_manager: &Arc<StoreManager>,
    session_id: &str,
) -> Result<(StoreSelector, SessionRow)> {
    let session_ref = parse_session_reference(session_id)?;
    let public_id = uuid::Uuid::parse_str(&session_ref.public_id)
        .with_context(|| format!("Invalid session id '{}'", session_ref.public_id))?;
    let store_selector = session_ref
        .store_selector
        .unwrap_or_else(|| StoreSelector::Alias("state".to_string()));
    let store = store_manager.open(&store_selector).await?;
    let row = store
        .get_session_row_by_public_id(public_id)
        .await?
        .ok_or_else(|| anyhow!("Session '{}' not found", session_id))?;
    Ok((store_selector, row))
}

struct SidestepBranchSource {
    turn_index: Option<u32>,
}

async fn resolve_sidestep_branch_source(
    store: &StateStore,
    row: &SessionRow,
    target: ExecutionContextTarget,
) -> Result<SidestepBranchSource> {
    match target {
        ExecutionContextTarget::BranchHead { branch_head_id } => {
            let branch = match branch_head_id {
                Some(branch_head_id) => store.get_branch_head(row.id, branch_head_id).await?,
                None => store.get_active_branch_head(row.id).await?,
            };
            sidestep_branch_source_from_branch(branch)
        }
        ExecutionContextTarget::TurnId { turn_id } => {
            sidestep_branch_source_from_turn(store, row.id, turn_id).await
        }
        ExecutionContextTarget::SelectedPath { turn_ids } => {
            let Some(turn_id) = turn_ids.last().copied() else {
                anyhow::bail!("Selected sidestep path must include at least one turn");
            };
            sidestep_branch_source_from_turn(store, row.id, turn_id).await
        }
        ExecutionContextTarget::SummarySource { source_turn_id } => {
            sidestep_branch_source_from_turn(store, row.id, source_turn_id).await
        }
        ExecutionContextTarget::ExternalReference { .. } => {
            anyhow::bail!("fork_sibling sidesteps do not support external_reference targets")
        }
    }
}

fn snapshot_target_from_branch_head(
    branch: Option<BranchHeadRow>,
    explicit_branch_head_id: Option<i64>,
) -> ExecutionContextTarget {
    match branch.and_then(|branch| branch.head_turn_id) {
        Some(turn_id) => ExecutionContextTarget::TurnId { turn_id },
        None => ExecutionContextTarget::BranchHead {
            branch_head_id: explicit_branch_head_id,
        },
    }
}

fn sidestep_branch_source_from_branch(
    branch: Option<BranchHeadRow>,
) -> Result<SidestepBranchSource> {
    let Some(branch) = branch else {
        anyhow::bail!("No branch head available for sidestep source");
    };
    Ok(SidestepBranchSource {
        turn_index: branch.head_turn_depth,
    })
}

async fn validate_session_turn_target(
    store: &StateStore,
    session_internal_id: i64,
    turn_id: i64,
    label: &str,
) -> Result<()> {
    let Some(turn) = store.get_turn_row(turn_id).await? else {
        anyhow::bail!("{} '{}' not found", label, turn_id);
    };
    if turn.session_id != session_internal_id {
        anyhow::bail!(
            "{} '{}' does not belong to the target session",
            label,
            turn_id
        );
    }
    Ok(())
}

async fn sidestep_branch_source_from_turn(
    store: &StateStore,
    session_internal_id: i64,
    turn_id: i64,
) -> Result<SidestepBranchSource> {
    validate_session_turn_target(store, session_internal_id, turn_id, "sidestep source turn")
        .await?;
    let turn = store
        .get_turn_row(turn_id)
        .await?
        .expect("validated sidestep source turn should exist");
    Ok(SidestepBranchSource {
        turn_index: Some(turn.branch_depth),
    })
}
