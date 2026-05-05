use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{RwLock, watch as watch_channel};
use tracing::{error, info};

use crate::daemon::state::DaemonState;

const SCHEDULER_IDLE_POLL: Duration = Duration::from_secs(60);
const SCHEDULER_ERROR_BACKOFF: Duration = Duration::from_secs(5);

pub(super) fn start_internal_scheduler(
    state: Arc<RwLock<DaemonState>>,
    mut shutdown_rx: watch_channel::Receiver<bool>,
) {
    let wake = Arc::new(tokio::sync::Notify::new());
    let wake_for_task = Arc::clone(&wake);
    let state_for_task = Arc::clone(&state);

    tokio::spawn(async move {
        {
            let mut guard = state_for_task.write().await;
            guard.set_scheduler_wake(Arc::clone(&wake_for_task));
        }

        loop {
            let next_sleep = {
                let mut guard = state_for_task.write().await;
                match guard.scheduler_tick().await {
                    Ok(next_due) => next_due.unwrap_or(SCHEDULER_IDLE_POLL),
                    Err(err) => {
                        error!(error = %err, "Internal scheduler tick failed");
                        SCHEDULER_ERROR_BACKOFF
                    }
                }
            };

            tokio::select! {
                _ = tokio::time::sleep(next_sleep) => {}
                _ = wake_for_task.notified() => {}
                changed = shutdown_rx.changed() => {
                    if changed.is_ok() && *shutdown_rx.borrow() {
                        info!("Internal scheduler shutting down");
                        break;
                    }
                }
            }
        }
    });
}
