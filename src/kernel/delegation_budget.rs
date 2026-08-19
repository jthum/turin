use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use tokio_util::sync::CancellationToken;

const UNLIMITED: u64 = u64::MAX;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DelegationBudgetLimits {
    pub(crate) max_total_tokens: Option<u64>,
    pub(crate) max_duration_ms: Option<u64>,
    pub(crate) max_tool_calls: Option<u64>,
}

impl DelegationBudgetLimits {
    pub(crate) fn is_unbounded(self) -> bool {
        self.max_total_tokens.is_none()
            && self.max_duration_ms.is_none()
            && self.max_tool_calls.is_none()
    }
}

#[derive(Debug)]
pub(crate) struct DelegationBudget {
    started: Instant,
    max_total_tokens: AtomicU64,
    max_duration_ms: AtomicU64,
    max_tool_calls: AtomicU64,
    total_tokens: AtomicU64,
    tool_calls: AtomicU64,
    cancellation: CancellationToken,
}

impl DelegationBudget {
    pub(crate) fn new(limits: DelegationBudgetLimits) -> Arc<Self> {
        let budget = Arc::new(Self {
            started: Instant::now(),
            max_total_tokens: AtomicU64::new(limit_value(limits.max_total_tokens)),
            max_duration_ms: AtomicU64::new(limit_value(limits.max_duration_ms)),
            max_tool_calls: AtomicU64::new(limit_value(limits.max_tool_calls)),
            total_tokens: AtomicU64::new(0),
            tool_calls: AtomicU64::new(0),
            cancellation: CancellationToken::new(),
        });
        budget.schedule_deadline(limits.max_duration_ms);
        budget
    }

    pub(crate) fn tighten(self: &Arc<Self>, limits: DelegationBudgetLimits) {
        tighten_limit(&self.max_total_tokens, limits.max_total_tokens);
        tighten_limit(&self.max_tool_calls, limits.max_tool_calls);
        if let Some(duration_ms) = limits.max_duration_ms {
            let previous = self
                .max_duration_ms
                .fetch_min(duration_ms, Ordering::AcqRel);
            if duration_ms < previous {
                self.schedule_deadline(Some(duration_ms));
            }
        }
        self.cancel_if_exceeded();
    }

    pub(crate) fn check_admission(&self) -> anyhow::Result<()> {
        self.cancel_if_exceeded();
        anyhow::ensure!(
            !self.cancellation.is_cancelled(),
            "Policy denial: delegated task-family budget exhausted"
        );
        Ok(())
    }

    pub(crate) fn child_cancellation_token(&self) -> CancellationToken {
        self.cancellation.child_token()
    }

    pub(crate) fn record_tokens(&self, tokens: u64) {
        self.total_tokens.fetch_add(tokens, Ordering::AcqRel);
        self.cancel_if_exceeded();
    }

    pub(crate) fn reserve_tool_calls(&self, requested: usize) -> usize {
        let requested = u64::try_from(requested).unwrap_or(u64::MAX);
        let limit = self.max_tool_calls.load(Ordering::Acquire);
        if limit == UNLIMITED {
            self.tool_calls.fetch_add(requested, Ordering::AcqRel);
            return usize::try_from(requested).unwrap_or(usize::MAX);
        }

        let mut used = self.tool_calls.load(Ordering::Acquire);
        loop {
            let granted = requested.min(limit.saturating_sub(used));
            match self.tool_calls.compare_exchange_weak(
                used,
                used.saturating_add(granted),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    if granted < requested {
                        self.cancellation.cancel();
                    }
                    return usize::try_from(granted).unwrap_or(usize::MAX);
                }
                Err(current) => used = current,
            }
        }
    }

    fn cancel_if_exceeded(&self) {
        let token_limit = self.max_total_tokens.load(Ordering::Acquire);
        let duration_limit = self.max_duration_ms.load(Ordering::Acquire);
        let tool_limit = self.max_tool_calls.load(Ordering::Acquire);
        if (token_limit != UNLIMITED && self.total_tokens.load(Ordering::Acquire) > token_limit)
            || (duration_limit != UNLIMITED
                && self.started.elapsed().as_millis() >= u128::from(duration_limit))
            || (tool_limit != UNLIMITED && self.tool_calls.load(Ordering::Acquire) > tool_limit)
        {
            self.cancellation.cancel();
        }
    }

    fn schedule_deadline(self: &Arc<Self>, duration_ms: Option<u64>) {
        let Some(duration_ms) = duration_ms else {
            return;
        };
        let budget = Arc::clone(self);
        tokio::spawn(async move {
            let elapsed_ms =
                u64::try_from(budget.started.elapsed().as_millis()).unwrap_or(u64::MAX);
            tokio::time::sleep(std::time::Duration::from_millis(
                duration_ms.saturating_sub(elapsed_ms),
            ))
            .await;
            if budget.max_duration_ms.load(Ordering::Acquire) <= duration_ms {
                budget.cancellation.cancel();
            }
        });
    }
}

fn limit_value(limit: Option<u64>) -> u64 {
    limit.unwrap_or(UNLIMITED)
}

fn tighten_limit(target: &AtomicU64, limit: Option<u64>) {
    if let Some(limit) = limit {
        target.fetch_min(limit, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_and_tool_limits_cancel_the_shared_budget() {
        let token_budget = DelegationBudget::new(DelegationBudgetLimits {
            max_total_tokens: Some(10),
            ..DelegationBudgetLimits::default()
        });
        token_budget.record_tokens(10);
        token_budget.check_admission().unwrap();
        token_budget.record_tokens(1);
        assert!(token_budget.check_admission().is_err());

        let tool_budget = DelegationBudget::new(DelegationBudgetLimits {
            max_tool_calls: Some(3),
            ..DelegationBudgetLimits::default()
        });
        assert_eq!(tool_budget.reserve_tool_calls(2), 2);
        assert_eq!(tool_budget.reserve_tool_calls(2), 1);
        assert!(tool_budget.check_admission().is_err());
    }

    #[tokio::test]
    async fn duration_limit_cancels_child_tokens() {
        let budget = DelegationBudget::new(DelegationBudgetLimits {
            max_duration_ms: Some(1),
            ..DelegationBudgetLimits::default()
        });
        let child = budget.child_cancellation_token();
        tokio::time::timeout(std::time::Duration::from_millis(100), child.cancelled())
            .await
            .expect("duration budget should cancel its descendants");
        assert!(budget.check_admission().is_err());
    }

    #[test]
    fn nested_policy_can_only_tighten_a_budget() {
        let budget = DelegationBudget::new(DelegationBudgetLimits {
            max_total_tokens: Some(100),
            ..DelegationBudgetLimits::default()
        });
        budget.record_tokens(60);
        budget.tighten(DelegationBudgetLimits {
            max_total_tokens: Some(50),
            ..DelegationBudgetLimits::default()
        });
        assert!(budget.check_admission().is_err());
    }
}
