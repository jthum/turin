use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::kernel::delegation_budget::{DelegationBudget, DelegationBudgetLimits};

use super::{PeerAgentTaskResult, PromotedTaskBranch};

#[derive(Default)]
pub(super) struct CompletedTaskCache {
    pub(super) results: HashMap<String, PeerAgentTaskResult>,
    order: VecDeque<String>,
}

impl CompletedTaskCache {
    const MAX_ENTRIES: usize = 1024;

    pub(super) fn insert(&mut self, result: PeerAgentTaskResult) {
        let request_id = result.request_id.clone();
        if !self.results.contains_key(&request_id) {
            self.order.push_back(request_id.clone());
        }
        self.results.insert(request_id, result);
        while self.order.len() > Self::MAX_ENTRIES {
            if let Some(evicted) = self.order.pop_front() {
                self.results.remove(&evicted);
            }
        }
    }

    pub(super) fn mark_promoted(&mut self, request_id: &str, branch: PromotedTaskBranch) {
        if let Some(result) = self.results.get_mut(request_id) {
            result.promoted_branch = Some(branch);
        }
    }
}

#[derive(Default)]
pub(super) struct DelegationBudgetCache {
    order: VecDeque<String>,
    budgets: HashMap<String, Arc<DelegationBudget>>,
}

impl DelegationBudgetCache {
    const MAX_ENTRIES: usize = 1024;

    pub(super) fn get_or_create(
        &mut self,
        trace_id: &str,
        limits: DelegationBudgetLimits,
    ) -> Arc<DelegationBudget> {
        if let Some(budget) = self.budgets.get(trace_id) {
            budget.tighten(limits);
            return Arc::clone(budget);
        }
        let budget = DelegationBudget::new(limits);
        self.order.push_back(trace_id.to_string());
        self.budgets
            .insert(trace_id.to_string(), Arc::clone(&budget));
        let mut eviction_attempts = self.order.len();
        while self.budgets.len() > Self::MAX_ENTRIES && eviction_attempts > 0 {
            eviction_attempts -= 1;
            let Some(candidate) = self.order.pop_front() else {
                break;
            };
            let can_evict = self
                .budgets
                .get(&candidate)
                .is_some_and(|budget| Arc::strong_count(budget) == 1);
            if can_evict {
                self.budgets.remove(&candidate);
            } else {
                self.order.push_back(candidate);
            }
        }
        budget
    }
}
