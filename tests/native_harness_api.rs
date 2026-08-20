use std::collections::BTreeSet;
use std::sync::Arc;

use anyhow::Result;
use turin::kernel::config::InferenceOverrideConfig;
use turin::kernel::harness_contract::{HarnessTurnRequest, RequestOptionsOverride, ToolExposure};
use turin::kernel::native_harness::{NativeHarness, NativeHarnessFactory, Verdict};

struct FixedHarness;

impl NativeHarness for FixedHarness {
    fn on_turn_prepare(&mut self, request: &mut HarnessTurnRequest) -> Result<Verdict> {
        request.system_prompt.push_str("\nUse concise answers.");
        request
            .tool_exposure
            .exclude(BTreeSet::from(["shell_exec".to_string()]));
        Ok(Verdict::Allow)
    }
}

#[test]
fn public_native_harness_contract_mutates_requests_without_lua_types() -> Result<()> {
    let factory: Arc<dyn NativeHarnessFactory> =
        Arc::new(|| Ok(Box::new(FixedHarness) as Box<dyn NativeHarness>));
    let mut harness = factory.create()?;
    let mut request = HarnessTurnRequest {
        inference: None,
        model: "model".to_string(),
        provider: "provider".to_string(),
        system_prompt: "Base instructions.".to_string(),
        messages: Vec::new(),
        turn_index: 1,
        task_turn_index: 1,
        is_first_turn_in_task: true,
        task_id: "task".to_string(),
        plan_id: None,
        token_count: 0,
        token_limit: 8_192,
        thinking_budget: 0,
        request_options: RequestOptionsOverride::default(),
        agent_id: "default".to_string(),
        session_inference: InferenceOverrideConfig::default(),
        session_id: "session".to_string(),
        session_title: None,
        available_tools: BTreeSet::from(["shell_exec".to_string()]),
        tool_exposure: ToolExposure::default(),
    };

    assert_eq!(harness.on_turn_prepare(&mut request)?, Verdict::Allow);
    assert!(request.system_prompt.ends_with("Use concise answers."));
    assert!(!request.tool_exposure.exposes("shell_exec"));
    Ok(())
}
