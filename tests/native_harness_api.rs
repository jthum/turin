use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use turin::kernel::config::InferenceOverrideConfig;
use turin::kernel::harness_contract::{
    HarnessSignal, HarnessTurnRequest, RequestOptionsOverride, ToolExposure,
};
use turin::kernel::native_harness::{NativeHarness, NativeHarnessFactory, Verdict};

struct FixedHarness {
    received_signals: Arc<Mutex<Vec<String>>>,
}

impl NativeHarness for FixedHarness {
    fn runtime_signal_topics(&self) -> Vec<String> {
        vec!["build.*".to_string()]
    }

    fn on_turn_prepare(&mut self, request: &mut HarnessTurnRequest) -> Result<Verdict> {
        request.system_prompt.push_str("\nUse concise answers.");
        request
            .tool_exposure
            .exclude(BTreeSet::from(["shell_exec".to_string()]));
        Ok(Verdict::Allow)
    }

    fn on_signal(&mut self, signal: HarnessSignal<'_>) -> Result<()> {
        self.received_signals
            .lock()
            .expect("signal recording mutex poisoned")
            .push(format!("{}:{}", signal.topic, signal.payload));
        Ok(())
    }
}

#[test]
fn public_native_harness_contract_mutates_requests_without_lua_types() -> Result<()> {
    let received_signals = Arc::new(Mutex::new(Vec::new()));
    let factory_signals = Arc::clone(&received_signals);
    let factory: Arc<dyn NativeHarnessFactory> = Arc::new(move || {
        Ok(Box::new(FixedHarness {
            received_signals: Arc::clone(&factory_signals),
        }) as Box<dyn NativeHarness>)
    });
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

    assert_eq!(harness.runtime_signal_topics(), ["build.*"]);
    harness.on_signal(HarnessSignal {
        signal_id: None,
        topic: "build.complete",
        source_agent_id: "builder",
        target_agent_id: "default",
        source_session_id: Some("source-session"),
        target_session_id: Some("target-session"),
        payload: r#"{"status":"passed"}"#,
        created_at: "2026-08-20T00:00:00Z",
    })?;
    assert_eq!(
        *received_signals
            .lock()
            .expect("signal recording mutex poisoned"),
        [r#"build.complete:{"status":"passed"}"#]
    );
    Ok(())
}
