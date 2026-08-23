use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;
use futures::future::BoxFuture;
use tempfile::tempdir;
use turin::inference::provider::{
    InferenceEvent, InferenceProvider, InferenceRequest, InferenceStream, ProviderClient,
    RequestOptions, SdkError,
};
use turin::kernel::Kernel;
use turin::kernel::config::{
    AgentConfig, EmbeddingConfig, HarnessConfig, InferenceOverrideConfig, PersistenceConfig,
    ProviderConfig, TurinConfig,
};
use turin::kernel::harness::{Harness, HarnessFactory, Verdict};
use turin::kernel::harness_contract::{
    HarnessActionRequest, HarnessSignal, HarnessTurnRequest, RequestOptionsOverride, ToolExposure,
};

struct FixedHarness {
    received_signals: Arc<Mutex<Vec<String>>>,
}

struct CaptureProvider {
    requests: Arc<Mutex<Vec<InferenceRequest>>>,
}

#[async_trait]
impl InferenceProvider for CaptureProvider {
    fn stream<'a>(
        &'a self,
        request: InferenceRequest,
        _options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        self.requests
            .lock()
            .expect("request mutex poisoned")
            .push(request);
        Box::pin(async move {
            let stream = futures::stream::iter(vec![
                Ok(InferenceEvent::MessageStart {
                    role: "assistant".to_string(),
                    model: "capture-model".to_string(),
                    provider_id: "capture".to_string(),
                }),
                Ok(InferenceEvent::MessageDelta {
                    content: "CAPTURED".to_string(),
                }),
                Ok(InferenceEvent::MessageEnd {
                    input_tokens: 2,
                    output_tokens: 1,
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                    stop_reason: None,
                }),
            ]);
            Ok(Box::pin(stream) as InferenceStream)
        })
    }
}

struct PromptHarness(&'static str);

impl Harness for PromptHarness {
    fn on_turn_prepare(&mut self, request: &mut HarnessTurnRequest) -> Result<Verdict> {
        request.system_prompt.push_str(self.0);
        Ok(Verdict::Allow)
    }
}

impl Harness for FixedHarness {
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

    fn on_action(
        &mut self,
        request: HarnessActionRequest<'_>,
    ) -> Result<Option<serde_json::Value>> {
        Ok((request.name == "build.status").then_some(request.params))
    }
}

#[test]
fn public_rust_harness_contract_mutates_requests_without_lua_types() -> Result<()> {
    let received_signals = Arc::new(Mutex::new(Vec::new()));
    let factory_signals = Arc::clone(&received_signals);
    let factory: Arc<dyn HarnessFactory> = Arc::new(move || {
        Ok(Box::new(FixedHarness {
            received_signals: Arc::clone(&factory_signals),
        }) as Box<dyn Harness>)
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
    assert_eq!(
        harness.on_action(HarnessActionRequest {
            agent_id: "default",
            name: "build.status",
            params: serde_json::json!({ "state": "passed" }),
        })?,
        Some(serde_json::json!({ "state": "passed" }))
    );
    Ok(())
}

#[tokio::test]
async fn rust_harness_mutation_reaches_provider_without_lua() -> Result<()> {
    let tmp = tempdir()?;
    let mut config = TurinConfig::default();
    config.kernel.workspace_root = tmp.path().to_string_lossy().into_owned();
    config.persistence = PersistenceConfig::with_state_path(
        tmp.path().join("state.db").to_string_lossy().into_owned(),
    );
    config.embeddings = Some(EmbeddingConfig::noop());
    config.agent = AgentConfig {
        id: "default".to_string(),
        model: "capture-model".to_string(),
        provider: "capture".to_string(),
        system_prompt: "Base instructions.".to_string(),
        ..AgentConfig::default()
    };
    config.providers = HashMap::from([(
        "capture".to_string(),
        ProviderConfig {
            kind: "capture".to_string(),
            ..ProviderConfig::default()
        },
    )]);

    let signals = Arc::new(Mutex::new(Vec::new()));
    let factory_signals = Arc::clone(&signals);
    let factory: Arc<dyn HarnessFactory> = Arc::new(move || {
        Ok(Box::new(FixedHarness {
            received_signals: Arc::clone(&factory_signals),
        }) as Box<dyn Harness>)
    });
    let captured = Arc::new(Mutex::new(Vec::new()));
    let mut kernel = Kernel::builder(config)
        .with_default_harness(factory)
        .build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;
    kernel.add_client(
        "capture".to_string(),
        ProviderClient::new(
            "capture",
            Arc::new(CaptureProvider {
                requests: Arc::clone(&captured),
            }),
        ),
    );

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Exercise the Rust harness.".to_string()))
        .await?;

    let request = captured
        .lock()
        .expect("request mutex poisoned")
        .first()
        .cloned()
        .expect("provider received an inference request");
    assert_eq!(
        request.system.as_deref(),
        Some("Base instructions.\nUse concise answers.")
    );
    Ok(())
}

#[tokio::test]
async fn agents_can_bind_to_distinct_rust_harnesses_without_lua() -> Result<()> {
    let tmp = tempdir()?;
    let mut config = TurinConfig::default();
    config.kernel.workspace_root = tmp.path().to_string_lossy().into_owned();
    config.persistence = PersistenceConfig::with_state_path(
        tmp.path().join("state.db").to_string_lossy().into_owned(),
    );
    config.embeddings = Some(EmbeddingConfig::noop());
    config.agent = AgentConfig {
        id: "default".to_string(),
        model: "capture-model".to_string(),
        provider: "capture".to_string(),
        system_prompt: "Default agent.".to_string(),
        ..AgentConfig::default()
    };
    config.agents.insert(
        "reviewer".to_string(),
        AgentConfig {
            id: "reviewer".to_string(),
            model: "capture-model".to_string(),
            provider: "capture".to_string(),
            system_prompt: "Reviewer agent.".to_string(),
            harness: Some("review".to_string()),
            ..AgentConfig::default()
        },
    );
    config
        .harnesses
        .insert("review".to_string(), HarnessConfig::default());
    config.providers = HashMap::from([(
        "capture".to_string(),
        ProviderConfig {
            kind: "capture".to_string(),
            ..ProviderConfig::default()
        },
    )]);

    let default_factory: Arc<dyn HarnessFactory> =
        Arc::new(|| Ok(Box::new(PromptHarness(" [default harness]")) as Box<dyn Harness>));
    let review_factory: Arc<dyn HarnessFactory> =
        Arc::new(|| Ok(Box::new(PromptHarness(" [review harness]")) as Box<dyn Harness>));
    let captured = Arc::new(Mutex::new(Vec::new()));
    let mut kernel = Kernel::builder(config)
        .with_default_harness(default_factory)
        .with_harness("review", review_factory)
        .build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;
    kernel.add_client(
        "capture".to_string(),
        ProviderClient::new(
            "capture",
            Arc::new(CaptureProvider {
                requests: Arc::clone(&captured),
            }),
        ),
    );

    let mut default_session = kernel.create_session().await;
    kernel
        .run(&mut default_session, Some("Default task".to_string()))
        .await?;
    let mut review_session = kernel.create_session_for_agent("reviewer").await;
    kernel
        .run(&mut review_session, Some("Review task".to_string()))
        .await?;

    let requests = captured.lock().expect("request mutex poisoned");
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[0].system.as_deref(),
        Some("Default agent. [default harness]")
    );
    assert_eq!(
        requests[1].system.as_deref(),
        Some("Reviewer agent. [review harness]")
    );
    Ok(())
}

#[test]
fn named_harness_without_registered_implementation_fails() -> Result<()> {
    let tmp = tempdir()?;
    let mut config = TurinConfig::default();
    config.kernel.workspace_root = tmp.path().to_string_lossy().into_owned();
    config.persistence = PersistenceConfig::with_state_path(
        tmp.path().join("state.db").to_string_lossy().into_owned(),
    );
    config
        .harnesses
        .insert("missing".to_string(), HarnessConfig::default());

    let factory: Arc<dyn HarnessFactory> =
        Arc::new(|| Ok(Box::new(PromptHarness(" [default harness]")) as Box<dyn Harness>));
    let result = Kernel::builder(config)
        .with_default_harness(factory)
        .build();
    let error = match result {
        Ok(_) => anyhow::bail!("missing harness implementation unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("Harness 'missing' has no registered implementation")
    );
    Ok(())
}

#[test]
fn rust_harness_registration_rejects_undeclared_id() -> Result<()> {
    let factory: Arc<dyn HarnessFactory> =
        Arc::new(|| Ok(Box::new(PromptHarness(" [typo]")) as Box<dyn Harness>));
    let result = Kernel::builder(TurinConfig::default())
        .with_harness("typo", factory)
        .build();

    let error = match result {
        Ok(_) => anyhow::bail!("undeclared Rust harness ID unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(
        error.to_string(),
        "Rust harness 'typo' is not declared in config.harnesses"
    );
    Ok(())
}
