use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
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
    AgentConfig, EmbeddingConfig, GovernanceConfig, GovernanceUnmatchedCapability, HarnessConfig,
    PersistenceConfig, ProviderConfig, TurinConfig,
};
use turin::kernel::harness::{Harness, Verdict};
use turin::kernel::harness_contract::HarnessTurnRequest;
use turin::tools::registry::ToolRegistry;
use turin::tools::{Tool, ToolContext, ToolEffect, ToolError, ToolOutput};

struct PromptHarness(&'static str);

impl Harness for PromptHarness {
    fn on_turn_prepare(&mut self, request: &mut HarnessTurnRequest) -> Result<Verdict> {
        request.system_prompt.push_str(self.0);
        Ok(Verdict::Allow)
    }
}

struct RecordFactTool;

#[async_trait]
impl Tool for RecordFactTool {
    fn name(&self) -> &str {
        "record_fact"
    }

    fn description(&self) -> &str {
        "Persist a fact in the embedded application's state store"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"]
        })
    }

    fn capability(&self) -> Option<&str> {
        Some("records.write")
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolEffect, ToolError> {
        let value = params
            .get("value")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ToolError::InvalidParams("value must be a string".into()))?;
        let manager = ctx
            .store_manager
            .as_ref()
            .ok_or_else(|| ToolError::ExecutionError("state store unavailable".into()))?;
        let store = manager
            .get_default()
            .await
            .map_err(|error| ToolError::ExecutionError(error.to_string()))?;
        let connection = store
            .get_connection()
            .await
            .map_err(|error| ToolError::ExecutionError(error.to_string()))?;
        connection
            .execute(
                "CREATE TABLE IF NOT EXISTS embedded_facts (value TEXT NOT NULL, agent_id TEXT NOT NULL)",
                (),
            )
            .await
            .map_err(|error| ToolError::ExecutionError(error.to_string()))?;
        connection
            .execute(
                "INSERT INTO embedded_facts (value, agent_id) VALUES (?1, ?2)",
                turso::params![value, ctx.agent_id.clone()],
            )
            .await
            .map_err(|error| ToolError::ExecutionError(error.to_string()))?;

        Ok(ToolEffect::Output(ToolOutput::new("fact recorded".into())))
    }
}

struct ForbiddenTool;

#[async_trait]
impl Tool for ForbiddenTool {
    fn name(&self) -> &str {
        "forbidden_record"
    }

    fn description(&self) -> &str {
        "A fixture tool whose capability must be denied"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }

    fn capability(&self) -> Option<&str> {
        Some("records.forbidden")
    }

    async fn execute(
        &self,
        _params: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<ToolEffect, ToolError> {
        panic!("governance must deny this tool before execution")
    }
}

struct FixtureProvider {
    call_count: AtomicUsize,
    requests: Arc<Mutex<Vec<InferenceRequest>>>,
}

impl InferenceProvider for FixtureProvider {
    fn stream<'a>(
        &'a self,
        request: InferenceRequest,
        _options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        self.requests.lock().unwrap().push(request);
        let call = self.call_count.fetch_add(1, Ordering::Relaxed);
        Box::pin(async move {
            let mut events = vec![Ok(InferenceEvent::MessageStart {
                role: "assistant".into(),
                model: "fixture-model".into(),
                provider_id: "fixture".into(),
            })];
            if call < 2 {
                events.extend([
                    Ok(InferenceEvent::ToolCallStart {
                        id: format!("record-{call}"),
                        name: if call == 0 {
                            "forbidden_record".into()
                        } else {
                            "record_fact".into()
                        },
                    }),
                    Ok(InferenceEvent::ToolCallDelta {
                        delta: if call == 0 {
                            "{}".into()
                        } else {
                            r#"{"value":"adapter boundary verified"}"#.into()
                        },
                    }),
                ]);
            } else {
                events.push(Ok(InferenceEvent::MessageDelta {
                    content: if call == 2 {
                        "The fact was recorded.".into()
                    } else {
                        "Review complete.".into()
                    },
                }));
            }
            events.push(Ok(InferenceEvent::MessageEnd {
                input_tokens: 4,
                output_tokens: 3,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
                stop_reason: None,
            }));
            Ok(Box::pin(futures::stream::iter(events)) as InferenceStream)
        })
    }
}

#[tokio::test]
async fn public_embedding_path_supports_harnesses_tools_governance_and_persistence() -> Result<()> {
    let tmp = tempdir()?;
    let mut config = TurinConfig::default();
    config.kernel.workspace_root = tmp.path().to_string_lossy().into_owned();
    config.persistence = PersistenceConfig::with_state_path(
        tmp.path().join("state.db").to_string_lossy().into_owned(),
    );
    config.embeddings = Some(EmbeddingConfig::noop());
    config.agent = AgentConfig {
        id: "default".into(),
        model: "fixture-model".into(),
        provider: "fixture".into(),
        system_prompt: "Main agent.".into(),
        ..AgentConfig::default()
    };
    config.agents.insert(
        "reviewer".into(),
        AgentConfig {
            id: "reviewer".into(),
            model: "fixture-model".into(),
            provider: "fixture".into(),
            system_prompt: "Review agent.".into(),
            harness: Some("review".into()),
            ..AgentConfig::default()
        },
    );
    config
        .harnesses
        .insert("review".into(), HarnessConfig::default());
    config.providers = HashMap::from([(
        "fixture".into(),
        ProviderConfig {
            kind: "fixture".into(),
            ..ProviderConfig::default()
        },
    )]);
    config.tools.selection.allow = Some(vec!["forbidden_record".into(), "record_fact".into()]);
    config.governance = GovernanceConfig {
        profile: "embedded".into(),
        enforcement_enabled: true,
        unmatched_capability: GovernanceUnmatchedCapability::Deny,
        capabilities: [("records.write".into(), serde_json::Value::Bool(true))]
            .into_iter()
            .collect(),
        ..GovernanceConfig::default()
    };

    let mut tools = ToolRegistry::new();
    tools.register(Box::new(ForbiddenTool))?;
    tools.register(Box::new(RecordFactTool))?;
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut kernel = Kernel::builder(config)
        .with_tool_registry(tools)
        .with_default_harness(|| Ok(Box::new(PromptHarness(" [main harness]")) as Box<dyn Harness>))
        .with_harness("review", || {
            Ok(Box::new(PromptHarness(" [review harness]")) as Box<dyn Harness>)
        })
        .build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;
    kernel.add_client(
        "fixture".into(),
        ProviderClient::new(
            "fixture",
            Arc::new(FixtureProvider {
                call_count: AtomicUsize::new(0),
                requests: Arc::clone(&requests),
            }),
        ),
    );

    let mut main_session = kernel.create_session().await;
    kernel
        .run(
            &mut main_session,
            Some("Record the verification fact.".into()),
        )
        .await?;
    let main_session_id = uuid::Uuid::parse_str(main_session.identity.session_id())?;
    kernel.end_session(&mut main_session).await?;

    let mut review_session = kernel.create_session_for_agent("reviewer").await;
    kernel
        .run(
            &mut review_session,
            Some("Review the embedding result.".into()),
        )
        .await?;
    kernel.end_session(&mut review_session).await?;

    {
        let captured = requests.lock().unwrap();
        assert_eq!(captured.len(), 4);
        assert_eq!(
            captured[0].system.as_deref(),
            Some("Main agent. [main harness]")
        );
        assert_eq!(
            captured[3].system.as_deref(),
            Some("Review agent. [review harness]")
        );
        assert!(captured[0].tools.as_ref().is_some_and(|tools| {
            tools
                .iter()
                .any(|definition| definition.name == "record_fact")
        }));
        assert!(captured[0].tools.as_ref().is_some_and(|tools| {
            tools
                .iter()
                .any(|definition| definition.name == "forbidden_record")
        }));
    }

    let governance = kernel
        .governance_manager()
        .snapshot_for_agent(Some("default"));
    assert!(governance.enforcement_enabled);
    assert_eq!(
        governance.capabilities.get("records.write"),
        Some(&serde_json::Value::Bool(true))
    );

    let store = kernel.store_manager().get_default().await?;
    assert!(
        store
            .get_session_row_by_public_id(main_session_id)
            .await?
            .is_some()
    );
    let connection = store.get_connection().await?;
    let mut rows = connection
        .query(
            "SELECT value, agent_id FROM embedded_facts ORDER BY rowid DESC LIMIT 1",
            (),
        )
        .await?;
    let row = rows
        .next()
        .await?
        .expect("custom tool should persist a fact");
    assert_eq!(row.get::<String>(0)?, "adapter boundary verified");
    assert_eq!(row.get::<String>(1)?, "default");
    Ok(())
}
