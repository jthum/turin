use anyhow::Result;
use futures::future::BoxFuture;
use futures::stream;
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::tempdir;
use turin::inference::provider::{
    InferenceContent, InferenceEvent, InferenceProvider, InferenceRequest, InferenceStream,
    ProviderClient, RequestOptions, SdkError,
};
use turin::kernel::Kernel;
use turin::kernel::config::{
    AgentConfig, EmbeddingConfig, HarnessConfig, PersistenceConfig, ProviderConfig, TurinConfig,
};
use turin::kernel::identity::ContextSelector;

struct ToolMockProvider {
    tool_name: String,
    tool_args: serde_json::Value,
}

impl InferenceProvider for ToolMockProvider {
    fn stream<'a>(
        &'a self,
        _request: InferenceRequest,
        _options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        let tool_name = self.tool_name.clone();
        let tool_args = self.tool_args.clone();
        Box::pin(async move {
            let events = vec![
                Ok(InferenceEvent::MessageStart {
                    role: "assistant".to_string(),
                    model: "mock-model".to_string(),
                    provider_id: "mock".to_string(),
                }),
                Ok(InferenceEvent::ToolCallStart {
                    id: "test-call-id".to_string(),
                    name: tool_name,
                }),
                Ok(InferenceEvent::ToolCallDelta {
                    delta: tool_args.to_string(),
                }),
                Ok(InferenceEvent::MessageEnd {
                    input_tokens: 10,
                    output_tokens: 5,
                    stop_reason: None,
                }),
            ];
            Ok(Box::pin(stream::iter(events)) as InferenceStream)
        })
    }
}

struct HeaderCaptureProvider {
    seen: Arc<std::sync::Mutex<(bool, bool, Option<u32>)>>,
}

impl InferenceProvider for HeaderCaptureProvider {
    fn stream<'a>(
        &'a self,
        _request: InferenceRequest,
        options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        let seen = self.seen.clone();
        Box::pin(async move {
            let (static_header, dynamic_header, max_retries) = if let Some(opts) = options {
                (
                    opts.headers.get("x-static").is_some(),
                    opts.headers.get("x-dynamic").is_some(),
                    opts.max_retries,
                )
            } else {
                (false, false, None)
            };
            *seen.lock().unwrap() = (static_header, dynamic_header, max_retries);

            let events = vec![
                Ok(InferenceEvent::MessageStart {
                    role: "assistant".to_string(),
                    model: "mock-model".to_string(),
                    provider_id: "mock".to_string(),
                }),
                Ok(InferenceEvent::MessageDelta {
                    content: "ok".to_string(),
                }),
                Ok(InferenceEvent::MessageEnd {
                    input_tokens: 1,
                    output_tokens: 1,
                    stop_reason: None,
                }),
            ];
            Ok(Box::pin(stream::iter(events)) as InferenceStream)
        })
    }
}

#[tokio::test]
async fn test_harness_rejection() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    // Create a safety harness that blocks 'shell_exec'
    let harness_code = r#"
        function on_tool_call(call)
            if call.name == "shell_exec" then
                return REJECT, "Security policy: shell_exec is forbidden"
            end
            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("safety.lua"), harness_code)?;

    let mut providers = HashMap::new();
    // We add a dummy config, but we will inject our custom provider client
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: None,
            ..ProviderConfig::default()
        },
    );

    let config = TurinConfig {
        agent: AgentConfig {
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "You are a test assistant.".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: None,
            idle_grace_secs: None,
        },
        agents: std::collections::HashMap::new(),
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 5,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        persistence: PersistenceConfig {
            database_path: db_path.to_str().unwrap().to_string(),
        },
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
        },
        providers,
        embeddings: Some(EmbeddingConfig::NoOp),
        governance: turin::kernel::config::GovernanceConfig::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;

    // Inject custom provider
    let mock_provider = Arc::new(ToolMockProvider {
        tool_name: "shell_exec".to_string(),
        tool_args: serde_json::json!({"command": "rm -rf /"}),
    });
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new("mock", mock_provider),
    );

    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;

    // Run the agent. The mock provider will trigger 'shell_exec'.
    // The harness should reject it.
    kernel
        .run(&mut session, Some("Run a dangerous command".to_string()))
        .await?;

    // Verify turn index incremented
    assert!(session.turn_index > 0);

    // Verify that the assistant history contains the rejection message
    // Turn 0: User message
    // Turn 1: Assistant ToolUse -> [Harness Rejection] -> [Assistant responds to rejection?]
    // Actually, Kernel::run_task does:
    // 1. Assistant streams ToolCall
    // 2. Kernel evaluates harness
    // 3. Harness REJECTS.
    // 4. Kernel appends [HARNESS REJECTED] as ToolResult to history.
    // 5. Kernel enters next turn of loop because tool_results were added.
    // 6. Next turn: LLM sees rejection.

    // In our test, the MockProvider ALWAYS returns the same ToolCall.
    // To prevent infinite loop, max_turns=5 will eventually stop it.

    // Let's check history for the rejection string
    let mut found_rejection = false;
    for msg in &session.history {
        for content in &msg.content {
            if let InferenceContent::ToolResult { content, .. } = content
                && content.contains("Security policy: shell_exec is forbidden")
            {
                found_rejection = true;
            }
        }
    }

    assert!(
        found_rejection,
        "Harness rejection message not found in session history"
    );

    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test]
async fn test_governed_mode_denies_shell_exec_tool_at_kernel_fallback() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_governed_tool_fallback.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    // No rejection hook here; governance should deny at kernel tool execution layer.
    std::fs::write(
        harness_dir.join("allow.lua"),
        r#"
            function on_tool_call(call)
                return ALLOW
            end
        "#,
    )?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: None,
            ..ProviderConfig::default()
        },
    );

    let config = TurinConfig {
        agent: AgentConfig {
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Governed tool fallback test".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: None,
            idle_grace_secs: None,
        },
        agents: std::collections::HashMap::new(),
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 3,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        persistence: PersistenceConfig {
            database_path: db_path.to_str().unwrap().to_string(),
        },
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
        },
        providers,
        embeddings: Some(EmbeddingConfig::NoOp),
        governance: turin::kernel::config::GovernanceConfig {
            profile: turin::kernel::config::GovernanceProfile::Governed,
            enforcement_enabled: true,
            ..turin::kernel::config::GovernanceConfig::default()
        },
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;

    let mock_provider = Arc::new(ToolMockProvider {
        tool_name: "shell_exec".to_string(),
        tool_args: serde_json::json!({"command": "echo governed-check"}),
    });
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new("mock", mock_provider),
    );

    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Try shell exec".to_string()))
        .await?;

    let mut found_governance_denial = false;
    for msg in &session.history {
        for content in &msg.content {
            if let InferenceContent::ToolResult { content, .. } = content
                && content.contains("Governance denial")
                && content.contains("shell.exec")
            {
                found_governance_denial = true;
            }
        }
    }

    assert!(
        found_governance_denial,
        "Expected governed-mode shell_exec denial from kernel tool fallback"
    );

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test]
async fn test_harness_request_options_passthrough() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_headers.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness_code = r#"
        function on_turn_prepare(ctx)
            local opts = ctx.request_options or {}
            opts.headers = opts.headers or {}
            opts.headers["x-dynamic"] = "from-harness"
            opts.max_retries = 1
            ctx.request_options = opts
            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("headers.lua"), harness_code)?;

    let mut static_headers = HashMap::new();
    static_headers.insert("x-static".to_string(), "from-config".to_string());

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: None,
            headers: static_headers,
            max_retries: Some(2),
            ..ProviderConfig::default()
        },
    );

    let config = TurinConfig {
        agent: AgentConfig {
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Header test".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: None,
            idle_grace_secs: None,
        },
        agents: std::collections::HashMap::new(),
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 2,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        persistence: PersistenceConfig {
            database_path: db_path.to_str().unwrap().to_string(),
        },
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
        },
        providers,
        embeddings: Some(EmbeddingConfig::NoOp),
        governance: turin::kernel::config::GovernanceConfig::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;

    let seen = Arc::new(std::sync::Mutex::new((false, false, None)));
    let provider = Arc::new(HeaderCaptureProvider { seen: seen.clone() });
    kernel.add_client("mock".to_string(), ProviderClient::new("mock", provider));

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("emit headers".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;

    let captured = *seen.lock().unwrap();
    assert!(captured.0, "expected config header to be passed through");
    assert!(captured.1, "expected harness header to be passed through");
    assert_eq!(captured.2, Some(1), "expected harness override for retries");

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_stdlib_context_api_kv_memory_and_tier2() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_stdlib_ctx.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness_code = r#"
        function on_turn_prepare(ctx)
            local ident = agent.session.identity()
            if not ident.agent_id or not ident.session_id then
                error("agent.session.identity missing fields")
            end

            local project = runtime.context("project", "alpha", { namespace = "notes" })
            local ok, err = runtime.kv.set("raw_key", "raw_val", project)
            if not ok then error("runtime.kv.set failed: " .. tostring(err)) end

            local raw_val, raw_err = runtime.kv.get("raw_key", project)
            if raw_err ~= nil then error("runtime.kv.get err: " .. tostring(raw_err)) end
            if raw_val ~= "raw_val" then error("runtime.kv.get mismatch") end

            local k = kv.as(project)
            local ok2, err2 = k.set("scoped_key", "scoped_val")
            if not ok2 then error("kv.as.set failed: " .. tostring(err2)) end
            local scoped_val, scoped_err = k.get("scoped_key")
            if scoped_err ~= nil then error("kv.as.get err: " .. tostring(scoped_err)) end
            if scoped_val ~= "scoped_val" then error("kv.as.get mismatch") end

            local m = memory.as(project)
            local okm, em = m.store("alpha memory", { source = "test" })
            if not okm then error("memory.as.store failed: " .. tostring(em)) end
            local hits, hm = m.search("alpha", { limit = 5 })
            if hits == nil then error("memory.as.search failed: " .. tostring(hm)) end
            if #hits < 1 then error("memory.as.search returned no hits") end

            local oks, es = session.kv.set("session_seen", "1")
            if not oks then error("session.kv.set failed: " .. tostring(es)) end
            local sv, se = session.kv.get("session_seen")
            if se ~= nil then error("session.kv.get err: " .. tostring(se)) end
            if sv ~= "1" then error("session.kv.get mismatch") end

            local oku, ue = user.kv.set("x", "y")
            if oku ~= false or ue == nil then
                error("user.kv.set should fail without user_id")
            end

            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("stdlib_ctx.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let config = TurinConfig {
        agent: AgentConfig {
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Stdlib API test".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: None,
            idle_grace_secs: None,
        },
        agents: std::collections::HashMap::new(),
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 1,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        persistence: PersistenceConfig {
            database_path: db_path.to_str().unwrap().to_string(),
        },
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
        },
        providers,
        embeddings: Some(EmbeddingConfig::NoOp),
        governance: turin::kernel::config::GovernanceConfig::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    let session_public_id = session.identity.session_id().to_string();
    kernel
        .run(&mut session, Some("exercise stdlib".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;

    let project_selector = ContextSelector {
        tags: vec!["project:alpha".to_string()],
        namespace: "notes".to_string(),
        visibility: "private".to_string(),
    };
    let project_store = kernel
        .store_manager()
        .open(&turin::persistence::manager::StoreSelector::Alias(
            project_selector.to_alias(),
        ))
        .await?;
    assert_eq!(
        project_store.kv_get("raw_key").await?,
        Some("raw_val".to_string())
    );
    assert_eq!(
        project_store.kv_get("scoped_key").await?,
        Some("scoped_val".to_string())
    );

    let ctx_session_uuid: uuid::Uuid = project_store
        .kv_get("__turin_context_session_public_id")
        .await?
        .expect("context session uuid kv missing")
        .parse()?;
    let ctx_internal_id = project_store
        .get_session_by_public_id(ctx_session_uuid)
        .await?
        .expect("context session row missing");
    let hits = project_store
        .search_memories(ctx_internal_id, None, Some("alpha"), 5)
        .await?;
    assert!(!hits.is_empty(), "expected context memory rows");

    let session_selector = ContextSelector {
        tags: vec![format!("session:{}", session_public_id)],
        namespace: "default".to_string(),
        visibility: "private".to_string(),
    };
    let session_store = kernel
        .store_manager()
        .open(&turin::persistence::manager::StoreSelector::Alias(
            session_selector.to_alias(),
        ))
        .await?;
    assert_eq!(
        session_store.kv_get("session_seen").await?,
        Some("1".to_string())
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runtime_policy_api_round_trip() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_runtime_policy.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness_code = r#"
        function on_turn_prepare(ctx)
            local ok, err = runtime.policy.set("spawn.max_depth", 2)
            if not ok then error("global policy set failed: " .. tostring(err)) end

            local v, ge = runtime.policy.get("spawn.max_depth")
            if ge ~= nil then error("global policy get err: " .. tostring(ge)) end
            if v ~= 2 then error("global policy mismatch: " .. tostring(v)) end

            local aok, ae = runtime.policy.set("queue.max_depth", 9, {
                scope = "agent",
                agent_id = "default",
            })
            if not aok then error("agent policy set failed: " .. tostring(ae)) end

            local av, ave = runtime.policy.get("queue.max_depth", {
                scope = "agent",
                agent_id = "default",
            })
            if ave ~= nil then error("agent policy get err: " .. tostring(ave)) end
            if av ~= 9 then error("agent policy mismatch: " .. tostring(av)) end

            local bad_ok, bad_err = runtime.policy.set("spawn.max_depth", "bad")
            if bad_ok ~= false or bad_err == nil then
                error("invalid policy value should return false + err")
            end

            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("policy.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let config = TurinConfig {
        agent: AgentConfig {
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Runtime policy test".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: None,
            idle_grace_secs: None,
        },
        agents: std::collections::HashMap::new(),
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 1,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        persistence: PersistenceConfig {
            database_path: db_path.to_str().unwrap().to_string(),
        },
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
        },
        providers,
        embeddings: Some(EmbeddingConfig::NoOp),
        governance: turin::kernel::config::GovernanceConfig::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("exercise runtime policy".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;

    let global_value = kernel
        .policy_manager()
        .get(
            "spawn.max_depth",
            &turin::kernel::policy::PolicyScope::default(),
        )
        .await?;
    assert_eq!(global_value, Some(serde_json::json!(2)));

    let agent_value = kernel
        .policy_manager()
        .get(
            "queue.max_depth",
            &turin::kernel::policy::PolicyScope {
                agent_id: Some("default".to_string()),
                ..turin::kernel::policy::PolicyScope::default()
            },
        )
        .await?;
    assert_eq!(agent_value, Some(serde_json::json!(9)));

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runtime_governance_observability_api() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_runtime_governance.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness_code = r#"
        function on_turn_prepare(ctx)
            local profile = runtime.governance.profile()
            if profile ~= "balanced" then
                error("runtime.governance.profile mismatch: " .. tostring(profile))
            end

            local snap, se = runtime.governance.snapshot()
            if snap == nil then error("runtime.governance.snapshot failed: " .. tostring(se)) end
            if snap.profile ~= "balanced" then error("snapshot.profile mismatch") end
            if snap.enforcement_enabled ~= false then error("snapshot.enforcement_enabled mismatch") end
            if snap.capabilities_observability_only ~= true then error("snapshot should be observability-only in G1") end
            if snap.audit_mode ~= "observational" then error("snapshot.audit_mode mismatch") end
            if snap.import_mode ~= "mixed" then error("snapshot.import_mode mismatch") end
            if snap.grants_enabled ~= true then error("snapshot.grants_enabled mismatch") end

            local saw_db_query = false
            for k, v in pairs(snap.preset_capabilities or {}) do
                if k == "runtime.db.query" and v == true then
                    saw_db_query = true
                end
            end
            if not saw_db_query then error("expected preset runtime.db.query capability") end

            local reviewer, re = runtime.governance.agent("reviewer")
            if reviewer == nil then error("runtime.governance.agent failed: " .. tostring(re)) end
            if reviewer.subject_agent_id ~= "reviewer" then error("reviewer subject_agent_id mismatch") end

            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("governance.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut roots = std::collections::HashMap::new();
    roots.insert(
        "core".to_string(),
        turin::kernel::config::GovernanceRootConfig {
            path: "harness/core".to_string(),
            writable_hint: false,
            default_profile: Some("core_full".to_string()),
            max_capabilities: std::collections::HashMap::new(),
        },
    );
    let mut agents = std::collections::HashMap::new();
    agents.insert(
        "reviewer".to_string(),
        turin::kernel::config::GovernanceAgentCapabilitiesConfig {
            capability_profile: Some("reviewer_ro".to_string()),
            max_capabilities: std::collections::HashMap::new(),
            allowed_child_agents: vec!["worker".to_string()],
        },
    );

    let config = TurinConfig {
        agent: AgentConfig {
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Runtime governance test".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: None,
            idle_grace_secs: None,
        },
        agents: std::collections::HashMap::new(),
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 1,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        persistence: PersistenceConfig {
            database_path: db_path.to_str().unwrap().to_string(),
        },
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
        },
        providers,
        embeddings: Some(EmbeddingConfig::NoOp),
        governance: turin::kernel::config::GovernanceConfig {
            profile: turin::kernel::config::GovernanceProfile::Balanced,
            enforcement_enabled: false,
            audit: turin::kernel::config::GovernanceAuditConfig {
                mode: turin::kernel::config::GovernanceAuditMode::Observational,
                include_capability_context: true,
                persist_before_hooks: None,
            },
            import: turin::kernel::config::GovernanceImportConfig {
                mode: turin::kernel::config::GovernanceImportMode::Mixed,
                default_root: Some("core".to_string()),
                allow_unscoped_in_open: false,
            },
            roots,
            agents,
            grants: turin::kernel::config::GovernanceGrantsConfig {
                enabled: true,
                max_ttl_ms: Some(60_000),
                require_audit_reason: true,
            },
        },
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise runtime governance api".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_governance_profile_enforcement_blocks_high_risk_runtime_apis() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_governed_enforcement.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness_code = r#"
        function on_turn_prepare(ctx)
            local d, de = runtime.governance.check("runtime.db.exec")
            if d == nil then error("runtime.governance.check failed: " .. tostring(de)) end
            if d.allowed ~= false then error("runtime.db.exec should be denied in governed mode") end
            if d.enforcement_enabled ~= true then error("enforcement_enabled should be true") end

            local q, qe = runtime.db.query("SELECT 1 AS n")
            if q == nil then error("runtime.db.query should be allowed: " .. tostring(qe)) end
            if #q < 1 then error("runtime.db.query should return one row") end

            local changed, ce = runtime.db.exec("CREATE TABLE IF NOT EXISTS g2_guard (id INTEGER)")
            if changed ~= nil or ce == nil then
                error("runtime.db.exec should be denied in governed mode")
            end

            local okp, ep = runtime.policy.set("spawn.max_depth", 1)
            if okp ~= false or ep == nil then
                error("runtime.policy.set should be denied in governed mode")
            end

            local status_list, sle = runtime.agent.list()
            if status_list == nil then error("runtime.agent.list should still be allowed: " .. tostring(sle)) end

            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("governed.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let config = TurinConfig {
        agent: AgentConfig {
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Governed enforcement test".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: None,
            idle_grace_secs: None,
        },
        agents: std::collections::HashMap::new(),
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 1,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        persistence: PersistenceConfig {
            database_path: db_path.to_str().unwrap().to_string(),
        },
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
        },
        providers,
        embeddings: Some(EmbeddingConfig::NoOp),
        governance: turin::kernel::config::GovernanceConfig {
            profile: turin::kernel::config::GovernanceProfile::Governed,
            enforcement_enabled: true,
            ..turin::kernel::config::GovernanceConfig::default()
        },
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise governed enforcement".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runtime_db_api_and_context_glob() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_runtime_db_api.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness_code = r#"
        function on_turn_prepare(ctx)
            local aliases, ae = runtime.context.glob("*")
            if ae ~= nil then error("runtime.context.glob err: " .. tostring(ae)) end
            local saw_state = false
            for _, a in ipairs(aliases) do
                if a == "state" then saw_state = true end
            end
            if not saw_state then error("runtime.context.glob missing state alias") end

            local h, he = runtime.db.open({ path = ".turin/runtime/test_dynamic.db" })
            if h == nil then error("runtime.db.open failed: " .. tostring(he)) end
            if h.handle == nil then error("runtime.db.open missing handle field") end

            local changed, ce = runtime.db.exec(
                "CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, name TEXT)",
                nil,
                { handle = h.handle }
            )
            if changed == nil and ce ~= nil then error("runtime.db.exec create failed: " .. tostring(ce)) end

            local ins, ie = runtime.db.exec(
                "INSERT INTO items (name) VALUES (?1)",
                { "alice" },
                { handle = h.handle }
            )
            if ins == nil then error("runtime.db.exec insert failed: " .. tostring(ie)) end

            local rows, qe = runtime.db.query(
                "SELECT name FROM items WHERE name = ?1",
                { "alice" },
                { handle = h.handle }
            )
            if rows == nil then error("runtime.db.query failed: " .. tostring(qe)) end
            if #rows < 1 or rows[1].name ~= "alice" then error("runtime.db.query mismatch") end

            local list, le = runtime.db.list()
            if list == nil then error("runtime.db.list failed: " .. tostring(le)) end
            if #list < 1 then error("runtime.db.list should include open handle") end

            local closed, clo_err = runtime.db.close(h)
            if not closed then error("runtime.db.close failed: " .. tostring(clo_err)) end

            local rows2, qe2 = runtime.db.query(
                "SELECT name FROM items",
                nil,
                { handle = h.handle }
            )
            if rows2 ~= nil or qe2 == nil then
                error("runtime.db.query on closed handle should fail")
            end

            local okp, ep = runtime.policy.set("db.allow_dynamic_open", false)
            if not okp then error("policy set failed: " .. tostring(ep)) end
            local h2, he2 = runtime.db.open({ path = ".turin/runtime/blocked.db" })
            if h2 ~= nil or he2 == nil then
                error("runtime.db.open should be blocked when db.allow_dynamic_open=false")
            end

            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("runtime_db.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let config = TurinConfig {
        agent: AgentConfig {
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Runtime DB API test".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: None,
            idle_grace_secs: None,
        },
        agents: std::collections::HashMap::new(),
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 1,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        persistence: PersistenceConfig {
            database_path: db_path.to_str().unwrap().to_string(),
        },
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
        },
        providers,
        embeddings: Some(EmbeddingConfig::NoOp),
        governance: turin::kernel::config::GovernanceConfig::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("exercise runtime db api".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;

    let dyn_db = tmp
        .path()
        .join(".turin")
        .join("runtime")
        .join("test_dynamic.db");
    assert!(
        dyn_db.exists(),
        "expected runtime.db.open path to create database"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runtime_agent_peer_submit_await_and_status() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_runtime_agent_peer.db");
    let orchestrator_harness_dir = tmp.path().join("harnesses_orchestrator");
    let worker_harness_dir = tmp.path().join("harnesses_worker");
    std::fs::create_dir(&orchestrator_harness_dir)?;
    std::fs::create_dir(&worker_harness_dir)?;

    let orchestrator_harness = r#"
        function on_turn_prepare(ctx)
            local before, be = runtime.agent.get_status("worker")
            if before == nil then error("runtime.agent.get_status before submit failed: " .. tostring(be)) end
            if before.running ~= false then
                error("worker should not be running before submit")
            end

            local task_id, se = runtime.agent.submit("worker", { prompt = "say hello", title = "hello" })
            if task_id == nil then error("runtime.agent.submit failed: " .. tostring(se)) end

            local res, ae = runtime.agent.await(task_id, { timeout_ms = 5000 })
            if res == nil then error("runtime.agent.await failed: " .. tostring(ae)) end
            if res.agent_id ~= "worker" then error("runtime.agent.await wrong agent") end
            if res.status ~= "success" then error("runtime.agent.await status should be success") end
            if res.output ~= "worker-ok" then
                error("runtime.agent.await output mismatch: " .. tostring(res.output))
            end

            local list, le = runtime.agent.list()
            if list == nil then error("runtime.agent.list failed: " .. tostring(le)) end
            local saw_worker = false
            for _, item in ipairs(list) do
                if item.agent_id == "worker" then saw_worker = true end
            end
            if not saw_worker then error("runtime.agent.list missing worker") end

            local after, afe = runtime.agent.get_status("worker")
            if after == nil then error("runtime.agent.get_status after submit failed: " .. tostring(afe)) end
            if after.awaiting_results ~= 0 then error("awaiting_results should be 0 after await") end

            return ALLOW
        end
    "#;
    std::fs::write(
        orchestrator_harness_dir.join("orchestrator.lua"),
        orchestrator_harness,
    )?;

    // Worker harness intentionally empty to validate per-agent harness_dir routing.
    std::fs::write(worker_harness_dir.join("worker.lua"), "-- worker harness\n")?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("worker-ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut agents = std::collections::HashMap::new();
    agents.insert(
        "worker".to_string(),
        AgentConfig {
            id: "worker".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Worker".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Stateless,
            harness_dir: Some(worker_harness_dir.to_str().unwrap().to_string()),
            idle_grace_secs: None,
        },
    );

    let config = TurinConfig {
        agent: AgentConfig {
            id: "orchestrator".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Orchestrator".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: None,
            idle_grace_secs: None,
        },
        agents,
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 1,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        persistence: PersistenceConfig {
            database_path: db_path.to_str().unwrap().to_string(),
        },
        harness: HarnessConfig {
            directory: orchestrator_harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
        },
        providers,
        embeddings: Some(EmbeddingConfig::NoOp),
        governance: turin::kernel::config::GovernanceConfig::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise runtime agent peer".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}
