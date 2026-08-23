use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;

use crate::inference::embeddings::EmbeddingProvider;
use crate::kernel::governance::GovernanceManager;
use crate::kernel::harness::RustHarnessFactories;
use crate::kernel::harness_runtime::HarnessAdapterFactory;
use crate::kernel::policy::RuntimePolicyManager;
use crate::kernel::{
    Kernel, TurinConfig,
    agent_manager::{AgentManager, SharedPeerRuntimeContext},
    execution_host::{ExecutionHost, SessionPersistenceCoordinator},
    harness_manager::HarnessManager,
};
use crate::persistence::manager::StoreManager;
use crate::tools::builtins::create_default_registry;
use crate::tools::registry::ToolRegistry;

/// Builder for constructing a `Kernel` instance.
pub struct RuntimeBuilder {
    config: TurinConfig,
    json: bool,
    tool_registry: ToolRegistry,

    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    rust_harness_factories: RustHarnessFactories,
    script_harness_adapter: Option<Arc<dyn HarnessAdapterFactory>>,
}

impl RuntimeBuilder {
    /// Create a new builder with the given configuration.
    pub fn new(config: TurinConfig) -> Self {
        Self {
            config,
            json: false,
            tool_registry: create_default_registry(),

            embedding_provider: None,
            rust_harness_factories: HashMap::new(),
            script_harness_adapter: crate::kernel::harness_runtime::default_script_adapter_factory(
            )
            .ok(),
        }
    }

    /// Enable JSON output mode (NDJSON).
    pub fn json_mode(mut self, json: bool) -> Self {
        self.json = json;
        self
    }

    /// Register a custom tool registry (overwriting defaults).
    pub fn with_tool_registry(mut self, registry: ToolRegistry) -> Self {
        self.tool_registry = registry;
        self
    }

    /// Use a Rust API harness for the default harness binding.
    pub fn with_default_harness<F>(self, factory: F) -> Self
    where
        F: crate::kernel::harness::HarnessFactory + 'static,
    {
        self.with_harness("default", factory)
    }

    /// Register a Rust API harness factory for a configured harness ID.
    pub fn with_harness<F>(mut self, harness_id: impl Into<String>, factory: F) -> Self
    where
        F: crate::kernel::harness::HarnessFactory + 'static,
    {
        self.rust_harness_factories
            .insert(harness_id.into(), Arc::new(factory));
        self
    }

    /// Use a script/runtime adapter for harness IDs without a Rust factory.
    pub fn with_harness_adapter(mut self, adapter: Arc<dyn HarnessAdapterFactory>) -> Self {
        self.script_harness_adapter = Some(adapter);
        self
    }

    /// Build the Kernel.
    pub fn build(self) -> Result<Kernel> {
        let available_tools = self.tool_registry.names();
        for agent_id in std::iter::once(self.config.agent.id.as_str())
            .chain(self.config.agents.keys().map(String::as_str))
        {
            crate::tools::policy::resolve_effective_tools_config_for_registry(
                &self.config,
                agent_id,
                None,
                &available_tools,
            )
            .map_err(|error| {
                anyhow::anyhow!("invalid tool registry for agent '{}': {}", agent_id, error)
            })?;
        }
        let store_manager = Arc::new(StoreManager::new(
            &self.config.kernel.workspace_root,
            &self.config.layout.stores_dir,
        ));
        let config_arc = Arc::new(self.config);
        let agent_manager = Arc::new(AgentManager::new(config_arc.clone(), store_manager.clone()));
        let policy_manager = Arc::new(RuntimePolicyManager::new());
        let governance_manager = Arc::new(GovernanceManager::new(config_arc.governance.clone()));
        let rust_harness_factories = Arc::new(self.rust_harness_factories);
        let harness_manager = Arc::new(HarnessManager::from_config_with_harnesses(
            config_arc.as_ref(),
            rust_harness_factories.as_ref(),
            self.script_harness_adapter.as_ref(),
        )?);
        let shared_harness_manager = Arc::new(std::sync::RwLock::new(Arc::clone(&harness_manager)));
        let persistence_locks = Arc::new(SessionPersistenceCoordinator::default());
        agent_manager.bind_shared_runtime(SharedPeerRuntimeContext {
            json: self.json,
            tool_registry: self.tool_registry.clone(),
            policy_manager: Arc::clone(&policy_manager),
            governance_manager: Arc::clone(&governance_manager),
            harness_manager: shared_harness_manager,
            persistence_locks: Arc::clone(&persistence_locks),
            script_harness_adapter: self.script_harness_adapter.clone(),
        });
        Ok(Kernel {
            host: ExecutionHost {
                config: config_arc,
                json: self.json,
                tool_registry: self.tool_registry,
                store_manager,
                agent_manager,
                policy_manager,
                governance_manager,
                harness_manager,
                scheduler: None,
                persistence_locks,
                clients: HashMap::new(),
                embedding_provider: self.embedding_provider,
                rust_harness_factories: Some(rust_harness_factories),
                script_harness_adapter: self.script_harness_adapter,
                mcp_clients: Vec::new(),
            },
            check_watcher: Arc::new(std::sync::Mutex::new(None)),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::harness::verdict::Verdict;
    use crate::kernel::harness::{Harness, HarnessFactory};
    use crate::kernel::harness_contract::{HarnessActionRequest, HarnessHook, HarnessSignal};

    struct RecordingHarness {
        calls: Arc<AtomicUsize>,
        signals: Arc<AtomicUsize>,
        actions: Arc<AtomicUsize>,
    }

    impl Harness for RecordingHarness {
        fn runtime_signal_topics(&self) -> Vec<String> {
            vec!["build.*".to_string()]
        }

        fn on_hook(&mut self, _hook: HarnessHook<'_>) -> Result<Verdict> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(Verdict::Allow)
        }

        fn on_signal(&mut self, signal: HarnessSignal<'_>) -> Result<()> {
            assert_eq!(signal.topic, "build.complete");
            self.signals.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn on_action(
            &mut self,
            request: HarnessActionRequest<'_>,
        ) -> Result<Option<serde_json::Value>> {
            self.actions.fetch_add(1, Ordering::Relaxed);
            Ok((request.name == "build.status").then_some(request.params))
        }
    }

    #[test]
    fn rust_factory_creates_isolated_session_harnesses_without_watch_roots() -> Result<()> {
        let calls = Arc::new(AtomicUsize::new(0));
        let signals = Arc::new(AtomicUsize::new(0));
        let actions = Arc::new(AtomicUsize::new(0));
        let factory_calls = Arc::clone(&calls);
        let factory_signals = Arc::clone(&signals);
        let factory_actions = Arc::clone(&actions);
        let factory: Arc<dyn HarnessFactory> = Arc::new(move || {
            Ok(Box::new(RecordingHarness {
                calls: Arc::clone(&factory_calls),
                signals: Arc::clone(&factory_signals),
                actions: Arc::clone(&factory_actions),
            }) as Box<dyn Harness>)
        });
        let kernel = RuntimeBuilder::new(TurinConfig::default())
            .with_default_harness(factory)
            .build()?;
        let runtime = kernel.host.harness_definition_for_agent("default");

        assert!(runtime.watch_roots().is_empty());
        let first = runtime.create_instance(kernel.host.harness_init_context())?;
        let second = runtime.create_instance(kernel.host.harness_init_context())?;
        let args = serde_json::json!({});
        let first_verdict = first.evaluate_hook(HarnessHook::ToolCall {
            name: "test",
            id: "call-1",
            args: &args,
        })?;
        let second_verdict = second.evaluate_hook(HarnessHook::ToolCall {
            name: "test",
            id: "call-2",
            args: &args,
        })?;

        assert_eq!(first_verdict, Verdict::Allow);
        assert_eq!(second_verdict, Verdict::Allow);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        assert_eq!(first.runtime_signal_topics(), ["build.*"]);
        assert_eq!(
            first.dispatch_runtime_signal(HarnessSignal {
                signal_id: None,
                topic: "build.complete",
                source_agent_id: "builder",
                target_agent_id: "default",
                source_session_id: None,
                target_session_id: None,
                payload: "{}",
                created_at: "2026-08-20T00:00:00Z",
            })?,
            1
        );
        assert_eq!(signals.load(Ordering::Relaxed), 1);
        assert_eq!(
            first.invoke_action(HarnessActionRequest {
                agent_id: "default",
                name: "build.status",
                params: serde_json::json!({ "state": "passed" }),
            })?,
            Some(serde_json::json!({ "state": "passed" }))
        );
        assert_eq!(actions.load(Ordering::Relaxed), 1);
        Ok(())
    }
}
