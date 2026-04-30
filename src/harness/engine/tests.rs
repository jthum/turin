use super::*;
use crate::harness::context::{ContextWrapper, RequestOptionsOverride};
use crate::inference::provider::{
    InferenceEvent, InferenceProvider, InferenceRequest, InferenceStream, ProviderClient, SdkError,
};
use crate::kernel::config::InferenceOverrideConfig;
use crate::persistence::manager::StoreManager;
use futures::future::BoxFuture;
use futures::stream;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

#[derive(Clone)]
struct CountingTextProvider {
    counter: Arc<Mutex<u32>>,
}

impl InferenceProvider for CountingTextProvider {
    fn stream<'a>(
        &'a self,
        _request: InferenceRequest,
        _options: Option<crate::inference::provider::RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        let counter = Arc::clone(&self.counter);
        Box::pin(async move {
            let next = {
                let mut lock = counter.lock().unwrap();
                *lock += 1;
                *lock
            };
            let text = format!("summary-{next}");
            let events = vec![
                Ok(InferenceEvent::MessageStart {
                    role: "assistant".to_string(),
                    model: "mock-model".to_string(),
                    provider_id: "mock".to_string(),
                }),
                Ok(InferenceEvent::MessageDelta { content: text }),
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

fn test_app_data_for_root_and_session(root: PathBuf, session_id: &str) -> HarnessAppData {
    HarnessAppData {
        fs_root: root.clone(),
        workspace_root: root.clone(),
        harness_directory: root.clone(),
        store_manager: Arc::new(StoreManager::new(
            root.clone(),
            turin_types::layout::default_stores_dir_for_workspace(&root),
        )),
        agent_manager: Arc::new(crate::kernel::agent_manager::AgentManager::new(
            std::sync::Arc::new(crate::kernel::config::TurinConfig::default()),
            Arc::new(StoreManager::new(
                root.clone(),
                turin_types::layout::default_stores_dir_for_workspace(&root),
            )),
        )),
        policy_manager: Arc::new(crate::kernel::policy::RuntimePolicyManager::new()),
        governance_manager: Arc::new(crate::kernel::governance::GovernanceManager::new(
            crate::kernel::config::GovernanceConfig::default(),
        )),
        clients: std::collections::HashMap::new(),
        embedding_provider: None,
        execution_ctx: std::sync::Arc::new(std::sync::Mutex::new(
            crate::harness::globals::HarnessExecutionContext {
                session_id: Some(session_id.to_string()),
                queue: Some(std::sync::Arc::new(tokio::sync::Mutex::new(
                    std::collections::VecDeque::new(),
                ))),
                ..Default::default()
            },
        )),
        config: std::sync::Arc::new(crate::kernel::config::TurinConfig::default()),
        spawn_depth: 0,
        active_modules: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        watch_roots: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        loading_phase: std::sync::Arc::new(std::sync::Mutex::new(true)),
    }
}

fn test_app_data_for_root(root: PathBuf) -> HarnessAppData {
    test_app_data_for_root_and_session(root, "test-session")
}

fn test_app_data() -> HarnessAppData {
    test_app_data_for_root(PathBuf::from("."))
}

#[test]
fn test_engine_no_scripts() {
    let engine = HarnessEngine::new(test_app_data()).unwrap();
    assert!(engine.loaded_scripts().is_empty());

    let verdict = engine
        .evaluate("on_tool_call", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[test]
fn test_engine_load_empty_dir() {
    let dir = TempDir::new().unwrap();
    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();
    assert!(engine.loaded_scripts().is_empty());
}

#[test]
fn test_engine_load_nonexistent_dir() {
    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(Path::new("/nonexistent/path")).unwrap();
    assert!(engine.loaded_scripts().is_empty());
}

#[test]
fn test_engine_allow_verdict() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("allow.lua"),
        r#"
            function on_tool_call(call)
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate(
            "on_tool_call",
            serde_json::json!({"name": "read_file", "args": {}}),
        )
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[test]
fn test_engine_reject_verdict() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("safety.lua"),
        r#"
            function on_tool_call(call)
                if call.name == "shell_exec" then
                    return REJECT, "Shell commands are not allowed"
                end
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate(
            "on_tool_call",
            serde_json::json!({"name": "shell_exec", "args": {"command": "ls"}}),
        )
        .unwrap();
    assert!(verdict.is_rejected());
    assert_eq!(verdict.reason(), Some("Shell commands are not allowed"));

    let verdict = engine
        .evaluate(
            "on_tool_call",
            serde_json::json!({"name": "read_file", "args": {"path": "foo.txt"}}),
        )
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[test]
fn test_engine_escalate_verdict() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("escalation.lua"),
        r#"
            function on_tool_call(call)
                if call.name == "write_file" then
                    return ESCALATE, "File writes need human approval"
                end
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate(
            "on_tool_call",
            serde_json::json!({"name": "write_file", "args": {}}),
        )
        .unwrap();
    assert!(verdict.is_escalated());
    assert_eq!(verdict.reason(), Some("File writes need human approval"));
}

#[test]
fn test_engine_composition_reject_wins() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("a_permissive.lua"),
        r#"
            function on_tool_call(call)
                return ALLOW
            end
            "#,
    )
    .unwrap();

    std::fs::write(
        dir.path().join("b_safety.lua"),
        r#"
            function on_tool_call(call)
                if call.name == "shell_exec" then
                    return REJECT, "Blocked by safety harness"
                end
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    assert_eq!(engine.loaded_scripts(), vec!["a_permissive", "b_safety"]);

    let verdict = engine
        .evaluate(
            "on_tool_call",
            serde_json::json!({"name": "shell_exec", "args": {}}),
        )
        .unwrap();
    assert_eq!(
        verdict,
        Verdict::Reject("Blocked by safety harness".to_string())
    );
}

#[test]
fn test_engine_collects_declared_virtual_tools() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            tool.declare("play_song", {
                description = "Play an audio file",
                params = {
                    filename = { type = "string", required = true }
                },
                handler = function(args)
                    return tool.call("shell_exec", {
                        command = "mpg123 " .. shell.quote(args.filename)
                    })
                end
            })
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let tools = engine.declared_virtual_tools().unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "play_song");
    assert_eq!(tools[0].description, "Play an audio file");
    assert_eq!(
        tools[0].input_schema["properties"]["filename"]["type"],
        "string"
    );
}

#[test]
fn test_engine_invokes_virtual_tool_handler_sequence() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            tool.declare("play_playlist", {
                description = "Play multiple songs",
                params = {
                    first = { type = "string", required = true },
                    second = { type = "string", required = true }
                },
                handler = function(args)
                    return tool.sequence({
                        tool.call("shell_exec", {
                            command = "mpg123 " .. shell.quote(args.first)
                        }),
                        tool.call("shell_exec", {
                            command = "mpg123 " .. shell.quote(args.second)
                        })
                    })
                end
            })
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let plan = engine
        .invoke_virtual_tool(
            "play_playlist",
            serde_json::json!({ "first": "one.mp3", "second": "two.mp3" }),
        )
        .unwrap()
        .unwrap();

    assert_eq!(plan.calls.len(), 2);
    assert_eq!(plan.calls[0].name, "shell_exec");
    assert_eq!(plan.calls[0].args["command"], "mpg123 'one.mp3'");
    assert_eq!(plan.calls[1].args["command"], "mpg123 'two.mp3'");
}

#[test]
fn test_engine_invokes_virtual_tool_result_handler() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            tool.declare("read_note_wrapped", {
                description = "Read and wrap a note",
                params = {
                    path = { type = "string", required = true }
                },
                handler = function(args)
                    return tool.call("read_file", { path = args.path }, function(result)
                        return {
                            content = "wrapped: " .. result.content,
                            is_error = result.is_error
                        }
                    end)
                end
            })
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let plan = engine
        .invoke_virtual_tool(
            "read_note_wrapped",
            serde_json::json!({ "path": "note.txt" }),
        )
        .unwrap()
        .unwrap();

    let handler_key = plan
        .result_handler_key
        .clone()
        .expect("expected result handler key");

    let output = engine
        .invoke_virtual_tool_result_handler(
            &handler_key,
            serde_json::json!({
                "id": "tc_1",
                "name": "read_file",
                "args": { "path": "note.txt" },
                "verdict": "allow",
                "duration_ms": 2,
                "content": "hello",
                "is_error": false
            }),
            false,
        )
        .unwrap();

    assert_eq!(
        output,
        VirtualToolResultResolution::Output(
            crate::harness::virtual_tools::VirtualToolResultOutput {
                content: "wrapped: hello".to_string(),
                is_error: false,
            }
        )
    );
}

#[test]
fn test_engine_imports_nested_module_from_subdirectory() {
    let dir = TempDir::new().unwrap();
    std::fs::create_dir_all(dir.path().join("modules")).unwrap();
    std::fs::write(
        dir.path().join("modules").join("helper.lua"),
        r#"
            return {
                ping = function()
                    return "pong"
                end
            }
            "#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            function on_turn_prepare(ctx)
                local helper = import("modules/helper")
                if helper.ping() ~= "pong" then
                    error("nested import returned wrong value")
                end
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(dir.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[test]
fn test_engine_use_activates_script_and_table_blocks() {
    let dir = TempDir::new().unwrap();
    std::fs::create_dir_all(dir.path().join("blocks")).unwrap();
    std::fs::write(
        dir.path().join("blocks").join("script_style.lua"),
        r#"
            function on_turn_prepare(ctx)
                return ESCALATE, "script-style block"
            end
            "#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("blocks").join("table_style.lua"),
        r#"
            return {
                on_turn_prepare = function(ctx)
                    return REJECT, "table-style block"
                end
            }
            "#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            use("blocks/script_style")
            use("blocks/table_style", {
                when = function(hook, payload)
                    return hook == "on_turn_prepare"
                end
            })
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(dir.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();

    assert_eq!(
        engine.loaded_scripts(),
        vec![
            "blocks/script_style#use1".to_string(),
            "blocks/table_style#use1".to_string(),
            "main".to_string()
        ]
    );

    let verdict = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Reject("table-style block".to_string()));
}

#[test]
fn test_engine_use_rejected_outside_load_phase() {
    let dir = TempDir::new().unwrap();
    std::fs::create_dir_all(dir.path().join("blocks")).unwrap();
    std::fs::write(
        dir.path().join("blocks").join("late.lua"),
        r#"
            function on_turn_prepare(ctx)
                return ALLOW
            end
            "#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            function on_turn_prepare(ctx)
                use("blocks/late")
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(dir.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let err = engine
        .evaluate("on_turn_prepare", serde_json::json!({}))
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("use(...) can only be called during harness load")
    );
}

#[test]
fn test_engine_watch_registers_explicit_roots() {
    let dir = TempDir::new().unwrap();
    std::fs::create_dir_all(dir.path().join("blocks")).unwrap();
    std::fs::write(
        dir.path().join("main.lua"),
        r#"
            watch("blocks")
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(dir.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();

    assert_eq!(
        engine.explicit_watch_roots(),
        vec![dir.path().join("blocks")]
    );
}

#[test]
fn test_engine_rm_rf_blocked() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("safety.lua"),
        r#"
            function on_tool_call(call)
                if call.name == "shell_exec" then
                    local cmd = call.args.command
                    if cmd and cmd:find("rm %-rf") then
                        return REJECT, "Destructive command 'rm -rf' is not allowed"
                    end
                end
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate(
            "on_tool_call",
            serde_json::json!({"name": "shell_exec", "args": {"command": "rm -rf /"}}),
        )
        .unwrap();
    assert_eq!(
        verdict,
        Verdict::Reject("Destructive command 'rm -rf' is not allowed".to_string())
    );

    let verdict = engine
        .evaluate(
            "on_tool_call",
            serde_json::json!({"name": "shell_exec", "args": {"command": "ls -la"}}),
        )
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[test]
fn test_engine_undefined_hook_returns_allow() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("partial.lua"),
        r#"
            function on_tool_call(call)
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate("on_token_usage", serde_json::json!({}))
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);
}

#[test]
fn test_engine_token_usage_hook() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("budget.lua"),
        r#"
            function on_token_usage(usage)
                if usage.total_cost_usd and usage.total_cost_usd > 1.0 then
                    return REJECT, "Budget exceeded: $" .. tostring(usage.total_cost_usd)
                end
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate(
            "on_token_usage",
            serde_json::json!({"total_cost_usd": 0.5, "input_tokens": 100, "output_tokens": 50}),
        )
        .unwrap();
    assert_eq!(verdict, Verdict::Allow);

    let verdict = engine
        .evaluate(
            "on_token_usage",
            serde_json::json!({"total_cost_usd": 1.5, "input_tokens": 100, "output_tokens": 50}),
        )
        .unwrap();
    assert!(verdict.is_rejected());
}

#[test]
fn test_engine_modify_verdict() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("modify.lua"),
        r#"
            function on_plan_submit(payload)
                return MODIFY, { "Modified Task 1", "Modified Task 2" }
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate(
            "on_plan_submit",
            serde_json::json!({"action": "submit_plan"}),
        )
        .unwrap();

    match verdict {
        Verdict::Modify(val) => {
            let arr = val.as_array().unwrap();
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0].as_str().unwrap(), "Modified Task 1");
            assert_eq!(arr[1].as_str().unwrap(), "Modified Task 2");
        }
        _ => panic!("Expected Modify verdict, got {:?}", verdict),
    }
}

#[derive(Clone)]
struct MockContext;
impl mlua::UserData for MockContext {}

#[test]
fn test_on_turn_prepare_reject() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("reject.lua"),
        r#"
            function on_turn_prepare(ctx)
                return REJECT, "Blocked by harness"
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_rejected());
    assert_eq!(verdict.reason(), Some("Blocked by harness"));
}

#[test]
fn test_verdict_helpers_support_or_chains() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("verdict_dx.lua"),
        r#"
            function on_tool_call(call)
                return verdict.reject_if(call.name == "shell_exec", "blocked by dx")
                    or verdict.allow()
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate(
            "on_tool_call",
            serde_json::json!({"name": "shell_exec", "args": {}}),
        )
        .unwrap();
    assert!(verdict.is_rejected());
    assert_eq!(verdict.reason(), Some("blocked by dx"));
}

#[test]
fn test_verdict_modify_helper() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("verdict_modify_dx.lua"),
        r#"
            function on_plan_submit(payload)
                return verdict.modify({ "A", "B" })
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let verdict = engine
        .evaluate(
            "on_plan_submit",
            serde_json::json!({ "action": "submit_plan" }),
        )
        .unwrap();

    match verdict {
        Verdict::Modify(val) => {
            let arr = val.as_array().unwrap();
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0].as_str().unwrap(), "A");
            assert_eq!(arr[1].as_str().unwrap(), "B");
        }
        other => panic!("Expected Modify verdict, got {:?}", other),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_access_helpers() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("access_dx.lua"),
        r#"
            function on_turn_prepare(ctx)
                if not allowed("db.exec") then
                    return REJECT, "expected allowed in default config"
                end
                local decision = access.check("db.exec")
                if type(decision) ~= "table" then
                    return REJECT, "access.check did not return table"
                end
                needs("db.exec")
                return ALLOW
            end
            "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data()).unwrap();
    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_session_user_kv_helpers() {
    let root = TempDir::new().unwrap();
    let mut engine = HarnessEngine::new(test_app_data_for_root(root.path().to_path_buf())).unwrap();

    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("data_dx.lua"),
        r#"
            function on_turn_prepare(ctx)
                session.set("counter", "0")
                local a = session.incr("counter")
                local b = session.incr("counter", 2)
                if a ~= 1 or b ~= 3 then
                    return REJECT, "session.incr mismatch"
                end

                if session.get("counter") ~= "3" then
                    return REJECT, "session.get mismatch"
                end

                user.set("tz", "UTC")
                if user.get("tz") ~= "UTC" then
                    return REJECT, "user.set/user.get mismatch"
                end
                user.del("tz")
                if user.get("tz") ~= nil then
                    return REJECT, "user.del mismatch"
                end
                return ALLOW
            end
            "#,
    )
    .unwrap();

    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_scope_helper_supports_custom_scopes() {
    let root = TempDir::new().unwrap();
    let mut engine = HarnessEngine::new(test_app_data_for_root(root.path().to_path_buf())).unwrap();

    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("scope_dx.lua"),
        r#"
            function on_turn_prepare(ctx)
                local project = scope("project", "my-app", {
                    namespace = "notes",
                    visibility = "shared",
                })

                project.set("counter", "0")
                local a = project.incr("counter")
                local b = project.incr("counter", 4)
                if a ~= 1 or b ~= 5 then
                    return REJECT, "scope.incr mismatch"
                end
                if project.get("counter") ~= "5" then
                    return REJECT, "scope.get mismatch"
                end

                project.remember("My app uses event sourcing.", { topic = "architecture" })
                local hits = project.recall("event sourcing")
                if hits == nil or #hits < 1 then
                    return REJECT, "scope.recall returned no hits"
                end
                if hits[1].metadata == nil or hits[1].metadata.topic ~= "architecture" then
                    return REJECT, "scope.recall metadata mismatch"
                end

                project.del("counter")
                if project.get("counter") ~= nil then
                    return REJECT, "scope.del mismatch"
                end

                local global_scope = scope("global")
                global_scope.set("banner", "shared")
                if global_scope.get("banner") ~= "shared" then
                    return REJECT, "scope(global) mismatch"
                end

                return ALLOW
            end
        "#,
    )
    .unwrap();

    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_runtime_db_proxy_one_and_with_error_precedence() {
    let root = TempDir::new().unwrap();
    let mut engine = HarnessEngine::new(test_app_data_for_root(root.path().to_path_buf())).unwrap();

    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("db_dx.lua"),
        r#"
            function on_turn_prepare(ctx)
                runtime.db.with("state", function(db)
                    db:exec("CREATE TABLE IF NOT EXISTS dx_users(id INTEGER PRIMARY KEY, name TEXT)")
                    db:exec("DELETE FROM dx_users")
                    db:exec("INSERT INTO dx_users(name) VALUES (?)", {"alice"})

                    local missing = db:one("SELECT name FROM dx_users WHERE id = ?", { 999 })
                    if missing ~= nil then
                        error("runtime.db:one should return nil when no rows")
                    end

                    local first = db:one("SELECT name FROM dx_users ORDER BY id LIMIT 1")
                    if first == nil or first.name ~= "alice" then
                        error("runtime.db:one returned wrong row")
                    end
                end)

                local ok, err = pcall(function()
                    runtime.db.with("state", function(db)
                        db:close()
                        error("callback error sentinel")
                    end)
                end)
                if ok then
                    return REJECT, "runtime.db.with should have failed"
                end
                if not tostring(err):find("callback error sentinel", 1, true) then
                    return REJECT, "runtime.db.with should prioritize callback error"
                end

                return ALLOW
            end
            "#,
    )
    .unwrap();

    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_try_helper_captures_runtime_errors() {
    let root = TempDir::new().unwrap();
    let mut engine = HarnessEngine::new(test_app_data_for_root(root.path().to_path_buf())).unwrap();

    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("try_dx.lua"),
        r#"
            function on_turn_prepare(ctx)
                local ok, err = try(function()
                    error("boom")
                end)
                if ok ~= nil then
                    return REJECT, "try should return nil on raised error"
                end
                if err == nil or not tostring(err):find("boom", 1, true) then
                    return REJECT, "try should expose the raised error"
                end

                local sum = try(function(a, b)
                    return a + b
                end, 2, 3)
                if sum ~= 5 then
                    return REJECT, "try should preserve successful return values"
                end

                return ALLOW
            end
        "#,
    )
    .unwrap();

    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_runtime_agent_status_proxy_and_fs_json_helpers() {
    let root = TempDir::new().unwrap();
    let mut engine = HarnessEngine::new(test_app_data_for_root(root.path().to_path_buf())).unwrap();

    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("agent_fs_dx.lua"),
        r#"
            function on_turn_prepare(ctx)
                local status = runtime.agent("default"):status()
                if status == nil or status.agent_id ~= "default" then
                    return REJECT, "runtime.agent(...):status() mismatch"
                end

                fs.write_json("dx-config.json", { enabled = true, count = 3 }, { pretty = true })
                local cfg = fs.read_json("dx-config.json")
                if cfg.enabled ~= true or cfg.count ~= 3 then
                    return REJECT, "fs.read_json/fs.write_json mismatch"
                end

                return ALLOW
            end
            "#,
    )
    .unwrap();

    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_hash_sha256_and_fs_stat_track_changes_per_session() {
    let root = TempDir::new().unwrap();
    std::fs::write(root.path().join("SPEC.md"), "alpha").unwrap();

    let script = r#"
        function on_turn_prepare(ctx)
            local run = session.incr("run")
            local stat = fs.stat("SPEC.md")

            if hash.sha256("hello") ~= "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824" then
                return REJECT, "hash.sha256 mismatch"
            end

            if stat.path ~= "SPEC.md" then
                return REJECT, "fs.stat path mismatch"
            end

            if run == 1 then
                if stat.bytes ~= 5 or stat.hash ~= hash.sha256("alpha") then
                    return REJECT, "run1 stat facts mismatch"
                end
                if stat.seen_before ~= false or stat.changed ~= true or stat.previous_hash ~= nil then
                    return REJECT, "run1 tracking mismatch"
                end
            elseif run == 2 then
                if stat.hash ~= hash.sha256("alpha") then
                    return REJECT, "run2 hash mismatch"
                end
                if stat.seen_before ~= true or stat.changed ~= false or stat.previous_hash ~= hash.sha256("alpha") then
                    return REJECT, "run2 tracking mismatch"
                end
            elseif run == 3 then
                if stat.bytes ~= 4 or stat.hash ~= hash.sha256("beta") then
                    return REJECT, "run3 stat facts mismatch"
                end
                if stat.seen_before ~= true or stat.changed ~= true or stat.previous_hash ~= hash.sha256("alpha") then
                    return REJECT, "run3 tracking mismatch"
                end
            else
                return REJECT, "unexpected run"
            end

            return ALLOW
        end
    "#;

    let harness_dir = TempDir::new().unwrap();
    std::fs::write(harness_dir.path().join("stat.lua"), script).unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(root.path().to_path_buf())).unwrap();
    engine.load_dir(harness_dir.path()).unwrap();

    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());

    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());

    std::fs::write(root.path().join("SPEC.md"), "beta").unwrap();

    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());

    let other_session_script = r#"
        function on_turn_prepare(ctx)
            local run = session.incr("run")
            local stat = fs.stat("SPEC.md")

            if run ~= 1 then
                return REJECT, "new session counter mismatch"
            end
            if stat.hash ~= hash.sha256("beta") or stat.bytes ~= 4 then
                return REJECT, "new session stat facts mismatch"
            end
            if stat.seen_before ~= false or stat.changed ~= true or stat.previous_hash ~= nil then
                return REJECT, "new session tracking mismatch"
            end

            return ALLOW
        end
    "#;
    let other_harness_dir = TempDir::new().unwrap();
    std::fs::write(
        other_harness_dir.path().join("other_session.lua"),
        other_session_script,
    )
    .unwrap();

    let mut other_engine = HarnessEngine::new(test_app_data_for_root_and_session(
        root.path().to_path_buf(),
        "other-session",
    ))
    .unwrap();
    other_engine.load_dir(other_harness_dir.path()).unwrap();

    let verdict = other_engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_fs_summary_reuses_cached_summary_until_file_changes() {
    let root = TempDir::new().unwrap();
    std::fs::write(root.path().join("SPEC.md"), "alpha").unwrap();

    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("summary.lua"),
        r#"
            function on_turn_prepare(ctx)
                local run = session.incr("summary_run")
                local default = fs.summary("SPEC.md")
                local default_again = fs.summary("SPEC.md")
                local custom = fs.summary("SPEC.md", {
                    prompt = "Summarize only the constraints and obligations."
                })

                if default ~= default_again then
                    return REJECT, "default summary should be cached within one turn"
                end

                if run == 1 then
                    if default ~= "summary-1" then
                        return REJECT, "run1 default summary mismatch"
                    end
                    if custom ~= "summary-2" then
                        return REJECT, "run1 custom summary mismatch"
                    end
                elseif run == 2 then
                    if default ~= "summary-1" then
                        return REJECT, "run2 default summary should be reused"
                    end
                    if custom ~= "summary-2" then
                        return REJECT, "run2 custom summary should be reused"
                    end
                elseif run == 3 then
                    if default ~= "summary-3" then
                        return REJECT, "run3 default summary mismatch"
                    end
                    if custom ~= "summary-4" then
                        return REJECT, "run3 custom summary mismatch"
                    end
                else
                    return REJECT, "unexpected summary run"
                end

                return ALLOW
            end
        "#,
    )
    .unwrap();

    let mut engine = HarnessEngine::new(test_app_data_for_root(root.path().to_path_buf())).unwrap();
    engine.load_dir(dir.path()).unwrap();

    let provider_calls = Arc::new(Mutex::new(0u32));
    let provider = Arc::new(CountingTextProvider {
        counter: Arc::clone(&provider_calls),
    });
    let mut clients = std::collections::HashMap::new();
    clients.insert("mock".to_string(), ProviderClient::new("mock", provider));

    let make_ctx = || {
        ContextWrapper::new(
            None,
            "mock-model".to_string(),
            "mock".to_string(),
            "Summarize files.".to_string(),
            Vec::new(),
            1,
            1,
            true,
            "task-1".to_string(),
            None,
            0,
            100000,
            0,
            RequestOptionsOverride::default(),
            clients.clone(),
            Arc::new(crate::kernel::config::TurinConfig::default()),
            "default".to_string(),
            InferenceOverrideConfig::default(),
        )
    };

    let verdict = engine
        .evaluate_userdata("on_turn_prepare", make_ctx())
        .unwrap();
    assert!(verdict.is_allowed());

    let verdict = engine
        .evaluate_userdata("on_turn_prepare", make_ctx())
        .unwrap();
    assert!(verdict.is_allowed());

    std::fs::write(root.path().join("SPEC.md"), "beta").unwrap();

    let verdict = engine
        .evaluate_userdata("on_turn_prepare", make_ctx())
        .unwrap();
    assert!(verdict.is_allowed());

    assert_eq!(*provider_calls.lock().unwrap(), 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_runtime_governance_grant_wrapper() {
    let mut engine = HarnessEngine::new(test_app_data()).unwrap();

    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("governance_dx.lua"),
        r#"
            function on_turn_prepare(ctx)
                local gid = nil
                local result = runtime.governance.grant({
                    ttl_ms = 5000,
                    capabilities = {
                        ["db.query"] = true,
                        ["governance.grant.get"] = true,
                    }
                }, function()
                    local query_dec = access.check("db.query")
                    if query_dec == nil or not query_dec.allowed then
                        error("runtime.db.query should be allowed inside grant")
                    end

                    local policy_dec = access.check("policy.set")
                    if policy_dec == nil then
                        error("runtime.policy.set decision missing")
                    end
                    if policy_dec.allowed then
                        error("runtime.policy.set should be denied by grant ceiling")
                    end

                    gid = query_dec.subject_grant_id
                    if gid == nil or gid == "" then
                        error("missing subject_grant_id in access decision")
                    end

                    return "grant_wrapper_ok"
                end)

                if result ~= "grant_wrapper_ok" then
                    return REJECT, "runtime.governance.grant result mismatch"
                end

                local grant, ge = runtime.governance.grant_get(gid)
                if grant ~= nil then
                    return REJECT, "grant should be revoked after callback returns"
                end
                if ge == nil then
                    return REJECT, "grant_get should report missing grant after revoke"
                end

                local ok, err = pcall(function()
                    runtime.governance.grant({
                        ttl_ms = 5000,
                        capabilities = {
                            ["db.query"] = true,
                            ["governance.grant.revoke"] = true,
                        }
                    }, function()
                        local dec = access.check("db.query")
                        local inner_gid = dec.subject_grant_id
                        local revoked, re = runtime.governance.grant_revoke(inner_gid)
                        if not revoked then
                            error("inner grant_revoke failed: " .. tostring(re))
                        end
                        error("grant callback sentinel")
                    end)
                end)
                if ok then
                    return REJECT, "runtime.governance.grant should fail when callback errors"
                end
                if not tostring(err):find("grant callback sentinel", 1, true) then
                    return REJECT, "runtime.governance.grant should prioritize callback error"
                end

                return ALLOW
            end
            "#,
    )
    .unwrap();

    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_time_helpers() {
    let mut engine = HarnessEngine::new(test_app_data()).unwrap();

    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("time_dx.lua"),
        r#"
            function on_turn_prepare(ctx)
                local now = tonumber(time.now_utc())
                if now == nil then
                    return REJECT, "time.now_utc should be numeric string"
                end

                local since_num = time.since(now - 2)
                if since_num < 1 then
                    return REJECT, "time.since(number) should be positive"
                end

                local since_str = time.since(tostring(now - 1))
                if since_str < 0 then
                    return REJECT, "time.since(string) should parse numeric string"
                end

                if not time.after(now - 1, 0.5) then
                    return REJECT, "time.after should be true when elapsed >= threshold"
                end

                if time.after(now - 1, 10) then
                    return REJECT, "time.after should be false for large threshold"
                end

                return ALLOW
            end
            "#,
    )
    .unwrap();

    engine.load_dir(dir.path()).unwrap();
    let verdict = engine
        .evaluate_userdata("on_turn_prepare", MockContext)
        .unwrap();
    assert!(verdict.is_allowed());
}
