use mlua::{LuaSerdeExt, MetaMethod, UserData, UserDataMethods, Value};
use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::harness::globals::block_on_current;
use crate::inference::content::{infer_prompt_from_messages, replace_user_text_content};
use crate::inference::provider::{InferenceMessage, ProviderClient};
use crate::kernel::config::{InferenceOverrideConfig, TurinConfig};
use crate::kernel::estimate_history_input_tokens;
use crate::kernel::harness_contract::{HarnessTurnRequest, HarnessTurnServices};

mod structured_call;
mod tool_exposure;

use tool_exposure::ToolExposureProxy;

pub(crate) use crate::kernel::harness_contract::build_merged_request_options;
pub use crate::kernel::harness_contract::{RequestOptionsOverride, ToolExposure};

/// Inner state shareable between Rust and Lua
#[derive(Clone, Debug)]
pub struct ContextState {
    pub inference: Option<String>,
    pub model: String,
    pub provider: String,
    pub system_prompt: String,
    pub messages: Vec<InferenceMessage>,
    pub prompt: Option<String>,
    pub turn_index: u32,
    pub task_turn_index: u32,
    pub is_first_turn_in_task: bool,
    pub task_id: String,
    pub plan_id: Option<String>,
    pub token_count: u32,
    pub token_limit: u32,
    pub thinking_budget: u32,
    pub request_options: RequestOptionsOverride,
    pub session_id: String,
    pub session_title: Option<String>,
    pub user_message_count: usize,
    pub tool_exposure: ToolExposure,
}

/// UserData wrapper for Context validation and mutation
#[derive(Clone)]
pub struct ContextWrapper {
    pub state: Arc<Mutex<ContextState>>,
    pub clients: HashMap<String, ProviderClient>,
    pub config: Arc<TurinConfig>,
    pub agent_id: String,
    pub session_inference: InferenceOverrideConfig,
    available_tools: Arc<BTreeSet<String>>,
}

pub struct ContextInit {
    pub inference: Option<String>,
    pub model: String,
    pub provider: String,
    pub system_prompt: String,
    pub messages: Vec<InferenceMessage>,
    pub turn_index: u32,
    pub task_turn_index: u32,
    pub is_first_turn_in_task: bool,
    pub task_id: String,
    pub plan_id: Option<String>,
    pub token_count: u32,
    pub token_limit: u32,
    pub thinking_budget: u32,
    pub request_options: RequestOptionsOverride,
    pub clients: HashMap<String, ProviderClient>,
    pub config: Arc<TurinConfig>,
    pub agent_id: String,
    pub session_inference: InferenceOverrideConfig,
    pub session_id: String,
    pub session_title: Option<String>,
    pub available_tools: BTreeSet<String>,
}

impl ContextWrapper {
    pub(crate) fn from_harness_request(
        request: &mut HarnessTurnRequest,
        services: HarnessTurnServices<'_>,
    ) -> Self {
        let context = Self::new(ContextInit {
            inference: request.inference.take(),
            model: std::mem::take(&mut request.model),
            provider: std::mem::take(&mut request.provider),
            system_prompt: std::mem::take(&mut request.system_prompt),
            messages: std::mem::take(&mut request.messages),
            turn_index: request.turn_index,
            task_turn_index: request.task_turn_index,
            is_first_turn_in_task: request.is_first_turn_in_task,
            task_id: request.task_id.clone(),
            plan_id: request.plan_id.clone(),
            token_count: request.token_count,
            token_limit: request.token_limit,
            thinking_budget: request.thinking_budget,
            request_options: std::mem::take(&mut request.request_options),
            clients: services.clients.clone(),
            config: Arc::clone(services.config),
            agent_id: request.agent_id.clone(),
            session_inference: request.session_inference.clone(),
            session_id: request.session_id.clone(),
            session_title: request.session_title.clone(),
            available_tools: std::mem::take(&mut request.available_tools),
        });
        context.lock_state().tool_exposure = std::mem::take(&mut request.tool_exposure);
        context
    }

    pub(crate) fn apply_to_harness_request(self, request: &mut HarnessTurnRequest) {
        let state = self.into_state();
        request.inference = state.inference;
        request.model = state.model;
        request.provider = state.provider;
        request.system_prompt = state.system_prompt;
        request.messages = state.messages;
        request.thinking_budget = state.thinking_budget;
        request.request_options = state.request_options;
        request.tool_exposure = state.tool_exposure;
    }

    pub fn new(init: ContextInit) -> Self {
        let ContextInit {
            inference,
            model,
            provider,
            system_prompt,
            messages,
            turn_index,
            task_turn_index,
            is_first_turn_in_task,
            task_id,
            plan_id,
            token_count,
            token_limit,
            thinking_budget,
            request_options,
            clients,
            config,
            agent_id,
            session_inference,
            session_id,
            session_title,
            available_tools,
        } = init;
        let prompt = infer_prompt_from_messages(&messages);
        let user_message_count = messages
            .iter()
            .filter(|message| message.role == crate::inference::provider::InferenceRole::User)
            .count();

        Self {
            state: Arc::new(Mutex::new(ContextState {
                inference,
                model,
                provider,
                system_prompt,
                messages,
                prompt,
                turn_index,
                task_turn_index,
                is_first_turn_in_task,
                task_id,
                plan_id,
                token_count,
                token_limit,
                thinking_budget,
                request_options,
                session_id,
                session_title,
                user_message_count,
                tool_exposure: ToolExposure::default(),
            })),
            clients,
            config,
            agent_id,
            session_inference,
            available_tools: Arc::new(available_tools),
        }
    }

    /// Lock the context state mutex.
    ///
    /// Panics if poisoned (previous holder panicked — unrecoverable).
    fn lock_state(&self) -> MutexGuard<'_, ContextState> {
        self.state.lock().expect("context state mutex poisoned")
    }

    /// Retrieve the inner state (cloning the data out)
    pub fn get_state(&self) -> ContextState {
        self.lock_state().clone()
    }

    pub fn into_state(self) -> ContextState {
        match Arc::try_unwrap(self.state) {
            Ok(state) => state.into_inner().expect("context state mutex poisoned"),
            Err(state) => state.lock().expect("context state mutex poisoned").clone(),
        }
    }
}

fn recompute_token_count(state: &mut ContextState) {
    state.token_count = estimate_history_input_tokens(&state.system_prompt, &state.messages);
}

fn refresh_message_state(state: &mut ContextState) {
    state.prompt = infer_prompt_from_messages(&state.messages);
    recompute_token_count(state);
}

fn replace_messages(state: &mut ContextState, messages: Vec<InferenceMessage>) {
    state.messages = messages;
    refresh_message_state(state);
}

impl UserData for ContextWrapper {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        // Properties
        methods.add_method("get_inference", |_, this, ()| {
            Ok(this.lock_state().inference.clone())
        });

        methods.add_method("set_inference", |_, this, val: Option<String>| {
            this.lock_state().inference = normalize_inference_context_name(val);
            Ok(())
        });

        methods.add_method("get_model", |_, this, ()| {
            Ok(this.lock_state().model.clone())
        });

        methods.add_method("get_provider", |_, this, ()| {
            Ok(this.lock_state().provider.clone())
        });

        methods.add_method("get_token_count", |_, this, ()| {
            Ok(this.lock_state().token_count)
        });

        methods.add_method("get_estimated_input_tokens", |_, this, ()| {
            Ok(this.lock_state().token_count)
        });

        methods.add_method("get_turn_index", |_, this, ()| {
            Ok(this.lock_state().turn_index)
        });

        methods.add_method("get_task_turn_index", |_, this, ()| {
            Ok(this.lock_state().task_turn_index)
        });

        methods.add_method("is_first_turn_in_task", |_, this, ()| {
            Ok(this.lock_state().is_first_turn_in_task)
        });

        methods.add_method("get_task_id", |_, this, ()| {
            Ok(this.lock_state().task_id.clone())
        });

        methods.add_method("get_plan_id", |_, this, ()| {
            Ok(this.lock_state().plan_id.clone())
        });

        methods.add_method("get_token_limit", |_, this, ()| {
            Ok(this.lock_state().token_limit)
        });

        methods.add_method("get_max_input_tokens", |_, this, ()| {
            Ok(this.lock_state().token_limit)
        });

        methods.add_method("get_system_prompt", |_, this, ()| {
            Ok(this.lock_state().system_prompt.clone())
        });

        methods.add_method("set_system_prompt", |_, this, val: String| {
            let mut state = this.lock_state();
            state.system_prompt = val;
            recompute_token_count(&mut state);
            Ok(())
        });

        methods.add_method("get_thinking_budget", |_, this, ()| {
            Ok(this.lock_state().thinking_budget)
        });

        methods.add_method("set_thinking_budget", |_, this, val: u32| {
            this.lock_state().thinking_budget = val;
            Ok(())
        });

        methods.add_method("get_request_options", |lua, this, ()| {
            let state = this.lock_state();
            lua.to_value(&state.request_options)
                .map_err(mlua::Error::external)
        });

        methods.add_method("set_request_options", |lua, this, val: Value| {
            let request_options: RequestOptionsOverride =
                lua.from_value(val).map_err(mlua::Error::external)?;
            this.lock_state().request_options = request_options;
            Ok(())
        });

        // Messages Property (Copy)
        methods.add_method("get_messages", |lua, this: &ContextWrapper, ()| {
            let state = this.lock_state();
            let val = lua
                .to_value(&state.messages)
                .map_err(mlua::Error::external)?;
            Ok(val)
        });

        methods.add_method("set_messages", |lua, this: &ContextWrapper, val: Value| {
            let messages: Vec<InferenceMessage> =
                lua.from_value(val).map_err(mlua::Error::external)?;
            let mut state = this.lock_state();
            replace_messages(&mut state, messages);
            Ok(())
        });

        // Metatable __index/__newindex for properties
        methods.add_meta_method(
            MetaMethod::Index,
            |lua, this: &ContextWrapper, key: String| match key.as_str() {
                "inference" => {
                    let state = this.lock_state();
                    state
                        .inference
                        .as_deref()
                        .map(|s| lua.create_string(s).map(Value::String))
                        .transpose()
                        .map(|value| value.unwrap_or(Value::Nil))
                }
                "model" => {
                    let state = this.lock_state();
                    Ok(Value::String(lua.create_string(&state.model)?))
                }
                "provider" => {
                    let state = this.lock_state();
                    Ok(Value::String(lua.create_string(&state.provider)?))
                }
                "token_count" => {
                    let state = this.lock_state();
                    Ok(Value::Integer(state.token_count as i64))
                }
                "estimated_input_tokens" => {
                    let state = this.lock_state();
                    Ok(Value::Integer(state.token_count as i64))
                }
                "turn_index" => {
                    let state = this.lock_state();
                    Ok(Value::Integer(state.turn_index as i64))
                }
                "task_turn_index" => {
                    let state = this.lock_state();
                    Ok(Value::Integer(state.task_turn_index as i64))
                }
                "is_first_turn_in_task" => {
                    let state = this.lock_state();
                    Ok(Value::Boolean(state.is_first_turn_in_task))
                }
                "task_id" => {
                    let state = this.lock_state();
                    Ok(Value::String(lua.create_string(&state.task_id)?))
                }
                "plan_id" => {
                    let state = this.lock_state();
                    state
                        .plan_id
                        .as_deref()
                        .map(|s| lua.create_string(s).map(Value::String))
                        .transpose()
                        .map(|value| value.unwrap_or(Value::Nil))
                }
                "token_limit" => {
                    let state = this.lock_state();
                    Ok(Value::Integer(state.token_limit as i64))
                }
                "max_input_tokens" => {
                    let state = this.lock_state();
                    Ok(Value::Integer(state.token_limit as i64))
                }
                "system_prompt" => {
                    let state = this.lock_state();
                    Ok(Value::String(lua.create_string(&state.system_prompt)?))
                }
                "thinking_budget" => {
                    let state = this.lock_state();
                    Ok(Value::Integer(state.thinking_budget as i64))
                }
                "request_options" => {
                    let state = this.lock_state();
                    lua.to_value(&state.request_options)
                        .map_err(mlua::Error::external)
                }
                "prompt" => {
                    let state = this.lock_state();
                    state
                        .prompt
                        .as_deref()
                        .map(|s| lua.create_string(s).map(Value::String))
                        .transpose()
                        .map(|value| value.unwrap_or(Value::Nil))
                }
                "messages" => {
                    let state = this.lock_state();
                    lua.to_value(&state.messages).map_err(mlua::Error::external)
                }
                "session" => {
                    let state = this.lock_state();
                    let session = lua.create_table()?;
                    session.set("id", state.session_id.clone())?;
                    session.set("title", state.session_title.clone())?;
                    session.set("user_message_count", state.user_message_count)?;
                    session.set("is_first_user_message", state.user_message_count == 1)?;
                    Ok(Value::Table(session))
                }
                "tools" => Ok(Value::UserData(lua.create_userdata(ToolExposureProxy {
                    state: Arc::clone(&this.state),
                    available_tools: Arc::clone(&this.available_tools),
                })?)),
                _ => Ok(Value::Nil),
            },
        );

        methods.add_meta_method(
            MetaMethod::NewIndex,
            |lua, this: &ContextWrapper, (key, val): (String, Value)| {
                match key.as_str() {
                    "inference" => {
                        let s: Option<String> =
                            lua.from_value(val).map_err(mlua::Error::external)?;
                        this.lock_state().inference = normalize_inference_context_name(s);
                        Ok(())
                    }
                    "system_prompt" => {
                        let s: String = lua.from_value(val).map_err(mlua::Error::external)?;
                        let mut state = this.lock_state();
                        state.system_prompt = s;
                        recompute_token_count(&mut state);
                        Ok(())
                    }
                    "provider" => {
                        let s: String = lua.from_value(val).map_err(mlua::Error::external)?;
                        this.lock_state().provider = s;
                        Ok(())
                    }
                    "thinking_budget" => {
                        let b: u32 = lua.from_value(val).map_err(mlua::Error::external)?;
                        this.lock_state().thinking_budget = b;
                        Ok(())
                    }
                    "request_options" => {
                        let opts: RequestOptionsOverride =
                            lua.from_value(val).map_err(mlua::Error::external)?;
                        this.lock_state().request_options = opts;
                        Ok(())
                    }
                    "prompt" => {
                        let s: Option<String> =
                            lua.from_value(val).map_err(mlua::Error::external)?;
                        let mut state = this.lock_state();
                        state.prompt = s.clone();
                        // Sync back to messages if it's the last message
                        if let Some(msg) = state.messages.last_mut()
                            && msg.role == crate::inference::provider::InferenceRole::User
                        {
                            msg.content = replace_user_text_content(&msg.content, s.as_deref());
                        }
                        recompute_token_count(&mut state);
                        Ok(())
                    }
                    "messages" => {
                        let msgs: Vec<InferenceMessage> =
                            lua.from_value(val).map_err(mlua::Error::external)?;
                        let mut state = this.lock_state();
                        replace_messages(&mut state, msgs);
                        Ok(())
                    }
                    _ => Err(mlua::Error::RuntimeError(format!(
                        "Cannot set read-only or unknown property: {}",
                        key
                    ))),
                }
            },
        );

        // Mutation Helpers
        methods.add_method("add_message", |lua, this: &ContextWrapper, val: Value| {
            let msg: InferenceMessage = lua.from_value(val).map_err(mlua::Error::external)?;
            let mut state = this.lock_state();
            state.messages.push(msg);
            refresh_message_state(&mut state);
            Ok(())
        });

        methods.add_method("remove_message", |_, this, idx: usize| {
            let mut state = this.lock_state();
            // Lua is 1-indexed, Rust is 0-indexed
            if idx > 0 && idx <= state.messages.len() {
                state.messages.remove(idx - 1);
                refresh_message_state(&mut state);
                Ok(())
            } else {
                Err(mlua::Error::RuntimeError(format!(
                    "Index out of bounds: {}",
                    idx
                )))
            }
        });

        methods.add_method("clear_messages", |_, this, ()| {
            let mut state = this.lock_state();
            state.messages.clear();
            state.prompt = None;
            recompute_token_count(&mut state);
            Ok(())
        });

        // Summarize Capability (Sync wrapper)
        methods.add_method("summarize", |lua, this: &ContextWrapper, args: Value| {
            let clients = this.clients.clone();
            let state_arc = this.state.clone();

            let messages_opt: Option<Vec<InferenceMessage>> = if args.is_nil() {
                None
            } else {
                Some(lua.from_value(args).map_err(mlua::Error::external)?)
            };

            // Bridge async provider completion into sync Lua callback.
            let res = block_on_current(async {
                let (messages, model, provider_name) = {
                    let state = state_arc.lock().expect("context state mutex poisoned");
                    let msgs = messages_opt.unwrap_or_else(|| state.messages.clone());
                    (msgs, state.model.clone(), state.provider.clone())
                };

                // Helper to map error
                let get_client = || -> Result<ProviderClient, String> {
                    clients
                        .get(&provider_name)
                        .cloned()
                        .ok_or_else(|| format!("Provider '{}' not initialized", provider_name))
                };

                match get_client() {
                    Ok(client) => {
                        let system_prompt = "Summarize the following conversation concisely.";
                        client
                            .completion(&model, system_prompt, &messages)
                            .await
                            .map_err(|e| format!("Completion failed: {}", e))
                    }
                    Err(e) => Err(e),
                }
            });

            match res {
                Ok(summary) => Ok(Some(summary)), // Return string
                Err(e) => Err(mlua::Error::RuntimeError(format!(
                    "Summarization failed: {}",
                    e
                ))),
            }
        });

        methods.add_method("structured", |lua, this: &ContextWrapper, args: Value| {
            let parsed = structured_call::parse(lua, args)?;
            let structured = block_on_current(structured_call::run(this, parsed));

            match structured {
                Ok(value) => lua.to_value(&value).map_err(mlua::Error::external),
                Err(err) => Err(mlua::Error::RuntimeError(format!(
                    "Structured inference failed: {}",
                    err
                ))),
            }
        });
    }
}

fn normalize_inference_context_name(value: Option<String>) -> Option<String> {
    value.and_then(|text| {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn structured_messages(
    prompt: Option<String>,
    messages: Option<Vec<InferenceMessage>>,
    current_messages: Vec<InferenceMessage>,
) -> Result<Vec<InferenceMessage>, String> {
    if prompt.is_some() && messages.is_some() {
        return Err("structured opts may define prompt or messages, not both".to_string());
    }

    if let Some(prompt) = prompt {
        return Ok(vec![InferenceMessage {
            role: crate::inference::provider::InferenceRole::User,
            content: vec![crate::inference::provider::InferenceContent::Text { text: prompt }],
            tool_call_id: None,
        }]);
    }

    Ok(messages.unwrap_or(current_messages))
}
