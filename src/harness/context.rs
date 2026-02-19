use mlua::{LuaSerdeExt, MetaMethod, UserData, UserDataMethods, Value};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::inference::provider::{InferenceMessage, ProviderClient};
use std::collections::HashMap;

/// Inner state shareable between Rust and Lua
#[derive(Clone, Debug)]
pub struct ContextState {
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
}

/// UserData wrapper for Context validation and mutation
#[derive(Clone)]
pub struct ContextWrapper {
    pub state: Arc<Mutex<ContextState>>,
    pub clients: HashMap<String, ProviderClient>,
}

impl ContextWrapper {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: String,
        provider: String,
        system_prompt: String,
        messages: Vec<InferenceMessage>,
        turn_index: u32,
        task_turn_index: u32,
        is_first_turn_in_task: bool,
        task_id: String,
        plan_id: Option<String>,
        token_count: u32,
        token_limit: u32,
        thinking_budget: u32,
        clients: HashMap<String, ProviderClient>,
    ) -> Self {
        let prompt = messages.iter().last().and_then(|m| {
            if m.role == crate::inference::provider::InferenceRole::User {
                Some(
                    m.content
                        .iter()
                        .filter_map(|c| match c {
                            crate::inference::provider::InferenceContent::Text { text } => {
                                Some(text.clone())
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                )
            } else {
                None
            }
        });

        Self {
            state: Arc::new(Mutex::new(ContextState {
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
            })),
            clients,
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
}

impl UserData for ContextWrapper {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        // Properties
        methods.add_method("get_model", |_, this, ()| {
            Ok(this.lock_state().model.clone())
        });

        methods.add_method("get_provider", |_, this, ()| {
            Ok(this.lock_state().provider.clone())
        });

        methods.add_method("get_token_count", |_, this, ()| {
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

        methods.add_method("get_system_prompt", |_, this, ()| {
            Ok(this.lock_state().system_prompt.clone())
        });

        methods.add_method("set_system_prompt", |_, this, val: String| {
            this.lock_state().system_prompt = val;
            Ok(())
        });

        methods.add_method("get_thinking_budget", |_, this, ()| {
            Ok(this.lock_state().thinking_budget)
        });

        methods.add_method("set_thinking_budget", |_, this, val: u32| {
            this.lock_state().thinking_budget = val;
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
            this.lock_state().messages = messages;
            Ok(())
        });

        // Metatable __index/__newindex for properties
        methods.add_meta_method(
            MetaMethod::Index,
            |lua, this: &ContextWrapper, key: String| match key.as_str() {
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
                    Ok(state
                        .plan_id
                        .clone()
                        .map(|s| Value::String(lua.create_string(&s).unwrap()))
                        .unwrap_or(Value::Nil))
                }
                "token_limit" => {
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
                "prompt" => {
                    let state = this.lock_state();
                    Ok(state
                        .prompt
                        .clone()
                        .map(|s| Value::String(lua.create_string(&s).unwrap()))
                        .unwrap_or(Value::Nil))
                }
                "messages" => {
                    let state = this.lock_state();
                    lua.to_value(&state.messages).map_err(mlua::Error::external)
                }
                _ => Ok(Value::Nil),
            },
        );

        methods.add_meta_method(
            MetaMethod::NewIndex,
            |lua, this: &ContextWrapper, (key, val): (String, Value)| {
                match key.as_str() {
                    "system_prompt" => {
                        let s: String = lua.from_value(val).map_err(mlua::Error::external)?;
                        this.lock_state().system_prompt = s;
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
                    "prompt" => {
                        let s: Option<String> =
                            lua.from_value(val).map_err(mlua::Error::external)?;
                        let mut state = this.lock_state();
                        state.prompt = s.clone();
                        // Sync back to messages if it's the last message
                        if let Some(msg) = state.messages.last_mut()
                            && msg.role == crate::inference::provider::InferenceRole::User
                            && let Some(new_text) = s
                        {
                            msg.content =
                                vec![crate::inference::provider::InferenceContent::Text {
                                    text: new_text,
                                }];
                        }
                        Ok(())
                    }
                    "messages" => {
                        let msgs: Vec<InferenceMessage> =
                            lua.from_value(val).map_err(mlua::Error::external)?;
                        this.lock_state().messages = msgs;
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
            this.lock_state().messages.push(msg);
            Ok(())
        });

        methods.add_method("remove_message", |_, this, idx: usize| {
            let mut state = this.lock_state();
            // Lua is 1-indexed, Rust is 0-indexed
            if idx > 0 && idx <= state.messages.len() {
                state.messages.remove(idx - 1);
                Ok(())
            } else {
                Err(mlua::Error::RuntimeError(format!(
                    "Index out of bounds: {}",
                    idx
                )))
            }
        });

        methods.add_method("clear_messages", |_, this, ()| {
            this.lock_state().messages.clear();
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

            // Use block_in_place to bridge async client to sync Lua
            let res = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
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
                })
            });

            match res {
                Ok(summary) => Ok(Some(summary)), // Return string
                Err(e) => Err(mlua::Error::RuntimeError(format!(
                    "Summarization failed: {}",
                    e
                ))),
            }
        });
    }
}
