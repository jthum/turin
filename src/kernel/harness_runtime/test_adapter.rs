use std::collections::BTreeSet;
use std::path::Path;

use anyhow::Result;

use super::{
    HarnessAdapterFactory, HarnessDefinition, HarnessInstance, HarnessRuntimeInitContext,
    HarnessSourceOverlay,
};
use crate::harness::scheduler::HarnessSchedulerAccess;
use crate::harness::verdict::Verdict;
use crate::kernel::harness_contract::{
    HarnessActionRequest, HarnessHook, HarnessSignal, HarnessTurnRequest, HarnessTurnServices,
};

pub(super) struct TestHarnessAdapterFactory;

impl HarnessAdapterFactory for TestHarnessAdapterFactory {
    fn name(&self) -> &'static str {
        "test"
    }

    fn watches_sources(&self) -> bool {
        true
    }

    fn create(
        &self,
        definition: &HarnessDefinition,
        ctx: HarnessRuntimeInitContext,
    ) -> Result<Box<dyn HarnessInstance>> {
        let loaded_scripts = source_files(definition.directory())?;
        validate_disk_sources(definition.directory(), &loaded_scripts)?;
        let runtime_signal_topics = test_signal_topics(definition.directory(), &loaded_scripts)?;
        Ok(Box::new(TestHarnessInstance {
            loaded_scripts,
            runtime_signal_topics,
            scheduler: ctx.scheduler,
        }))
    }

    fn validate_sources(
        &self,
        definition: &HarnessDefinition,
        _ctx: HarnessRuntimeInitContext,
        source_overlay: std::sync::Arc<HarnessSourceOverlay>,
    ) -> Result<usize> {
        let mut sources = source_files(definition.directory())?
            .into_iter()
            .collect::<BTreeSet<_>>();
        for (path, source) in source_overlay.entries() {
            let path = source_name(path);
            if let Some(source) = source {
                validate_test_source(source)?;
                sources.insert(path);
            } else {
                sources.remove(&path);
            }
        }
        Ok(sources.len())
    }

    fn run_source(
        &self,
        _definition: &HarnessDefinition,
        _ctx: HarnessRuntimeInitContext,
        _source: &str,
    ) -> Result<()> {
        Ok(())
    }
}

struct TestHarnessInstance {
    loaded_scripts: Vec<String>,
    runtime_signal_topics: Vec<String>,
    scheduler: Option<std::sync::Arc<HarnessSchedulerAccess>>,
}

impl HarnessInstance for TestHarnessInstance {
    fn loaded_scripts(&self) -> Vec<String> {
        self.loaded_scripts.clone()
    }

    fn runtime_signal_topics(&self) -> Vec<String> {
        self.runtime_signal_topics.clone()
    }

    fn evaluate_hook(&self, _hook: HarnessHook<'_>) -> Result<Verdict> {
        Ok(Verdict::Allow)
    }

    fn prepares_turn(&self) -> bool {
        false
    }

    fn prepare_turn(
        &self,
        _request: &mut HarnessTurnRequest,
        _services: HarnessTurnServices<'_>,
    ) -> Result<Verdict> {
        Ok(Verdict::Allow)
    }

    fn invoke_action(
        &self,
        request: HarnessActionRequest<'_>,
    ) -> Result<Option<serde_json::Value>> {
        match request.name {
            "test.echo" => Ok(Some(serde_json::json!({
                "status": request
                    .params
                    .get("status")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("ok")
            }))),
            "test.enqueue_followup" => {
                let prompt = request
                    .params
                    .get("prompt")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("Follow up")
                    .to_string();
                let after_seconds = request
                    .params
                    .get("after_seconds")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(30);
                self.create_scheduled_job(request.agent_id, Some(prompt), None, after_seconds)?;
                Ok(Some(serde_json::json!({ "status": "queued followup" })))
            }
            "test.defer" => {
                let after_seconds = request
                    .params
                    .get("after_seconds")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(30);
                self.create_scheduled_job(
                    request.agent_id,
                    None,
                    Some(turin_daemon_protocol::ScheduleActionParams {
                        name: request.name.to_string(),
                        params: Some(request.params.clone()),
                    }),
                    after_seconds,
                )?;
                Ok(Some(serde_json::json!({ "status": "paused" })))
            }
            _ => Ok(None),
        }
    }

    fn dispatch_runtime_signal(&self, _signal: HarnessSignal<'_>) -> Result<usize> {
        Ok(1)
    }
}

impl TestHarnessInstance {
    fn create_scheduled_job(
        &self,
        agent_id: &str,
        prompt: Option<String>,
        action: Option<turin_daemon_protocol::ScheduleActionParams>,
        after_seconds: u64,
    ) -> Result<()> {
        let scheduler = self
            .scheduler
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("test action requires scheduler access"))?;
        let next_run_unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis() as i64
            + (after_seconds.saturating_mul(1000)) as i64;
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(scheduler.create_job(
                turin_daemon_protocol::ScheduleCreateParams {
                    agent_id: agent_id.to_string(),
                    prompt,
                    content: None,
                    tools: None,
                    conflict_policy: None,
                    action,
                    next_run_unix_ms,
                    interval_seconds: None,
                    recurring_pattern: None,
                    overlap_policy: Some("skip".to_string()),
                    work_key: None,
                    max_concurrency: None,
                    persistence: None,
                    enabled: true,
                },
            ))
        })?;
        Ok(())
    }
}

fn source_files(root: &Path) -> Result<Vec<String>> {
    let mut pending = vec![root.to_path_buf()];
    let mut files = Vec::new();

    while let Some(directory) = pending.pop() {
        let entries = match std::fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        for entry in entries {
            let path = entry?.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().and_then(|value| value.to_str()) == Some("lua") {
                files.push(source_name(path.strip_prefix(root).unwrap_or(&path)));
            }
        }
    }

    files.sort();
    Ok(files)
}

fn source_name(path: &Path) -> String {
    path.with_extension("")
        .to_string_lossy()
        .replace(std::path::MAIN_SEPARATOR, "/")
}

fn validate_disk_sources(root: &Path, sources: &[String]) -> Result<()> {
    for source in sources {
        let path = root.join(source).with_extension("lua");
        validate_test_source(&std::fs::read_to_string(path)?)?;
    }
    Ok(())
}

fn test_signal_topics(root: &Path, sources: &[String]) -> Result<Vec<String>> {
    let mut topics = BTreeSet::new();
    for source in sources {
        let body = std::fs::read_to_string(root.join(source).with_extension("lua"))?;
        for line in body.lines() {
            if let Some(topic) = line.trim().strip_prefix("TEST SUBSCRIBE ") {
                topics.insert(topic.trim().to_string());
            }
        }
    }
    Ok(topics.into_iter().collect())
}

fn validate_test_source(source: &str) -> Result<()> {
    if source.contains("INVALID TEST SOURCE") {
        anyhow::bail!("test harness source is invalid");
    }
    Ok(())
}
