use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use tokio::runtime::Builder;
use tokio::runtime::Handle;
use tokio::sync::{mpsc, watch};
use tokio::time;
use turin_control_client::{ConnectionSpec, ControlClient, SessionDetail, SessionSearchHit};
use turin_daemon_protocol::{EventEnvelope, RuntimeEventsSubscribeParams, SessionSearchScope};

use crate::{DashboardSnapshot, DashboardState};

pub const DEFAULT_REFRESH_INTERVAL: Duration = Duration::from_secs(5);
pub const MAX_RECENT_CONNECTION_DRAFTS: usize = 5;

#[derive(Debug, Clone)]
pub struct ConnectionOptions {
    pub config_path: Option<PathBuf>,
    pub endpoint: Option<PathBuf>,
    pub remote_url: Option<String>,
    pub auth_token: Option<String>,
    pub auth_token_env: Option<String>,
    pub profile: Option<String>,
    pub profiles_file: Option<PathBuf>,
    pub suppress_profile_resolution: bool,
}

#[derive(Debug)]
pub struct UiController {
    pub update_rx: mpsc::UnboundedReceiver<UiUpdate>,
    pub command_tx: mpsc::UnboundedSender<OperatorCommand>,
    shutdown_tx: watch::Sender<bool>,
}

#[derive(Debug)]
pub enum UiUpdate {
    Snapshot(Box<DashboardSnapshot>),
    SessionDetail(Box<SessionDetail>),
    SearchResults {
        query: String,
        scope: SessionSearchScope,
        offset: usize,
        limit: usize,
        has_more: bool,
        hits: Vec<SessionSearchHit>,
    },
    Event(EventEnvelope),
    SessionEvent(EventEnvelope),
    RefreshTelemetry {
        duration_ms: u64,
        success: bool,
    },
    Error(String),
    Info(String),
}

#[derive(Debug, Clone)]
pub enum OperatorCommand {
    Refresh,
    FocusSessionStream {
        session_id: Option<String>,
    },
    LoadSessionDetail {
        session_id: String,
    },
    SearchSessions {
        query: String,
        scope: SessionSearchScope,
        limit: usize,
        offset: usize,
    },
    OpenSession {
        agent_id: String,
    },
    ResumeSession {
        session_id: String,
    },
    SetSessionTitle {
        session_id: String,
        title: Option<String>,
    },
    SubmitPrompt {
        session_id: String,
        prompt: String,
    },
    CancelSession {
        session_id: String,
    },
    KillSession {
        session_id: String,
    },
    CancelTask {
        request_id: String,
    },
}

#[derive(Debug, Clone)]
pub struct ConnectionProfileCatalog {
    source_path: PathBuf,
    default_profile: Option<String>,
    profiles: Vec<ConnectionProfileSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectionProfileSummary {
    pub name: String,
    pub kind: ConnectionProfileKind,
    pub target: String,
    pub auth: Option<ConnectionProfileAuth>,
    pub is_default: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionProfileKind {
    LocalConfig,
    LocalEndpoint,
    Remote,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionProfileAuth {
    TokenEnv(String),
    InlineToken,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectionProfileDraft {
    pub kind: ConnectionProfileKind,
    pub target: String,
    pub auth_mode: ConnectionProfileDraftAuthMode,
    pub auth_value: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectionProfileDraftValidation {
    pub target_error: Option<String>,
    pub auth_error: Option<String>,
    pub target_notice: Option<String>,
    pub auth_notice: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectionProfileDraftFieldDiff {
    pub field: String,
    pub draft_value: String,
    pub comparison_value: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectionProfileDraftDiff {
    #[serde(default)]
    pub changed_fields: Vec<ConnectionProfileDraftFieldDiff>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConnectionDraftHistory {
    drafts: Vec<ConnectionProfileDraft>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectionPreflightOutcome {
    Ready,
    Degraded,
    Invalid,
    ConnectFailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectionPreflightReport {
    pub outcome: ConnectionPreflightOutcome,
    pub connection_kind: String,
    pub target: String,
    pub auth: String,
    pub resolved_profile: Option<String>,
    pub message: String,
    pub latency_ms: Option<u64>,
    pub ready: Option<bool>,
    pub transport: Option<String>,
    pub wire_format: Option<String>,
    pub issue_count: Option<usize>,
    pub agent_count: Option<usize>,
    pub channel_count: Option<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectionProfileActivityBook {
    #[serde(default)]
    entries: BTreeMap<String, ConnectionProfileActivity>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectionProfileActivity {
    pub connect_count: u64,
    pub successful_connect_count: u64,
    pub preflight_count: u64,
    pub failure_count: u64,
    pub last_connect_unix_ms: Option<u64>,
    pub last_successful_connect_unix_ms: Option<u64>,
    pub last_preflight_unix_ms: Option<u64>,
    pub last_connect_result: Option<String>,
    pub last_preflight_result: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionProfileDraftAuthMode {
    None,
    TokenEnv,
    InlineToken,
}

impl Default for ConnectionProfileDraft {
    fn default() -> Self {
        Self {
            kind: ConnectionProfileKind::LocalConfig,
            target: "turin.toml".to_string(),
            auth_mode: ConnectionProfileDraftAuthMode::None,
            auth_value: String::new(),
        }
    }
}

impl ConnectionProfileDraft {
    pub fn validate(&self) -> ConnectionProfileDraftValidation {
        let target = self.target.trim();
        let auth_value = self.auth_value.trim();

        match self.kind {
            ConnectionProfileKind::LocalConfig => ConnectionProfileDraftValidation {
                target_error: None,
                auth_error: None,
                target_notice: target
                    .is_empty()
                    .then(|| "Blank config path will default to turin.toml".to_string()),
                auth_notice: None,
            },
            ConnectionProfileKind::LocalEndpoint => ConnectionProfileDraftValidation {
                target_error: target
                    .is_empty()
                    .then(|| "Local endpoint profiles require an endpoint path".to_string()),
                auth_error: None,
                target_notice: None,
                auth_notice: None,
            },
            ConnectionProfileKind::Remote => ConnectionProfileDraftValidation {
                target_error: validate_remote_target(target),
                auth_error: validate_remote_auth(self.auth_mode, auth_value),
                target_notice: None,
                auth_notice: (self.auth_mode == ConnectionProfileDraftAuthMode::InlineToken
                    && !auth_value.is_empty())
                .then(|| "Inline tokens are stored in plaintext in the profiles file".to_string()),
            },
        }
    }

    pub fn display_target(&self) -> String {
        let target = self.target.trim();
        match self.kind {
            ConnectionProfileKind::LocalConfig => {
                if target.is_empty() {
                    "turin.toml".to_string()
                } else {
                    target.to_string()
                }
            }
            ConnectionProfileKind::LocalEndpoint | ConnectionProfileKind::Remote => {
                target.to_string()
            }
        }
    }

    pub fn auth_summary(&self) -> String {
        let auth_value = self.auth_value.trim();
        match self.auth_mode {
            ConnectionProfileDraftAuthMode::None => "none".to_string(),
            ConnectionProfileDraftAuthMode::TokenEnv => {
                if auth_value.is_empty() {
                    "env:<unset>".to_string()
                } else {
                    format!("env:{auth_value}")
                }
            }
            ConnectionProfileDraftAuthMode::InlineToken => "inline token".to_string(),
        }
    }

    pub fn summary_label(&self) -> String {
        format!(
            "{} [{} | {}]",
            self.display_target(),
            profile_kind_label(self.kind),
            self.auth_summary()
        )
    }

    pub fn diff_against(&self, other: &Self) -> ConnectionProfileDraftDiff {
        let draft = self.canonicalized();
        let comparison = other.canonicalized();
        let mut changed_fields = Vec::new();

        if draft.kind != comparison.kind {
            changed_fields.push(ConnectionProfileDraftFieldDiff {
                field: "kind".to_string(),
                draft_value: profile_kind_label(draft.kind).to_string(),
                comparison_value: profile_kind_label(comparison.kind).to_string(),
            });
        }

        if draft.display_target() != comparison.display_target() {
            changed_fields.push(ConnectionProfileDraftFieldDiff {
                field: "target".to_string(),
                draft_value: draft.display_target(),
                comparison_value: comparison.display_target(),
            });
        }

        if draft.auth_mode != comparison.auth_mode {
            changed_fields.push(ConnectionProfileDraftFieldDiff {
                field: "auth_mode".to_string(),
                draft_value: profile_draft_auth_label(draft.auth_mode).to_string(),
                comparison_value: profile_draft_auth_label(comparison.auth_mode).to_string(),
            });
        }

        if draft.auth_value != comparison.auth_value {
            changed_fields.push(ConnectionProfileDraftFieldDiff {
                field: "auth_value".to_string(),
                draft_value: redacted_auth_value(&draft),
                comparison_value: redacted_auth_value(&comparison),
            });
        }

        ConnectionProfileDraftDiff { changed_fields }
    }

    fn canonicalized(&self) -> Self {
        StoredConnectionProfile::from_draft(self)
            .map(|profile| profile.to_draft())
            .unwrap_or_else(|_| {
                let mut draft = self.clone();
                draft.target = draft.target.trim().to_string();
                draft.auth_value = draft.auth_value.trim().to_string();
                if draft.kind != ConnectionProfileKind::Remote {
                    draft.auth_mode = ConnectionProfileDraftAuthMode::None;
                    draft.auth_value.clear();
                }
                draft
            })
    }
}

impl ConnectionProfileDraftValidation {
    pub fn is_valid(&self) -> bool {
        self.target_error.is_none() && self.auth_error.is_none()
    }

    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if let Some(message) = self.target_error.as_ref() {
            parts.push(message.clone());
        }
        if let Some(message) = self.auth_error.as_ref() {
            parts.push(message.clone());
        }
        if parts.is_empty() {
            if let Some(message) = self.target_notice.as_ref() {
                parts.push(message.clone());
            }
            if let Some(message) = self.auth_notice.as_ref() {
                parts.push(message.clone());
            }
        }
        if parts.is_empty() {
            "Draft is ready to save".to_string()
        } else {
            parts.join(" ")
        }
    }
}

impl ConnectionProfileDraftDiff {
    pub fn is_empty(&self) -> bool {
        self.changed_fields.is_empty()
    }

    pub fn changed_field_names(&self) -> Vec<&str> {
        self.changed_fields
            .iter()
            .map(|field| field.field.as_str())
            .collect()
    }

    pub fn summary(&self) -> String {
        if self.changed_fields.is_empty() {
            "Draft matches the comparison target".to_string()
        } else {
            format!(
                "Changed fields: {}",
                self.changed_fields
                    .iter()
                    .map(|field| field.field.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }
}

impl ConnectionDraftHistory {
    pub fn record_success(&mut self, draft: &ConnectionProfileDraft) {
        let draft = draft.canonicalized();
        self.drafts.retain(|existing| *existing != draft);
        self.drafts.insert(0, draft);
        if self.drafts.len() > MAX_RECENT_CONNECTION_DRAFTS {
            self.drafts.truncate(MAX_RECENT_CONNECTION_DRAFTS);
        }
    }

    pub fn drafts(&self) -> &[ConnectionProfileDraft] {
        &self.drafts
    }

    pub fn latest(&self) -> Option<&ConnectionProfileDraft> {
        self.drafts.first()
    }

    pub fn is_empty(&self) -> bool {
        self.drafts.is_empty()
    }
}

fn profile_kind_label(kind: ConnectionProfileKind) -> &'static str {
    match kind {
        ConnectionProfileKind::LocalConfig => "local-config",
        ConnectionProfileKind::LocalEndpoint => "local-endpoint",
        ConnectionProfileKind::Remote => "remote",
    }
}

fn profile_draft_auth_label(mode: ConnectionProfileDraftAuthMode) -> &'static str {
    match mode {
        ConnectionProfileDraftAuthMode::None => "none",
        ConnectionProfileDraftAuthMode::TokenEnv => "token-env",
        ConnectionProfileDraftAuthMode::InlineToken => "inline-token",
    }
}

fn redacted_auth_value(draft: &ConnectionProfileDraft) -> String {
    match draft.auth_mode {
        ConnectionProfileDraftAuthMode::None => "none".to_string(),
        ConnectionProfileDraftAuthMode::TokenEnv => {
            let value = draft.auth_value.trim();
            if value.is_empty() {
                "<unset env>".to_string()
            } else {
                value.to_string()
            }
        }
        ConnectionProfileDraftAuthMode::InlineToken => {
            if draft.auth_value.trim().is_empty() {
                "<unset inline token>".to_string()
            } else {
                "<inline token set>".to_string()
            }
        }
    }
}

impl ConnectionOptions {
    pub fn to_spec(&self) -> Result<ConnectionSpec> {
        let resolved = self.resolve_profile()?;

        if let Some(base_url) = &resolved.remote_url {
            if let Some(auth_token) = &resolved.auth_token {
                return Ok(ConnectionSpec::Remote {
                    base_url: base_url.clone(),
                    auth_token: auth_token.clone(),
                });
            }
            if let Some(auth_token_env) = &resolved.auth_token_env {
                return Ok(ConnectionSpec::RemoteEnv {
                    base_url: base_url.clone(),
                    auth_token_env: auth_token_env.clone(),
                });
            }
            return Err(anyhow!(
                "--remote-url requires either --auth-token or --auth-token-env"
            ));
        }

        if let Some(endpoint) = &resolved.endpoint {
            return Ok(ConnectionSpec::LocalEndpoint {
                endpoint: endpoint.clone(),
            });
        }

        Ok(ConnectionSpec::LocalConfig {
            config_path: resolved
                .config_path
                .unwrap_or_else(|| PathBuf::from("turin.toml")),
        })
    }

    pub fn resolved_profile_name(&self) -> Result<Option<String>> {
        if self.suppress_profile_resolution {
            return Ok(self.profile.clone());
        }
        if self.profile.is_none() && self.profiles_file.is_none() {
            return Ok(None);
        }

        let profiles_path = self.profiles_path();
        let profiles = ConnectionProfiles::load(&profiles_path)?;
        Ok(self
            .profile
            .clone()
            .or_else(|| profiles.default_profile.clone()))
    }

    pub fn load_profiles(&self) -> Result<Option<ConnectionProfileCatalog>> {
        let profiles_path = self.profiles_path();
        if !profiles_path.exists() {
            if self.profile.is_some() || self.profiles_file.is_some() {
                let _ = ConnectionProfiles::load(&profiles_path)?;
            }
            return Ok(None);
        }

        let profiles = ConnectionProfiles::load(&profiles_path)?;
        Ok(Some(ConnectionProfileCatalog::from_raw(
            profiles_path,
            profiles,
        )))
    }

    pub fn materialized(&self) -> Result<Self> {
        let resolved = if self.profile.is_some() || self.profiles_path().exists() {
            self.resolve_profile()?
        } else {
            self.clone()
        };
        Ok(Self {
            config_path: resolved.config_path,
            endpoint: resolved.endpoint,
            remote_url: resolved.remote_url,
            auth_token: resolved.auth_token,
            auth_token_env: resolved.auth_token_env,
            profile: None,
            profiles_file: Some(self.profiles_path()),
            suppress_profile_resolution: self.suppress_profile_resolution,
        })
    }

    pub fn current_profile_draft(&self) -> Result<ConnectionProfileDraft> {
        let materialized = self.materialized()?;
        Ok(StoredConnectionProfile::from_options(&materialized).to_draft())
    }

    pub fn connection_options_for_draft(&self, draft: &ConnectionProfileDraft) -> Result<Self> {
        let profile = StoredConnectionProfile::from_draft(draft)?;
        Ok(Self {
            config_path: profile.config_path,
            endpoint: profile.endpoint,
            remote_url: profile.remote_url,
            auth_token: profile.auth_token,
            auth_token_env: profile.auth_token_env,
            profile: None,
            profiles_file: Some(self.profiles_path()),
            suppress_profile_resolution: true,
        })
    }

    pub fn draft_to_spec(&self, draft: &ConnectionProfileDraft) -> Result<ConnectionSpec> {
        self.connection_options_for_draft(draft)?.to_spec()
    }

    pub fn draft_diff_against_profile(
        &self,
        draft: &ConnectionProfileDraft,
        profile_name: &str,
    ) -> Result<ConnectionProfileDraftDiff> {
        let profile_draft = self.load_profile_draft(profile_name)?;
        Ok(draft.diff_against(&profile_draft))
    }

    pub fn load_profile_draft(&self, name: &str) -> Result<ConnectionProfileDraft> {
        let name = validate_profile_name(name)?;
        let profiles_path = self.profiles_path();
        let profiles = ConnectionProfiles::load(&profiles_path)?;
        let profile = profiles.profiles.get(name).with_context(|| {
            format!(
                "Connection profile '{}' was not found in '{}'",
                name,
                profiles_path.display()
            )
        })?;
        Ok(profile.to_draft())
    }

    pub fn save_profile(&self, name: &str, make_default: bool) -> Result<ConnectionProfileCatalog> {
        let draft = self.current_profile_draft()?;
        self.save_profile_draft(name, &draft, make_default)
    }

    pub fn save_profile_draft(
        &self,
        name: &str,
        draft: &ConnectionProfileDraft,
        make_default: bool,
    ) -> Result<ConnectionProfileCatalog> {
        let name = validate_profile_name(name)?;

        let profiles_path = self.profiles_path();
        let mut profiles = ConnectionProfiles::load_optional(&profiles_path)?;
        profiles.profiles.insert(
            name.to_string(),
            StoredConnectionProfile::from_draft(draft)?,
        );
        if make_default || profiles.default_profile.is_none() {
            profiles.default_profile = Some(name.to_string());
        }
        profiles.save(&profiles_path)?;
        Ok(ConnectionProfileCatalog::from_raw(profiles_path, profiles))
    }

    pub fn duplicate_profile(
        &self,
        source_name: &str,
        new_name: &str,
        make_default: bool,
    ) -> Result<ConnectionProfileCatalog> {
        let source_name = validate_profile_name(source_name)?;
        let new_name = validate_profile_name(new_name)?;
        if source_name == new_name {
            return Err(anyhow!(
                "New profile name must be different from the source profile"
            ));
        }

        let profiles_path = self.profiles_path();
        let mut profiles = ConnectionProfiles::load(&profiles_path)?;
        if profiles.profiles.contains_key(new_name) {
            return Err(anyhow!(
                "Connection profile '{}' already exists in '{}'",
                new_name,
                profiles_path.display()
            ));
        }
        let source = profiles
            .profiles
            .get(source_name)
            .cloned()
            .with_context(|| {
                format!(
                    "Connection profile '{}' was not found in '{}'",
                    source_name,
                    profiles_path.display()
                )
            })?;
        profiles.profiles.insert(new_name.to_string(), source);
        if make_default || profiles.default_profile.is_none() {
            profiles.default_profile = Some(new_name.to_string());
        }
        profiles.save(&profiles_path)?;
        Ok(ConnectionProfileCatalog::from_raw(profiles_path, profiles))
    }

    pub fn rename_profile(
        &self,
        source_name: &str,
        new_name: &str,
        make_default: bool,
    ) -> Result<ConnectionProfileCatalog> {
        let source_name = validate_profile_name(source_name)?;
        let new_name = validate_profile_name(new_name)?;
        if source_name == new_name {
            return Err(anyhow!(
                "New profile name must be different from the source profile"
            ));
        }

        let profiles_path = self.profiles_path();
        let mut profiles = ConnectionProfiles::load(&profiles_path)?;
        if profiles.profiles.contains_key(new_name) {
            return Err(anyhow!(
                "Connection profile '{}' already exists in '{}'",
                new_name,
                profiles_path.display()
            ));
        }
        let source_was_default = profiles.default_profile.as_deref() == Some(source_name);
        let profile = profiles.profiles.remove(source_name).with_context(|| {
            format!(
                "Connection profile '{}' was not found in '{}'",
                source_name,
                profiles_path.display()
            )
        })?;
        profiles.profiles.insert(new_name.to_string(), profile);
        if source_was_default || make_default {
            profiles.default_profile = Some(new_name.to_string());
        }
        profiles.save(&profiles_path)?;
        Ok(ConnectionProfileCatalog::from_raw(profiles_path, profiles))
    }

    pub fn delete_profile(&self, name: &str) -> Result<ConnectionProfileCatalog> {
        let name = validate_profile_name(name)?;

        let profiles_path = self.profiles_path();
        let mut profiles = ConnectionProfiles::load(&profiles_path)?;
        if profiles.profiles.remove(name).is_none() {
            return Err(anyhow!(
                "Connection profile '{}' was not found in '{}'",
                name,
                profiles_path.display()
            ));
        }
        if profiles.default_profile.as_deref() == Some(name) {
            profiles.default_profile = profiles.profiles.keys().next().cloned();
        }
        profiles.save(&profiles_path)?;
        Ok(ConnectionProfileCatalog::from_raw(profiles_path, profiles))
    }

    fn resolve_profile(&self) -> Result<Self> {
        if self.profile.is_none() && self.profiles_file.is_none() {
            return Ok(self.clone());
        }
        if self.suppress_profile_resolution {
            return Ok(self.clone());
        }

        let profiles_path = self.profiles_path();
        let profiles = ConnectionProfiles::load(&profiles_path)?;
        let Some(profile_name) = self
            .profile
            .clone()
            .or_else(|| profiles.default_profile.clone())
        else {
            return Ok(self.clone());
        };
        let profile = profiles.profiles.get(&profile_name).with_context(|| {
            format!(
                "Connection profile '{}' was not found in '{}'",
                profile_name,
                profiles_path.display()
            )
        })?;

        Ok(Self {
            config_path: self
                .config_path
                .clone()
                .or_else(|| profile.config_path.clone()),
            endpoint: self.endpoint.clone().or_else(|| profile.endpoint.clone()),
            remote_url: self
                .remote_url
                .clone()
                .or_else(|| profile.remote_url.clone()),
            auth_token: self
                .auth_token
                .clone()
                .or_else(|| profile.auth_token.clone()),
            auth_token_env: self
                .auth_token_env
                .clone()
                .or_else(|| profile.auth_token_env.clone()),
            profile: Some(profile_name),
            profiles_file: Some(profiles_path),
            suppress_profile_resolution: false,
        })
    }

    pub fn profiles_path(&self) -> PathBuf {
        self.profiles_file
            .clone()
            .unwrap_or_else(|| PathBuf::from("ui-profiles.toml"))
    }
}

impl ConnectionPreflightReport {
    pub fn is_success(&self) -> bool {
        matches!(
            self.outcome,
            ConnectionPreflightOutcome::Ready | ConnectionPreflightOutcome::Degraded
        )
    }

    pub fn summary_label(&self) -> String {
        match self.latency_ms {
            Some(latency) => format!(
                "{} in {}ms: {}",
                preflight_outcome_label(self.outcome),
                latency,
                self.message
            ),
            None => format!(
                "{}: {}",
                preflight_outcome_label(self.outcome),
                self.message
            ),
        }
    }
}

impl ConnectionProfileActivityBook {
    pub fn record_connect_result(
        &mut self,
        profile_name: impl Into<String>,
        success: bool,
        message: impl Into<String>,
    ) {
        let now = now_unix_ms();
        let entry = self.entries.entry(profile_name.into()).or_default();
        entry.connect_count += 1;
        entry.last_connect_unix_ms = Some(now);
        entry.last_connect_result = Some(message.into());
        if success {
            entry.successful_connect_count += 1;
            entry.last_successful_connect_unix_ms = Some(now);
        } else {
            entry.failure_count += 1;
        }
    }

    pub fn record_preflight_result(
        &mut self,
        profile_name: impl Into<String>,
        success: bool,
        message: impl Into<String>,
    ) {
        let now = now_unix_ms();
        let entry = self.entries.entry(profile_name.into()).or_default();
        entry.preflight_count += 1;
        entry.last_preflight_unix_ms = Some(now);
        entry.last_preflight_result = Some(message.into());
        if !success {
            entry.failure_count += 1;
        }
    }

    pub fn entry(&self, profile_name: &str) -> Option<&ConnectionProfileActivity> {
        self.entries.get(profile_name)
    }
}

pub fn preflight_connection_blocking(options: &ConnectionOptions) -> ConnectionPreflightReport {
    let resolved = match options.materialized() {
        Ok(resolved) => resolved,
        Err(err) => {
            return invalid_preflight_report(
                &options
                    .current_profile_draft()
                    .unwrap_or_else(|_| ConnectionProfileDraft::default()),
                options.profile.clone(),
                err.to_string(),
            );
        }
    };

    let draft = resolved
        .current_profile_draft()
        .unwrap_or_else(|_| ConnectionProfileDraft::default());
    if let Some(report) = preflight_env_error_report(&resolved, &draft, options.profile.clone()) {
        return report;
    }

    let spec = match resolved.to_spec() {
        Ok(spec) => spec,
        Err(err) => {
            return invalid_preflight_report(&draft, options.profile.clone(), err.to_string());
        }
    };

    let started = Instant::now();
    match blocking_connect_dashboard(spec) {
        Ok((_, dashboard)) => {
            let latency_ms = started.elapsed().as_millis() as u64;
            let ready = dashboard
                .health
                .as_ref()
                .map(|health| health.ready)
                .unwrap_or(false);
            let transport = dashboard
                .health
                .as_ref()
                .map(|health| health.transport.clone());
            let wire_format = dashboard
                .health
                .as_ref()
                .map(|health| health.wire_format.clone());
            let issue_count = dashboard.health.as_ref().map(|health| health.issue_count);
            let agent_count = dashboard.health.as_ref().map(|health| health.agent_count);
            let channel_count = dashboard.health.as_ref().map(|health| health.channel_count);
            ConnectionPreflightReport {
                outcome: if ready {
                    ConnectionPreflightOutcome::Ready
                } else {
                    ConnectionPreflightOutcome::Degraded
                },
                connection_kind: profile_kind_label(draft.kind).to_string(),
                target: draft.display_target(),
                auth: draft.auth_summary(),
                resolved_profile: options.profile.clone(),
                message: if ready {
                    "Connected and daemon reported ready".to_string()
                } else {
                    "Connected, but the daemon reported degraded readiness".to_string()
                },
                latency_ms: Some(latency_ms),
                ready: Some(ready),
                transport,
                wire_format,
                issue_count,
                agent_count,
                channel_count,
            }
        }
        Err(err) => ConnectionPreflightReport {
            outcome: ConnectionPreflightOutcome::ConnectFailed,
            connection_kind: profile_kind_label(draft.kind).to_string(),
            target: draft.display_target(),
            auth: draft.auth_summary(),
            resolved_profile: options.profile.clone(),
            message: err.to_string(),
            latency_ms: Some(started.elapsed().as_millis() as u64),
            ready: None,
            transport: None,
            wire_format: None,
            issue_count: None,
            agent_count: None,
            channel_count: None,
        },
    }
}

pub fn preflight_draft_blocking(
    options: &ConnectionOptions,
    draft: &ConnectionProfileDraft,
) -> ConnectionPreflightReport {
    let validation = draft.validate();
    if !validation.is_valid() {
        return invalid_preflight_report(draft, None, validation.summary());
    }

    match options.connection_options_for_draft(draft) {
        Ok(options) => preflight_connection_blocking(&options),
        Err(err) => invalid_preflight_report(draft, None, err.to_string()),
    }
}

pub fn ensure_local_daemon_for_options(options: &ConnectionOptions) -> Result<String> {
    let resolved = options.materialized()?;
    let draft = resolved.current_profile_draft()?;
    run_local_daemon_ensure(&resolved, &draft)
}

pub fn ensure_local_daemon_for_draft(
    options: &ConnectionOptions,
    draft: &ConnectionProfileDraft,
) -> Result<String> {
    let resolved = options.connection_options_for_draft(draft)?;
    run_local_daemon_ensure(&resolved, draft)
}

fn validate_profile_name(name: &str) -> Result<&str> {
    let name = name.trim();
    if name.is_empty() {
        return Err(anyhow!("Connection profile name cannot be empty"));
    }
    Ok(name)
}

fn invalid_preflight_report(
    draft: &ConnectionProfileDraft,
    resolved_profile: Option<String>,
    message: String,
) -> ConnectionPreflightReport {
    ConnectionPreflightReport {
        outcome: ConnectionPreflightOutcome::Invalid,
        connection_kind: profile_kind_label(draft.kind).to_string(),
        target: draft.display_target(),
        auth: draft.auth_summary(),
        resolved_profile,
        message,
        latency_ms: None,
        ready: None,
        transport: None,
        wire_format: None,
        issue_count: None,
        agent_count: None,
        channel_count: None,
    }
}

fn preflight_env_error_report(
    options: &ConnectionOptions,
    draft: &ConnectionProfileDraft,
    resolved_profile: Option<String>,
) -> Option<ConnectionPreflightReport> {
    let env_name = options.auth_token_env.as_deref()?;
    if std::env::var_os(env_name).is_some() {
        return None;
    }
    Some(invalid_preflight_report(
        draft,
        resolved_profile,
        format!("Environment variable '{}' is not set", env_name),
    ))
}

fn blocking_connect_dashboard(spec: ConnectionSpec) -> Result<(ControlClient, DashboardState)> {
    std::thread::spawn(move || {
        let runtime = Builder::new_current_thread().enable_all().build()?;
        runtime.block_on(async move { connect_dashboard(&spec).await })
    })
    .join()
    .map_err(|_| anyhow!("Connection preflight thread panicked"))?
}

fn run_local_daemon_ensure(
    options: &ConnectionOptions,
    draft: &ConnectionProfileDraft,
) -> Result<String> {
    let config_path = match draft.kind {
        ConnectionProfileKind::LocalConfig => options
            .config_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("turin.toml")),
        ConnectionProfileKind::LocalEndpoint => {
            return Err(anyhow!(
                "Local endpoint profiles do not imply a config file. Run `turin daemon ensure --config <path>` manually."
            ));
        }
        ConnectionProfileKind::Remote => {
            return Err(anyhow!(
                "Local daemon ensure is only available for local connection profiles."
            ));
        }
    };

    let (program, command_label) = resolve_turin_binary();
    let output = Command::new(&program)
        .arg("daemon")
        .arg("ensure")
        .arg("--config")
        .arg(&config_path)
        .output()
        .with_context(|| {
            format!(
                "Failed to launch `{command_label} daemon ensure --config {}`",
                config_path.display()
            )
        })?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let detail = if !stdout.is_empty() {
            stdout
        } else if !stderr.is_empty() {
            stderr
        } else {
            "daemon ensure completed successfully".to_string()
        };
        Ok(format!(
            "Ensured local daemon with `{command_label} daemon ensure --config {}`: {}",
            config_path.display(),
            detail
        ))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = if !stderr.is_empty() { stderr } else { stdout };
        Err(anyhow!(
            "`{command_label} daemon ensure --config {}` failed: {}",
            config_path.display(),
            detail
        ))
    }
}

fn resolve_turin_binary() -> (PathBuf, String) {
    let fallback = PathBuf::from("turin");
    let current_exe = std::env::current_exe().ok();
    let sibling_name = if cfg!(windows) { "turin.exe" } else { "turin" };
    if let Some(path) = current_exe
        .as_ref()
        .and_then(|path| path.parent())
        .map(|dir| dir.join(sibling_name))
        && path.exists()
    {
        return (path.clone(), path.display().to_string());
    }
    (fallback.clone(), fallback.display().to_string())
}

fn preflight_outcome_label(outcome: ConnectionPreflightOutcome) -> &'static str {
    match outcome {
        ConnectionPreflightOutcome::Ready => "ready",
        ConnectionPreflightOutcome::Degraded => "degraded",
        ConnectionPreflightOutcome::Invalid => "invalid",
        ConnectionPreflightOutcome::ConnectFailed => "connect-failed",
    }
}

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

fn validate_remote_target(target: &str) -> Option<String> {
    if target.is_empty() {
        return Some("Remote profiles require a base URL".to_string());
    }
    if target.chars().any(char::is_whitespace) {
        return Some("Remote base URLs cannot contain whitespace".to_string());
    }

    let Some((scheme, remainder)) = target.split_once("://") else {
        return Some("Remote base URLs must start with http:// or https://".to_string());
    };
    if scheme != "http" && scheme != "https" {
        return Some("Remote base URLs must start with http:// or https://".to_string());
    }

    let authority = remainder
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default()
        .trim();
    if authority.is_empty() || authority.starts_with(':') {
        return Some("Remote base URLs must include a host".to_string());
    }

    None
}

fn validate_remote_auth(
    auth_mode: ConnectionProfileDraftAuthMode,
    auth_value: &str,
) -> Option<String> {
    match auth_mode {
        ConnectionProfileDraftAuthMode::None => {
            Some("Remote profiles require either a token env var or an inline token".to_string())
        }
        ConnectionProfileDraftAuthMode::TokenEnv => {
            if auth_value.is_empty() {
                return Some("Remote profiles using env auth require an env var name".to_string());
            }
            let mut chars = auth_value.chars();
            match chars.next() {
                Some(first) if first == '_' || first.is_ascii_alphabetic() => {}
                _ => {
                    return Some(
                        "Env var names must start with a letter or underscore".to_string(),
                    );
                }
            }
            if chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric()) {
                None
            } else {
                Some("Env var names may only contain letters, numbers, and underscores".to_string())
            }
        }
        ConnectionProfileDraftAuthMode::InlineToken => {
            if auth_value.is_empty() {
                Some("Remote profiles using inline auth require a token value".to_string())
            } else {
                None
            }
        }
    }
}

pub async fn connect_dashboard(spec: &ConnectionSpec) -> Result<(ControlClient, DashboardState)> {
    let client = ControlClient::connect(spec).await?;
    let dashboard = DashboardState::load(&client).await?;
    Ok((client, dashboard))
}

pub fn spawn_controller(handle: &Handle, client: ControlClient) -> UiController {
    spawn_controller_with_interval(handle, client, DEFAULT_REFRESH_INTERVAL)
}

pub fn spawn_controller_with_interval(
    handle: &Handle,
    client: ControlClient,
    refresh_interval: Duration,
) -> UiController {
    let (update_tx, update_rx) = mpsc::unbounded_channel::<UiUpdate>();
    let (command_tx, command_rx) = mpsc::unbounded_channel::<OperatorCommand>();
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let (focused_session_tx, focused_session_rx) = watch::channel::<Option<String>>(None);

    spawn_event_task(
        handle,
        client.clone(),
        update_tx.clone(),
        shutdown_rx.clone(),
    );
    spawn_focused_session_event_task(
        handle,
        client.clone(),
        update_tx.clone(),
        focused_session_rx,
        shutdown_rx.clone(),
    );
    spawn_refresh_task(
        handle,
        client.clone(),
        update_tx.clone(),
        refresh_interval,
        shutdown_rx.clone(),
    );
    spawn_command_task(
        handle,
        client,
        command_rx,
        update_tx,
        focused_session_tx,
        shutdown_rx,
    );

    UiController {
        update_rx,
        command_tx,
        shutdown_tx,
    }
}

impl UiController {
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }
}

impl ConnectionProfileCatalog {
    pub fn source_path(&self) -> &Path {
        &self.source_path
    }

    pub fn default_profile(&self) -> Option<&str> {
        self.default_profile.as_deref()
    }

    pub fn profiles(&self) -> &[ConnectionProfileSummary] {
        &self.profiles
    }

    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    pub fn connection_options(&self, name: &str) -> Option<ConnectionOptions> {
        self.profiles
            .iter()
            .any(|profile| profile.name == name)
            .then(|| ConnectionOptions {
                config_path: None,
                endpoint: None,
                remote_url: None,
                auth_token: None,
                auth_token_env: None,
                profile: Some(name.to_string()),
                profiles_file: Some(self.source_path.clone()),
                suppress_profile_resolution: false,
            })
    }

    fn from_raw(source_path: PathBuf, raw: ConnectionProfiles) -> Self {
        let default_profile = raw.default_profile.clone();
        let profiles = raw
            .profiles
            .into_iter()
            .map(|(name, profile)| ConnectionProfileSummary {
                is_default: default_profile.as_deref() == Some(name.as_str()),
                name,
                kind: profile.kind(),
                target: profile.target(),
                auth: profile.auth_label(),
            })
            .collect();

        Self {
            source_path,
            default_profile,
            profiles,
        }
    }
}

fn spawn_event_task(
    handle: &Handle,
    client: ControlClient,
    tx: mpsc::UnboundedSender<UiUpdate>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    handle.spawn(async move {
        match client
            .subscribe_managed(RuntimeEventsSubscribeParams::default())
            .await
        {
            Ok(mut stream) => loop {
                tokio::select! {
                    changed = shutdown_rx.changed() => {
                        if changed.is_ok() && *shutdown_rx.borrow() {
                            break;
                        }
                    }
                    next = stream.next_event() => match next {
                    Ok(event) => {
                        if tx.send(UiUpdate::Event(event)).is_err() {
                            break;
                        }
                    }
                    Err(err) => {
                        if tx.send(UiUpdate::Error(err.to_string())).is_err() {
                            break;
                        }
                        break;
                    }
                    }
                }
            },
            Err(err) => {
                let _ = tx.send(UiUpdate::Error(err.to_string()));
            }
        }
    });
}

fn spawn_focused_session_event_task(
    handle: &Handle,
    client: ControlClient,
    tx: mpsc::UnboundedSender<UiUpdate>,
    mut focused_session_rx: watch::Receiver<Option<String>>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    handle.spawn(async move {
        let mut current_session_id = focused_session_rx.borrow().clone();
        loop {
            let Some(session_id) = current_session_id.clone() else {
                tokio::select! {
                    changed = shutdown_rx.changed() => {
                        if changed.is_ok() && *shutdown_rx.borrow() {
                            break;
                        }
                    }
                    changed = focused_session_rx.changed() => {
                        if changed.is_err() {
                            break;
                        }
                        current_session_id = focused_session_rx.borrow().clone();
                    }
                }
                continue;
            };

            match client
                .subscribe_managed(RuntimeEventsSubscribeParams {
                    agent_id: None,
                    session_id: Some(session_id.clone()),
                })
                .await
            {
                Ok(mut stream) => loop {
                    tokio::select! {
                        changed = shutdown_rx.changed() => {
                            if changed.is_ok() && *shutdown_rx.borrow() {
                                return;
                            }
                        }
                        changed = focused_session_rx.changed() => {
                            if changed.is_err() {
                                return;
                            }
                            let next_session_id = focused_session_rx.borrow().clone();
                            if next_session_id != Some(session_id.clone()) {
                                current_session_id = next_session_id;
                                break;
                            }
                        }
                        next = stream.next_event() => match next {
                            Ok(event) => {
                                if tx.send(UiUpdate::SessionEvent(event)).is_err() {
                                    return;
                                }
                            }
                            Err(err) => {
                                if tx.send(UiUpdate::Error(format!(
                                    "Focused session stream failed for {}: {}",
                                    session_id, err
                                ))).is_err() {
                                    return;
                                }
                                current_session_id = focused_session_rx.borrow().clone();
                                break;
                            }
                        }
                    }
                },
                Err(err) => {
                    if tx
                        .send(UiUpdate::Error(format!(
                            "Failed to subscribe to focused session {}: {}",
                            session_id, err
                        )))
                        .is_err()
                    {
                        break;
                    }
                    current_session_id = focused_session_rx.borrow().clone();
                }
            }
        }
    });
}

fn spawn_refresh_task(
    handle: &Handle,
    client: ControlClient,
    tx: mpsc::UnboundedSender<UiUpdate>,
    refresh_interval: Duration,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    handle.spawn(async move {
        let mut interval = time::interval(refresh_interval);
        loop {
            tokio::select! {
                changed = shutdown_rx.changed() => {
                    if changed.is_ok() && *shutdown_rx.borrow() {
                        break;
                    }
                }
                _ = interval.tick() => {}
            }
            let started = Instant::now();
            match DashboardState::snapshot(&client).await {
                Ok(snapshot) => {
                    if tx.send(UiUpdate::Snapshot(Box::new(snapshot))).is_err() {
                        break;
                    }
                    if tx
                        .send(UiUpdate::RefreshTelemetry {
                            duration_ms: started.elapsed().as_millis() as u64,
                            success: true,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
                Err(err) => {
                    if tx.send(UiUpdate::Error(err.to_string())).is_err() {
                        break;
                    }
                    if tx
                        .send(UiUpdate::RefreshTelemetry {
                            duration_ms: started.elapsed().as_millis() as u64,
                            success: false,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            }
        }
    });
}

fn spawn_command_task(
    handle: &Handle,
    client: ControlClient,
    mut command_rx: mpsc::UnboundedReceiver<OperatorCommand>,
    tx: mpsc::UnboundedSender<UiUpdate>,
    focused_session_tx: watch::Sender<Option<String>>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    handle.spawn(async move {
        loop {
            let command = tokio::select! {
                changed = shutdown_rx.changed() => {
                    if changed.is_ok() && *shutdown_rx.borrow() {
                        break;
                    }
                    continue;
                }
                maybe_command = command_rx.recv() => {
                    match maybe_command {
                        Some(command) => command,
                        None => break,
                    }
                }
            };

            if let OperatorCommand::FocusSessionStream { session_id } = &command {
                if focused_session_tx.send(session_id.clone()).is_err()
                    && tx
                        .send(UiUpdate::Error(
                            "Failed to update focused session stream".to_string(),
                        ))
                        .is_err()
                {
                    break;
                }
                continue;
            }

            if let OperatorCommand::LoadSessionDetail { session_id } = &command {
                match client.get_session(session_id.as_str()).await {
                    Ok(detail) => {
                        if tx.send(UiUpdate::SessionDetail(Box::new(detail))).is_err() {
                            break;
                        }
                    }
                    Err(err) => {
                        if tx.send(UiUpdate::Error(err.to_string())).is_err() {
                            break;
                        }
                    }
                }
                continue;
            }

            if let OperatorCommand::SearchSessions {
                query,
                scope,
                limit,
                offset,
            } = &command
            {
                match client
                    .search_sessions(
                        query.as_str(),
                        *scope,
                        limit.saturating_add(1),
                        *offset,
                    )
                    .await
                {
                    Ok(mut hits) => {
                        let has_more = hits.len() > *limit;
                        if has_more {
                            hits.truncate(*limit);
                        }
                        if tx
                            .send(UiUpdate::SearchResults {
                                query: query.clone(),
                                scope: *scope,
                                offset: *offset,
                                limit: *limit,
                                has_more,
                                hits,
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(err) => {
                        if tx.send(UiUpdate::Error(err.to_string())).is_err() {
                            break;
                        }
                    }
                }
                continue;
            }

            match execute_operator_command(&client, command).await {
                Ok(message) => {
                    if tx.send(UiUpdate::Info(message)).is_err() {
                        break;
                    }
                    let started = Instant::now();
                    match DashboardState::snapshot(&client).await {
                        Ok(snapshot) => {
                            if tx.send(UiUpdate::Snapshot(Box::new(snapshot))).is_err() {
                                break;
                            }
                            if tx
                                .send(UiUpdate::RefreshTelemetry {
                                    duration_ms: started.elapsed().as_millis() as u64,
                                    success: true,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        Err(err) => {
                            if tx.send(UiUpdate::Error(err.to_string())).is_err() {
                                break;
                            }
                            if tx
                                .send(UiUpdate::RefreshTelemetry {
                                    duration_ms: started.elapsed().as_millis() as u64,
                                    success: false,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                    }
                }
                Err(err) => {
                    if tx.send(UiUpdate::Error(err.to_string())).is_err() {
                        break;
                    }
                }
            }
        }
    });
}

pub async fn execute_operator_command(
    client: &ControlClient,
    command: OperatorCommand,
) -> Result<String> {
    match command {
        OperatorCommand::Refresh => Ok("Refreshed Turin state".to_string()),
        OperatorCommand::FocusSessionStream { .. } => {
            Ok("Updated focused session stream".to_string())
        }
        OperatorCommand::LoadSessionDetail { .. } => Ok("Loaded session detail".to_string()),
        OperatorCommand::SearchSessions { .. } => {
            Ok("Loaded persisted session search results".to_string())
        }
        OperatorCommand::OpenSession { agent_id } => {
            let session = client.open_session(&agent_id, None).await?;
            Ok(format!(
                "Opened live session {} for agent {}",
                session.session_id, session.agent_id
            ))
        }
        OperatorCommand::ResumeSession { session_id } => {
            let session = client.resume_session(&session_id, None).await?;
            Ok(format!(
                "Resumed session {} into live slot {}",
                session.session_id, session.slot_id
            ))
        }
        OperatorCommand::SetSessionTitle { session_id, title } => {
            let session = client.set_session_title(&session_id, title.clone()).await?;
            let title = session
                .metadata
                .as_ref()
                .and_then(|value| value.get("title"))
                .and_then(|value| value.as_str())
                .unwrap_or("cleared");
            Ok(format!(
                "Updated session {} title to {}",
                session.session_id, title
            ))
        }
        OperatorCommand::SubmitPrompt { session_id, prompt } => {
            let task = client
                .submit_task(None, Some(session_id.clone()), prompt)
                .await?;
            Ok(format!(
                "Submitted task {} to session {}",
                task.request_id, session_id
            ))
        }
        OperatorCommand::CancelSession { session_id } => {
            let result = client.cancel_session(&session_id).await?;
            Ok(format!(
                "Requested cancel for session {} ({})",
                result.session_id, result.agent_id
            ))
        }
        OperatorCommand::KillSession { session_id } => {
            let result = client.kill_session(&session_id).await?;
            Ok(format!(
                "Killed session {} ({})",
                result.session_id, result.agent_id
            ))
        }
        OperatorCommand::CancelTask { request_id } => {
            let task = client.cancel_task(&request_id).await?;
            Ok(format!(
                "Cancelled task {} -> {}",
                task.request_id, task.state
            ))
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct ConnectionProfiles {
    #[serde(skip_serializing_if = "Option::is_none")]
    default_profile: Option<String>,
    #[serde(default)]
    profiles: BTreeMap<String, StoredConnectionProfile>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct StoredConnectionProfile {
    #[serde(
        default,
        rename = "config",
        alias = "config_path",
        skip_serializing_if = "Option::is_none"
    )]
    config_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    endpoint: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    remote_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    auth_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    auth_token_env: Option<String>,
}

impl StoredConnectionProfile {
    fn from_options(options: &ConnectionOptions) -> Self {
        if let Some(remote_url) = &options.remote_url {
            return Self {
                config_path: None,
                endpoint: None,
                remote_url: Some(remote_url.clone()),
                auth_token: options.auth_token.clone(),
                auth_token_env: options.auth_token_env.clone(),
            };
        }

        if let Some(endpoint) = &options.endpoint {
            return Self {
                config_path: None,
                endpoint: Some(endpoint.clone()),
                remote_url: None,
                auth_token: None,
                auth_token_env: None,
            };
        }

        Self {
            config_path: Some(
                options
                    .config_path
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("turin.toml")),
            ),
            endpoint: None,
            remote_url: None,
            auth_token: None,
            auth_token_env: None,
        }
    }

    fn from_draft(draft: &ConnectionProfileDraft) -> Result<Self> {
        let target = draft.target.trim();
        let validation = draft.validate();
        if let Some(message) = validation.target_error.as_ref() {
            return Err(anyhow!(message.clone()));
        }
        if let Some(message) = validation.auth_error.as_ref() {
            return Err(anyhow!(message.clone()));
        }

        match draft.kind {
            ConnectionProfileKind::LocalConfig => Ok(Self {
                config_path: Some(if target.is_empty() {
                    PathBuf::from("turin.toml")
                } else {
                    PathBuf::from(target)
                }),
                endpoint: None,
                remote_url: None,
                auth_token: None,
                auth_token_env: None,
            }),
            ConnectionProfileKind::LocalEndpoint => Ok(Self {
                config_path: None,
                endpoint: Some(PathBuf::from(target)),
                remote_url: None,
                auth_token: None,
                auth_token_env: None,
            }),
            ConnectionProfileKind::Remote => {
                let auth_value = draft.auth_value.trim();
                let (auth_token, auth_token_env) = match draft.auth_mode {
                    ConnectionProfileDraftAuthMode::None => unreachable!("validated above"),
                    ConnectionProfileDraftAuthMode::TokenEnv => {
                        (None, Some(auth_value.to_string()))
                    }
                    ConnectionProfileDraftAuthMode::InlineToken => {
                        (Some(auth_value.to_string()), None)
                    }
                };

                Ok(Self {
                    config_path: None,
                    endpoint: None,
                    remote_url: Some(target.to_string()),
                    auth_token,
                    auth_token_env,
                })
            }
        }
    }

    fn kind(&self) -> ConnectionProfileKind {
        if self.remote_url.is_some() {
            ConnectionProfileKind::Remote
        } else if self.endpoint.is_some() {
            ConnectionProfileKind::LocalEndpoint
        } else {
            ConnectionProfileKind::LocalConfig
        }
    }

    fn target(&self) -> String {
        self.remote_url
            .clone()
            .or_else(|| {
                self.endpoint
                    .as_ref()
                    .map(|path| path.display().to_string())
            })
            .or_else(|| {
                self.config_path
                    .as_ref()
                    .map(|path| path.display().to_string())
            })
            .unwrap_or_else(|| "turin.toml".to_string())
    }

    fn auth_label(&self) -> Option<ConnectionProfileAuth> {
        self.auth_token_env
            .as_ref()
            .map(|env| ConnectionProfileAuth::TokenEnv(env.clone()))
            .or_else(|| {
                self.auth_token
                    .as_ref()
                    .map(|_| ConnectionProfileAuth::InlineToken)
            })
    }

    fn to_draft(&self) -> ConnectionProfileDraft {
        let (auth_mode, auth_value) = if let Some(env) = &self.auth_token_env {
            (ConnectionProfileDraftAuthMode::TokenEnv, env.clone())
        } else if let Some(token) = &self.auth_token {
            (ConnectionProfileDraftAuthMode::InlineToken, token.clone())
        } else {
            (ConnectionProfileDraftAuthMode::None, String::new())
        };

        ConnectionProfileDraft {
            kind: self.kind(),
            target: self.target(),
            auth_mode,
            auth_value,
        }
    }
}

impl ConnectionProfiles {
    fn load(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path).with_context(|| {
            format!(
                "Failed to read connection profiles from '{}'",
                path.display()
            )
        })?;
        toml::from_str(&raw).with_context(|| {
            format!(
                "Failed to parse connection profiles TOML from '{}'",
                path.display()
            )
        })
    }

    fn load_optional(path: &Path) -> Result<Self> {
        if path.exists() {
            Self::load(path)
        } else {
            Ok(Self::default())
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "Failed to create connection profile directory '{}'",
                    parent.display()
                )
            })?;
        }
        let raw = toml::to_string_pretty(self).context("Failed to encode connection profiles")?;
        fs::write(path, raw).with_context(|| {
            format!(
                "Failed to write connection profiles to '{}'",
                path.display()
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connection_options_default_to_local_config() {
        let options = ConnectionOptions {
            config_path: None,
            endpoint: None,
            remote_url: None,
            auth_token: None,
            auth_token_env: None,
            profile: None,
            profiles_file: None,
            suppress_profile_resolution: false,
        };

        match options.to_spec().expect("spec") {
            ConnectionSpec::LocalConfig { config_path } => {
                assert_eq!(config_path, PathBuf::from("turin.toml"));
            }
            other => panic!("unexpected spec: {other:?}"),
        }
    }

    #[test]
    fn connection_options_require_remote_auth_material() {
        let options = ConnectionOptions {
            config_path: None,
            endpoint: None,
            remote_url: Some("http://example.test".to_string()),
            auth_token: None,
            auth_token_env: None,
            profile: None,
            profiles_file: None,
            suppress_profile_resolution: false,
        };

        let err = options.to_spec().expect_err("missing auth should error");
        assert!(
            err.to_string()
                .contains("--remote-url requires either --auth-token or --auth-token-env")
        );
    }

    #[test]
    fn connection_options_apply_profile_overrides() {
        let temp = tempfile::NamedTempFile::new().expect("temp profile file");
        fs::write(
            temp.path(),
            r#"
[profiles.lab]
remote_url = "http://example.test"
auth_token_env = "TURIN_REMOTE_TOKEN"
"#,
        )
        .expect("write profile file");

        let options = ConnectionOptions {
            config_path: None,
            endpoint: None,
            remote_url: None,
            auth_token: None,
            auth_token_env: None,
            profile: Some("lab".to_string()),
            profiles_file: Some(temp.path().to_path_buf()),
            suppress_profile_resolution: false,
        };

        match options.to_spec().expect("spec") {
            ConnectionSpec::RemoteEnv {
                base_url,
                auth_token_env,
            } => {
                assert_eq!(base_url, "http://example.test");
                assert_eq!(auth_token_env, "TURIN_REMOTE_TOKEN");
            }
            other => panic!("unexpected spec: {other:?}"),
        }
    }

    #[test]
    fn connection_options_can_save_and_delete_profiles() {
        let temp = tempfile::tempdir().expect("temp dir");
        let profiles_path = temp.path().join("ui-profiles.toml");

        let remote = ConnectionOptions {
            config_path: None,
            endpoint: None,
            remote_url: Some("http://example.test".to_string()),
            auth_token: None,
            auth_token_env: Some("TURIN_REMOTE_TOKEN".to_string()),
            profile: None,
            profiles_file: Some(profiles_path.clone()),
            suppress_profile_resolution: false,
        };

        let catalog = remote
            .save_profile("lab", true)
            .expect("save remote profile");
        assert_eq!(catalog.default_profile(), Some("lab"));
        assert_eq!(catalog.profiles().len(), 1);

        let local = ConnectionOptions {
            config_path: Some(PathBuf::from("turin-dev.toml")),
            endpoint: None,
            remote_url: None,
            auth_token: None,
            auth_token_env: None,
            profile: None,
            profiles_file: Some(profiles_path.clone()),
            suppress_profile_resolution: false,
        };

        let catalog = local
            .save_profile("local", false)
            .expect("save local profile");
        assert_eq!(catalog.default_profile(), Some("lab"));
        assert_eq!(catalog.profiles().len(), 2);

        let deleted = local.delete_profile("lab").expect("delete profile");
        assert_eq!(deleted.default_profile(), Some("local"));
        assert_eq!(deleted.profiles().len(), 1);
        assert_eq!(deleted.profiles()[0].name, "local");

        let raw = fs::read_to_string(&profiles_path).expect("read saved file");
        assert!(raw.contains("default_profile = \"local\""));
        assert!(raw.contains("[profiles.local]"));
    }

    #[test]
    fn connection_options_can_duplicate_and_rename_profiles() {
        let temp = tempfile::tempdir().expect("temp dir");
        let profiles_path = temp.path().join("ui-profiles.toml");
        fs::write(
            &profiles_path,
            r#"
default_profile = "lab"

[profiles.lab]
remote_url = "http://example.test"
auth_token_env = "TURIN_REMOTE_TOKEN"
"#,
        )
        .expect("write initial profiles");

        let options = ConnectionOptions {
            config_path: None,
            endpoint: None,
            remote_url: None,
            auth_token: None,
            auth_token_env: None,
            profile: None,
            profiles_file: Some(profiles_path.clone()),
            suppress_profile_resolution: false,
        };

        let duplicated = options
            .duplicate_profile("lab", "lab-copy", false)
            .expect("duplicate profile");
        assert_eq!(duplicated.profiles().len(), 2);
        assert_eq!(duplicated.default_profile(), Some("lab"));

        let renamed = options
            .rename_profile("lab-copy", "lab-stage", true)
            .expect("rename profile");
        assert_eq!(renamed.profiles().len(), 2);
        assert_eq!(renamed.default_profile(), Some("lab-stage"));
        assert!(
            renamed
                .profiles()
                .iter()
                .any(|profile| profile.name == "lab-stage")
        );
    }

    #[test]
    fn connection_profile_drafts_roundtrip_through_profile_storage() {
        let temp = tempfile::tempdir().expect("temp dir");
        let profiles_path = temp.path().join("ui-profiles.toml");
        let options = ConnectionOptions {
            config_path: None,
            endpoint: None,
            remote_url: None,
            auth_token: None,
            auth_token_env: None,
            profile: None,
            profiles_file: Some(profiles_path),
            suppress_profile_resolution: false,
        };

        let draft = ConnectionProfileDraft {
            kind: ConnectionProfileKind::Remote,
            target: "http://example.test:9324".to_string(),
            auth_mode: ConnectionProfileDraftAuthMode::TokenEnv,
            auth_value: "TURIN_REMOTE_TOKEN".to_string(),
        };

        options
            .save_profile_draft("lab", &draft, true)
            .expect("save draft");
        let loaded = options.load_profile_draft("lab").expect("load draft");

        assert_eq!(loaded, draft);
    }

    #[test]
    fn connection_options_can_materialize_and_resolve_remote_drafts() {
        let options = ConnectionOptions {
            config_path: Some(PathBuf::from("turin.toml")),
            endpoint: None,
            remote_url: None,
            auth_token: None,
            auth_token_env: None,
            profile: Some("ignored".to_string()),
            profiles_file: Some(PathBuf::from("ui-profiles.toml")),
            suppress_profile_resolution: false,
        };
        let draft = ConnectionProfileDraft {
            kind: ConnectionProfileKind::Remote,
            target: "https://turin.example.com".to_string(),
            auth_mode: ConnectionProfileDraftAuthMode::TokenEnv,
            auth_value: "TURIN_REMOTE_TOKEN".to_string(),
        };

        let materialized = options
            .connection_options_for_draft(&draft)
            .expect("materialize draft");
        assert_eq!(
            materialized.remote_url.as_deref(),
            Some("https://turin.example.com")
        );
        assert_eq!(
            materialized.auth_token_env.as_deref(),
            Some("TURIN_REMOTE_TOKEN")
        );
        assert!(materialized.profile.is_none());
        assert_eq!(
            materialized.profiles_file.as_deref(),
            Some(Path::new("ui-profiles.toml"))
        );

        match options.draft_to_spec(&draft).expect("draft spec") {
            ConnectionSpec::RemoteEnv {
                base_url,
                auth_token_env,
            } => {
                assert_eq!(base_url, "https://turin.example.com");
                assert_eq!(auth_token_env, "TURIN_REMOTE_TOKEN");
            }
            other => panic!("unexpected spec: {other:?}"),
        }
    }

    #[test]
    fn remote_profile_draft_validation_reports_target_and_auth_errors() {
        let validation = ConnectionProfileDraft {
            kind: ConnectionProfileKind::Remote,
            target: "ftp://bad host".to_string(),
            auth_mode: ConnectionProfileDraftAuthMode::TokenEnv,
            auth_value: "bad-token-name".to_string(),
        }
        .validate();

        assert!(!validation.is_valid());
        assert_eq!(
            validation.target_error.as_deref(),
            Some("Remote base URLs cannot contain whitespace")
        );
        assert_eq!(
            validation.auth_error.as_deref(),
            Some("Env var names may only contain letters, numbers, and underscores")
        );
    }

    #[test]
    fn remote_inline_token_draft_validation_reports_plaintext_notice() {
        let validation = ConnectionProfileDraft {
            kind: ConnectionProfileKind::Remote,
            target: "https://turin.example.com".to_string(),
            auth_mode: ConnectionProfileDraftAuthMode::InlineToken,
            auth_value: "secret".to_string(),
        }
        .validate();

        assert!(validation.is_valid());
        assert_eq!(
            validation.auth_notice.as_deref(),
            Some("Inline tokens are stored in plaintext in the profiles file")
        );
    }

    #[test]
    fn local_config_draft_validation_reports_default_path_notice() {
        let validation = ConnectionProfileDraft {
            kind: ConnectionProfileKind::LocalConfig,
            target: String::new(),
            auth_mode: ConnectionProfileDraftAuthMode::None,
            auth_value: String::new(),
        }
        .validate();

        assert!(validation.is_valid());
        assert_eq!(
            validation.target_notice.as_deref(),
            Some("Blank config path will default to turin.toml")
        );
    }

    #[test]
    fn recent_connection_drafts_are_deduped_and_bounded() {
        let mut history = ConnectionDraftHistory::default();
        let remote = ConnectionProfileDraft {
            kind: ConnectionProfileKind::Remote,
            target: "https://turin.example.com".to_string(),
            auth_mode: ConnectionProfileDraftAuthMode::TokenEnv,
            auth_value: "TURIN_REMOTE_TOKEN".to_string(),
        };

        history.record_success(&ConnectionProfileDraft {
            kind: ConnectionProfileKind::LocalConfig,
            target: String::new(),
            auth_mode: ConnectionProfileDraftAuthMode::None,
            auth_value: String::new(),
        });
        history.record_success(&ConnectionProfileDraft {
            kind: ConnectionProfileKind::LocalEndpoint,
            target: ".turin/daemon.sock".to_string(),
            auth_mode: ConnectionProfileDraftAuthMode::None,
            auth_value: String::new(),
        });
        history.record_success(&remote);
        history.record_success(&ConnectionProfileDraft {
            kind: ConnectionProfileKind::Remote,
            target: "https://turin-backup.example.com".to_string(),
            auth_mode: ConnectionProfileDraftAuthMode::InlineToken,
            auth_value: "secret".to_string(),
        });
        history.record_success(&ConnectionProfileDraft {
            kind: ConnectionProfileKind::LocalConfig,
            target: "turin-stage.toml".to_string(),
            auth_mode: ConnectionProfileDraftAuthMode::None,
            auth_value: String::new(),
        });
        history.record_success(&remote);

        assert_eq!(history.drafts().len(), MAX_RECENT_CONNECTION_DRAFTS);
        assert_eq!(history.latest(), Some(&remote));
        assert_eq!(
            history
                .drafts()
                .iter()
                .filter(|draft| draft.target == "https://turin.example.com")
                .count(),
            1
        );
    }

    #[test]
    fn profile_draft_diff_redacts_inline_tokens() {
        let baseline = ConnectionProfileDraft {
            kind: ConnectionProfileKind::Remote,
            target: "https://turin.example.com".to_string(),
            auth_mode: ConnectionProfileDraftAuthMode::InlineToken,
            auth_value: "secret-a".to_string(),
        };
        let changed = ConnectionProfileDraft {
            auth_value: "secret-b".to_string(),
            ..baseline.clone()
        };

        let diff = changed.diff_against(&baseline);
        assert_eq!(diff.changed_field_names(), vec!["auth_value"]);
        assert_eq!(diff.changed_fields[0].draft_value, "<inline token set>");
        assert_eq!(
            diff.changed_fields[0].comparison_value,
            "<inline token set>"
        );
    }

    #[test]
    fn remote_preflight_reports_missing_env_before_connect() {
        let options = ConnectionOptions {
            config_path: None,
            endpoint: None,
            remote_url: Some("https://turin.example.com".to_string()),
            auth_token: None,
            auth_token_env: Some("TURIN_UI_CORE_MISSING_TOKEN".to_string()),
            profile: None,
            profiles_file: None,
            suppress_profile_resolution: false,
        };

        let report = preflight_connection_blocking(&options);
        assert_eq!(report.outcome, ConnectionPreflightOutcome::Invalid);
        assert!(
            report
                .message
                .contains("Environment variable 'TURIN_UI_CORE_MISSING_TOKEN' is not set")
        );
    }

    #[test]
    fn connection_profile_activity_book_tracks_success_and_failure() {
        let mut book = ConnectionProfileActivityBook::default();
        book.record_preflight_result("lab", true, "ready");
        book.record_connect_result("lab", false, "failed");
        book.record_connect_result("lab", true, "connected");

        let entry = book.entry("lab").expect("activity entry");
        assert_eq!(entry.preflight_count, 1);
        assert_eq!(entry.connect_count, 2);
        assert_eq!(entry.successful_connect_count, 1);
        assert_eq!(entry.failure_count, 1);
        assert_eq!(entry.last_connect_result.as_deref(), Some("connected"));
        assert_eq!(entry.last_preflight_result.as_deref(), Some("ready"));
    }
}
