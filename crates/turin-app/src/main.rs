use anyhow::{Result, anyhow};
use clap::Parser;
use eframe::egui::{self, Color32, RichText, ScrollArea, Vec2};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;
use tokio::sync::mpsc;
use tokio::time;
use turin_control_client::{ConnectionSpec, ControlClient};
use turin_daemon_protocol::{EventEnvelope, RuntimeEventsSubscribeParams};
use turin_ui_core::{DashboardSnapshot, DashboardState};

#[derive(Parser, Debug)]
#[command(name = "turin-app", version, about)]
struct Args {
    #[arg(long, default_value = "turin.toml")]
    config: PathBuf,
    #[arg(long)]
    endpoint: Option<PathBuf>,
    #[arg(long)]
    remote_url: Option<String>,
    #[arg(long)]
    auth_token: Option<String>,
    #[arg(long)]
    auth_token_env: Option<String>,
}

enum UiUpdate {
    Snapshot(DashboardSnapshot),
    Event(EventEnvelope),
    Error(String),
}

fn main() -> Result<()> {
    let args = Args::parse();
    let spec = connection_spec_from_args(&args)?;
    let runtime = Arc::new(Runtime::new()?);
    let client = runtime.block_on(ControlClient::connect(&spec))?;
    let dashboard = runtime.block_on(DashboardState::load(&client))?;
    let (tx, rx) = mpsc::unbounded_channel::<UiUpdate>();

    spawn_event_task(runtime.clone(), client.clone(), tx.clone());
    spawn_refresh_task(runtime.clone(), client, tx);

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(Vec2::new(1180.0, 760.0))
            .with_min_inner_size(Vec2::new(980.0, 640.0)),
        ..Default::default()
    };

    eframe::run_native(
        "Turin App",
        native_options,
        Box::new(move |cc| {
            configure_visuals(&cc.egui_ctx);
            Ok(Box::new(TurinDesktopApp {
                dashboard,
                rx,
                _runtime: runtime,
            }))
        }),
    )
    .map_err(|err| anyhow!(err.to_string()))
}

fn connection_spec_from_args(args: &Args) -> Result<ConnectionSpec> {
    if let Some(base_url) = &args.remote_url {
        if let Some(auth_token) = &args.auth_token {
            return Ok(ConnectionSpec::Remote {
                base_url: base_url.clone(),
                auth_token: auth_token.clone(),
            });
        }
        if let Some(auth_token_env) = &args.auth_token_env {
            return Ok(ConnectionSpec::RemoteEnv {
                base_url: base_url.clone(),
                auth_token_env: auth_token_env.clone(),
            });
        }
        return Err(anyhow!(
            "--remote-url requires either --auth-token or --auth-token-env"
        ));
    }

    if let Some(endpoint) = &args.endpoint {
        return Ok(ConnectionSpec::LocalEndpoint {
            endpoint: endpoint.clone(),
        });
    }

    Ok(ConnectionSpec::LocalConfig {
        config_path: args.config.clone(),
    })
}

fn spawn_event_task(
    runtime: Arc<Runtime>,
    client: ControlClient,
    tx: mpsc::UnboundedSender<UiUpdate>,
) {
    runtime.spawn(async move {
        match client
            .subscribe_managed(RuntimeEventsSubscribeParams::default())
            .await
        {
            Ok(mut stream) => loop {
                match stream.next_event().await {
                    Ok(event) => {
                        let _ = tx.send(UiUpdate::Event(event));
                    }
                    Err(err) => {
                        let _ = tx.send(UiUpdate::Error(err.to_string()));
                        break;
                    }
                }
            },
            Err(err) => {
                let _ = tx.send(UiUpdate::Error(err.to_string()));
            }
        }
    });
}

fn spawn_refresh_task(
    runtime: Arc<Runtime>,
    client: ControlClient,
    tx: mpsc::UnboundedSender<UiUpdate>,
) {
    runtime.spawn(async move {
        let mut interval = time::interval(Duration::from_secs(5));
        loop {
            interval.tick().await;
            match DashboardState::snapshot(&client).await {
                Ok(snapshot) => {
                    let _ = tx.send(UiUpdate::Snapshot(snapshot));
                }
                Err(err) => {
                    let _ = tx.send(UiUpdate::Error(err.to_string()));
                }
            }
        }
    });
}

fn configure_visuals(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.override_text_color = Some(Color32::from_rgb(240, 234, 224));
    visuals.widgets.active.bg_fill = Color32::from_rgb(31, 78, 96);
    visuals.widgets.inactive.bg_fill = Color32::from_rgb(23, 33, 40);
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(46, 88, 101);
    visuals.panel_fill = Color32::from_rgb(13, 18, 22);
    visuals.window_fill = Color32::from_rgb(17, 24, 29);
    ctx.set_visuals(visuals);
}

struct TurinDesktopApp {
    dashboard: DashboardState,
    rx: mpsc::UnboundedReceiver<UiUpdate>,
    _runtime: Arc<Runtime>,
}

impl eframe::App for TurinDesktopApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(update) = self.rx.try_recv() {
            match update {
                UiUpdate::Snapshot(snapshot) => self.dashboard.apply_snapshot(snapshot),
                UiUpdate::Event(event) => self.dashboard.record_event(event),
                UiUpdate::Error(message) => self.dashboard.record_error(message),
            }
        }

        ctx.request_repaint_after(Duration::from_millis(250));

        let ready = self
            .dashboard
            .health
            .as_ref()
            .is_some_and(|health| health.ready);
        let accent = if ready {
            Color32::from_rgb(111, 214, 161)
        } else {
            Color32::from_rgb(255, 196, 107)
        };

        egui::TopBottomPanel::top("top_banner").show(ctx, |ui| {
            ui.add_space(8.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    RichText::new("Turin App")
                        .size(26.0)
                        .color(Color32::from_rgb(142, 214, 255))
                        .strong(),
                );
                ui.add_space(12.0);
                ui.label(
                    RichText::new(if ready { "CONNECTED" } else { "DEGRADED" })
                        .color(accent)
                        .strong(),
                );
                ui.add_space(16.0);
                ui.label(
                    RichText::new(self.dashboard.connection_target.clone())
                        .color(Color32::from_rgb(201, 195, 187)),
                );
            });
            ui.add_space(6.0);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let health = self.dashboard.health.as_ref();

            ui.columns(2, |columns| {
                columns[0].group(|ui| {
                    ui.heading("Runtime");
                    ui.add_space(8.0);
                    if let Some(health) = health {
                        metric_row(
                            ui,
                            "Agents",
                            health.agent_count,
                            "Harnesses",
                            health.harness_count,
                            "Channels",
                            health.channel_count,
                        );
                        metric_row(
                            ui,
                            "Running",
                            health.running_agent_count,
                            "Active",
                            health.active_task_count,
                            "Queued",
                            health.queued_task_count,
                        );
                        metric_row(
                            ui,
                            "Awaiting",
                            health.awaiting_result_count,
                            "Issues",
                            health.issue_count,
                            "Failed Channels",
                            health.failed_channel_count,
                        );
                        ui.add_space(8.0);
                        ui.label(
                            RichText::new(format!(
                                "Version {} / protocol {} / {}",
                                health.version, health.protocol_version, health.transport
                            ))
                            .color(Color32::from_rgb(173, 167, 159)),
                        );
                    } else {
                        ui.label("Loading runtime health...");
                    }
                });

                columns[1].group(|ui| {
                    ui.heading("Recent Events");
                    ui.add_space(8.0);
                    ScrollArea::vertical().max_height(240.0).show(ui, |ui| {
                        for event in self.dashboard.recent_events.iter().rev().take(18) {
                            ui.label(
                                RichText::new(event.event.clone())
                                    .color(Color32::from_rgb(121, 188, 233))
                                    .strong(),
                            );
                            ui.label(
                                RichText::new(
                                    serde_json::to_string(&event.data)
                                        .unwrap_or_else(|_| "{}".to_string()),
                                )
                                .color(Color32::from_rgb(171, 165, 155))
                                .small(),
                            );
                            ui.add_space(6.0);
                        }
                    });
                });
            });

            ui.add_space(12.0);

            ui.group(|ui| {
                ui.heading("Daemon Status JSON");
                ui.add_space(8.0);
                if let Some(error) = &self.dashboard.last_error {
                    ui.label(
                        RichText::new(error)
                            .color(Color32::from_rgb(255, 171, 145))
                            .strong(),
                    );
                    ui.add_space(6.0);
                }
                ScrollArea::vertical().show(ui, |ui| {
                    ui.code(self.dashboard.status_pretty_json());
                });
            });
        });
    }
}

fn metric_row(
    ui: &mut egui::Ui,
    left_label: &str,
    left_value: usize,
    middle_label: &str,
    middle_value: usize,
    right_label: &str,
    right_value: usize,
) {
    ui.horizontal(|ui| {
        metric_chip(ui, left_label, left_value);
        metric_chip(ui, middle_label, middle_value);
        metric_chip(ui, right_label, right_value);
    });
}

fn metric_chip(ui: &mut egui::Ui, label: &str, value: usize) {
    ui.label(
        RichText::new(format!("{}: {}", label, value)).color(Color32::from_rgb(240, 234, 224)),
    );
    ui.add_space(12.0);
}
