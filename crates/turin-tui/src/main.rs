use anyhow::{Context, Result, anyhow};
use clap::Parser;
use crossterm::event::{self, Event as CEvent, KeyCode};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};
use ratatui::{DefaultTerminal, Frame};
use std::io::stdout;
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time;
use turin_control_client::{ConnectionSpec, ControlClient};
use turin_daemon_protocol::{EventEnvelope, RuntimeEventsSubscribeParams};
use turin_ui_core::{DashboardSnapshot, DashboardState};

#[derive(Parser, Debug)]
#[command(name = "turin-tui", version, about)]
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

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let spec = connection_spec_from_args(&args)?;
    let client = ControlClient::connect(&spec).await?;
    let mut dashboard = DashboardState::load(&client).await?;

    let (tx, mut rx) = mpsc::unbounded_channel::<UiUpdate>();
    spawn_event_task(client.clone(), tx.clone());
    spawn_refresh_task(client.clone(), tx);

    enable_raw_mode()?;
    execute!(stdout(), EnterAlternateScreen)?;
    let mut terminal = ratatui::init();

    let loop_result = run_app(&mut terminal, &mut dashboard, &mut rx);

    ratatui::restore();
    disable_raw_mode()?;
    execute!(stdout(), LeaveAlternateScreen)?;

    loop_result
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

fn spawn_event_task(client: ControlClient, tx: mpsc::UnboundedSender<UiUpdate>) {
    tokio::spawn(async move {
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

fn spawn_refresh_task(client: ControlClient, tx: mpsc::UnboundedSender<UiUpdate>) {
    tokio::spawn(async move {
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

fn run_app(
    terminal: &mut DefaultTerminal,
    dashboard: &mut DashboardState,
    rx: &mut mpsc::UnboundedReceiver<UiUpdate>,
) -> Result<()> {
    loop {
        while let Ok(update) = rx.try_recv() {
            match update {
                UiUpdate::Snapshot(snapshot) => dashboard.apply_snapshot(snapshot),
                UiUpdate::Event(event) => dashboard.record_event(event),
                UiUpdate::Error(message) => dashboard.record_error(message),
            }
        }

        terminal.draw(|frame| render(frame, dashboard))?;

        if event::poll(Duration::from_millis(120)).context("Failed to poll terminal events")?
            && let CEvent::Key(key) = event::read().context("Failed to read terminal event")?
        {
            match key.code {
                KeyCode::Char('q') => return Ok(()),
                KeyCode::Esc => return Ok(()),
                _ => {}
            }
        }
    }
}

fn render(frame: &mut Frame<'_>, dashboard: &DashboardState) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Length(7),
            Constraint::Min(10),
        ])
        .split(frame.area());

    let health = dashboard.health.as_ref();
    let ready = health.is_some_and(|health| health.ready);
    let status_color = if ready {
        Color::LightGreen
    } else {
        Color::LightYellow
    };

    let banner = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(
                "Turin TUI",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                if ready { "CONNECTED" } else { "DEGRADED" },
                Style::default()
                    .fg(status_color)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Target: ", Style::default().fg(Color::Gray)),
            Span::raw(dashboard.connection_target.clone()),
        ]),
        Line::from(vec![
            Span::styled("Quit: ", Style::default().fg(Color::Gray)),
            Span::raw("q / Esc"),
        ]),
    ])
    .block(
        Block::default()
            .title("Connection")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    )
    .wrap(Wrap { trim: true });
    frame.render_widget(banner, layout[0]);

    let metrics = if let Some(health) = health {
        vec![
            Line::from(vec![
                metric_span("Agents", health.agent_count),
                Span::raw("  "),
                metric_span("Harnesses", health.harness_count),
                Span::raw("  "),
                metric_span("Channels", health.channel_count),
            ]),
            Line::from(vec![
                metric_span("Running", health.running_agent_count),
                Span::raw("  "),
                metric_span("Active", health.active_task_count),
                Span::raw("  "),
                metric_span("Queued", health.queued_task_count),
                Span::raw("  "),
                metric_span("Awaiting", health.awaiting_result_count),
            ]),
            Line::from(vec![
                metric_span("Issues", health.issue_count),
                Span::raw("  "),
                metric_span("Failed Channels", health.failed_channel_count),
                Span::raw("  "),
                Span::styled(
                    format!("v{} / protocol {}", health.version, health.protocol_version),
                    Style::default().fg(Color::Gray),
                ),
            ]),
        ]
    } else {
        vec![Line::from("Loading health...")]
    };
    let metrics = Paragraph::new(metrics).block(
        Block::default()
            .title("Runtime")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    frame.render_widget(metrics, layout[1]);

    let main = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(48), Constraint::Percentage(52)])
        .split(layout[2]);

    let events: Vec<ListItem<'_>> = dashboard
        .recent_events
        .iter()
        .rev()
        .take(20)
        .map(|event| {
            let preview = serde_json::to_string(&event.data).unwrap_or_else(|_| "{}".to_string());
            ListItem::new(vec![
                Line::from(Span::styled(
                    event.event.clone(),
                    Style::default()
                        .fg(Color::LightBlue)
                        .add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::styled(
                    preview.chars().take(120).collect::<String>(),
                    Style::default().fg(Color::Gray),
                )),
            ])
        })
        .collect();
    let events = List::new(events).block(
        Block::default()
            .title("Recent Events")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    frame.render_widget(events, main[0]);

    let mut status_text = dashboard.status_pretty_json();
    if let Some(error) = &dashboard.last_error {
        status_text = format!("Last error:\n{}\n\n{}", error, status_text);
    }
    let status = Paragraph::new(status_text)
        .block(
            Block::default()
                .title("Daemon Status JSON")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(status, main[1]);
}

fn metric_span(label: &str, value: usize) -> Span<'static> {
    Span::styled(
        format!("{}: {}", label, value),
        Style::default().fg(Color::White),
    )
}
