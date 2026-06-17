use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::event;
use ratatui::{DefaultTerminal, Frame};

use crate::app::{TuiApp, TuiSignal};

pub async fn run(mut app: TuiApp) -> Result<()> {
    let mut terminal = ratatui::init();
    let result = run_loop(&mut terminal, &mut app).await;
    ratatui::restore();
    app.shutdown();
    result
}

async fn run_loop(terminal: &mut DefaultTerminal, app: &mut TuiApp) -> Result<()> {
    while !app.should_quit() {
        app.drain_updates();
        app.ensure_visible_data()?;
        terminal.draw(|frame| render(frame, app))?;

        if event::poll(Duration::from_millis(50)).context("failed to poll terminal events")? {
            let event = event::read().context("failed to read terminal event")?;
            if matches!(app.handle_terminal_event(event)?, TuiSignal::Quit) {
                break;
            }
        }
    }
    Ok(())
}

fn render(frame: &mut Frame<'_>, app: &mut TuiApp) {
    app.render(frame);
}
