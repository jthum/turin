use ratatui::style::{Color, Modifier, Style};

pub const BG: Color = Color::Rgb(10, 13, 15);
pub const PANEL_HOT: Color = Color::Rgb(26, 34, 38);
pub const TEXT: Color = Color::Rgb(219, 224, 223);
pub const MUTED: Color = Color::Rgb(117, 130, 132);
pub const CYAN: Color = Color::Rgb(56, 189, 208);
pub const AMBER: Color = Color::Rgb(221, 164, 72);
pub const GREEN: Color = Color::Rgb(94, 190, 119);
pub const RED: Color = Color::Rgb(217, 90, 83);

pub fn base() -> Style {
    Style::default().fg(TEXT).bg(BG)
}

pub fn muted() -> Style {
    Style::default().fg(MUTED).bg(BG)
}

pub fn title() -> Style {
    Style::default()
        .fg(TEXT)
        .bg(BG)
        .add_modifier(Modifier::BOLD)
}

pub fn accent() -> Style {
    Style::default()
        .fg(CYAN)
        .bg(BG)
        .add_modifier(Modifier::BOLD)
}

pub fn warning() -> Style {
    Style::default()
        .fg(AMBER)
        .bg(BG)
        .add_modifier(Modifier::BOLD)
}

pub fn danger() -> Style {
    Style::default().fg(RED).bg(BG).add_modifier(Modifier::BOLD)
}

pub fn success() -> Style {
    Style::default()
        .fg(GREEN)
        .bg(BG)
        .add_modifier(Modifier::BOLD)
}

pub fn selected() -> Style {
    Style::default()
        .fg(TEXT)
        .bg(PANEL_HOT)
        .add_modifier(Modifier::BOLD)
}
