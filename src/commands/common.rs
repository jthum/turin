use turin::display;
use turin::kernel::session::SessionState;

pub(crate) fn print_session_summary(session: &SessionState) {
    let ansi = display::stdout_ansi();
    println!("\n{}", display::header("Session Summary", ansi));
    println!(
        "  {}  {} ({} in, {} out)",
        display::bold("Total Tokens:", ansi),
        session.total_input_tokens + session.total_output_tokens,
        session.total_input_tokens,
        session.total_output_tokens
    );
    println!(
        "  {}         {}",
        display::bold("Turns:", ansi),
        session.turn_index
    );
}
