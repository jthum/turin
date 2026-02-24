use std::io::{self, IsTerminal};

pub fn stdout_ansi() -> bool {
    io::stdout().is_terminal()
}

pub fn stderr_ansi() -> bool {
    io::stderr().is_terminal()
}

pub fn paint(text: &str, codes: &str, ansi: bool) -> String {
    if ansi {
        format!("\x1b[{codes}m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

pub fn bold(text: &str, ansi: bool) -> String {
    paint(text, "1", ansi)
}

pub fn header(label: &str, ansi: bool) -> String {
    if ansi {
        format!("\x1b[36m\x1b[1m── {label} ──\x1b[0m")
    } else {
        format!("── {label} ──")
    }
}

pub fn turn_header(turn_index_1based: u32, ansi: bool) -> String {
    header(&format!("Turn {}", turn_index_1based), ansi)
}

pub fn repl_prompt(ansi: bool) -> String {
    if ansi {
        "\x1b[36m\x1b[1mturin\x1b[0m\x1b[34m>\x1b[0m ".to_string()
    } else {
        "turin> ".to_string()
    }
}

pub fn approval_prompt_prefix(ansi: bool) -> String {
    if ansi {
        "\x1b[33m\x1b[1m! Approval Required:\x1b[0m".to_string()
    } else {
        "! Approval Required:".to_string()
    }
}

pub fn thinking_label(ansi: bool) -> String {
    paint("💭 Thinking...", "35", ansi)
}

pub fn tool_call_line(name: &str, args: &serde_json::Value, ansi: bool) -> String {
    if ansi {
        format!(
            "\n{} {}({})",
            paint("⚒️  Tool Call:", "33", true),
            bold(name, true),
            args
        )
    } else {
        format!("\n⚒️  Tool Call: {}({})", name, args)
    }
}

pub fn tool_status_line(name: &str, success: bool, ansi: bool) -> String {
    if success {
        if ansi {
            format!("{} Tool '{}' complete", paint("✓", "32;1", true), name)
        } else {
            format!("✓ Tool '{}' complete", name)
        }
    } else if ansi {
        format!("{} Tool '{}' failed", paint("✗", "31", true), name)
    } else {
        format!("✗ Tool '{}' failed", name)
    }
}

pub fn rejection_line(prefix: &str, reason: &str, ansi: bool) -> String {
    if ansi {
        format!("{} {}", paint(prefix, "31", true), reason)
    } else {
        format!("{prefix} {reason}")
    }
}

pub fn approval_line(approved: bool, ansi: bool) -> String {
    match (approved, ansi) {
        (true, true) => paint("✓ Approved by user", "32", true),
        (true, false) => "✓ Approved by user".to_string(),
        (false, true) => paint("✗ Denied by user", "31", true),
        (false, false) => "✗ Denied by user".to_string(),
    }
}
