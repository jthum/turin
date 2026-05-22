use pulldown_cmark::{CodeBlockKind, Event, Options, Parser, Tag, TagEnd};
use turin_channel_core::{MessageBlock, OutboundMessage};

use super::TELEGRAM_MESSAGE_MAX_LEN;

pub(super) fn render_html_chunks(
    message: &OutboundMessage,
    final_thinking: Option<&str>,
) -> Vec<String> {
    let mut segments = Vec::new();
    if let Some(thinking) = final_thinking {
        segments.push("<i>Thinking</i>".to_string());
        segments.extend(split_wrapped_segment(thinking, "<pre>", "</pre>"));
        segments.push("<i>Reply</i>".to_string());
    }
    for block in &message.blocks {
        segments.extend(render_html_segments_for_block(block));
    }

    pack_segments(segments)
}

fn render_html_segments_for_block(block: &MessageBlock) -> Vec<String> {
    match block {
        MessageBlock::Text { text } => render_markdown_segments(text),
        MessageBlock::CodeBlock { code, .. } => split_wrapped_segment(code, "<pre>", "</pre>"),
    }
}

#[derive(Debug, Clone, Copy)]
struct MarkdownListState {
    ordered: bool,
    next_index: u64,
}

#[derive(Debug, Default, Clone)]
struct MarkdownTableState {
    rows: Vec<Vec<String>>,
    current_row: Vec<String>,
    current_cell: String,
    header_rows: usize,
}

fn render_markdown_segments(markdown: &str) -> Vec<String> {
    let trimmed = markdown.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    let mut options = Options::empty();
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_TASKLISTS);

    let parser = Parser::new_ext(trimmed, options);
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut blockquote_depth = 0usize;
    let mut list_stack: Vec<MarkdownListState> = Vec::new();
    let mut code_block: Option<String> = None;
    let mut table_state: Option<MarkdownTableState> = None;

    for event in parser {
        match event {
            Event::Start(tag) => match tag {
                Tag::Paragraph => {}
                Tag::Heading { .. } => {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str("<b>");
                }
                Tag::BlockQuote(_) => {
                    blockquote_depth = blockquote_depth.saturating_add(1);
                }
                Tag::List(start) => {
                    list_stack.push(MarkdownListState {
                        ordered: start.is_some(),
                        next_index: start.unwrap_or(1),
                    });
                }
                Tag::Item => {
                    flush_rich_segment(&mut segments, &mut current);
                    current.push_str(&blockquote_prefix(blockquote_depth));
                    if let Some(state) = list_stack.last_mut() {
                        if state.ordered {
                            current.push_str(&format!("{}. ", state.next_index));
                            state.next_index = state.next_index.saturating_add(1);
                        } else {
                            current.push_str("• ");
                        }
                    }
                }
                Tag::Emphasis => {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str("<i>");
                }
                Tag::Strong => {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str("<b>");
                }
                Tag::Strikethrough => {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str("<s>");
                }
                Tag::Link { dest_url, .. } => {
                    if table_state.is_some() {
                        continue;
                    }
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str("<a href=\"");
                    current.push_str(&escape_html(dest_url.as_ref()));
                    current.push_str("\">");
                }
                Tag::Table(_) => {
                    flush_rich_segment(&mut segments, &mut current);
                    table_state = Some(MarkdownTableState::default());
                }
                Tag::TableHead => {}
                Tag::TableRow => {
                    if let Some(table) = table_state.as_mut() {
                        table.current_row.clear();
                    }
                }
                Tag::TableCell => {
                    if let Some(table) = table_state.as_mut() {
                        table.current_cell.clear();
                    }
                }
                Tag::CodeBlock(kind) => {
                    flush_rich_segment(&mut segments, &mut current);
                    let mut rendered = String::new();
                    if let CodeBlockKind::Fenced(language) = kind {
                        let language = language.trim();
                        if !language.is_empty() {
                            rendered.push_str(language);
                            rendered.push('\n');
                        }
                    }
                    code_block = Some(rendered);
                }
                _ => {}
            },
            Event::End(tag) => match tag {
                TagEnd::Paragraph if list_stack.is_empty() => {
                    flush_rich_segment(&mut segments, &mut current);
                }
                TagEnd::Heading(_) => {
                    current.push_str("</b>");
                    flush_rich_segment(&mut segments, &mut current);
                }
                TagEnd::BlockQuote(_) => {
                    flush_rich_segment(&mut segments, &mut current);
                    blockquote_depth = blockquote_depth.saturating_sub(1);
                }
                TagEnd::List(_) => {
                    flush_rich_segment(&mut segments, &mut current);
                    list_stack.pop();
                }
                TagEnd::Item => {
                    flush_rich_segment(&mut segments, &mut current);
                }
                TagEnd::Emphasis => current.push_str("</i>"),
                TagEnd::Strong => current.push_str("</b>"),
                TagEnd::Strikethrough => current.push_str("</s>"),
                TagEnd::Table => {
                    if let Some(table) = table_state.take() {
                        let rendered = render_markdown_table(&table);
                        if !rendered.trim().is_empty() {
                            segments.extend(split_wrapped_segment(&rendered, "<pre>", "</pre>"));
                        }
                    }
                }
                TagEnd::TableHead => {
                    if let Some(table) = table_state.as_mut() {
                        if !table.current_row.is_empty() {
                            table.rows.push(std::mem::take(&mut table.current_row));
                        }
                        table.header_rows = table.rows.len();
                    }
                }
                TagEnd::TableRow => {
                    if let Some(table) = table_state.as_mut()
                        && !table.current_row.is_empty()
                    {
                        table.rows.push(std::mem::take(&mut table.current_row));
                    }
                }
                TagEnd::TableCell => {
                    if let Some(table) = table_state.as_mut() {
                        table
                            .current_row
                            .push(normalize_table_cell(&table.current_cell));
                        table.current_cell.clear();
                    }
                }
                TagEnd::Link if table_state.is_none() => {
                    current.push_str("</a>");
                }
                TagEnd::CodeBlock => {
                    if let Some(rendered) = code_block.take() {
                        segments.extend(split_wrapped_segment(&rendered, "<pre>", "</pre>"));
                    }
                }
                _ => {}
            },
            Event::Text(text) => {
                if let Some(code) = code_block.as_mut() {
                    code.push_str(text.as_ref());
                } else if let Some(table) = table_state.as_mut() {
                    table.current_cell.push_str(text.as_ref());
                } else {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str(&escape_html(text.as_ref()));
                }
            }
            Event::Code(text) => {
                if let Some(table) = table_state.as_mut() {
                    table.current_cell.push_str(text.as_ref());
                } else {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str("<code>");
                    current.push_str(&escape_html(text.as_ref()));
                    current.push_str("</code>");
                }
            }
            Event::SoftBreak | Event::HardBreak => {
                if let Some(code) = code_block.as_mut() {
                    code.push('\n');
                } else if let Some(table) = table_state.as_mut() {
                    if !table.current_cell.ends_with(' ') && !table.current_cell.is_empty() {
                        table.current_cell.push(' ');
                    }
                } else {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push('\n');
                }
            }
            Event::Rule => {
                flush_rich_segment(&mut segments, &mut current);
                segments.push("────────".to_string());
            }
            Event::TaskListMarker(checked) => {
                if let Some(table) = table_state.as_mut() {
                    table
                        .current_cell
                        .push_str(if checked { "[x] " } else { "[ ] " });
                } else {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str(if checked { "[x] " } else { "[ ] " });
                }
            }
            Event::Html(html) | Event::InlineHtml(html) => {
                if let Some(code) = code_block.as_mut() {
                    code.push_str(html.as_ref());
                } else if let Some(table) = table_state.as_mut() {
                    table.current_cell.push_str(html.as_ref());
                } else {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str(&escape_html(html.as_ref()));
                }
            }
            Event::InlineMath(text) | Event::DisplayMath(text) => {
                if let Some(table) = table_state.as_mut() {
                    table.current_cell.push_str(text.as_ref());
                } else {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str(&escape_html(text.as_ref()));
                }
            }
            Event::FootnoteReference(reference) => {
                if let Some(table) = table_state.as_mut() {
                    table.current_cell.push('[');
                    table.current_cell.push_str(reference.as_ref());
                    table.current_cell.push(']');
                } else {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push('[');
                    current.push_str(&escape_html(reference.as_ref()));
                    current.push(']');
                }
            }
        }
    }

    flush_rich_segment(&mut segments, &mut current);
    pack_segments(segments)
}

fn ensure_prefix(current: &mut String, blockquote_depth: usize) {
    if current.is_empty() {
        current.push_str(&blockquote_prefix(blockquote_depth));
    }
}

fn blockquote_prefix(depth: usize) -> String {
    "&gt; ".repeat(depth)
}

fn flush_rich_segment(segments: &mut Vec<String>, current: &mut String) {
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        segments.extend(split_rich_segment(trimmed));
    }
    current.clear();
}

fn normalize_table_cell(cell: &str) -> String {
    cell.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn render_markdown_table(table: &MarkdownTableState) -> String {
    if table.rows.is_empty() {
        return String::new();
    }

    let column_count = table.rows.iter().map(Vec::len).max().unwrap_or(0);
    if column_count == 0 {
        return String::new();
    }

    let mut widths = vec![0usize; column_count];
    for row in &table.rows {
        for (index, cell) in row.iter().enumerate() {
            widths[index] = widths[index].max(cell.chars().count());
        }
    }

    let format_row = |row: &[String]| {
        let mut out = String::from("|");
        for (index, width) in widths.iter().enumerate() {
            let cell = row.get(index).map(String::as_str).unwrap_or("");
            out.push(' ');
            out.push_str(cell);
            let padding = width.saturating_sub(cell.chars().count());
            if padding > 0 {
                out.push_str(&" ".repeat(padding));
            }
            out.push(' ');
            out.push('|');
        }
        out
    };

    let separator = {
        let mut out = String::from("|");
        for width in &widths {
            out.push(' ');
            out.push_str(&"-".repeat((*width).max(3)));
            out.push(' ');
            out.push('|');
        }
        out
    };

    let mut lines = Vec::new();
    for (index, row) in table.rows.iter().enumerate() {
        lines.push(format_row(row));
        if table.header_rows > 0 && index + 1 == table.header_rows {
            lines.push(separator.clone());
        }
    }

    lines.join("\n")
}

fn split_rich_segment(content: &str) -> Vec<String> {
    if content.chars().count() <= TELEGRAM_MESSAGE_MAX_LEN {
        return vec![content.to_string()];
    }

    let mut out = Vec::new();
    let mut current = String::new();
    for line in content.lines() {
        let tentative = if current.is_empty() {
            line.to_string()
        } else {
            format!("{current}\n{line}")
        };
        if tentative.chars().count() > TELEGRAM_MESSAGE_MAX_LEN {
            if !current.is_empty() {
                out.push(current.clone());
                current.clear();
            }
            if line.chars().count() > TELEGRAM_MESSAGE_MAX_LEN {
                out.extend(split_plain_segment(line));
            } else {
                current = line.to_string();
            }
        } else {
            current = tentative;
        }
    }

    if !current.is_empty() {
        out.push(current);
    }

    out
}

fn split_plain_segment(content: &str) -> Vec<String> {
    split_content_to_limit(content, TELEGRAM_MESSAGE_MAX_LEN)
}

fn split_wrapped_segment(content: &str, prefix: &str, suffix: &str) -> Vec<String> {
    let limit = TELEGRAM_MESSAGE_MAX_LEN
        .saturating_sub(prefix.chars().count())
        .saturating_sub(suffix.chars().count())
        .max(1);
    split_content_to_limit(&escape_html(content), limit)
        .into_iter()
        .map(|chunk| format!("{prefix}{chunk}{suffix}"))
        .collect()
}

fn split_content_to_limit(content: &str, limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return out;
    }

    let mut current = String::new();
    for ch in trimmed.chars() {
        current.push(ch);
        if current.chars().count() >= limit {
            out.push(current.clone());
            current.clear();
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn pack_segments(segments: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();

    for segment in segments {
        let segment = segment.trim().to_string();
        if segment.is_empty() {
            continue;
        }

        let tentative = if current.is_empty() {
            segment.clone()
        } else {
            format!("{current}\n\n{segment}")
        };
        if tentative.chars().count() > TELEGRAM_MESSAGE_MAX_LEN {
            if !current.is_empty() {
                out.push(current.clone());
                current.clear();
            }
            current = segment;
        } else {
            current = tentative;
        }
    }

    if !current.is_empty() {
        out.push(current);
    }

    out
}

fn escape_html(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
