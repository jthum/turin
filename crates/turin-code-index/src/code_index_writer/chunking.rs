use super::CodeChunkRecord;

pub(super) const CHUNK_LINES: usize = 40;
const CHUNK_OVERLAP: usize = 8;
const MAX_SNIPPET_CHARS: usize = 1800;

#[derive(Debug, Clone)]
struct DiscoveredSymbol {
    start_line: usize,
    kind: String,
    name: String,
    signature: String,
}

pub(super) fn build_chunks(
    relative_path: &str,
    language: &str,
    content: &str,
) -> Vec<CodeChunkRecord> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let symbols = discover_symbols(language, &lines);
    if symbols.is_empty() {
        return build_chunk_segment(
            relative_path,
            language,
            &lines,
            0,
            fallback_symbol(relative_path, 1),
        );
    }

    let mut out = Vec::new();
    if symbols[0].start_line > 0 {
        out.extend(build_chunk_segment(
            relative_path,
            language,
            &lines[..symbols[0].start_line],
            0,
            fallback_symbol(relative_path, 1),
        ));
    }

    for (index, symbol) in symbols.iter().enumerate() {
        let segment_end = symbols
            .get(index + 1)
            .map(|next| next.start_line)
            .unwrap_or(lines.len());
        out.extend(build_chunk_segment(
            relative_path,
            language,
            &lines[symbol.start_line..segment_end],
            symbol.start_line,
            (
                symbol.kind.clone(),
                symbol.name.clone(),
                Some(symbol.signature.clone()),
            ),
        ));
    }
    out
}

fn fallback_symbol(relative_path: &str, start_line: usize) -> (String, String, Option<String>) {
    (
        "chunk".to_string(),
        format!("{relative_path}:{start_line}"),
        None,
    )
}

fn discover_symbols(language: &str, lines: &[&str]) -> Vec<DiscoveredSymbol> {
    let mut out = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || is_comment_line(language, trimmed) {
            continue;
        }
        if let Some((kind, name)) = parse_signature(language, trimmed) {
            out.push(DiscoveredSymbol {
                start_line: index,
                kind: kind.to_string(),
                name,
                signature: trimmed.to_string(),
            });
        }
    }
    out
}

fn build_chunk_segment(
    relative_path: &str,
    language: &str,
    lines: &[&str],
    base_start: usize,
    metadata: (String, String, Option<String>),
) -> Vec<CodeChunkRecord> {
    if lines.is_empty() {
        return Vec::new();
    }

    let step = CHUNK_LINES.saturating_sub(CHUNK_OVERLAP).max(1);
    let mut start = 0_usize;
    let mut out = Vec::new();
    while start < lines.len() {
        let end = (start + CHUNK_LINES).min(lines.len());
        let chunk_lines = &lines[start..end];
        let snippet = truncate_chars(&chunk_lines.join("\n"), MAX_SNIPPET_CHARS);
        let (kind, name, signature) = metadata.clone();
        let search_text = match &signature {
            Some(signature) => format!("{relative_path}\n{name}\n{signature}\n{snippet}"),
            None => format!("{relative_path}\n{name}\n{snippet}"),
        };
        out.push(CodeChunkRecord {
            chunk_key: format!("{relative_path}:{}", base_start + start + 1),
            path: relative_path.to_string(),
            language: language.to_string(),
            kind,
            name,
            signature,
            snippet,
            search_text,
            embedding: None,
            start_line: (base_start + start + 1) as i64,
            end_line: (base_start + end) as i64,
        });
        if end == lines.len() {
            break;
        }
        start += step;
    }
    out
}

fn is_comment_line(language: &str, line: &str) -> bool {
    match language {
        "python" => line.starts_with('#'),
        _ => line.starts_with("//") || line.starts_with("--") || line.starts_with("/*"),
    }
}

fn parse_signature(language: &str, line: &str) -> Option<(&'static str, String)> {
    match language {
        "rust" => parse_from_prefixes(
            line,
            &[
                ("function", "pub async fn "),
                ("function", "pub fn "),
                ("function", "async fn "),
                ("function", "fn "),
                ("type", "pub struct "),
                ("type", "struct "),
                ("type", "pub enum "),
                ("type", "enum "),
                ("type", "pub trait "),
                ("type", "trait "),
                ("impl", "impl "),
            ],
        ),
        "lua" => parse_from_prefixes(
            line,
            &[("function", "local function "), ("function", "function ")],
        ),
        "python" => parse_from_prefixes(line, &[("function", "def "), ("type", "class ")]),
        "go" => parse_from_prefixes(
            line,
            &[
                ("function", "func "),
                ("type", "type "),
                ("value", "const "),
                ("value", "var "),
            ],
        ),
        "javascript" | "typescript" => parse_from_prefixes(
            line,
            &[
                ("function", "export async function "),
                ("function", "export function "),
                ("function", "async function "),
                ("function", "function "),
                ("type", "export class "),
                ("type", "class "),
                ("value", "export const "),
                ("value", "const "),
            ],
        ),
        "php" => parse_from_prefixes(line, &[("function", "function "), ("type", "class ")]),
        _ => None,
    }
}

fn parse_from_prefixes(
    line: &str,
    prefixes: &[(&'static str, &'static str)],
) -> Option<(&'static str, String)> {
    for (kind, prefix) in prefixes {
        if let Some(rest) = line.strip_prefix(prefix)
            && let Some(identifier) = extract_identifier(rest)
        {
            return Some((*kind, identifier));
        }
    }
    None
}

fn extract_identifier(input: &str) -> Option<String> {
    let mut identifier = String::new();
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | ':') {
            identifier.push(ch);
        } else if !identifier.is_empty() {
            break;
        } else if ch == '(' {
            break;
        } else if ch.is_whitespace() {
            if identifier.is_empty() {
                continue;
            }
            break;
        } else {
            break;
        }
    }
    if identifier.is_empty() {
        None
    } else {
        Some(identifier)
    }
}

fn truncate_chars(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    input.chars().take(max_chars).collect::<String>()
}
