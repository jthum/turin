use turin_channel_core::{MessageBlock, OutboundMessage};

pub(crate) fn render_whatsapp_message(message: &OutboundMessage) -> String {
    let mut parts = Vec::new();
    for block in &message.blocks {
        match block {
            MessageBlock::Text { text } => {
                if !text.trim().is_empty() {
                    parts.push(text.trim().to_string());
                }
            }
            MessageBlock::CodeBlock { language, code } => {
                let mut fenced = String::from("```");
                if let Some(language) = language.as_deref()
                    && !language.trim().is_empty()
                {
                    fenced.push_str(language.trim());
                }
                fenced.push('\n');
                fenced.push_str(code.trim_end());
                fenced.push_str("\n```");
                parts.push(fenced);
            }
        }
    }
    parts.join("\n\n")
}
