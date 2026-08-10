import DOMPurify from "dompurify";
import { marked } from "marked";

export function renderMarkdown(source: string): string {
  const rendered = marked.parse(source, {
    async: false,
    breaks: true,
    gfm: true,
  });

  return DOMPurify.sanitize(rendered, {
    USE_PROFILES: { html: true },
    FORBID_TAGS: ["img", "style"],
    FORBID_ATTR: ["style"],
  });
}
