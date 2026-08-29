import { Marked, Renderer } from 'marked';

function escapeHtml(value: string): string {
	return value
		.replaceAll('&', '&amp;')
		.replaceAll('<', '&lt;')
		.replaceAll('>', '&gt;')
		.replaceAll('"', '&quot;')
		.replaceAll("'", '&#039;');
}

function safeHref(value: string): string | null {
	const trimmed = value.trim();
	if (trimmed.startsWith('/') || trimmed.startsWith('#')) return trimmed;
	try {
		const url = new URL(trimmed);
		return ['http:', 'https:', 'mailto:'].includes(url.protocol) ? trimmed : null;
	} catch {
		return null;
	}
}

const renderer = new Renderer();

// Provider output is untrusted. Markdown is supported, raw HTML is always text.
renderer.html = ({ text }) => escapeHtml(text);
renderer.link = function ({ href, title, tokens }) {
	const label = this.parser.parseInline(tokens);
	const safe = safeHref(href);
	if (!safe) return label;
	const titleAttribute = title ? ` title="${escapeHtml(title)}"` : '';
	return `<a href="${escapeHtml(safe)}"${titleAttribute} target="_blank" rel="noopener noreferrer">${label}</a>`;
};
renderer.image = function ({ href, text }) {
	const safe = safeHref(href);
	const label = `Image: ${escapeHtml(text || 'attachment')}`;
	return safe
		? `<a href="${escapeHtml(safe)}" target="_blank" rel="noopener noreferrer">${label}</a>`
		: label;
};

const markdown = new Marked({
	async: false,
	breaks: true,
	gfm: true,
	renderer
});

export function renderMarkdown(source: string): string {
	return markdown.parse(source) as string;
}
