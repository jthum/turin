import { describe, expect, test } from 'bun:test';
import { renderMarkdown } from '../src/lib/markdown.js';

describe('renderMarkdown', () => {
	test('renders useful Markdown', () => {
		const html = renderMarkdown('## Result\n\n- one\n- two\n\n`cargo test`');
		expect(html).toContain('<h2>Result</h2>');
		expect(html).toContain('<li>one</li>');
		expect(html).toContain('<code>cargo test</code>');
	});

	test('does not execute provider HTML or unsafe links', () => {
		const html = renderMarkdown('<script>alert(1)</script>\n\n[unsafe](javascript:alert(1))');
		expect(html).not.toContain('<script>');
		expect(html).not.toContain('href="javascript:');
		expect(html).toContain('&lt;script&gt;');
	});

	test('does not automatically load remote images', () => {
		const html = renderMarkdown('![diagram](https://example.com/diagram.png)');
		expect(html).not.toContain('<img');
		expect(html).toContain('Image: diagram');
	});
});
