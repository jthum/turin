<script lang="ts">
	import { renderMarkdown } from '#lib/markdown.js';

	let { content, streaming = false }: { content: string; streaming?: boolean } = $props();
	let html = $derived(renderMarkdown(content));
</script>

{#if streaming}
	<div class="whitespace-pre-wrap break-words text-[0.9375rem] leading-7">{content}</div>
{:else}
	<div class="prose-message">{@html html}</div>
{/if}

<style>
	.prose-message { font-size: 0.9375rem; line-height: 1.72; overflow-wrap: anywhere; }
	.prose-message :global(> :first-child) { margin-top: 0; }
	.prose-message :global(> :last-child) { margin-bottom: 0; }
	.prose-message :global(p) { margin: 0.7rem 0; }
	.prose-message :global(h1),
	.prose-message :global(h2),
	.prose-message :global(h3),
	.prose-message :global(h4) { margin: 1.35rem 0 0.55rem; color: var(--foreground); font-weight: 650; letter-spacing: -0.018em; line-height: 1.25; }
	.prose-message :global(h1) { font-size: 1.45rem; }
	.prose-message :global(h2) { font-size: 1.25rem; }
	.prose-message :global(h3) { font-size: 1.08rem; }
	.prose-message :global(ul), .prose-message :global(ol) { margin: 0.75rem 0; padding-left: 1.35rem; }
	.prose-message :global(li) { margin: 0.28rem 0; padding-left: 0.18rem; }
	.prose-message :global(li::marker) { color: var(--muted-foreground); }
	.prose-message :global(a) { color: var(--primary); font-weight: 520; text-decoration: underline; text-decoration-color: color-mix(in oklab, var(--primary) 35%, transparent); text-underline-offset: 0.18em; }
	.prose-message :global(a:hover) { text-decoration-color: var(--primary); }
	.prose-message :global(strong) { font-weight: 650; }
	.prose-message :global(blockquote) { margin: 0.9rem 0; border-left: 3px solid var(--border); padding: 0.15rem 0 0.15rem 1rem; color: var(--muted-foreground); }
	.prose-message :global(code) { border: 1px solid var(--border); border-radius: 0.4rem; background: var(--muted); padding: 0.12rem 0.34rem; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 0.84em; }
	.prose-message :global(pre) { margin: 1rem 0; overflow-x: auto; border: 1px solid var(--border); border-radius: 0.85rem; background: color-mix(in oklab, var(--muted) 72%, var(--background)); padding: 0.9rem 1rem; }
	.prose-message :global(pre code) { border: 0; background: transparent; padding: 0; font-size: 0.8rem; line-height: 1.65; }
	.prose-message :global(hr) { margin: 1.25rem 0; border: 0; border-top: 1px solid var(--border); }
	.prose-message :global(table) { width: 100%; margin: 1rem 0; border-collapse: separate; border-spacing: 0; overflow: hidden; border: 1px solid var(--border); border-radius: 0.75rem; font-size: 0.86rem; }
	.prose-message :global(th), .prose-message :global(td) { border-bottom: 1px solid var(--border); padding: 0.55rem 0.7rem; text-align: left; }
	.prose-message :global(th) { background: var(--muted); font-weight: 600; }
	.prose-message :global(tr:last-child td) { border-bottom: 0; }
</style>
