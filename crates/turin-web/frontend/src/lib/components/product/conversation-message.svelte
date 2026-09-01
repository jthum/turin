<script lang="ts">
	import { BrainCircuit, Check, ChevronRight, Copy, GitBranch, MoreHorizontal, Share2, Wrench } from '@lucide/svelte';
	import { Button } from '#lib/components/ui/button/index.js';
	import * as DropdownMenu from '#lib/components/ui/dropdown-menu/index.js';
	import * as Tooltip from '#lib/components/ui/tooltip/index.js';
	import type { ConversationMessage } from '#lib/api/contracts.js';
	import AgentMarker from './agent-marker.svelte';
	import RichMessage from './rich-message.svelte';

	let {
		message, agentName, streaming, focused, onFork
	}: {
		message: ConversationMessage;
		agentName: string;
		streaming: boolean;
		focused: boolean;
		onFork?: (message: ConversationMessage, activate: boolean) => void;
	} = $props();
	let copied = $state(false);
	let menuOpen = $state(false);
	let copyTimer: ReturnType<typeof setTimeout> | null = null;
	let isUser = $derived(message.role === 'user');
	let isTool = $derived(message.role === 'tool');
	let canFork = $derived(/^\d+$/.test(message.turn_id));
	let author = $derived(isUser ? 'You' : message.role === 'system' ? 'System' : agentName);

	function formatTime(value: string) {
		return new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(new Date(value));
	}

	function formatDuration(milliseconds: number) {
		return milliseconds < 1_000 ? `${milliseconds}ms` : `${(milliseconds / 1_000).toFixed(1)}s`;
	}

	async function copyMessage() {
		try {
			await navigator.clipboard.writeText(message.content);
			copied = true;
			if (copyTimer) clearTimeout(copyTimer);
			copyTimer = setTimeout(() => copied = false, 1_500);
		} catch {
			copied = false;
		}
	}

	async function shareMessage() {
		if (navigator.share) {
			try {
				await navigator.share({ text: message.content });
				return;
			} catch (cause) {
				if (cause instanceof DOMException && cause.name === 'AbortError') return;
			}
		}
		await copyMessage();
	}

</script>

{#if isTool}
	<article class="rounded-xl py-1.5 pl-5 sm:pl-6" class:focused-result={focused}>
		<details class="tool-disclosure text-sm text-muted-foreground">
			<summary class="flex w-fit cursor-pointer list-none items-center gap-2 rounded-md py-1 pr-2 transition-colors hover:text-foreground">
				<ChevronRight class="disclosure-chevron size-3" />
				<Wrench class="size-3.5" />
				<span>Tool activity</span>
				<time class="text-[11px] text-muted-foreground/70">{formatTime(message.created_at)}</time>
			</summary>
			<pre class="mb-1 ml-5 mt-1.5 overflow-x-auto whitespace-pre-wrap break-words rounded-lg border bg-muted/25 px-3 py-2.5 text-xs leading-5 text-foreground">{message.content}</pre>
		</details>
	</article>
{:else}
<article class="group/message flex items-start gap-2 rounded-xl py-4 transition-colors duration-500" class:flex-row-reverse={isUser} class:focused-result={focused}>
	{#if !isUser}<AgentMarker name={author} class="mt-1 size-3" />{/if}
	<div class="min-w-0" class:flex-1={!isUser} class:max-w-[85%]={isUser}>
		<div class="mb-2.5 flex items-start" class:justify-end={isUser}>
			<strong class="text-sm font-semibold capitalize leading-5">{author}</strong>
		</div>
		{#if message.reasoning}
			<details class="reasoning-disclosure mb-2 rounded-lg border border-border/70 bg-muted/30 px-3 py-2 text-sm">
				<summary class="flex cursor-pointer list-none items-center gap-2 text-muted-foreground"><ChevronRight class="disclosure-chevron size-3.5" /><BrainCircuit class="size-3.5" />Thought for {formatDuration(message.reasoning.duration_ms)}</summary>
				<p class="mb-0 mt-2 pl-7 text-sm leading-6 text-muted-foreground">{message.reasoning.summary}</p>
			</details>
		{/if}
		<div class:user-surface={isUser} class="text-foreground" class:ml-auto={isUser}>
			<RichMessage content={message.content} {streaming} />
			{#if streaming}<span class="ml-0.5 inline-block h-4 w-1 animate-pulse bg-primary align-text-bottom"></span>{/if}
		</div>

		<div class="message-footer mt-1.5 flex min-h-7 items-center gap-1 text-muted-foreground" class:justify-end={isUser}>
			<time class="mr-1 text-[11px]">{formatTime(message.created_at)}</time>
			<Tooltip.Root>
				<Tooltip.Trigger>
					{#snippet child({ props })}
						<Button {...props} variant="ghost" size="icon-sm" onclick={copyMessage} aria-label="Copy message">
							{#if copied}<Check class="size-3.5 text-emerald-600" />{:else}<Copy class="size-3.5" />{/if}
						</Button>
					{/snippet}
				</Tooltip.Trigger>
				<Tooltip.Content>{copied ? 'Copied' : 'Copy message'}</Tooltip.Content>
			</Tooltip.Root>
			<DropdownMenu.Root bind:open={menuOpen}>
				<DropdownMenu.Trigger>
					{#snippet child({ props })}
						<Button {...props} variant="ghost" size="icon-sm" aria-label="More message actions"><MoreHorizontal class="size-3.5" /></Button>
					{/snippet}
				</DropdownMenu.Trigger>
				<DropdownMenu.Content align={isUser ? 'end' : 'start'} class="w-52">
					<DropdownMenu.Item onclick={shareMessage}><Share2 />Share</DropdownMenu.Item>
					<DropdownMenu.Sub>
						<DropdownMenu.SubTrigger><GitBranch />Branch from here</DropdownMenu.SubTrigger>
						<DropdownMenu.SubContent class="w-52">
							<DropdownMenu.Item disabled={!onFork || !canFork} onclick={() => onFork?.(message, true)}>Fork and switch</DropdownMenu.Item>
							<DropdownMenu.Item disabled={!onFork || !canFork} onclick={() => onFork?.(message, false)}>Fork in background</DropdownMenu.Item>
						</DropdownMenu.SubContent>
					</DropdownMenu.Sub>
				</DropdownMenu.Content>
			</DropdownMenu.Root>
		</div>

		{#if !isUser && (message.metrics || message.token_count !== null)}
			<details class="response-details text-xs text-muted-foreground">
				<summary class="flex w-fit cursor-pointer list-none items-center gap-1.5 py-1"><ChevronRight class="disclosure-chevron size-3" />Response details</summary>
				<div class="mt-1 grid grid-cols-2 gap-x-6 gap-y-2 rounded-lg border border-border/70 bg-muted/25 px-3 py-2.5 sm:grid-cols-3">
					{#if message.metrics}
						<span><b>Input</b>{message.metrics.input_tokens.toLocaleString()} tokens</span>
						<span><b>Output</b>{message.metrics.output_tokens.toLocaleString()} tokens</span>
						{#if message.metrics.cache_read_input_tokens !== undefined}<span><b>Cache read</b>{message.metrics.cache_read_input_tokens.toLocaleString()} tokens</span>{/if}
						{#if message.metrics.cache_creation_input_tokens !== undefined}<span><b>Cache write</b>{message.metrics.cache_creation_input_tokens.toLocaleString()} tokens</span>{/if}
						{#if message.metrics.model}<span><b>Model</b>{message.metrics.model}</span>{/if}
					{:else if message.token_count !== null}
						<span><b>Message</b>{message.token_count.toLocaleString()} tokens</span>
					{/if}
				</div>
			</details>
		{/if}
	</div>
</article>
{/if}

<style>
	.user-surface {
		display: inline-block;
		border: 1px solid color-mix(in oklab, var(--primary) 16%, var(--border));
		border-radius: 0.35rem 1rem 1rem 1rem;
		background: color-mix(in oklab, var(--primary) 8%, var(--background));
		padding: 0.58rem 0.85rem;
	}
	.focused-result {
		outline: 2px solid color-mix(in oklab, var(--primary) 45%, transparent);
		outline-offset: 2px;
	}
	:global(.reasoning-disclosure[open] .disclosure-chevron),
	:global(.tool-disclosure[open] .disclosure-chevron),
	:global(.response-details[open] .disclosure-chevron) { transform: rotate(90deg); }
	:global(.disclosure-chevron) { transition: transform 150ms ease; }
	.response-details b { display: block; margin-bottom: 0.1rem; color: var(--foreground); font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
</style>
