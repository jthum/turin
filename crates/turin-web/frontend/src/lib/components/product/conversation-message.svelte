<script lang="ts">
	import { Check, Copy, Wrench } from '@lucide/svelte';
	import * as Avatar from '#lib/components/ui/avatar/index.js';
	import { Button } from '#lib/components/ui/button/index.js';
	import * as Tooltip from '#lib/components/ui/tooltip/index.js';
	import type { ConversationMessage } from '#lib/api/contracts.js';
	import RichMessage from './rich-message.svelte';

	let { message, agentId, streaming }: { message: ConversationMessage; agentId: string; streaming: boolean } = $props();
	let copied = $state(false);
	let copyTimer: ReturnType<typeof setTimeout> | null = null;
	let isUser = $derived(message.role === 'user');
	let isTool = $derived(message.role === 'tool');
	let author = $derived(isUser ? 'You' : isTool ? 'Tool result' : message.role === 'system' ? 'System' : agentId);
	let initials = $derived(isUser ? 'Y' : isTool ? 'T' : message.role === 'system' ? 'S' : 'A');

	function formatTime(value: string) {
		return new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(new Date(value));
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
</script>

<article class="group/message flex gap-3 py-4 sm:gap-4" class:tool-message={isTool}>
	<Avatar.Root size="sm" class={isUser ? 'mt-0.5 bg-primary text-primary-foreground' : isTool ? 'mt-0.5 bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-200' : 'mt-0.5 bg-muted'}>
		<Avatar.Fallback class={isUser ? 'bg-primary text-primary-foreground' : isTool ? 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-200' : 'bg-muted text-foreground'}>
			{#if isTool}<Wrench class="size-3" />{:else}{initials}{/if}
		</Avatar.Fallback>
	</Avatar.Root>
	<div class="min-w-0 flex-1">
		<div class="mb-2 flex min-h-6 items-center gap-2">
			<strong class="text-sm font-semibold capitalize">{author}</strong>
			<time class="text-xs text-muted-foreground">{formatTime(message.created_at)}</time>
			{#if message.token_count !== null}<span class="text-[11px] text-muted-foreground/70">{message.token_count.toLocaleString()} tokens</span>{/if}
			<div class="message-actions ml-auto opacity-0 transition-opacity focus-within:opacity-100 group-hover/message:opacity-100">
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
			</div>
		</div>
		<div class:user-surface={isUser} class:tool-surface={isTool} class="text-foreground">
			<RichMessage content={message.content} {streaming} />
			{#if streaming}<span class="ml-0.5 inline-block h-4 w-1 animate-pulse bg-primary align-text-bottom"></span>{/if}
		</div>
	</div>
</article>

<style>
	.user-surface {
		display: inline-block;
		border: 1px solid var(--border);
		border-radius: 0.35rem 1rem 1rem 1rem;
		background: var(--muted);
		padding: 0.58rem 0.85rem;
	}
	.tool-surface {
		border-left: 2px solid color-mix(in oklab, #d97706 35%, transparent);
		padding-left: 0.9rem;
		color: var(--muted-foreground);
	}
	@media (hover: none) {
		.message-actions { opacity: 1; }
	}
</style>
