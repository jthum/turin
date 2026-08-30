<script lang="ts">
	import { tick } from 'svelte';
	import { ArrowUp, LoaderCircle, Sparkles } from '@lucide/svelte';
	import { Badge } from '#lib/components/ui/badge/index.js';
	import { Button } from '#lib/components/ui/button/index.js';
	import { Textarea } from '#lib/components/ui/textarea/index.js';
	import * as Tooltip from '#lib/components/ui/tooltip/index.js';

	let {
		value = $bindable(), agentName, model, submitting, connected, onSend
	}: {
		value: string;
		agentName: string;
		model: string;
		submitting: boolean;
		connected: boolean;
		onSend: () => void;
	} = $props();
	let textarea: HTMLTextAreaElement | null = $state(null);
	let expanded = $state(false);

	async function measureInput() {
		await tick();
		if (!textarea) return;
		expanded = value.includes('\n') || textarea.scrollHeight > 44;
	}

	$effect(() => {
		value;
		void measureInput();
	});

	function keydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			onSend();
		}
	}
</script>

{#snippet sendButton()}
	<Tooltip.Root>
		<Tooltip.Trigger>
			{#snippet child({ props })}
				<Button {...props} size="icon" class="size-9 shrink-0 rounded-xl" onclick={onSend} disabled={!value.trim() || submitting || !connected} aria-label="Send message">
					{#if submitting}<LoaderCircle class="size-4 animate-spin" />{:else}<ArrowUp class="size-4" />{/if}
				</Button>
			{/snippet}
		</Tooltip.Trigger>
		<Tooltip.Content>{connected ? 'Send message' : 'Waiting for live connection'}</Tooltip.Content>
	</Tooltip.Root>
{/snippet}

<div class="relative z-10 border-t border-border bg-background px-4 py-3 sm:px-6">
	<div class="mx-auto max-w-3xl rounded-2xl border border-border bg-background p-1.5 transition-[border-color,box-shadow] focus-within:border-ring/60 focus-within:ring-2 focus-within:ring-ring/10">
		<div class:items-end={!expanded} class="flex gap-2" class:flex-col={expanded}>
			<Textarea bind:ref={textarea} bind:value onkeydown={keydown} placeholder={`Message ${agentName}`} rows={1} aria-label="Message" class="max-h-48 min-h-9 flex-1 resize-none border-0 bg-transparent px-2.5 py-2 text-[15px] leading-5 shadow-none focus-visible:border-transparent focus-visible:ring-0" />
			{#if !expanded}{@render sendButton()}{/if}
		</div>
		{#if expanded}
		<div class="mt-1 flex items-center justify-between gap-3 border-t border-border/70 px-1 pt-1.5">
			<Badge variant="secondary" class="max-w-[70%] gap-1.5 rounded-full px-2.5 py-1 font-normal text-muted-foreground">
				<Sparkles class="size-3" /><span class="truncate">{model}</span>
			</Badge>
			<span class="ml-auto hidden text-[11px] text-muted-foreground sm:inline">Enter to send · Shift + Enter for a new line</span>
			{@render sendButton()}
		</div>
		{/if}
	</div>
</div>
