<script lang="ts">
	import { ArrowUp, LoaderCircle, Sparkles } from '@lucide/svelte';
	import { Badge } from '#lib/components/ui/badge/index.js';
	import { Button } from '#lib/components/ui/button/index.js';
	import { Textarea } from '#lib/components/ui/textarea/index.js';
	import * as Tooltip from '#lib/components/ui/tooltip/index.js';

	let {
		value = $bindable(), agentId, model, submitting, connected, onSend
	}: {
		value: string;
		agentId: string;
		model: string;
		submitting: boolean;
		connected: boolean;
		onSend: () => void;
	} = $props();

	function keydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			onSend();
		}
	}
</script>

<div class="relative z-10 border-t border-border/60 bg-background/92 px-4 pb-4 pt-3 backdrop-blur-xl sm:px-6">
	<div class="mx-auto max-w-3xl rounded-3xl border border-border bg-card p-2 shadow-[0_12px_40px_-20px_rgba(15,23,42,0.28)] transition-shadow focus-within:shadow-[0_16px_48px_-22px_rgba(15,23,42,0.34)]">
		<Textarea bind:value onkeydown={keydown} placeholder={`Message ${agentId}`} rows={2} aria-label="Message" class="max-h-48 min-h-16 resize-none border-0 bg-transparent px-3 py-2 text-[15px] shadow-none focus-visible:border-transparent focus-visible:ring-0" />
		<div class="flex items-center justify-between gap-3 px-1 pb-1 pt-2">
			<Badge variant="secondary" class="max-w-[70%] gap-1.5 rounded-full px-2.5 py-1 font-normal text-muted-foreground">
				<Sparkles class="size-3" /><span class="truncate">{model}</span>
			</Badge>
			<Tooltip.Root>
				<Tooltip.Trigger>
					{#snippet child({ props })}
						<Button {...props} size="icon" class="rounded-full" onclick={onSend} disabled={!value.trim() || submitting || !connected} aria-label="Send message">
							{#if submitting}<LoaderCircle class="size-4 animate-spin" />{:else}<ArrowUp class="size-4" />{/if}
						</Button>
					{/snippet}
				</Tooltip.Trigger>
				<Tooltip.Content>{connected ? 'Send message' : 'Waiting for live connection'}</Tooltip.Content>
			</Tooltip.Root>
		</div>
	</div>
	<p class="mx-auto mt-2 max-w-3xl text-center text-[11px] text-muted-foreground">Enter to send · Shift + Enter for a new line</p>
</div>
