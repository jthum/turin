<script lang="ts">
	import { ArrowUp, ChevronUp, LoaderCircle, Sparkles } from '@lucide/svelte';
	import { Button } from '#lib/components/ui/button/index.js';

	let {
		value = $bindable(),
		agentId,
		model,
		submitting,
		connected,
		onSend
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

<div class="composer-dock">
	<div class="composer-card">
		<textarea bind:value onkeydown={keydown} placeholder={`Message ${agentId}`} rows="2" aria-label="Message"></textarea>
		<div class="composer-actions">
			<button class="model-pill"><Sparkles />{model}<ChevronUp /></button>
			<Button size="icon" onclick={onSend} disabled={!value.trim() || submitting || !connected} aria-label="Send message">
				{#if submitting}<LoaderCircle class="spin" />{:else}<ArrowUp />{/if}
			</Button>
		</div>
	</div>
	<p>Enter to send · Shift + Enter for a new line</p>
</div>

<style>
	.composer-dock { padding: 0 24px 17px; background: linear-gradient(transparent, #fbfbfa 22%); }
	.composer-card { width: min(760px, 100%); margin: auto; border: 1px solid #d9d9d4; border-radius: 16px; background: white; padding: 12px; box-shadow: 0 10px 35px #0000000b, 0 1px 2px #00000010; }
	.composer-card textarea { display: block; width: 100%; min-height: 50px; resize: none; border: 0; outline: 0; background: transparent; color: #242422; font-size: 14px; line-height: 1.5; }
	.composer-card textarea::placeholder { color: #a2a29b; }
	.composer-actions { display: flex; align-items: center; justify-content: space-between; padding-top: 5px; }
	.model-pill { display: flex; align-items: center; gap: 6px; border: 0; border-radius: 7px; background: transparent; padding: 5px 7px; color: #777771; font-size: 10px; }
	.model-pill :global(svg) { width: 12px; height: 12px; }
	.composer-dock > p { margin: 7px auto 0; color: #aaa9a3; font-size: 9px; text-align: center; }
	:global(.spin) { animation: spin 1s linear infinite; }
	@keyframes spin { to { transform: rotate(360deg); } }
	@media (max-width: 760px) {
		.composer-dock { padding: 0 10px 10px; }
		.composer-card { border-radius: 14px; }
	}
</style>
