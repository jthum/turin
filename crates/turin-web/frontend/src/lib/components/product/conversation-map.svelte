<script lang="ts">
	import type { ConversationMessage } from '#lib/api/contracts.js';

	let {
		messages,
		total,
		newestOffset,
		olderOffset,
		viewportProgress,
		loading,
		onJump
	}: {
		messages: ConversationMessage[];
		total: number;
		newestOffset: number;
		olderOffset: number;
		viewportProgress: number;
		loading: boolean;
		onJump: (position: number) => void;
	} = $props();

	const clamp = (value: number) => Math.min(1, Math.max(0, value));
	let windowStart = $derived(total > 0 ? clamp((total - olderOffset) / total) : 0);
	let windowEnd = $derived(total > 0 ? clamp((total - newestOffset) / total) : 1);
	let viewportPosition = $derived(clamp(windowStart + (windowEnd - windowStart) * viewportProgress));
	let userLandmarks = $derived.by(() => {
		if (total === 0) return [];
		const firstIndex = total - olderOffset;
		const all = messages.flatMap((message, index) => message.role === 'user'
			? [{ message, position: clamp((firstIndex + index + 0.5) / total) }]
			: []);
		if (all.length <= 36) return all;
		const stride = all.length / 36;
		return Array.from({ length: 36 }, (_, index) => all[Math.floor(index * stride)]);
	});

	function jumpFromTrack(event: MouseEvent) {
		const bounds = event.currentTarget instanceof HTMLElement ? event.currentTarget.getBoundingClientRect() : null;
		if (bounds) onJump(clamp((event.clientY - bounds.top) / bounds.height));
	}

	function preview(message: ConversationMessage) {
		const firstLine = message.content.replace(/\s+/g, ' ').trim();
		return firstLine.length > 90 ? `${firstLine.slice(0, 87)}…` : firstLine;
	}
</script>

{#if total > 1}
	<nav class="group/map absolute bottom-9 right-3.5 top-9 z-10 hidden w-3 opacity-45 transition-opacity hover:opacity-100 focus-within:opacity-100 lg:block" aria-label="Conversation map" aria-busy={loading}>
		<button
			type="button"
			class="absolute inset-y-0 left-1/2 w-3 -translate-x-1/2 cursor-pointer rounded-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/35"
			onclick={jumpFromTrack}
			aria-label="Jump within conversation"
		>
			<span class="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 rounded-full bg-border/70 transition-colors group-hover/map:bg-muted-foreground/30"></span>
		</button>

		<span
			class="pointer-events-none absolute left-1/2 w-0.5 -translate-x-1/2 rounded-full bg-primary/20"
			style:top={`${windowStart * 100}%`}
			style:height={`${Math.max(1.5, (windowEnd - windowStart) * 100)}%`}
		></span>

		{#each userLandmarks as landmark (landmark.message.id)}
			<button
				type="button"
				class="absolute left-1/2 h-1 w-0.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-muted-foreground/30 transition-[width,height,background-color] hover:h-1.5 hover:w-1 hover:bg-foreground/65 focus-visible:h-1.5 focus-visible:w-1 focus-visible:bg-foreground focus-visible:outline-none"
				style:top={`${landmark.position * 100}%`}
				onclick={(event) => { event.stopPropagation(); onJump(landmark.position); }}
				title={preview(landmark.message)}
				aria-label={`Jump to user message: ${preview(landmark.message)}`}
			></button>
		{/each}

		<span
			class="pointer-events-none absolute left-1/2 h-1.5 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-foreground/65 ring-1 ring-background"
			style:top={`${viewportPosition * 100}%`}
		></span>
	</nav>
{/if}
