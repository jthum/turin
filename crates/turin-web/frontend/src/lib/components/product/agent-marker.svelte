<script lang="ts">
	import { agentHue } from '#lib/agent-color.js';
	import { cn } from '#lib/utils.js';

	let { name, class: className }: { name: string; class?: string } = $props();
	let hue = $derived(agentHue(name));
</script>

<span
	aria-hidden="true"
	class={cn('agent-marker size-2.5 shrink-0 rounded-full ring-1 ring-black/5 dark:ring-white/10', className)}
	style={`--agent-hue:${hue};`}
></span>

<style>
	.agent-marker {
		background:
			radial-gradient(circle at 72% 24%, hsl(calc(var(--agent-hue) + 72) 88% 72%), transparent 38%),
			linear-gradient(145deg, hsl(var(--agent-hue) 72% 48%), hsl(calc(var(--agent-hue) + 42) 78% 56%));
	}
</style>
