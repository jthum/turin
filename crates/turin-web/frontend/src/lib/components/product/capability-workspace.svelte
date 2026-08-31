<script lang="ts">
	import { Bot } from '@lucide/svelte';
	import type { Agent, Harness } from '#lib/api/contracts.js';
	import { Badge } from '#lib/components/ui/badge/index.js';
	import * as Card from '#lib/components/ui/card/index.js';
	import type { WorkspaceSection } from '#lib/workspace.js';

	let {
		section, agents, harness, loading
	}: {
		section: Exclude<WorkspaceSection, 'overview' | 'conversations' | 'work' | 'memory'>;
		agents: Agent[];
		harness: Harness | undefined;
		loading: boolean;
	} = $props();

	let content = $derived({
		agents: { title: 'Agents', description: 'Choose which model and harness own a new conversation.' },
		settings: { title: 'Settings', description: 'Inspect the active workspace configuration.' }
	}[section]);
</script>

<div class="h-full overflow-y-auto bg-muted/20">
	<div class="mx-auto flex w-full max-w-[92rem] flex-col gap-6 px-5 py-8 lg:px-8 lg:py-10">
		<header><h1 class="font-heading text-3xl font-semibold tracking-tight">{content.title}</h1><p class="mt-2 text-sm text-muted-foreground">{content.description}</p></header>

		{#if loading}
			<Card.Root class="grid min-h-72 place-items-center shadow-none"><p class="text-sm text-muted-foreground">Loading…</p></Card.Root>
		{:else if section === 'agents'}
			<div class="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
				{#each agents as agent}<Card.Root class="border-primary/10 shadow-none"><Card.Header><span class="mb-2 grid size-9 place-items-center rounded-xl bg-primary/10 text-primary"><Bot class="size-4" /></span><Card.Title>{agent.name}</Card.Title><Card.Description>{agent.provider} / {agent.model}</Card.Description><Card.Action><Badge variant={agent.enabled ? 'secondary' : 'outline'}>{agent.enabled ? 'Ready' : 'Disabled'}</Badge></Card.Action></Card.Header></Card.Root>{/each}
			</div>
		{:else}
			<Card.Root class="max-w-3xl shadow-none"><Card.Content class="divide-y p-0"><div class="flex items-center justify-between px-5 py-4"><span class="text-sm text-muted-foreground">Harness</span><span class="text-sm font-medium">{harness?.name ?? 'Not selected'}</span></div><div class="flex items-center justify-between px-5 py-4"><span class="text-sm text-muted-foreground">Interface</span><span class="text-sm font-medium">{harness?.has_ui ? 'Harness-defined' : 'Standard'}</span></div></Card.Content></Card.Root>
		{/if}
	</div>
</div>
