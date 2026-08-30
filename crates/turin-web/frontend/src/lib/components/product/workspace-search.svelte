<script lang="ts">
	import { CalendarClock, FileText, LoaderCircle, MessageSquare, TerminalSquare } from '@lucide/svelte';
	import { turinWeb } from '#lib/api/client.js';
	import type { Agent, SearchHit } from '#lib/api/contracts.js';
	import * as Command from '#lib/components/ui/command/index.js';

	let {
		open = $bindable(false),
		agents,
		onSelect
	}: {
		open: boolean;
		agents: Agent[];
		onSelect: (hit: SearchHit, query: string) => void;
	} = $props();
	let query = $state('');
	let hits = $state<SearchHit[]>([]);
	let searching = $state(false);
	let failed = $state(false);
	let groups = $derived([
		{ kind: 'session' as const, label: 'Conversations', hits: hits.filter((hit) => hit.kind === 'session') },
		{ kind: 'message' as const, label: 'Messages', hits: hits.filter((hit) => hit.kind === 'message') },
		{ kind: 'tool_execution' as const, label: 'Tool executions', hits: hits.filter((hit) => hit.kind === 'tool_execution') },
		{ kind: 'event' as const, label: 'Runtime events', hits: hits.filter((hit) => hit.kind === 'event') }
	].filter((group) => group.hits.length > 0));

	$effect(() => {
		const value = query.trim();
		if (!open || value.length < 2) {
			hits = [];
			searching = false;
			failed = false;
			return;
		}
		const controller = new AbortController();
		const timer = setTimeout(async () => {
			searching = true;
			failed = false;
			try {
				hits = (await turinWeb.searchWorkspace(value, controller.signal)).hits;
			} catch (cause) {
				if (!(cause instanceof DOMException && cause.name === 'AbortError')) {
					hits = [];
					failed = true;
				}
			} finally {
				if (!controller.signal.aborted) searching = false;
			}
		}, 220);
		return () => { clearTimeout(timer); controller.abort(); };
	});

	function choose(hit: SearchHit) {
		open = false;
		onSelect(hit, query.trim());
	}

	function agentName(id: string) {
		return agents.find((agent) => agent.id === id)?.name ?? id;
	}

	function resultLabel(hit: SearchHit) {
		return hit.tool_name ?? hit.event_type ?? hit.role ?? 'Conversation';
	}
</script>

<Command.Dialog bind:open shouldFilter={false} title="Search Turin" description="Search persisted conversations, messages, tools, and runtime events." class="max-w-2xl">
	<Command.Input bind:value={query} placeholder="Search conversations, messages, tools, and events…" />
	<Command.List class="max-h-[min(32rem,65vh)]">
		{#if query.trim().length < 2}
			<div class="grid min-h-48 place-items-center px-8 text-center text-sm text-muted-foreground">
				<div><FileText class="mx-auto mb-3 size-5" /><p>Search across Turin's persisted workspace.</p><p class="mt-1 text-xs">Enter at least two characters.</p></div>
			</div>
		{:else if searching}
			<Command.Loading><div class="flex min-h-48 items-center justify-center gap-2 text-sm text-muted-foreground"><LoaderCircle class="size-4 animate-spin" />Searching persisted history</div></Command.Loading>
		{:else if failed}
			<Command.Empty>Search could not be completed.</Command.Empty>
		{:else if hits.length === 0}
			<Command.Empty>No persisted results found.</Command.Empty>
		{:else}
			{#each groups as group}
				<Command.Group heading={group.label}>
					{#each group.hits as hit, index (`${hit.kind}:${hit.session_id}:${hit.turn_id}:${index}`)}
						<Command.Item value={`${hit.kind}:${hit.session_id}:${hit.turn_id ?? ''}:${hit.snippet}`} onSelect={() => choose(hit)} class="items-start px-3 py-2.5">
							{#if hit.kind === 'session'}<MessageSquare class="mt-0.5 size-4 text-sky-600" />
							{:else if hit.kind === 'message'}<FileText class="mt-0.5 size-4 text-emerald-600" />
							{:else if hit.kind === 'tool_execution'}<TerminalSquare class="mt-0.5 size-4 text-amber-600" />
							{:else}<CalendarClock class="mt-0.5 size-4 text-rose-600" />{/if}
							<span class="grid min-w-0 flex-1 gap-0.5">
								<span class="flex min-w-0 items-center gap-2 text-xs text-muted-foreground"><b class="truncate font-medium text-foreground">{hit.title ?? 'Untitled conversation'}</b><span>·</span><span>{agentName(hit.agent_id)}</span>{#if hit.turn_index !== null}<span>· Turn {hit.turn_index + 1}</span>{/if}</span>
								<span class="line-clamp-2 text-sm leading-5">{hit.snippet}</span>
							</span>
							<Command.Shortcut class="capitalize">{resultLabel(hit).replaceAll('_', ' ')}</Command.Shortcut>
						</Command.Item>
					{/each}
				</Command.Group>
			{/each}
		{/if}
	</Command.List>
</Command.Dialog>
