<script lang="ts">
	import { ArrowUpRight, MessageSquare, MoreHorizontal, Plus, Search, Trash2 } from '@lucide/svelte';
	import type { Agent, SearchHit, Session } from '#lib/api/contracts.js';
	import { turinWeb } from '#lib/api/client.js';
	import { Button } from '#lib/components/ui/button/index.js';
	import * as Card from '#lib/components/ui/card/index.js';
	import * as DropdownMenu from '#lib/components/ui/dropdown-menu/index.js';
	import { Input } from '#lib/components/ui/input/index.js';
	import * as Select from '#lib/components/ui/select/index.js';
	import * as Table from '#lib/components/ui/table/index.js';
	import AgentMarker from './agent-marker.svelte';

	let {
		sessions, agents, loading, loadingMore, hasMore,
		onCreate, onSelect, onDelete, onLoadMore
	}: {
		sessions: Session[];
		agents: Agent[];
		loading: boolean;
		loadingMore: boolean;
		hasMore: boolean;
		onCreate: (agentId?: string) => void;
		onSelect: (session: Session) => void;
		onDelete: (session: Session) => void;
		onLoadMore: () => void;
	} = $props();

	let query = $state('');
	let agentFilter = $state('all');
	let searchHits = $state<SearchHit[]>([]);
	let searching = $state(false);
	let localFiltered = $derived(sessions.filter((session) => {
		const matchesQuery = session.title.toLowerCase().includes(query.trim().toLowerCase());
		return matchesQuery && (agentFilter === 'all' || session.agent_id === agentFilter);
	}));
	let searchSessions = $derived(searchHits
		.filter((hit) => agents.some((agent) => agent.id === hit.agent_id))
		.filter((hit) => agentFilter === 'all' || hit.agent_id === agentFilter)
		.map((hit): Session => ({
			id: hit.session_id,
			title: hit.title ?? 'Untitled conversation',
			agent_id: hit.agent_id,
			created_at: hit.created_at,
			message_count: null,
			visibility: 'private',
			relation_kind: null
		})));
	let filtered = $derived(query.trim().length >= 2 ? searchSessions : localFiltered);

	$effect(() => {
		const value = query.trim();
		if (value.length < 2) {
			searchHits = [];
			searching = false;
			return;
		}
		const controller = new AbortController();
		const timer = setTimeout(async () => {
			searching = true;
			try {
				searchHits = (await turinWeb.searchSessions(value, controller.signal)).hits;
			} catch (cause) {
				if (!(cause instanceof DOMException && cause.name === 'AbortError')) searchHits = [];
			} finally {
				if (!controller.signal.aborted) searching = false;
			}
		}, 250);
		return () => { clearTimeout(timer); controller.abort(); };
	});
	function agentName(agentId: string) {
		return agents.find((agent) => agent.id === agentId)?.name ?? agentId;
	}

	function formatDate(value: string) {
		const date = new Date(value);
		if (Number.isNaN(date.getTime())) return value;
		return new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', year: 'numeric' }).format(date);
	}

</script>

<div class="h-full overflow-y-auto bg-muted/20">
	<div class="mx-auto flex w-full max-w-[92rem] flex-col gap-6 px-5 py-8 lg:px-8 lg:py-10">
		<header class="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
			<div>
				<h1 class="font-heading text-3xl font-semibold tracking-tight">Conversations</h1>
				<p class="mt-2 text-sm text-muted-foreground">Search, filter, and return to an existing thread.</p>
			</div>
			<Button onclick={() => onCreate(agents[0]?.id)} disabled={agents.length === 0}><Plus />New conversation</Button>
		</header>

		<Card.Root class="gap-0 py-0 shadow-none">
			<div class="flex flex-col gap-3 border-b p-4 sm:flex-row sm:items-center sm:justify-between">
				<div class="relative min-w-0 flex-1 sm:max-w-md">
					<Search class="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
					<Input bind:value={query} class="pl-9" placeholder="Search conversations" aria-label="Search conversations" />
				</div>
				<Select.Root type="single" bind:value={agentFilter}>
					<Select.Trigger class="w-full sm:w-44">{agentFilter === 'all' ? 'All agents' : agentName(agentFilter)}</Select.Trigger>
					<Select.Content>
						<Select.Item value="all">All agents</Select.Item>
						{#each agents as agent}<Select.Item value={agent.id}>{agent.name}</Select.Item>{/each}
					</Select.Content>
				</Select.Root>
			</div>

			{#if loading || searching}
				<div class="grid min-h-72 place-items-center text-sm text-muted-foreground">Loading conversations…</div>
			{:else if filtered.length === 0}
				<div class="grid min-h-72 place-items-center px-6 text-center">
					<div><MessageSquare class="mx-auto mb-3 size-6 text-muted-foreground" /><p class="font-medium">No matching conversations</p><p class="mt-1 text-sm text-muted-foreground">Try another title or agent.</p></div>
				</div>
			{:else}
				<Table.Root>
					<Table.Header>
						<Table.Row><Table.Head>Title</Table.Head><Table.Head>Agent</Table.Head><Table.Head>Created</Table.Head><Table.Head class="w-12"><span class="sr-only">Actions</span></Table.Head></Table.Row>
					</Table.Header>
					<Table.Body>
						{#each filtered as session (session.id)}
							<Table.Row class="cursor-pointer" onclick={() => onSelect(session)}>
								<Table.Cell class="min-w-72"><p class="font-medium">{session.title}</p>{#if session.latest_message_preview}<p class="mt-1 max-w-xl truncate text-xs font-normal text-muted-foreground">{session.latest_message_preview}</p>{/if}</Table.Cell>
								<Table.Cell><span class="inline-flex items-center gap-2"><AgentMarker name={agentName(session.agent_id)} />{agentName(session.agent_id)}</span></Table.Cell>
								<Table.Cell class="text-muted-foreground">{formatDate(session.created_at)}</Table.Cell>
								<Table.Cell onclick={(event) => event.stopPropagation()}>
									<DropdownMenu.Root>
										<DropdownMenu.Trigger>{#snippet child({ props })}<Button {...props} variant="ghost" size="icon-sm" aria-label={`Actions for ${session.title}`}><MoreHorizontal /></Button>{/snippet}</DropdownMenu.Trigger>
										<DropdownMenu.Content align="end"><DropdownMenu.Item onclick={() => onSelect(session)}><ArrowUpRight />Open</DropdownMenu.Item><DropdownMenu.Separator /><DropdownMenu.Item variant="destructive" onclick={() => onDelete(session)}><Trash2 />Delete</DropdownMenu.Item></DropdownMenu.Content>
									</DropdownMenu.Root>
								</Table.Cell>
							</Table.Row>
						{/each}
					</Table.Body>
				</Table.Root>
			{/if}
			{#if hasMore}<Card.Footer class="justify-center border-t py-4"><Button variant="outline" onclick={onLoadMore} disabled={loadingMore}>{loadingMore ? 'Loading…' : 'Load more'}</Button></Card.Footer>{/if}
		</Card.Root>
	</div>
</div>
