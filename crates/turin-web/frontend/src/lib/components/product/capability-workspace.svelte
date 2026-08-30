<script lang="ts">
	import { ArrowLeft, ArrowRight, Bot, Brain, Search } from '@lucide/svelte';
	import type { Agent, Harness, Memory, WorkItem, Worklist } from '#lib/api/contracts.js';
	import { Badge } from '#lib/components/ui/badge/index.js';
	import { Button } from '#lib/components/ui/button/index.js';
	import * as Card from '#lib/components/ui/card/index.js';
	import { Input } from '#lib/components/ui/input/index.js';
	import * as Select from '#lib/components/ui/select/index.js';
	import * as Table from '#lib/components/ui/table/index.js';
	import type { WorkspaceSection } from '#lib/workspace.js';

	let {
		section, agents, harness, worklists, selectedWorklist, workItems, memories, memoryTotal,
		loading, loadingWorkItems, loadingMoreMemories, onOpenWorklist, onCloseWorklist, onLoadMoreMemories
	}: {
		section: Exclude<WorkspaceSection, 'overview' | 'conversations'>;
		agents: Agent[];
		harness: Harness | undefined;
		worklists: Worklist[];
		selectedWorklist: Worklist | null;
		workItems: WorkItem[];
		memories: Memory[];
		memoryTotal: number;
		loading: boolean;
		loadingWorkItems: boolean;
		loadingMoreMemories: boolean;
		onOpenWorklist: (worklist: Worklist) => void;
		onCloseWorklist: () => void;
		onLoadMoreMemories: () => void;
	} = $props();

	let query = $state('');
	let scopeFilter = $state('all');
	let content = $derived({
		work: { title: 'Work', description: 'Open a queue to inspect the items waiting inside it.' },
		memory: { title: 'Memory', description: 'Search durable knowledge and inspect how it is scoped and used.' },
		agents: { title: 'Agents', description: 'Choose which model and harness own a new conversation.' },
		settings: { title: 'Settings', description: 'Inspect the active workspace configuration.' }
	}[section]);
	let memoryScopes = $derived(Array.from(new Set(memories.map((memory) => `${memory.scope_kind}:${memory.scope_key}`))));
	let filteredMemories = $derived(memories.filter((memory) => {
		const needle = query.trim().toLowerCase();
		const scope = `${memory.scope_kind}:${memory.scope_key}`;
		return (!needle || memory.content.toLowerCase().includes(needle)) && (scopeFilter === 'all' || scopeFilter === scope);
	}));
	let filteredWorklists = $derived(worklists.filter((worklist) => worklist.name.toLowerCase().includes(query.trim().toLowerCase())));

	function formatDate(value: string) {
		return new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', year: 'numeric' }).format(new Date(value));
	}

	function statusVariant(status: string): 'default' | 'secondary' | 'outline' | 'destructive' {
		if (status === 'failed' || status === 'cancelled') return 'destructive';
		if (status === 'running' || status === 'completed') return 'default';
		if (status === 'ready') return 'secondary';
		return 'outline';
	}
</script>

<div class="h-full overflow-y-auto bg-muted/20">
	<div class="mx-auto flex w-full max-w-[92rem] flex-col gap-6 px-5 py-8 lg:px-8 lg:py-10">
		{#if section === 'work' && selectedWorklist}
			<header>
				<Button variant="ghost" size="sm" class="-ml-2 mb-3" onclick={onCloseWorklist}><ArrowLeft />All worklists</Button>
				<h1 class="font-heading text-3xl font-semibold tracking-tight">{selectedWorklist.name}</h1>
				<p class="mt-2 text-sm text-muted-foreground">Items in {selectedWorklist.scope}.</p>
			</header>
		{:else}
			<header>
				<h1 class="font-heading text-3xl font-semibold tracking-tight">{content.title}</h1>
				<p class="mt-2 text-sm text-muted-foreground">{content.description}</p>
			</header>
		{/if}

		{#if loading || loadingWorkItems}
			<Card.Root class="grid min-h-72 place-items-center shadow-none"><p class="text-sm text-muted-foreground">Loading…</p></Card.Root>
		{:else if section === 'work' && selectedWorklist}
			<Card.Root class="gap-0 py-0 shadow-none">
				{#if workItems.length === 0}
					<div class="grid min-h-72 place-items-center text-sm text-muted-foreground">This worklist is empty.</div>
				{:else}
					<Table.Root>
						<Table.Header><Table.Row><Table.Head>Item</Table.Head><Table.Head>Status</Table.Head><Table.Head>Kind</Table.Head><Table.Head>Priority</Table.Head><Table.Head>Owner</Table.Head><Table.Head>Updated</Table.Head></Table.Row></Table.Header>
						<Table.Body>{#each workItems as item}<Table.Row><Table.Cell class="font-medium">{item.title}</Table.Cell><Table.Cell><Badge variant={statusVariant(item.status)}>{item.paused ? 'paused' : item.status}</Badge></Table.Cell><Table.Cell class="text-muted-foreground">{item.kind}</Table.Cell><Table.Cell class="tabular-nums">{item.priority}</Table.Cell><Table.Cell class="text-muted-foreground">{item.claim_agent_id ?? 'Unclaimed'}</Table.Cell><Table.Cell class="text-muted-foreground">{formatDate(item.updated_at)}</Table.Cell></Table.Row>{/each}</Table.Body>
					</Table.Root>
				{/if}
			</Card.Root>
		{:else if section === 'work'}
			<Card.Root class="gap-0 py-0 shadow-none">
				<div class="border-b p-4"><div class="relative max-w-md"><Search class="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" /><Input bind:value={query} class="pl-9" placeholder="Search worklists" /></div></div>
				{#if filteredWorklists.length === 0}
					<div class="grid min-h-72 place-items-center text-sm text-muted-foreground">{worklists.length === 0 ? 'No worklists have been created.' : 'No matching worklists.'}</div>
				{:else}
					<Table.Root>
						<Table.Header><Table.Row><Table.Head>Name</Table.Head><Table.Head>Scope</Table.Head><Table.Head>Updated</Table.Head><Table.Head class="w-12"><span class="sr-only">Open</span></Table.Head></Table.Row></Table.Header>
						<Table.Body>{#each filteredWorklists as worklist}<Table.Row class="cursor-pointer" onclick={() => onOpenWorklist(worklist)}><Table.Cell class="font-medium">{worklist.name}</Table.Cell><Table.Cell><Badge variant="outline">{worklist.scope}</Badge></Table.Cell><Table.Cell class="text-muted-foreground">{formatDate(worklist.updated_at)}</Table.Cell><Table.Cell><ArrowRight class="size-4 text-muted-foreground" /></Table.Cell></Table.Row>{/each}</Table.Body>
					</Table.Root>
				{/if}
			</Card.Root>
		{:else if section === 'memory'}
			<Card.Root class="gap-0 py-0 shadow-none">
				<div class="flex flex-col gap-3 border-b p-4 sm:flex-row">
					<div class="relative min-w-0 flex-1 sm:max-w-md"><Search class="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" /><Input bind:value={query} class="pl-9" placeholder="Search memory content" /></div>
					<Select.Root type="single" bind:value={scopeFilter}><Select.Trigger class="w-full sm:w-52">{scopeFilter === 'all' ? 'All scopes' : scopeFilter}</Select.Trigger><Select.Content><Select.Item value="all">All scopes</Select.Item>{#each memoryScopes as scope}<Select.Item value={scope}>{scope}</Select.Item>{/each}</Select.Content></Select.Root>
				</div>
				{#if filteredMemories.length === 0}
					<div class="grid min-h-72 place-items-center text-center"><div><Brain class="mx-auto mb-3 size-5 text-primary" /><p class="text-sm font-medium">{memories.length === 0 ? 'No durable memories' : 'No matching memories'}</p></div></div>
				{:else}
					<Table.Root>
						<Table.Header><Table.Row><Table.Head>Content</Table.Head><Table.Head>Scope</Table.Head><Table.Head>Storage</Table.Head><Table.Head>Weight</Table.Head><Table.Head>Used</Table.Head><Table.Head>Created</Table.Head></Table.Row></Table.Header>
						<Table.Body>{#each filteredMemories as memory}<Table.Row><Table.Cell class="max-w-xl whitespace-normal"><p class="line-clamp-2 leading-5">{memory.content}</p></Table.Cell><Table.Cell><Badge variant="secondary">{memory.scope_kind}:{memory.scope_key}</Badge></Table.Cell><Table.Cell class="text-muted-foreground">{memory.storage}</Table.Cell><Table.Cell class="tabular-nums">{memory.weight.toFixed(2)}</Table.Cell><Table.Cell class="tabular-nums text-muted-foreground">{memory.retrieval_count}</Table.Cell><Table.Cell class="text-muted-foreground">{formatDate(memory.created_at)}</Table.Cell></Table.Row>{/each}</Table.Body>
					</Table.Root>
				{/if}
				{#if memories.length < memoryTotal}<Card.Footer class="justify-center border-t py-4"><Button variant="outline" onclick={onLoadMoreMemories} disabled={loadingMoreMemories}>{loadingMoreMemories ? 'Loading…' : 'Load more'}</Button></Card.Footer>{/if}
			</Card.Root>
		{:else if section === 'agents'}
			<div class="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
				{#each agents as agent}<Card.Root class="border-primary/10 shadow-none"><Card.Header><span class="mb-2 grid size-9 place-items-center rounded-xl bg-primary/10 text-primary"><Bot class="size-4" /></span><Card.Title>{agent.name}</Card.Title><Card.Description>{agent.provider} / {agent.model}</Card.Description><Card.Action><Badge variant={agent.enabled ? 'secondary' : 'outline'}>{agent.enabled ? 'Ready' : 'Disabled'}</Badge></Card.Action></Card.Header></Card.Root>{/each}
			</div>
		{:else}
			<Card.Root class="max-w-3xl shadow-none"><Card.Content class="divide-y p-0"><div class="flex items-center justify-between px-5 py-4"><span class="text-sm text-muted-foreground">Harness</span><span class="text-sm font-medium">{harness?.name ?? 'Not selected'}</span></div><div class="flex items-center justify-between px-5 py-4"><span class="text-sm text-muted-foreground">Interface</span><span class="text-sm font-medium">{harness?.has_ui ? 'Harness-defined' : 'Standard'}</span></div></Card.Content></Card.Root>
		{/if}
	</div>
</div>
