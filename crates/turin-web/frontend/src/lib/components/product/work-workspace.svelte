<script lang="ts">
	import { ArrowLeft, CirclePause, ExternalLink, ListChecks, RotateCcw, Search, Unplug } from '@lucide/svelte';
	import type { Agent, Harness, Session, WorkItem, WorkItemControlAction, Worklist } from '#lib/api/contracts.js';
	import * as AlertDialog from '#lib/components/ui/alert-dialog/index.js';
	import { Badge } from '#lib/components/ui/badge/index.js';
	import { Button } from '#lib/components/ui/button/index.js';
	import * as Card from '#lib/components/ui/card/index.js';
	import { Input } from '#lib/components/ui/input/index.js';
	import * as Select from '#lib/components/ui/select/index.js';
	import * as Sheet from '#lib/components/ui/sheet/index.js';
	import * as Table from '#lib/components/ui/table/index.js';

	let {
		worklists, selectedWorklist, workItems, selectedWorkItem, sessions, agents, harnesses, loading, controlling,
		onOpenWorklist, onCloseWorklist, onOpenItem, onCloseItem, onControlItem, onOpenSession
	}: {
		worklists: Worklist[];
		selectedWorklist: Worklist | null;
		workItems: WorkItem[];
		selectedWorkItem: WorkItem | null;
		sessions: Session[];
		agents: Agent[];
		harnesses: Harness[];
		loading: boolean;
		controlling: boolean;
		onOpenWorklist: (worklist: Worklist) => void;
		onCloseWorklist: () => void;
		onOpenItem: (item: WorkItem) => void;
		onCloseItem: () => void;
		onControlItem: (action: WorkItemControlAction, reason?: string) => void;
		onOpenSession: (item: WorkItem) => void;
	} = $props();

	let query = $state('');
	let status = $state('all');
	let ownership = $state('all');
	let pauseReason = $state('');
	let releaseOpen = $state(false);
	let filteredWorklists = $derived(worklists.filter((worklist) => worklist.name.toLowerCase().includes(query.trim().toLowerCase())));
	let filteredItems = $derived(workItems.filter((item) => {
		const needle = query.trim().toLowerCase();
		const visibleStatus = item.paused ? 'paused' : item.status;
		return (!needle || `${item.title} ${item.kind} ${item.claim_agent_id ?? ''}`.toLowerCase().includes(needle))
			&& (status === 'all' || visibleStatus === status)
			&& (ownership === 'all' || (ownership === 'claimed') === Boolean(item.claim_agent_id));
	}));

	function formatDate(value: string | null) {
		if (!value) return 'Not recorded';
		return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value));
	}

	function visibleStatus(item: WorkItem) {
		return item.paused ? 'paused' : item.status;
	}

	function agentName(id: string | null) {
		if (!id) return 'Unclaimed';
		return agents.find((agent) => agent.id === id)?.name ?? id;
	}

	function scopeLabel(scope: string) {
		if (scope === 'global') return 'All workspaces';
		const separator = scope.indexOf(':');
		if (separator < 0) return scope.replaceAll('_', ' ');
		const kind = scope.slice(0, separator);
		const key = scope.slice(separator + 1);
		if (kind === 'harness') return harnesses.find((harness) => harness.id === key)?.name ?? 'This workspace';
		if (kind === 'agent') return agents.find((agent) => agent.id === key)?.name ?? 'Assigned agent';
		if (kind === 'session') return sessions.find((session) => session.id === key)?.title ?? 'One conversation';
		return `${kind.replaceAll('_', ' ')} · ${key}`;
	}

	function statusVariant(item: WorkItem): 'success' | 'warning' | 'danger' | 'info' | 'outline' {
		const value = visibleStatus(item);
		if (value === 'failed' || value === 'cancelled') return 'danger';
		if (value === 'active' || value === 'running') return 'info';
		if (value === 'done' || value === 'completed') return 'success';
		if (value === 'paused' || value === 'pending') return 'warning';
		return 'outline';
	}

	function pause() {
		onControlItem('pause', pauseReason.trim() || undefined);
		pauseReason = '';
	}
</script>

<div class="h-full overflow-y-auto bg-muted/20">
	<div class="mx-auto flex w-full max-w-[92rem] flex-col gap-6 px-5 py-8 lg:px-8 lg:py-10">
		<header>
			{#if selectedWorklist}
				<Button variant="ghost" size="sm" class="-ml-2 mb-3" onclick={onCloseWorklist}><ArrowLeft />All worklists</Button>
				<div class="flex flex-wrap items-end justify-between gap-3">
					<div><h1 class="font-heading text-3xl font-semibold tracking-tight">{selectedWorklist.name}</h1><p class="mt-2 text-sm text-muted-foreground">Available to {scopeLabel(selectedWorklist.scope)}.</p></div>
					<Badge variant="secondary">{workItems.length} {workItems.length === 1 ? 'item' : 'items'}</Badge>
				</div>
			{:else}
				<h1 class="font-heading text-3xl font-semibold tracking-tight">Work</h1>
				<p class="mt-2 max-w-2xl text-sm text-muted-foreground">Inspect queues and intervene when work needs an operator. Execution remains owned by the harness.</p>
			{/if}
		</header>

		<Card.Root class="gap-0 overflow-hidden py-0 shadow-none">
			<div class="flex flex-col gap-3 border-b bg-background p-4 sm:flex-row">
				<div class="relative min-w-0 flex-1 sm:max-w-md"><Search class="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" /><Input bind:value={query} class="pl-9" placeholder={selectedWorklist ? 'Search this worklist' : 'Search worklists'} /></div>
				{#if selectedWorklist}
					<Select.Root type="single" bind:value={status}><Select.Trigger class="w-full sm:w-40">{status === 'all' ? 'All states' : status}</Select.Trigger><Select.Content><Select.Item value="all">All states</Select.Item><Select.Item value="pending">Pending</Select.Item><Select.Item value="active">Active</Select.Item><Select.Item value="paused">Paused</Select.Item><Select.Item value="done">Done</Select.Item><Select.Item value="failed">Failed</Select.Item></Select.Content></Select.Root>
					<Select.Root type="single" bind:value={ownership}><Select.Trigger class="w-full sm:w-40">{ownership === 'all' ? 'Any owner' : ownership === 'claimed' ? 'Claimed' : 'Unclaimed'}</Select.Trigger><Select.Content><Select.Item value="all">Any owner</Select.Item><Select.Item value="claimed">Claimed</Select.Item><Select.Item value="unclaimed">Unclaimed</Select.Item></Select.Content></Select.Root>
				{/if}
			</div>

			{#if loading}
				<div class="grid min-h-72 place-items-center text-sm text-muted-foreground">Loading work…</div>
			{:else if selectedWorklist && filteredItems.length > 0}
				<Table.Root>
					<Table.Header><Table.Row><Table.Head>Item</Table.Head><Table.Head>Status</Table.Head><Table.Head>Priority</Table.Head><Table.Head>Owner</Table.Head><Table.Head>Updated</Table.Head></Table.Row></Table.Header>
					<Table.Body>{#each filteredItems as item}<Table.Row class="cursor-pointer" onclick={() => onOpenItem(item)}><Table.Cell><p class="font-medium">{item.title}</p><p class="mt-0.5 text-xs text-muted-foreground">{item.action_name ?? item.kind}</p></Table.Cell><Table.Cell><Badge variant={statusVariant(item)}>{visibleStatus(item)}</Badge></Table.Cell><Table.Cell class="tabular-nums">{item.priority}</Table.Cell><Table.Cell class="text-muted-foreground">{agentName(item.claim_agent_id)}</Table.Cell><Table.Cell class="text-muted-foreground">{formatDate(item.updated_at)}</Table.Cell></Table.Row>{/each}</Table.Body>
				</Table.Root>
			{:else if selectedWorklist}
				<div class="grid min-h-72 place-items-center text-center"><div><ListChecks class="mx-auto mb-3 size-5 text-primary" /><p class="text-sm font-medium">{workItems.length === 0 ? 'This worklist is empty' : 'No items match these filters'}</p><p class="mt-1 text-xs text-muted-foreground">{workItems.length === 0 ? 'New work will appear here when a harness creates it.' : 'Change the state, ownership, or search filters.'}</p></div></div>
			{:else if filteredWorklists.length > 0}
				<Table.Root><Table.Header><Table.Row><Table.Head>Name</Table.Head><Table.Head>Available to</Table.Head><Table.Head>Updated</Table.Head></Table.Row></Table.Header><Table.Body>{#each filteredWorklists as worklist}<Table.Row class="cursor-pointer" onclick={() => onOpenWorklist(worklist)}><Table.Cell class="font-medium">{worklist.name}</Table.Cell><Table.Cell class="text-muted-foreground">{scopeLabel(worklist.scope)}</Table.Cell><Table.Cell class="text-muted-foreground">{formatDate(worklist.updated_at)}</Table.Cell></Table.Row>{/each}</Table.Body></Table.Root>
			{:else}
				<div class="grid min-h-72 place-items-center text-sm text-muted-foreground">{worklists.length === 0 ? 'No worklists have been created.' : 'No matching worklists.'}</div>
			{/if}
		</Card.Root>
	</div>
</div>

<Sheet.Root bind:open={() => selectedWorkItem !== null, (open) => { if (!open) onCloseItem(); }}>
	<Sheet.Content class="w-full max-w-none! overflow-y-auto bg-background sm:w-[min(92vw,60rem)]!">
		{#if selectedWorkItem}
			<Sheet.Header class="border-b bg-background pb-5 pr-12"><div class="mb-2 flex items-center gap-2"><Badge variant={statusVariant(selectedWorkItem)}>{visibleStatus(selectedWorkItem)}</Badge><span class="text-xs text-muted-foreground">Priority {selectedWorkItem.priority}</span></div><Sheet.Title class="font-heading text-xl leading-7">{selectedWorkItem.title}</Sheet.Title><Sheet.Description>{selectedWorkItem.action_name ?? selectedWorkItem.kind}</Sheet.Description></Sheet.Header>
			<Sheet.Body class="gap-5 px-7 py-7">
				{#if selectedWorkItem.prompt}<section class="rounded-xl border bg-background p-5"><h3 class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Task brief</h3><p class="mt-3 whitespace-pre-wrap text-[0.9375rem] leading-6">{selectedWorkItem.prompt}</p></section>{/if}
				{#if selectedWorkItem.pause_reason}<section class="rounded-lg border border-amber-500/25 bg-amber-500/5 p-4"><h3 class="text-sm font-medium">Paused</h3><p class="mt-1 text-sm text-muted-foreground">{selectedWorkItem.pause_reason}</p></section>{/if}
				{#if selectedWorkItem.failure_reason}<section class="rounded-lg border border-destructive/25 bg-destructive/5 p-4"><h3 class="text-sm font-medium text-destructive">Failure</h3><p class="mt-1 text-sm text-muted-foreground">{selectedWorkItem.failure_reason}</p></section>{/if}
				<section class="rounded-xl border bg-background p-5"><h3 class="mb-4 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Assignment</h3><div class="grid grid-cols-2 gap-x-5 gap-y-5 text-sm"><div><p class="text-xs text-muted-foreground">Owner</p><p class="mt-1 font-medium">{agentName(selectedWorkItem.claim_agent_id)}</p></div><div><p class="text-xs text-muted-foreground">Kind</p><p class="mt-1 font-medium">{selectedWorkItem.kind}</p></div><div><p class="text-xs text-muted-foreground">Claimed</p><p class="mt-1">{formatDate(selectedWorkItem.claimed_at)}</p></div><div><p class="text-xs text-muted-foreground">Updated</p><p class="mt-1">{formatDate(selectedWorkItem.updated_at)}</p></div></div>{#if selectedWorkItem.claim_session_id}<div class="mt-5 flex flex-col gap-3 border-t pt-5 sm:flex-row sm:items-center sm:justify-between"><div><p class="text-sm font-medium">Worker context</p><p class="mt-1 max-w-2xl text-xs leading-5 text-muted-foreground">Inspect the conversation in which {agentName(selectedWorkItem.claim_agent_id)} is processing this item. Turin records the conversation, but not an exact task message.</p></div><Button size="sm" variant="outline" onclick={() => onOpenSession(selectedWorkItem)}><ExternalLink />Inspect context</Button></div>{/if}</section>
				{#if selectedWorkItem.after.length > 0}<section class="rounded-xl border bg-background p-5"><h3 class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Dependencies</h3><div class="mt-3 flex flex-wrap gap-2">{#each selectedWorkItem.after as dependency}<Badge variant="outline">{dependency}</Badge>{/each}</div></section>{/if}
				{#if selectedWorkItem.status === 'pending' && !selectedWorkItem.claim_execution_id}<section class="rounded-xl border bg-background p-5"><div class="flex items-start gap-3"><span class="grid size-8 shrink-0 place-items-center rounded-lg bg-muted text-muted-foreground"><CirclePause class="size-4" /></span><div class="min-w-0 flex-1"><h3 class="text-sm font-semibold">Pause pending work</h3><p class="mt-1 text-xs leading-5 text-muted-foreground">Prevent a worker from claiming this item until an operator resumes it.</p><Input bind:value={pauseReason} class="mt-4 bg-background" placeholder="Reason (optional)" /><Button class="mt-3" variant="outline" size="sm" onclick={pause} disabled={controlling}>Pause item</Button></div></div></section>{/if}
			</Sheet.Body>
			{#if selectedWorkItem.status === 'paused' || selectedWorkItem.status === 'active'}
				<Sheet.Footer class="sticky bottom-0 mt-auto flex-row flex-wrap border-t bg-background/95 py-4 backdrop-blur sm:justify-start">
					{#if selectedWorkItem.status === 'paused'}<Button size="sm" onclick={() => onControlItem('resume')} disabled={controlling}><RotateCcw />Resume</Button>{/if}
					{#if selectedWorkItem.status === 'active'}<Button size="sm" variant="outline" onclick={() => releaseOpen = true} disabled={controlling}><Unplug />Release stale claim</Button>{/if}
				</Sheet.Footer>
			{/if}
		{/if}
	</Sheet.Content>
</Sheet.Root>

<AlertDialog.Root bind:open={releaseOpen}>
	<AlertDialog.Content><AlertDialog.Header><AlertDialog.Title>Release this claim?</AlertDialog.Title><AlertDialog.Description>Turin will release it only if its worker heartbeat is stale. A live worker keeps ownership.</AlertDialog.Description></AlertDialog.Header><AlertDialog.Footer><AlertDialog.Cancel>Cancel</AlertDialog.Cancel><AlertDialog.Action onclick={() => onControlItem('release_stale')}>Release stale claim</AlertDialog.Action></AlertDialog.Footer></AlertDialog.Content>
</AlertDialog.Root>
