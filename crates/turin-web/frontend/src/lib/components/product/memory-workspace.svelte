<script lang="ts">
	import { Brain, Database, History, Search, Trash2 } from '@lucide/svelte';
	import type { Memory, MemoryListOptions, MemoryScope } from '#lib/api/contracts.js';
	import * as AlertDialog from '#lib/components/ui/alert-dialog/index.js';
	import { Badge } from '#lib/components/ui/badge/index.js';
	import { Button } from '#lib/components/ui/button/index.js';
	import * as Card from '#lib/components/ui/card/index.js';
	import { Input } from '#lib/components/ui/input/index.js';
	import * as Select from '#lib/components/ui/select/index.js';
	import * as Sheet from '#lib/components/ui/sheet/index.js';
	import * as Table from '#lib/components/ui/table/index.js';
	import { Textarea } from '#lib/components/ui/textarea/index.js';

	let {
		memories, scopes, total, selectedMemory, loading, loadingMore, mutating,
		onFilter, onLoadMore, onOpen, onClose, onCorrect, onDelete
	}: {
		memories: Memory[];
		scopes: MemoryScope[];
		total: number;
		selectedMemory: Memory | null;
		loading: boolean;
		loadingMore: boolean;
		mutating: boolean;
		onFilter: (options: MemoryListOptions) => void;
		onLoadMore: () => void;
		onOpen: (memory: Memory) => void;
		onClose: () => void;
		onCorrect: (content: string) => void;
		onDelete: () => void;
	} = $props();

	let query = $state('');
	let scope = $state('all');
	let history = $state('current');
	let correction = $state('');
	let deleteOpen = $state(false);
	let initialized = false;

	$effect(() => {
		const filter = { query, scope, history };
		if (!initialized) {
			initialized = true;
			return;
		}
		const timer = setTimeout(() => {
			const selectedScope = scopes.find((candidate) => scopeValue(candidate) === filter.scope);
			onFilter({
				query: filter.query,
				scopeKind: selectedScope?.scope_kind,
				scopeKey: selectedScope?.scope_key,
				includeSuperseded: filter.history === 'all'
			});
		}, 250);
		return () => clearTimeout(timer);
	});

	$effect(() => {
		correction = selectedMemory?.content ?? '';
	});

	function scopeValue(value: MemoryScope) {
		return JSON.stringify([value.scope_kind, value.scope_key]);
	}

	function formatDate(value: string | null) {
		if (!value) return 'Never';
		return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value));
	}

	function metadata(value: unknown | null) {
		return value === null ? null : JSON.stringify(value, null, 2);
	}
</script>

<div class="h-full overflow-y-auto bg-muted/20">
	<div class="mx-auto flex w-full max-w-[92rem] flex-col gap-6 px-5 py-8 lg:px-8 lg:py-10">
		<header class="flex flex-wrap items-end justify-between gap-3">
			<div><h1 class="font-heading text-3xl font-semibold tracking-tight">Memory</h1><p class="mt-2 max-w-2xl text-sm text-muted-foreground">Search durable knowledge, inspect its provenance, and correct facts without rewriting history.</p></div>
			{#if total > 0}<p class="text-sm text-muted-foreground">{total.toLocaleString()} matching {total === 1 ? 'memory' : 'memories'}</p>{/if}
		</header>

		<Card.Root class="gap-0 overflow-hidden py-0 shadow-none">
			<div class="flex flex-col gap-3 border-b bg-background p-4 lg:flex-row">
				<div class="relative min-w-0 flex-1 lg:max-w-xl"><Search class="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" /><Input bind:value={query} class="pl-9" placeholder="Search durable knowledge" /></div>
				<Select.Root type="single" bind:value={scope}><Select.Trigger class="w-full lg:w-64">{scope === 'all' ? 'All scopes' : (scopes.find((item) => scopeValue(item) === scope)?.scope_key ?? 'Scope')}</Select.Trigger><Select.Content><Select.Item value="all">All scopes</Select.Item>{#each scopes as item}<Select.Item value={scopeValue(item)}>{item.scope_kind}:{item.scope_key} ({item.count})</Select.Item>{/each}</Select.Content></Select.Root>
				<Select.Root type="single" bind:value={history}><Select.Trigger class="w-full lg:w-44">{history === 'current' ? 'Current only' : 'Include history'}</Select.Trigger><Select.Content><Select.Item value="current">Current only</Select.Item><Select.Item value="all">Include history</Select.Item></Select.Content></Select.Root>
			</div>

			{#if loading}
				<div class="grid min-h-72 place-items-center text-sm text-muted-foreground">Searching memory…</div>
			{:else if memories.length === 0}
				<div class="grid min-h-72 place-items-center text-center"><div><Brain class="mx-auto mb-3 size-5 text-primary" /><p class="text-sm font-medium">{query.trim() || scope !== 'all' ? 'No matching memories' : 'No durable memories yet'}</p><p class="mt-1 text-xs text-muted-foreground">{query.trim() || scope !== 'all' ? 'Try a broader query or scope.' : 'Harnesses can store knowledge here when it should outlive a conversation.'}</p></div></div>
			{:else}
				<Table.Root>
					<Table.Header><Table.Row><Table.Head>Knowledge</Table.Head><Table.Head>Scope</Table.Head><Table.Head>Storage</Table.Head><Table.Head>Weight</Table.Head><Table.Head>Used</Table.Head><Table.Head>Created</Table.Head></Table.Row></Table.Header>
					<Table.Body>{#each memories as memory}<Table.Row class="cursor-pointer align-top" onclick={() => onOpen(memory)}><Table.Cell class="max-w-2xl whitespace-normal"><p class="line-clamp-2 font-medium leading-5">{memory.content}</p>{#if memory.superseded_at}<div class="mt-2"><Badge variant="outline"><History />Superseded</Badge></div>{/if}</Table.Cell><Table.Cell><Badge variant="secondary">{memory.scope_kind}:{memory.scope_key}</Badge></Table.Cell><Table.Cell class="text-muted-foreground">{memory.storage === 'embedded' ? 'Vector + text' : 'Text'}</Table.Cell><Table.Cell class="tabular-nums">{memory.weight.toFixed(2)}</Table.Cell><Table.Cell class="tabular-nums text-muted-foreground">{memory.retrieval_count}</Table.Cell><Table.Cell class="text-muted-foreground">{formatDate(memory.created_at)}</Table.Cell></Table.Row>{/each}</Table.Body>
				</Table.Root>
			{/if}
			{#if memories.length < total}<Card.Footer class="justify-center border-t py-4"><Button variant="outline" onclick={onLoadMore} disabled={loadingMore}>{loadingMore ? 'Loading…' : 'Load more'}</Button></Card.Footer>{/if}
		</Card.Root>
	</div>
</div>

<Sheet.Root bind:open={() => selectedMemory !== null, (open) => { if (!open) onClose(); }}>
	<Sheet.Content class="w-full overflow-y-auto sm:max-w-2xl">
		{#if selectedMemory}
			<Sheet.Header class="border-b pb-5 pr-10"><div class="mb-2 flex items-center gap-2"><Badge variant="secondary">{selectedMemory.scope_kind}:{selectedMemory.scope_key}</Badge>{#if selectedMemory.superseded_at}<Badge variant="outline">Superseded</Badge>{/if}</div><Sheet.Title class="font-heading text-xl">Durable memory</Sheet.Title><Sheet.Description>Stored {formatDate(selectedMemory.created_at)}</Sheet.Description></Sheet.Header>
			<div class="flex flex-1 flex-col gap-6 py-6">
				<section><h3 class="text-xs font-medium uppercase tracking-wide text-muted-foreground">Content</h3><p class="mt-2 whitespace-pre-wrap text-base leading-7">{selectedMemory.content}</p></section>
				<section class="grid grid-cols-2 gap-x-5 gap-y-4 border-y py-5 text-sm sm:grid-cols-3"><div><p class="text-xs text-muted-foreground">Storage</p><p class="mt-1 font-medium">{selectedMemory.storage}</p></div><div><p class="text-xs text-muted-foreground">Weight</p><p class="mt-1 font-medium tabular-nums">{selectedMemory.weight.toFixed(2)}</p></div><div><p class="text-xs text-muted-foreground">Retrievals</p><p class="mt-1 font-medium tabular-nums">{selectedMemory.retrieval_count}</p></div><div><p class="text-xs text-muted-foreground">Last used</p><p class="mt-1">{formatDate(selectedMemory.last_retrieved_at)}</p></div><div><p class="text-xs text-muted-foreground">Embedding</p><p class="mt-1">{selectedMemory.embedding_dimensions ? `${selectedMemory.embedding_dimensions} dimensions` : 'None'}</p></div><div><p class="text-xs text-muted-foreground">ID</p><p class="mt-1 truncate font-mono text-xs" title={selectedMemory.id}>{selectedMemory.id}</p></div></section>
				{#if selectedMemory.superseded_by_id}<section class="rounded-lg border bg-muted/30 p-4"><div class="flex items-start gap-3"><History class="mt-0.5 size-4 text-muted-foreground" /><div><h3 class="text-sm font-medium">Replaced by a correction</h3><p class="mt-1 font-mono text-xs text-muted-foreground">{selectedMemory.superseded_by_id}</p></div></div></section>{/if}
				{#if metadata(selectedMemory.metadata)}<section><h3 class="text-xs font-medium uppercase tracking-wide text-muted-foreground">Provenance metadata</h3><pre class="mt-2 overflow-x-auto rounded-lg border bg-muted/30 p-4 text-xs leading-5">{metadata(selectedMemory.metadata)}</pre></section>{/if}
				{#if !selectedMemory.superseded_at}<section class="rounded-lg border p-4"><div class="flex items-start gap-3"><Database class="mt-1 size-4 text-muted-foreground" /><div class="min-w-0 flex-1"><h3 class="text-sm font-medium">Correct this memory</h3><p class="mt-1 text-xs text-muted-foreground">Creates a replacement and preserves this version in history.</p><Textarea bind:value={correction} class="mt-3 min-h-32" /><Button class="mt-3" size="sm" onclick={() => onCorrect(correction)} disabled={mutating || !correction.trim() || correction.trim() === selectedMemory.content}>{mutating ? 'Saving…' : 'Save correction'}</Button></div></div></section>{/if}
			</div>
			<Sheet.Footer class="sticky bottom-0 -mx-6 mt-auto border-t bg-background px-6 py-4 sm:justify-start"><Button variant="ghost" class="text-destructive hover:text-destructive" onclick={() => deleteOpen = true} disabled={mutating}><Trash2 />Forget memory</Button></Sheet.Footer>
		{/if}
	</Sheet.Content>
</Sheet.Root>

<AlertDialog.Root bind:open={deleteOpen}><AlertDialog.Content><AlertDialog.Header><AlertDialog.Title>Forget this memory?</AlertDialog.Title><AlertDialog.Description>This permanently removes the memory and its feedback history. This cannot be undone.</AlertDialog.Description></AlertDialog.Header><AlertDialog.Footer><AlertDialog.Cancel>Cancel</AlertDialog.Cancel><AlertDialog.Action onclick={onDelete}>Forget memory</AlertDialog.Action></AlertDialog.Footer></AlertDialog.Content></AlertDialog.Root>
