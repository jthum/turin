<script lang="ts">
	import { tick } from 'svelte';
	import { LoaderCircle, Search, X } from '@lucide/svelte';
	import { turinWeb } from '#lib/api/client.js';
	import type { SearchHit } from '#lib/api/contracts.js';
	import { Button } from '#lib/components/ui/button/index.js';
	import { Input } from '#lib/components/ui/input/index.js';
	import * as Popover from '#lib/components/ui/popover/index.js';

	let { sessionId, onSelect }: { sessionId: string; onSelect: (hit: SearchHit, query: string) => void } = $props();
	let open = $state(false);
	let query = $state('');
	let hits = $state<SearchHit[]>([]);
	let searching = $state(false);
	let inputRef = $state<HTMLInputElement | null>(null);

	async function openSearch() {
		open = true;
		await tick();
		inputRef?.focus();
		inputRef?.select();
	}

	function handleShortcut(event: KeyboardEvent) {
		if ((event.metaKey || event.ctrlKey) && event.key.toLocaleLowerCase() === 'f') {
			event.preventDefault();
			void openSearch();
		}
	}

	$effect(() => {
		const value = query.trim();
		if (!open || value.length < 2) {
			hits = [];
			searching = false;
			return;
		}
		const controller = new AbortController();
		const timer = setTimeout(async () => {
			searching = true;
			try {
				hits = (await turinWeb.searchSessionMessages(sessionId, value, controller.signal)).hits;
			} catch (cause) {
				if (!(cause instanceof DOMException && cause.name === 'AbortError')) hits = [];
			} finally {
				if (!controller.signal.aborted) searching = false;
			}
		}, 220);
		return () => { clearTimeout(timer); controller.abort(); };
	});

	function select(hit: SearchHit) {
		onSelect(hit, query.trim());
		open = false;
	}

	function snippetParts(value: string) {
		const needle = query.trim().toLocaleLowerCase();
		const index = value.toLocaleLowerCase().indexOf(needle);
		if (!needle || index < 0) return [{ value, match: false }];
		return [
			{ value: value.slice(0, index), match: false },
			{ value: value.slice(index, index + needle.length), match: true },
			{ value: value.slice(index + needle.length), match: false }
		].filter((part) => part.value);
	}
</script>

<svelte:window onkeydown={handleShortcut} />

<Popover.Root bind:open>
	<Popover.Trigger>
		{#snippet child({ props })}<Button {...props} variant="outline" size="icon-sm" class="rounded-full bg-background" aria-label="Search this conversation"><Search /></Button>{/snippet}
	</Popover.Trigger>
	<Popover.Content align="end" class="w-[min(26rem,calc(100vw-2rem))] p-0">
		<div class="flex items-center gap-2 border-b p-3">
			<Search class="size-4 shrink-0 text-muted-foreground" />
			<Input bind:ref={inputRef} bind:value={query} class="h-8 border-0 px-0 shadow-none focus-visible:ring-0" placeholder="Search every message" aria-label="Search every message" />
			{#if searching}<LoaderCircle class="size-4 animate-spin text-muted-foreground" />{:else if query}<Button variant="ghost" size="icon-sm" onclick={() => query = ''} aria-label="Clear search"><X /></Button>{/if}
		</div>
		<div class="max-h-80 overflow-y-auto p-2">
			{#if query.trim().length < 2}
				<p class="px-3 py-8 text-center text-sm text-muted-foreground">Enter at least two characters.</p>
			{:else if !searching && hits.length === 0}
				<p class="px-3 py-8 text-center text-sm text-muted-foreground">No matching messages.</p>
			{:else}
				{#each hits as hit}
					<button class="w-full rounded-lg px-3 py-2.5 text-left hover:bg-muted" onclick={() => select(hit)}>
						<span class="mb-1 flex items-center gap-2 text-xs font-medium capitalize text-muted-foreground"><span>{hit.role ?? 'message'}</span>{#if hit.turn_index !== null}<span>Turn {hit.turn_index + 1}</span>{/if}</span>
						<span class="line-clamp-3 text-sm leading-5">{#each snippetParts(hit.snippet) as part}{#if part.match}<mark class="rounded-sm bg-primary/15 px-0.5 text-foreground">{part.value}</mark>{:else}{part.value}{/if}{/each}</span>
					</button>
				{/each}
			{/if}
		</div>
	</Popover.Content>
</Popover.Root>
