<script lang="ts">
	import { onMount } from 'svelte';
	import { Check, ChevronDown, GitBranch, Menu, MoreHorizontal, Pencil, Search, Sparkles, Trash2 } from '@lucide/svelte';
	import { Button } from '#lib/components/ui/button/index.js';
	import * as DropdownMenu from '#lib/components/ui/dropdown-menu/index.js';
	import * as Sidebar from '#lib/components/ui/sidebar/index.js';
	import type { Harness, Session } from '#lib/api/contracts.js';
	import { workspaceLabels, type WorkspaceSection } from '#lib/workspace.js';
	import LinkedConversationsMenu from './linked-conversations-menu.svelte';

	let {
		harnesses,
		selectedHarnessId,
		session,
		renamable,
		section,
		onSelect,
		onNavigate,
		onSearch,
		onOpenRelationship,
		onRename,
		onDelete
	}: {
		harnesses: Harness[];
		selectedHarnessId: string;
		session: Session | null;
		renamable: boolean;
		section: WorkspaceSection;
		onSelect: (harnessId: string) => void;
		onNavigate: (section: WorkspaceSection) => void;
		onSearch: () => void;
		onOpenRelationship: (sessionId: string, turnId?: string) => void;
		onRename: () => void;
		onDelete: () => void;
	} = $props();

	let selectedHarness = $derived(harnesses.find((harness) => harness.id === selectedHarnessId));
	let searchShortcut = $state('Ctrl K');
	onMount(() => {
		searchShortcut = navigator.userAgent.includes('Macintosh') ? '⌘K' : 'Ctrl K';
	});
</script>

<header class="grid h-14 shrink-0 grid-cols-[minmax(0,1fr)_auto] items-center border-b border-border bg-background px-3 md:grid-cols-[minmax(0,1fr)_minmax(12rem,2fr)_minmax(0,1fr)]">
	<div class="min-w-0">
		<DropdownMenu.Root>
			<DropdownMenu.Trigger>
				{#snippet child({ props })}
					<Button {...props} variant="ghost" size="sm" class="max-w-72 justify-start gap-2 px-1.5 font-medium">
						<span class="flex size-7 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground"><Sparkles class="size-3.5" /></span>
						<span class="font-semibold">Turin</span>
						<span class="text-muted-foreground/50">/</span>
						<span class="truncate text-muted-foreground">{selectedHarness?.name ?? 'Select harness'}</span>
						<ChevronDown class="size-3.5 shrink-0 text-muted-foreground" />
					</Button>
				{/snippet}
			</DropdownMenu.Trigger>
			<DropdownMenu.Content align="start" class="w-64">
				<DropdownMenu.Label>Switch harness</DropdownMenu.Label>
				<DropdownMenu.Separator />
				{#each harnesses as harness}
					<DropdownMenu.Item onclick={() => onSelect(harness.id)}>
						<div class="grid min-w-0 flex-1">
							<span class="truncate font-medium">{harness.name}</span>
							<span class="text-xs text-muted-foreground">{harness.bound_agent_ids.length} {harness.bound_agent_ids.length === 1 ? 'agent' : 'agents'}</span>
						</div>
						{#if selectedHarnessId === harness.id}<Check class="ml-auto size-4 text-primary" />{/if}
					</DropdownMenu.Item>
				{/each}
			</DropdownMenu.Content>
		</DropdownMenu.Root>
	</div>

	<div class="hidden min-w-0 px-6 text-center md:block">
		{#if session}
			<span class="block truncate text-sm font-medium">{session.title}</span>
			{#if session.parent_session_id}
				<a href={`/conversations/${encodeURIComponent(session.parent_session_id)}`} onclick={(event) => { event.preventDefault(); onOpenRelationship(session.parent_session_id!, session.origin_turn_id ?? undefined); }} class="mx-auto mt-0.5 flex w-fit max-w-full items-center gap-1 truncate text-[11px] text-muted-foreground transition-colors hover:text-foreground">
					<GitBranch class="size-3 shrink-0" />
					<span class="truncate">Linked from {session.parent_title ?? 'parent conversation'}</span>
				</a>
			{/if}
		{/if}
	</div>

	<div class="flex items-center justify-end gap-1">
		<Button variant="outline" size="sm" class="mr-1 hidden h-8 min-w-36 justify-between rounded-full bg-muted/20 px-3 text-muted-foreground sm:flex" onclick={onSearch}>
			<span class="flex items-center gap-2"><Search class="size-3.5" />Search</span><kbd class="text-[10px] font-medium">{searchShortcut}</kbd>
		</Button>
		<DropdownMenu.Root>
			<DropdownMenu.Trigger>
				{#snippet child({ props })}<Button {...props} variant="ghost" size="icon" class="md:hidden" aria-label="Open workspace navigation"><Menu class="size-4" /></Button>{/snippet}
			</DropdownMenu.Trigger>
			<DropdownMenu.Content align="end" class="w-52">
				<DropdownMenu.Label>Workspace</DropdownMenu.Label>
				<DropdownMenu.Separator />
				{#each Object.entries(workspaceLabels) as [id, label]}
					<DropdownMenu.Item onclick={() => onNavigate(id as WorkspaceSection)}>{label}{#if section === id}<Check class="ml-auto" />{/if}</DropdownMenu.Item>
				{/each}
			</DropdownMenu.Content>
		</DropdownMenu.Root>
		{#if section === 'conversations' && session}<Sidebar.Trigger aria-label="Toggle conversations" />{/if}
		{#if session}
			<LinkedConversationsMenu {session} onOpen={onOpenRelationship} />
			<DropdownMenu.Root>
				<DropdownMenu.Trigger>
					{#snippet child({ props })}<Button {...props} variant="ghost" size="icon" aria-label="Conversation actions"><MoreHorizontal class="size-4" /></Button>{/snippet}
				</DropdownMenu.Trigger>
				<DropdownMenu.Content align="end">
					<DropdownMenu.Label>Conversation</DropdownMenu.Label>
					<DropdownMenu.Separator />
					{#if renamable}<DropdownMenu.Item onclick={onRename}><Pencil />Rename conversation</DropdownMenu.Item><DropdownMenu.Separator />{/if}
					<DropdownMenu.Item variant="destructive" onclick={onDelete}><Trash2 />Delete conversation</DropdownMenu.Item>
				</DropdownMenu.Content>
			</DropdownMenu.Root>
		{/if}
	</div>
</header>
