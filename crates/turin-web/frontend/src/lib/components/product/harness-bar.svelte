<script lang="ts">
	import { Check, ChevronDown, Menu, MoreHorizontal, Sparkles, Trash2 } from '@lucide/svelte';
	import { Button } from '#lib/components/ui/button/index.js';
	import * as DropdownMenu from '#lib/components/ui/dropdown-menu/index.js';
	import * as Sidebar from '#lib/components/ui/sidebar/index.js';
	import type { Harness, Session } from '#lib/api/contracts.js';
	import { workspaceLabels, type WorkspaceSection } from '#lib/workspace.js';

	let {
		harnesses,
		selectedHarnessId,
		session,
		section,
		onSelect,
		onNavigate,
		onDelete
	}: {
		harnesses: Harness[];
		selectedHarnessId: string;
		session: Session | null;
		section: WorkspaceSection;
		onSelect: (harnessId: string) => void;
		onNavigate: (section: WorkspaceSection) => void;
		onDelete: () => void;
	} = $props();

	let selectedHarness = $derived(harnesses.find((harness) => harness.id === selectedHarnessId));
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
		{#if session}<span class="block truncate text-sm font-medium">{session.title}</span>{/if}
	</div>

	<div class="flex items-center justify-end gap-1">
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
			<DropdownMenu.Root>
				<DropdownMenu.Trigger>
					{#snippet child({ props })}<Button {...props} variant="ghost" size="icon" aria-label="Conversation actions"><MoreHorizontal class="size-4" /></Button>{/snippet}
				</DropdownMenu.Trigger>
				<DropdownMenu.Content align="end">
					<DropdownMenu.Label>Conversation</DropdownMenu.Label>
					<DropdownMenu.Separator />
					<DropdownMenu.Item variant="destructive" onclick={onDelete}><Trash2 />Delete conversation</DropdownMenu.Item>
				</DropdownMenu.Content>
			</DropdownMenu.Root>
		{/if}
	</div>
</header>
