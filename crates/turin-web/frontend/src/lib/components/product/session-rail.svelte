<script lang="ts">
	import { Bot, ChevronDown, Plus, Search, X } from '@lucide/svelte';
	import { Button } from '#lib/components/ui/button/index.js';
	import * as DropdownMenu from '#lib/components/ui/dropdown-menu/index.js';
	import * as Sidebar from '#lib/components/ui/sidebar/index.js';
	import type { Agent, Session } from '#lib/api/contracts.js';
	import ConversationListItem from './conversation-list-item.svelte';

	let {
		agents, sessions, selectedId, loading,
		search = $bindable(), newAgentId = $bindable(), onCreate, onSelect, onDelete
	}: {
		agents: Agent[];
		sessions: Session[];
		selectedId: string | null;
		loading: boolean;
		search: string;
		newAgentId: string;
		onCreate: (agentId?: string) => void;
		onSelect: (session: Session) => void;
		onDelete: (session: Session) => void;
	} = $props();

	const sidebar = Sidebar.useSidebar();
	let searching = $state(false);
	let searchInput = $state<HTMLInputElement | null>(null);
	let filtered = $derived(sessions.filter((session) => session.title.toLowerCase().includes(search.toLowerCase())));

	$effect(() => {
		if (!searching) return;
		const frame = requestAnimationFrame(() => searchInput?.focus());
		return () => cancelAnimationFrame(frame);
	});

	function createWithAgent(agentId: string) {
		newAgentId = agentId;
		onCreate(agentId);
	}

	function closeSearch() {
		search = '';
		searching = false;
	}

	function selectSession(session: Session) {
		onSelect(session);
		if (sidebar.isMobile) sidebar.setOpenMobile(false);
	}
</script>

<Sidebar.Root collapsible="offcanvas" class="top-(--global-bar-height)! h-[calc(100svh-var(--global-bar-height))]! border-r border-sidebar-border">
	<Sidebar.Header class="p-3">
		{#if searching}
			<div class="flex items-center gap-1.5">
				<div class="relative min-w-0 flex-1">
					<Search class="pointer-events-none absolute left-3 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
					<Sidebar.Input bind:ref={searchInput} bind:value={search} class="bg-background pl-8" placeholder="Search conversations" aria-label="Search conversations" />
				</div>
				<Button variant="ghost" size="icon" onclick={closeSearch} aria-label="Close search"><X class="size-4" /></Button>
			</div>
		{:else}
			<div class="flex items-center gap-1.5">
				{#if agents.length <= 1}
					<Button class="min-w-0 flex-1 justify-start" onclick={() => agents[0] && createWithAgent(agents[0].id)} disabled={agents.length === 0}>
						<Plus class="size-4" /><span>New conversation</span>
					</Button>
				{:else}
					<DropdownMenu.Root>
						<DropdownMenu.Trigger>
							{#snippet child({ props })}
								<Button {...props} class="min-w-0 flex-1 justify-start">
									<Plus class="size-4" /><span>New conversation</span><ChevronDown class="ml-auto size-3.5 opacity-70" />
								</Button>
							{/snippet}
						</DropdownMenu.Trigger>
						<DropdownMenu.Content align="start" class="w-64">
							<DropdownMenu.Label>Start a conversation with</DropdownMenu.Label>
							<DropdownMenu.Separator />
							{#each agents as agent}
								<DropdownMenu.Item onclick={() => createWithAgent(agent.id)}>
									<Bot class="text-muted-foreground" />
									<span class="grid min-w-0 flex-1">
										<span class="truncate font-medium">{agent.name}</span>
										<span class="truncate text-xs text-muted-foreground">{agent.model}</span>
									</span>
								</DropdownMenu.Item>
							{/each}
						</DropdownMenu.Content>
					</DropdownMenu.Root>
				{/if}
				<Button variant="outline" size="icon" onclick={() => searching = true} aria-label="Search conversations"><Search class="size-4" /></Button>
			</div>
		{/if}
	</Sidebar.Header>

	<Sidebar.Content>
		<Sidebar.Group class="min-h-0 flex-1">
			<Sidebar.GroupLabel>Conversations</Sidebar.GroupLabel>
			<Sidebar.GroupContent>
				<Sidebar.Menu>
					{#if loading}
						{#each Array(6) as _}<Sidebar.MenuSkeleton />{/each}
					{:else if filtered.length === 0}
						<p class="px-3 py-5 text-center text-xs text-muted-foreground">No conversations found.</p>
					{:else}
						{#each filtered as session (session.id)}
							<ConversationListItem {session} active={selectedId === session.id} onSelect={() => selectSession(session)} onDelete={() => onDelete(session)} />
						{/each}
					{/if}
				</Sidebar.Menu>
			</Sidebar.GroupContent>
		</Sidebar.Group>
	</Sidebar.Content>

	<Sidebar.Rail />
</Sidebar.Root>
