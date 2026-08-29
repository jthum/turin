<script lang="ts">
	import { Bot, MessageSquare, Plus, Search, Sparkles } from '@lucide/svelte';
	import { Button } from '#lib/components/ui/button/index.js';
	import * as Select from '#lib/components/ui/select/index.js';
	import * as Sidebar from '#lib/components/ui/sidebar/index.js';
	import type { Agent, Bootstrap, Session } from '#lib/api/contracts.js';

	let {
		bootstrap, agents, sessions, selectedId, loading,
		search = $bindable(), newAgentId = $bindable(), onCreate, onSelect
	}: {
		bootstrap: Bootstrap | null;
		agents: Agent[];
		sessions: Session[];
		selectedId: string | null;
		loading: boolean;
		search: string;
		newAgentId: string;
		onCreate: () => void;
		onSelect: (session: Session) => void;
	} = $props();

	const sidebar = Sidebar.useSidebar();
	let filtered = $derived(sessions.filter((session) => session.title.toLowerCase().includes(search.toLowerCase())));
	let selectedAgent = $derived(agents.find((agent) => agent.id === newAgentId));

	function selectSession(session: Session) {
		onSelect(session);
		if (sidebar.isMobile) sidebar.setOpenMobile(false);
	}
</script>

<Sidebar.Root collapsible="icon" class="border-r border-sidebar-border">
	<Sidebar.Header class="gap-3 p-3">
		<Sidebar.Menu>
			<Sidebar.MenuItem>
				<Sidebar.MenuButton size="lg" class="hover:bg-transparent active:bg-transparent">
					<div class="flex size-8 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-sm"><Sparkles class="size-4" /></div>
					<div class="grid min-w-0 flex-1 text-left leading-tight">
						<span class="truncate text-sm font-semibold">Turin</span>
						<span class="truncate text-xs text-muted-foreground">Agent workspace</span>
					</div>
				</Sidebar.MenuButton>
			</Sidebar.MenuItem>
		</Sidebar.Menu>

		<Button class="w-full justify-start group-data-[collapsible=icon]:size-8 group-data-[collapsible=icon]:p-0" onclick={onCreate} disabled={!newAgentId}>
			<Plus class="size-4" /><span class="group-data-[collapsible=icon]:hidden">New conversation</span>
		</Button>

		<div class="group-data-[collapsible=icon]:hidden">
			<Select.Root type="single" bind:value={newAgentId}>
				<Select.Trigger class="w-full bg-background/60" size="sm">
					<Bot class="size-4 text-muted-foreground" /><span class="truncate">{selectedAgent?.name ?? 'Choose agent'}</span>
				</Select.Trigger>
				<Select.Content>
					{#each agents as agent}
						<Select.Item value={agent.id} label={agent.name}>
							<span>{agent.name}</span><span class="text-xs text-muted-foreground">{agent.model}</span>
						</Select.Item>
					{/each}
				</Select.Content>
			</Select.Root>
		</div>
	</Sidebar.Header>

	<Sidebar.Content>
		<Sidebar.Group class="min-h-0 flex-1">
			<Sidebar.GroupLabel>Conversations</Sidebar.GroupLabel>
			<div class="relative mb-2 px-2 group-data-[collapsible=icon]:hidden">
				<Search class="pointer-events-none absolute left-5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
				<Sidebar.Input bind:value={search} class="pl-8" placeholder="Search conversations" aria-label="Search conversations" />
			</div>
			<Sidebar.GroupContent>
				<Sidebar.Menu>
					{#if loading}
						{#each Array(6) as _}<Sidebar.MenuSkeleton showIcon />{/each}
					{:else if filtered.length === 0}
						<p class="px-3 py-5 text-center text-xs text-muted-foreground group-data-[collapsible=icon]:hidden">No conversations found.</p>
					{:else}
						{#each filtered as session (session.id)}
							<Sidebar.MenuItem>
								<Sidebar.MenuButton size="lg" isActive={selectedId === session.id} tooltipContent={session.title} onclick={() => selectSession(session)}>
									<MessageSquare />
									<span class="grid min-w-0 flex-1 gap-0.5">
										<span class="truncate font-medium">{session.title}</span>
										<span class="text-xs font-normal text-muted-foreground">{session.message_count?.toLocaleString() ?? '0'} messages</span>
									</span>
								</Sidebar.MenuButton>
							</Sidebar.MenuItem>
						{/each}
					{/if}
				</Sidebar.Menu>
			</Sidebar.GroupContent>
		</Sidebar.Group>
	</Sidebar.Content>

	<Sidebar.Footer class="p-3">
		<div class="flex items-center gap-3 rounded-xl px-2 py-2 group-data-[collapsible=icon]:justify-center group-data-[collapsible=icon]:px-0">
			<span class="relative flex size-2 shrink-0">
				{#if bootstrap?.runtime.ready}<span class="absolute inline-flex size-full animate-ping rounded-full bg-emerald-400 opacity-50"></span>{/if}
				<span class:!bg-emerald-500={bootstrap?.runtime.ready} class="relative inline-flex size-2 rounded-full bg-muted-foreground/40"></span>
			</span>
			<div class="grid min-w-0 flex-1 group-data-[collapsible=icon]:hidden">
				<span class="text-xs font-medium">{bootstrap?.runtime.ready ? 'Runtime ready' : 'Runtime unavailable'}</span>
				<span class="truncate text-[11px] capitalize text-muted-foreground">{bootstrap?.runtime.connection_kind ?? 'Connecting'}</span>
			</div>
		</div>
	</Sidebar.Footer>
	<Sidebar.Rail />
</Sidebar.Root>
