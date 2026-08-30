<script lang="ts">
	import { Bot, Brain, LayoutDashboard, ListTodo, MessageSquare, Settings2 } from '@lucide/svelte';
	import * as Sidebar from '#lib/components/ui/sidebar/index.js';
	import type { Component } from 'svelte';
	import type { WorkspaceSection } from '#lib/workspace.js';

	let {
		active,
		onNavigate
	}: {
		active: WorkspaceSection;
		onNavigate: (section: WorkspaceSection) => void;
	} = $props();

	const primary: { id: WorkspaceSection; label: string; icon: Component }[] = [
		{ id: 'overview', label: 'Overview', icon: LayoutDashboard },
		{ id: 'conversations', label: 'Conversations', icon: MessageSquare },
		{ id: 'work', label: 'Work', icon: ListTodo },
		{ id: 'memory', label: 'Memory', icon: Brain }
	];
</script>

<Sidebar.Root collapsible="none" class="hidden w-52! shrink-0 border-r border-sidebar-border bg-sidebar md:flex">
	<Sidebar.Content class="px-2 py-3">
		<Sidebar.Group>
			<Sidebar.GroupLabel>Workspace</Sidebar.GroupLabel>
			<Sidebar.GroupContent>
				<Sidebar.Menu>
					{#each primary as item}
						<Sidebar.MenuItem>
							<Sidebar.MenuButton isActive={active === item.id} class="data-[active=true]:bg-primary/10 data-[active=true]:text-primary" onclick={() => onNavigate(item.id)}>
								<item.icon />
								<span>{item.label}</span>
							</Sidebar.MenuButton>
						</Sidebar.MenuItem>
					{/each}
				</Sidebar.Menu>
			</Sidebar.GroupContent>
		</Sidebar.Group>

		<Sidebar.Group>
			<Sidebar.GroupLabel>Runtime</Sidebar.GroupLabel>
			<Sidebar.GroupContent>
				<Sidebar.Menu>
					<Sidebar.MenuItem>
						<Sidebar.MenuButton isActive={active === 'agents'} class="data-[active=true]:bg-primary/10 data-[active=true]:text-primary" onclick={() => onNavigate('agents')}>
							<Bot /><span>Agents</span>
						</Sidebar.MenuButton>
					</Sidebar.MenuItem>
				</Sidebar.Menu>
			</Sidebar.GroupContent>
		</Sidebar.Group>
	</Sidebar.Content>

	<Sidebar.Footer class="p-3">
		<Sidebar.Menu>
			<Sidebar.MenuItem>
				<Sidebar.MenuButton isActive={active === 'settings'} class="data-[active=true]:bg-primary/10 data-[active=true]:text-primary" onclick={() => onNavigate('settings')}>
					<Settings2 /><span>Settings</span>
				</Sidebar.MenuButton>
			</Sidebar.MenuItem>
		</Sidebar.Menu>
	</Sidebar.Footer>
</Sidebar.Root>
