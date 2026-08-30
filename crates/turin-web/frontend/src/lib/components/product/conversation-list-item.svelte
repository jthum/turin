<script lang="ts">
	import { MoreHorizontal, Trash2 } from '@lucide/svelte';
	import * as DropdownMenu from '#lib/components/ui/dropdown-menu/index.js';
	import * as Sidebar from '#lib/components/ui/sidebar/index.js';
	import type { Session } from '#lib/api/contracts.js';

	let {
		session, active, onSelect, onDelete
	}: {
		session: Session;
		active: boolean;
		onSelect: () => void;
		onDelete: () => void;
	} = $props();
	let menuOpen = $state(false);

	function openContextMenu(event: MouseEvent) {
		event.preventDefault();
		menuOpen = true;
	}
</script>

<DropdownMenu.Root bind:open={menuOpen}>
	<Sidebar.MenuItem oncontextmenu={openContextMenu}>
		<Sidebar.MenuButton isActive={active} onclick={onSelect} class="h-8 pr-8 text-[13px]">
			<span class="truncate font-medium leading-none">{session.title}</span>
		</Sidebar.MenuButton>
		<DropdownMenu.Trigger>
			{#snippet child({ props })}
				<Sidebar.MenuAction {...props} showOnHover aria-label={`Actions for ${session.title}`}>
					<MoreHorizontal />
				</Sidebar.MenuAction>
			{/snippet}
		</DropdownMenu.Trigger>
		<DropdownMenu.Content align="start" side="right" class="w-48">
			<DropdownMenu.Item onclick={onSelect}>Open conversation</DropdownMenu.Item>
			<DropdownMenu.Separator />
			<DropdownMenu.Item variant="destructive" onclick={onDelete}><Trash2 />Delete conversation</DropdownMenu.Item>
		</DropdownMenu.Content>
	</Sidebar.MenuItem>
</DropdownMenu.Root>
