<script lang="ts">
	import { GitBranch, LoaderCircle } from '@lucide/svelte';
	import { turinWeb } from '#lib/api/client.js';
	import type { Session } from '#lib/api/contracts.js';
	import { Button } from '#lib/components/ui/button/index.js';
	import * as DropdownMenu from '#lib/components/ui/dropdown-menu/index.js';
	import * as Tooltip from '#lib/components/ui/tooltip/index.js';

	let { session, onOpen }: { session: Session; onOpen: (sessionId: string, turnId?: string) => void } = $props();
	let open = $state(false);
	let loading = $state(false);
	let loadedFor = $state<string | null>(null);
	let children = $state<Session[]>([]);
	let error = $state<string | null>(null);

	$effect(() => {
		if (!open || loadedFor === session.id) return;
		const controller = new AbortController();
		loading = true;
		error = null;
		void turinWeb.listLinkedSessions(session.id, 50, 0, controller.signal)
			.then((page) => {
				if (controller.signal.aborted) return;
				children = page.sessions;
				loadedFor = session.id;
			})
			.catch((cause: unknown) => {
				if (controller.signal.aborted) return;
				error = cause instanceof Error ? cause.message : 'Linked conversations could not be loaded.';
			})
			.finally(() => {
				if (!controller.signal.aborted) loading = false;
			});
		return () => controller.abort();
	});

	function openConversation(sessionId: string, turnId?: string) {
		open = false;
		onOpen(sessionId, turnId);
	}
</script>

<DropdownMenu.Root bind:open>
	<Tooltip.Root>
		<Tooltip.Trigger>
			{#snippet child({ props })}
				<DropdownMenu.Trigger>
					{#snippet child({ props: triggerProps })}
						<Button {...props} {...triggerProps} variant="ghost" size="icon" aria-label="Conversation relationships"><GitBranch class="size-4" /></Button>
					{/snippet}
				</DropdownMenu.Trigger>
			{/snippet}
		</Tooltip.Trigger>
		<Tooltip.Content>Conversation relationships</Tooltip.Content>
	</Tooltip.Root>
	<DropdownMenu.Content align="end" class="w-80">
		<DropdownMenu.Label>Conversation relationships</DropdownMenu.Label>
		{#if session.parent_session_id}
			<DropdownMenu.Separator />
			<DropdownMenu.Label class="text-xs font-normal text-muted-foreground">Parent</DropdownMenu.Label>
			<DropdownMenu.Item onclick={() => openConversation(session.parent_session_id!, session.origin_turn_id ?? undefined)}>
				<GitBranch />
				<div class="min-w-0">
					<p class="truncate font-medium">{session.parent_title ?? 'Parent conversation'}</p>
					<p class="text-xs text-muted-foreground">Open at the delegation point</p>
				</div>
			</DropdownMenu.Item>
		{/if}
		<DropdownMenu.Separator />
		<DropdownMenu.Label class="text-xs font-normal text-muted-foreground">Linked from this conversation</DropdownMenu.Label>
		{#if loading}
			<div class="flex items-center gap-2 px-3 py-3 text-sm text-muted-foreground"><LoaderCircle class="size-4 animate-spin" />Loading relationships</div>
		{:else if error}
			<p class="px-3 py-3 text-sm text-destructive">{error}</p>
		{:else if children.length === 0}
			<p class="px-3 py-3 text-sm text-muted-foreground">No linked conversations</p>
		{:else}
			{#each children as child (child.id)}
				<DropdownMenu.Item onclick={() => openConversation(child.id)}>
					<GitBranch />
					<div class="min-w-0">
						<p class="truncate font-medium">{child.title}</p>
						<p class="truncate text-xs text-muted-foreground">{child.relation_kind?.replaceAll('_', ' ') ?? 'Linked thread'}</p>
					</div>
				</DropdownMenu.Item>
			{/each}
		{/if}
	</DropdownMenu.Content>
</DropdownMenu.Root>
