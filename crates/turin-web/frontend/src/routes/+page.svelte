<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { CircleAlert, MoreHorizontal, RefreshCw, Trash2, X } from '@lucide/svelte';
	import * as AlertDialog from '#lib/components/ui/alert-dialog/index.js';
	import { Button } from '#lib/components/ui/button/index.js';
	import * as DropdownMenu from '#lib/components/ui/dropdown-menu/index.js';
	import * as Sidebar from '#lib/components/ui/sidebar/index.js';
	import MessageComposer from '#lib/components/product/message-composer.svelte';
	import ConversationTranscript from '#lib/components/product/conversation-transcript.svelte';
	import SessionRail from '#lib/components/product/session-rail.svelte';
	import type { Agent, Bootstrap, ConversationMessage, Session } from '#lib/api/contracts.js';
	import { loadBootstrap } from '#lib/api/bootstrap.js';
	import { turinWeb } from '#lib/api/client.js';
	import type { StreamConnectionState } from '#lib/api/client.js';

	const PAGE_SIZE = 80;
	const MAX_RESIDENT = 240;
	let bootstrap = $state<Bootstrap | null>(null);
	let agents = $state<Agent[]>([]);
	let sessions = $state<Session[]>([]);
	let selected = $state<Session | null>(null);
	let messages = $state<ConversationMessage[]>([]);
	let newestOffset = $state(0);
	let olderOffset = $state(0);
	let messageTotal = $state(0);
	let hasOlder = $state(false);
	let hasNewer = $state(false);
	let loading = $state(true);
	let loadingMessages = $state(false);
	let loadingOlder = $state(false);
	let submitting = $state(false);
	let error = $state<string | null>(null);
	let composer = $state('');
	let search = $state('');
	let newAgentId = $state('');
	let deleteDialogOpen = $state(false);
	let streamMessageId = $state<string | null>(null);
	let streamState = $state<StreamConnectionState>('connecting');
	let unsubscribe = $state<(() => void) | null>(null);
	let transcript = $state<HTMLElement | null>(null);

	function showError(cause: unknown, fallback: string) {
		error = cause instanceof Error ? cause.message : fallback;
	}

	async function scrollToBottom(behavior: ScrollBehavior = 'instant') {
		await tick();
		transcript?.scrollTo({ top: transcript.scrollHeight, behavior });
	}

	async function initialize(signal?: AbortSignal) {
		loading = true;
		error = null;
		try {
			const [health, agentPage, sessionPage] = await Promise.all([
				loadBootstrap(signal), turinWeb.listAgents(signal), turinWeb.listSessions(60, 0, signal)
			]);
			bootstrap = health;
			agents = agentPage.agents;
			newAgentId = agents[0]?.id ?? '';
			sessions = sessionPage.sessions;
			if (sessions[0]) await selectSession(sessions[0], signal);
		} catch (cause) {
			showError(cause, 'Turin could not be reached.');
		} finally {
			loading = false;
		}
	}

	async function selectSession(session: Session, signal?: AbortSignal) {
		unsubscribe?.();
		unsubscribe = null;
		selected = session;
		messages = [];
		newestOffset = 0;
		olderOffset = 0;
		loadingMessages = true;
		error = null;
		try {
			const page = await turinWeb.loadMessages(session.id, PAGE_SIZE, 0, signal);
			messages = page.messages;
			olderOffset = page.messages.length;
			messageTotal = page.total;
			hasOlder = page.has_more;
			hasNewer = false;
			unsubscribe = turinWeb.subscribe(session.id, {
				'conversation.task.started': () => submitting = true,
				'conversation.message.started': (event) => {
					streamMessageId = event.message_id;
					messages = [...messages, {
						id: event.message_id, turn_id: event.request_id, role: 'assistant', content: '',
						created_at: new Date().toISOString(), token_count: null
					}];
				},
				'conversation.message.delta': (event) => {
					streamMessageId = event.message_id;
					const index = messages.findIndex((message) => message.id === event.message_id);
					if (index >= 0) messages[index] = { ...messages[index], content: messages[index].content + event.delta };
					else messages = [...messages, {
						id: event.message_id, turn_id: event.request_id, role: 'assistant', content: event.delta,
						created_at: new Date().toISOString(), token_count: null
					}];
					void scrollToBottom('smooth');
				},
				'conversation.task.completed': () => {
					streamMessageId = null;
					submitting = false;
					messageTotal += 1;
				},
				'conversation.task.failed': (event) => {
					streamMessageId = null;
					submitting = false;
					error = event.message;
				}
			}, (state) => streamState = state);
			await scrollToBottom();
		} catch (cause) {
			showError(cause, 'The conversation could not be loaded.');
		} finally {
			loadingMessages = false;
		}
	}

	async function loadOlder() {
		if (!selected || !hasOlder || loadingOlder) return;
		loadingOlder = true;
		const previousHeight = transcript?.scrollHeight ?? 0;
		try {
			const page = await turinWeb.loadMessages(selected.id, PAGE_SIZE, olderOffset);
			const combined = [...page.messages, ...messages];
			const droppedNewest = Math.max(0, combined.length - MAX_RESIDENT);
			messages = combined.slice(0, MAX_RESIDENT);
			newestOffset += droppedNewest;
			olderOffset += page.messages.length;
			hasOlder = page.has_more;
			hasNewer = newestOffset > 0;
			await tick();
			transcript?.scrollTo({ top: transcript.scrollHeight - previousHeight });
		} catch (cause) {
			showError(cause, 'Older messages could not be loaded.');
		} finally {
			loadingOlder = false;
		}
	}

	async function loadNewer() {
		if (!selected || !hasNewer || loadingOlder) return;
		loadingOlder = true;
		try {
			const offset = Math.max(0, newestOffset - PAGE_SIZE);
			const page = await turinWeb.loadMessages(selected.id, PAGE_SIZE, offset);
			const combined = [...messages, ...page.messages];
			const droppedOldest = Math.max(0, combined.length - MAX_RESIDENT);
			messages = combined.slice(-MAX_RESIDENT);
			newestOffset = offset;
			olderOffset = Math.max(messages.length, olderOffset - droppedOldest);
			hasNewer = newestOffset > 0;
			hasOlder = olderOffset < page.total;
			await scrollToBottom();
		} catch (cause) {
			showError(cause, 'Newer messages could not be loaded.');
		} finally {
			loadingOlder = false;
		}
	}

	async function returnToLatest() {
		if (!selected || newestOffset === 0) return;
		const page = await turinWeb.loadMessages(selected.id, PAGE_SIZE, 0);
		messages = page.messages;
		newestOffset = 0;
		olderOffset = page.messages.length;
		messageTotal = page.total;
		hasNewer = false;
		hasOlder = page.has_more;
	}

	async function createSession() {
		if (!newAgentId) return;
		try {
			const created = await turinWeb.createSession(newAgentId);
			sessions = [created.session, ...sessions];
			await selectSession(created.session);
		} catch (cause) {
			showError(cause, 'A new conversation could not be created.');
		}
	}

	async function deleteSelected() {
		if (!selected) return;
		try {
			await turinWeb.deleteSession(selected.id);
			sessions = sessions.filter((item) => item.id !== selected?.id);
			selected = null;
			messages = [];
			deleteDialogOpen = false;
			if (sessions[0]) await selectSession(sessions[0]);
		} catch (cause) {
			showError(cause, 'The conversation could not be deleted.');
		}
	}

	async function sendMessage() {
		const content = composer.trim();
		if (!selected || !content || submitting || streamState !== 'open') return;
		try {
			await returnToLatest();
		} catch (cause) {
			showError(cause, 'The latest messages could not be loaded.');
			return;
		}
		composer = '';
		error = null;
		submitting = true;
		const optimistic: ConversationMessage = {
			id: `pending-${Date.now()}`, turn_id: 'pending', role: 'user', content,
			created_at: new Date().toISOString(), token_count: Math.ceil(content.length / 4)
		};
		messages = [...messages, optimistic];
		messageTotal += 1;
		await scrollToBottom('smooth');
		try {
			await turinWeb.submitMessage(selected.id, content);
		} catch (cause) {
			messages = messages.filter((message) => message.id !== optimistic.id);
			messageTotal = Math.max(0, messageTotal - 1);
			composer = content;
			submitting = false;
			showError(cause, 'The message could not be submitted.');
		}
	}

	onMount(() => {
		const controller = new AbortController();
		void initialize(controller.signal);
		return () => { controller.abort(); unsubscribe?.(); };
	});
</script>

<Sidebar.Provider>
	<SessionRail {bootstrap} {agents} {sessions} selectedId={selected?.id ?? null} {loading} bind:search bind:newAgentId onCreate={createSession} onSelect={selectSession} />

	<Sidebar.Inset class="h-svh min-w-0 overflow-hidden bg-background">
		<header class="flex h-16 shrink-0 items-center gap-3 border-b border-border/70 bg-background/90 px-4 backdrop-blur-xl sm:px-5">
			<Sidebar.Trigger class="-ml-1" />
			<div class="grid min-w-0 flex-1 gap-0.5">
				<strong class="truncate text-sm font-semibold">{selected?.title ?? 'Your Turin workspace'}</strong>
				{#if selected}
					<span class:!text-amber-600={streamState !== 'open'} class="truncate text-xs text-muted-foreground">
						{streamState === 'open' ? (agents.find((agent) => agent.id === selected?.agent_id)?.name ?? selected.agent_id) : 'Connecting live updates…'}
					</span>
				{/if}
			</div>
			{#if selected}
				<DropdownMenu.Root>
					<DropdownMenu.Trigger>
						{#snippet child({ props })}<Button {...props} variant="ghost" size="icon" aria-label="Conversation actions"><MoreHorizontal class="size-4" /></Button>{/snippet}
					</DropdownMenu.Trigger>
					<DropdownMenu.Content align="end">
						<DropdownMenu.Label>Conversation</DropdownMenu.Label>
						<DropdownMenu.Separator />
						<DropdownMenu.Item variant="destructive" onclick={() => deleteDialogOpen = true}><Trash2 />Delete conversation</DropdownMenu.Item>
					</DropdownMenu.Content>
				</DropdownMenu.Root>
			{:else}
				<Button variant="ghost" size="icon" onclick={() => initialize()} aria-label="Refresh"><RefreshCw class="size-4" /></Button>
			{/if}
		</header>

		{#if error}
			<div class="flex items-center gap-3 border-b border-destructive/20 bg-destructive/5 px-4 py-2.5 text-sm text-destructive sm:px-5">
				<CircleAlert class="size-4 shrink-0" /><span class="min-w-0 flex-1">{error}</span>
				<Button variant="ghost" size="icon-sm" onclick={() => error = null} aria-label="Dismiss error"><X class="size-4" /></Button>
			</div>
		{/if}

		<div class="min-h-0 flex-1">
			<ConversationTranscript bind:ref={transcript} session={selected} {messages} loading={loadingMessages} loadingWindow={loadingOlder} {hasOlder} {hasNewer} {messageTotal} {submitting} {streamMessageId} onLoadOlder={loadOlder} onLoadNewer={loadNewer} onCreate={createSession} />
		</div>
		{#if selected}<MessageComposer bind:value={composer} agentId={selected.agent_id} model={agents.find((agent) => agent.id === selected?.agent_id)?.model ?? selected.agent_id} {submitting} connected={streamState === 'open'} onSend={sendMessage} />{/if}
	</Sidebar.Inset>
</Sidebar.Provider>

<AlertDialog.Root bind:open={deleteDialogOpen}>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Media class="bg-destructive/10 text-destructive"><Trash2 /></AlertDialog.Media>
			<AlertDialog.Title>Delete this conversation?</AlertDialog.Title>
			<AlertDialog.Description>“{selected?.title}” and its stored turns will be permanently removed. This action cannot be undone.</AlertDialog.Description>
		</AlertDialog.Header>
		<AlertDialog.Footer>
			<AlertDialog.Cancel>Keep conversation</AlertDialog.Cancel>
			<AlertDialog.Action variant="destructive" onclick={deleteSelected}>Delete</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>

<style>
	:global(body) { margin: 0; overflow: hidden; }
</style>
