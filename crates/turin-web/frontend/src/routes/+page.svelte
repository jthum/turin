<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { PanelLeft, RefreshCw, Trash2 } from '@lucide/svelte';
	import { Button } from '#lib/components/ui/button/index.js';
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
	let sidebarOpen = $state(false);
	let streamMessageId = $state<string | null>(null);
	let streamState = $state<StreamConnectionState>('connecting');
	let unsubscribe = $state<(() => void) | null>(null);
	let transcript = $state<HTMLDivElement>();

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
		sidebarOpen = false;
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
		if (!selected || !confirm(`Delete “${selected.title}”? This cannot be undone.`)) return;
		try {
			await turinWeb.deleteSession(selected.id);
			sessions = sessions.filter((item) => item.id !== selected?.id);
			selected = null;
			messages = [];
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

<div class="workspace" class:sidebar-open={sidebarOpen}>
	<button class="sidebar-scrim" aria-label="Close conversations" onclick={() => sidebarOpen = false}></button>
	<SessionRail {bootstrap} {agents} {sessions} selectedId={selected?.id ?? null} {loading} bind:search bind:newAgentId onCreate={createSession} onSelect={selectSession} />

	<main class="conversation-shell">
		<header class="conversation-header">
			<Button class="mobile-menu" variant="ghost" size="icon" onclick={() => sidebarOpen = true} aria-label="Open conversations"><PanelLeft /></Button>
			<div class="conversation-title"><strong>{selected?.title ?? 'Your Turin workspace'}</strong>{#if selected}<span class:reconnecting={streamState !== 'open'}>{streamState === 'open' ? (agents.find((agent) => agent.id === selected?.agent_id)?.name ?? selected.agent_id) : 'Connecting live updates…'}</span>{/if}</div>
			{#if selected}<Button variant="ghost" size="icon" onclick={deleteSelected} aria-label="Delete conversation"><Trash2 /></Button>{:else}<Button variant="ghost" size="icon" onclick={() => initialize()} aria-label="Refresh"><RefreshCw /></Button>{/if}
		</header>
		{#if error}<div class="error-banner"><span>{error}</span><button onclick={() => error = null}>Dismiss</button></div>{/if}
		<ConversationTranscript bind:ref={transcript} session={selected} {messages} loading={loadingMessages} loadingWindow={loadingOlder} {hasOlder} {hasNewer} {messageTotal} {submitting} {streamMessageId} onLoadOlder={loadOlder} onLoadNewer={loadNewer} onCreate={createSession} />
		{#if selected}<MessageComposer bind:value={composer} agentId={selected.agent_id} model={agents.find((agent) => agent.id === selected?.agent_id)?.model ?? selected.agent_id} {submitting} connected={streamState === 'open'} onSend={sendMessage} />{/if}
	</main>
</div>

<style>
	:global(body) { margin: 0; overflow: hidden; background: #f7f7f5; }
	:global(button), :global(input), :global(textarea), :global(select) { font: inherit; }
	.workspace { --rail: 292px; display: grid; grid-template-columns: var(--rail) minmax(0, 1fr); height: 100dvh; color: #191918; background: #f7f7f5; }
	.conversation-header :global(svg) { width: 17px; height: 17px; }
	.conversation-shell { display: grid; min-width: 0; min-height: 0; grid-template-rows: auto auto minmax(0, 1fr) auto; background: #fbfbfa; }
	.conversation-header { display: flex; height: 68px; align-items: center; gap: 12px; padding: 0 22px; border-bottom: 1px solid #e9e9e5; background: #fbfbfae8; backdrop-filter: blur(16px); }
	.conversation-title { display: grid; min-width: 0; flex: 1; gap: 2px; }
	.conversation-title strong { overflow: hidden; font-size: 13px; font-weight: 620; text-overflow: ellipsis; white-space: nowrap; }
	.conversation-title span { color: #92928c; font-size: 10px; }
	.conversation-title span.reconnecting { color: #a66d22; }
	:global(.mobile-menu) { display: none; }
	.error-banner { display: flex; align-items: center; justify-content: space-between; gap: 16px; border-bottom: 1px solid #f0d4ce; background: #fff4f1; padding: 9px 22px; color: #9b392d; font-size: 12px; }
	.error-banner button { border: 0; background: transparent; color: inherit; font-size: 11px; font-weight: 650; cursor: pointer; }
	.sidebar-scrim { display: none; }
	:global(.spin) { animation: spin 1s linear infinite; }
	@keyframes spin { to { transform: rotate(360deg); } }

	@media (max-width: 760px) {
		.workspace { display: block; }
		:global(.session-rail) { position: fixed; z-index: 30; inset: 0 auto 0 0; width: min(86vw, 320px); transform: translateX(-102%); box-shadow: 20px 0 60px #0002; transition: transform .22s ease; }
		.sidebar-open :global(.session-rail) { transform: translateX(0); }
		.sidebar-scrim { position: fixed; z-index: 20; inset: 0; border: 0; background: #15151366; opacity: 0; pointer-events: none; transition: opacity .2s; }
		.sidebar-open .sidebar-scrim { display: block; opacity: 1; pointer-events: auto; }
		.conversation-shell { height: 100dvh; }
		:global(.mobile-menu) { display: inline-flex; }
		.conversation-header { height: 58px; padding: 0 12px; }
	}
</style>
