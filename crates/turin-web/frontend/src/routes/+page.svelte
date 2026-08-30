<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { CircleAlert, Trash2, X } from '@lucide/svelte';
	import * as AlertDialog from '#lib/components/ui/alert-dialog/index.js';
	import { Button } from '#lib/components/ui/button/index.js';
	import * as Sidebar from '#lib/components/ui/sidebar/index.js';
	import MessageComposer from '#lib/components/product/message-composer.svelte';
	import ConversationTranscript from '#lib/components/product/conversation-transcript.svelte';
	import HarnessBar from '#lib/components/product/harness-bar.svelte';
	import SessionRail from '#lib/components/product/session-rail.svelte';
	import type { Agent, ConversationMessage, Harness, Session } from '#lib/api/contracts.js';
	import { turinWeb } from '#lib/api/client.js';
	import type { StreamConnectionState } from '#lib/api/client.js';

	const PAGE_SIZE = 80;
	const MAX_RESIDENT = 240;
	const DRAFT_SESSION_PREFIX = 'draft:';
	let agents = $state<Agent[]>([]);
	let harnesses = $state<Harness[]>([]);
	let sessions = $state<Session[]>([]);
	let selectedHarnessId = $state('');
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
	let deleteTarget = $state<Session | null>(null);
	let streamMessageId = $state<string | null>(null);
	let streamState = $state<StreamConnectionState>('connecting');
	let unsubscribe = $state<(() => void) | null>(null);
	let transcript = $state<HTMLElement | null>(null);
	let transcriptView = $state<{
		restoreMessageAnchor: (id: string, viewportOffset: number) => Promise<void>;
		scrollToEnd: (behavior?: ScrollBehavior) => Promise<void>;
	} | null>(null);
	let windowRequest: { id: number; controller: AbortController } | null = null;
	let nextWindowRequestId = 0;
	let visibleAgents = $derived(agents.filter((agent) => agent.harness_id === selectedHarnessId));
	let visibleSessions = $derived(sessions.filter((session) => visibleAgents.some((agent) => agent.id === session.agent_id)));
	let selectedAgentName = $derived(agents.find((agent) => agent.id === selected?.agent_id)?.name ?? selected?.agent_id ?? 'Turin');

	function showError(cause: unknown, fallback: string) {
		error = cause instanceof Error ? cause.message : fallback;
	}

	function beginWindowRequest() {
		windowRequest?.controller.abort();
		const request = { id: ++nextWindowRequestId, controller: new AbortController() };
		windowRequest = request;
		loadingOlder = true;
		return request;
	}

	function isCurrentWindowRequest(request: { id: number }) {
		return windowRequest?.id === request.id;
	}

	function finishWindowRequest(request: { id: number }) {
		if (!isCurrentWindowRequest(request)) return;
		windowRequest = null;
		loadingOlder = false;
	}

	function cancelWindowRequest() {
		windowRequest?.controller.abort();
		windowRequest = null;
		loadingOlder = false;
	}

	function isAbortError(cause: unknown) {
		return cause instanceof DOMException && cause.name === 'AbortError';
	}

	function uniqueMessages(items: ConversationMessage[]) {
		const seen = new Set<string>();
		return items.filter((message) => {
			if (seen.has(message.id)) return false;
			seen.add(message.id);
			return true;
		});
	}

	async function scrollToBottom(behavior: ScrollBehavior = 'instant') {
		if (transcriptView) {
			await transcriptView.scrollToEnd(behavior);
			return;
		}
		await tick();
		transcript?.scrollTo({ top: transcript.scrollHeight, behavior });
	}

	function visibleMessageAnchor() {
		if (!transcript) return null;
		const viewportTop = transcript.getBoundingClientRect().top;
		const elements = transcript.querySelectorAll<HTMLElement>('[data-message-id]');
		const element = Array.from(elements).find((candidate) => candidate.getBoundingClientRect().bottom > viewportTop);
		return element ? { id: element.dataset.messageId ?? '', viewportOffset: element.getBoundingClientRect().top - viewportTop } : null;
	}

	async function restoreMessageAnchor(anchor: { id: string; viewportOffset: number } | null) {
		if (anchor) await transcriptView?.restoreMessageAnchor(anchor.id, anchor.viewportOffset);
	}

	async function initialize(signal?: AbortSignal) {
		loading = true;
		error = null;
		try {
			const [harnessPage, agentPage, sessionPage] = await Promise.all([
				turinWeb.listHarnesses(signal), turinWeb.listAgents(signal), turinWeb.listSessions(60, 0, signal)
			]);
			harnesses = harnessPage.harnesses;
			agents = agentPage.agents;
			sessions = sessionPage.sessions;
			const remembered = localStorage.getItem('turin.selectedHarness');
			selectedHarnessId = harnesses.some((harness) => harness.id === remembered)
				? remembered ?? ''
				: (harnesses.find((harness) => harness.id === 'default')?.id ?? harnesses[0]?.id ?? '');
			const initialAgents = agents.filter((agent) => agent.harness_id === selectedHarnessId);
			newAgentId = initialAgents[0]?.id ?? '';
			const initialSession = sessions.find((session) => initialAgents.some((agent) => agent.id === session.agent_id));
			if (initialSession) await selectSession(initialSession, signal);
		} catch (cause) {
			showError(cause, 'Turin could not be reached.');
		} finally {
			loading = false;
		}
	}

	async function selectHarness(harnessId: string) {
		if (harnessId === selectedHarnessId) return;
		selectedHarnessId = harnessId;
		localStorage.setItem('turin.selectedHarness', harnessId);
		const harnessAgents = agents.filter((agent) => agent.harness_id === harnessId);
		newAgentId = harnessAgents[0]?.id ?? '';
		const nextSession = sessions.find((session) => harnessAgents.some((agent) => agent.id === session.agent_id));
		if (nextSession) {
			await selectSession(nextSession);
			return;
		}
		cancelWindowRequest();
		unsubscribe?.();
		unsubscribe = null;
		selected = null;
		messages = [];
		messageTotal = 0;
		streamState = 'connecting';
	}

	async function selectSession(session: Session, signal?: AbortSignal) {
		cancelWindowRequest();
		unsubscribe?.();
		unsubscribe = null;
		selected = session;
		messages = [];
		newestOffset = 0;
		olderOffset = 0;
		loadingMessages = true;
		error = null;
		try {
			const page = await turinWeb.loadMessages(session.id, { limit: PAGE_SIZE, signal });
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

	function beginConversation(agentId = newAgentId) {
		if (!agentId) return;
		cancelWindowRequest();
		unsubscribe?.();
		unsubscribe = null;
		newAgentId = agentId;
		selected = {
			id: `${DRAFT_SESSION_PREFIX}${agentId}`,
			title: 'New conversation',
			agent_id: agentId,
			created_at: new Date().toISOString(),
			message_count: 0
		};
		messages = [];
		newestOffset = 0;
		olderOffset = 0;
		messageTotal = 0;
		hasOlder = false;
		hasNewer = false;
		streamState = 'open';
		error = null;
	}

	async function loadOlder() {
		if (!selected || !hasOlder || loadingOlder) return;
		const request = beginWindowRequest();
		const anchor = visibleMessageAnchor();
		try {
			const page = await turinWeb.loadMessages(selected.id, { limit: PAGE_SIZE, offset: olderOffset, total: messageTotal, signal: request.controller.signal });
			if (!isCurrentWindowRequest(request)) return;
			const combined = uniqueMessages([...page.messages, ...messages]);
			const droppedNewest = Math.max(0, combined.length - MAX_RESIDENT);
			messages = combined.slice(0, MAX_RESIDENT);
			newestOffset += droppedNewest;
			olderOffset = Math.max(olderOffset, page.offset + page.messages.length);
			hasOlder = olderOffset < page.total;
			hasNewer = newestOffset > 0;
			await restoreMessageAnchor(anchor);
		} catch (cause) {
			if (isCurrentWindowRequest(request) && !isAbortError(cause)) showError(cause, 'Older messages could not be loaded.');
		} finally {
			finishWindowRequest(request);
		}
	}

	async function loadNewer() {
		if (!selected || !hasNewer || loadingOlder) return;
		const request = beginWindowRequest();
		const anchor = visibleMessageAnchor();
		try {
			const offset = Math.max(0, newestOffset - PAGE_SIZE);
			const page = await turinWeb.loadMessages(selected.id, { limit: PAGE_SIZE, offset, total: messageTotal, signal: request.controller.signal });
			if (!isCurrentWindowRequest(request)) return;
			const combined = uniqueMessages([...messages, ...page.messages]);
			const droppedOldest = Math.max(0, combined.length - MAX_RESIDENT);
			messages = combined.slice(-MAX_RESIDENT);
			newestOffset = page.offset;
			olderOffset = newestOffset + messages.length;
			hasNewer = newestOffset > 0;
			hasOlder = olderOffset < page.total;
			await restoreMessageAnchor(anchor);
		} catch (cause) {
			if (isCurrentWindowRequest(request) && !isAbortError(cause)) showError(cause, 'Newer messages could not be loaded.');
		} finally {
			finishWindowRequest(request);
		}
	}

	async function returnToLatest(): Promise<boolean> {
		if (!selected) return false;
		if (newestOffset === 0) {
			cancelWindowRequest();
			return true;
		}
		const request = beginWindowRequest();
		try {
			const page = await turinWeb.loadMessages(selected.id, { limit: PAGE_SIZE, signal: request.controller.signal });
			if (!isCurrentWindowRequest(request)) return false;
			messages = page.messages;
			newestOffset = 0;
			olderOffset = page.messages.length;
			messageTotal = page.total;
			hasNewer = false;
			hasOlder = page.has_more;
			return true;
		} catch (cause) {
			if (isCurrentWindowRequest(request) && !isAbortError(cause)) throw cause;
			return false;
		} finally {
			finishWindowRequest(request);
		}
	}

	async function jumpToLatest() {
		if (await returnToLatest()) await scrollToBottom('smooth');
	}

	async function jumpToConversationPosition(position: number) {
		if (!selected || messageTotal === 0) return;
		const targetIndex = Math.round(Math.min(1, Math.max(0, position)) * (messageTotal - 1));
		const residentStart = messageTotal - olderOffset;
		const residentEnd = messageTotal - newestOffset;
		if (targetIndex >= residentStart && targetIndex < residentEnd) {
			cancelWindowRequest();
			const message = messages[targetIndex - residentStart];
			if (message) await transcriptView?.restoreMessageAnchor(message.id, (transcript?.clientHeight ?? 0) * 0.35);
			return;
		}

		const request = beginWindowRequest();
		try {
			const offset = Math.min(
				Math.max(0, messageTotal - PAGE_SIZE),
				Math.max(0, messageTotal - targetIndex - Math.ceil(PAGE_SIZE / 2))
			);
			const page = await turinWeb.loadMessages(selected.id, { limit: PAGE_SIZE, offset, total: messageTotal, signal: request.controller.signal });
			if (!isCurrentWindowRequest(request)) return;
			messages = page.messages;
			newestOffset = page.offset;
			olderOffset = page.offset + page.messages.length;
			messageTotal = page.total;
			hasNewer = newestOffset > 0;
			hasOlder = olderOffset < page.total;
			const firstIndex = page.total - olderOffset;
			const localIndex = Math.min(page.messages.length - 1, Math.max(0, targetIndex - firstIndex));
			const message = page.messages[localIndex];
			if (message) await transcriptView?.restoreMessageAnchor(message.id, (transcript?.clientHeight ?? 0) * 0.35);
		} catch (cause) {
			if (isCurrentWindowRequest(request) && !isAbortError(cause)) showError(cause, 'That point in the conversation could not be loaded.');
		} finally {
			finishWindowRequest(request);
		}
	}

	async function forkFromMessage(message: ConversationMessage, activate: boolean) {
		if (!selected || selected.id.startsWith(DRAFT_SESSION_PREFIX)) return;
		try {
			await turinWeb.createBranch(selected.id, message.turn_id, activate);
			if (activate) await selectSession(selected);
		} catch (cause) {
			showError(cause, 'The branch could not be created.');
		}
	}

	function requestDelete(session: Session) {
		deleteTarget = session;
		deleteDialogOpen = true;
	}

	async function deleteConversation() {
		const target = deleteTarget;
		if (!target) return;
		try {
			cancelWindowRequest();
			if (!target.id.startsWith(DRAFT_SESSION_PREFIX)) await turinWeb.deleteSession(target.id);
			sessions = sessions.filter((item) => item.id !== target.id);
			if (selected?.id === target.id) {
				selected = null;
				messages = [];
			}
			deleteDialogOpen = false;
			deleteTarget = null;
			const nextSession = sessions.find((session) => visibleAgents.some((agent) => agent.id === session.agent_id));
			if (!selected && nextSession) await selectSession(nextSession);
		} catch (cause) {
			showError(cause, 'The conversation could not be deleted.');
		}
	}

	async function sendMessage() {
		const content = composer.trim();
		if (!selected || !content || submitting || streamState !== 'open') return;
		let createdFromDraft: Session | null = null;
		if (selected.id.startsWith(DRAFT_SESSION_PREFIX)) {
			try {
				const created = await turinWeb.createSession(selected.agent_id);
				createdFromDraft = created.session;
				sessions = [created.session, ...sessions];
				await selectSession(created.session);
			} catch (cause) {
				showError(cause, 'A new conversation could not be created.');
				return;
			}
		}
		try {
			if (!await returnToLatest()) return;
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
			if (createdFromDraft) {
				unsubscribe?.();
				unsubscribe = null;
				sessions = sessions.filter((session) => session.id !== createdFromDraft?.id);
				try { await turinWeb.deleteSession(createdFromDraft.id); } catch { /* Preserve the submission error. */ }
				beginConversation(createdFromDraft.agent_id);
			}
			showError(cause, 'The message could not be submitted.');
		}
	}

	onMount(() => {
		const controller = new AbortController();
		void initialize(controller.signal);
		return () => { controller.abort(); cancelWindowRequest(); unsubscribe?.(); };
	});
</script>

<Sidebar.Provider class="h-svh min-h-0! flex-col overflow-hidden [--global-bar-height:3.5rem]">
	<HarnessBar {harnesses} {selectedHarnessId} session={selected} onSelect={selectHarness} onDelete={() => selected && requestDelete(selected)} />
	<div class="flex min-h-0 flex-1">
		<SessionRail agents={visibleAgents} sessions={visibleSessions} selectedId={selected?.id ?? null} {loading} bind:search bind:newAgentId onCreate={beginConversation} onSelect={selectSession} onDelete={requestDelete} />

	<Sidebar.Inset class="h-full min-w-0 overflow-hidden bg-background">
		{#if error}
			<div class="flex items-center gap-3 border-b border-destructive/20 bg-destructive/5 px-4 py-2.5 text-sm text-destructive sm:px-5">
				<CircleAlert class="size-4 shrink-0" /><span class="min-w-0 flex-1">{error}</span>
				<Button variant="ghost" size="icon-sm" onclick={() => error = null} aria-label="Dismiss error"><X class="size-4" /></Button>
			</div>
		{/if}

		<div class="min-h-0 flex-1">
			<ConversationTranscript bind:this={transcriptView} bind:ref={transcript} session={selected} agentName={selectedAgentName} {messages} loading={loadingMessages} loadingWindow={loadingOlder} {hasOlder} {hasNewer} {messageTotal} {newestOffset} {olderOffset} {submitting} {streamMessageId} onLoadOlder={loadOlder} onLoadNewer={loadNewer} onJumpToPosition={jumpToConversationPosition} onJumpToEnd={jumpToLatest} onFork={forkFromMessage} onCreate={beginConversation} />
		</div>
		{#if selected}<MessageComposer bind:value={composer} agentName={selectedAgentName} model={agents.find((agent) => agent.id === selected?.agent_id)?.model ?? selected.agent_id} {submitting} connected={streamState === 'open'} onSend={sendMessage} />{/if}
		</Sidebar.Inset>
	</div>
</Sidebar.Provider>

<AlertDialog.Root bind:open={deleteDialogOpen}>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Media class="bg-destructive/10 text-destructive"><Trash2 /></AlertDialog.Media>
			<AlertDialog.Title>Delete this conversation?</AlertDialog.Title>
			<AlertDialog.Description>“{deleteTarget?.title}” and its stored turns will be permanently removed. This action cannot be undone.</AlertDialog.Description>
		</AlertDialog.Header>
		<AlertDialog.Footer>
			<AlertDialog.Cancel>Keep conversation</AlertDialog.Cancel>
			<AlertDialog.Action variant="destructive" onclick={deleteConversation}>Delete</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>

<style>
	:global(body) { margin: 0; overflow: hidden; }
</style>
