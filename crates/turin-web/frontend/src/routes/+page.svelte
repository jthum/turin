<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { CircleAlert, Trash2, X } from '@lucide/svelte';
	import * as AlertDialog from '#lib/components/ui/alert-dialog/index.js';
	import { Button } from '#lib/components/ui/button/index.js';
	import * as Sidebar from '#lib/components/ui/sidebar/index.js';
	import MessageComposer from '#lib/components/product/message-composer.svelte';
	import ConversationTranscript from '#lib/components/product/conversation-transcript.svelte';
	import CapabilityWorkspace from '#lib/components/product/capability-workspace.svelte';
	import ConversationDashboard from '#lib/components/product/conversation-dashboard.svelte';
	import HarnessBar from '#lib/components/product/harness-bar.svelte';
	import SessionRail from '#lib/components/product/session-rail.svelte';
	import WorkspaceNav from '#lib/components/product/workspace-nav.svelte';
	import WorkspaceOverview from '#lib/components/product/workspace-overview.svelte';
	import WorkWorkspace from '#lib/components/product/work-workspace.svelte';
	import type { Agent, ConversationMessage, Harness, Memory, SearchHit, Session, WorkItem, WorkItemControlAction, Worklist } from '#lib/api/contracts.js';
	import { turinWeb } from '#lib/api/client.js';
	import type { StreamConnectionState } from '#lib/api/client.js';
	import type { WorkspaceSection } from '#lib/workspace.js';

	const PAGE_SIZE = 80;
	const MAX_RESIDENT = 240;
	const DRAFT_SESSION_PREFIX = 'draft:';
	let agents = $state<Agent[]>([]);
	let harnesses = $state<Harness[]>([]);
	let sessions = $state<Session[]>([]);
	let hasMoreSessions = $state(false);
	let worklists = $state<Worklist[]>([]);
	let selectedWorklist = $state<Worklist | null>(null);
	let workItems = $state<WorkItem[]>([]);
	let selectedWorkItem = $state<WorkItem | null>(null);
	let loadingWorkItems = $state(false);
	let controllingWorkItem = $state(false);
	let memories = $state<Memory[]>([]);
	let memoryTotal = $state(0);
	let loadingMoreMemories = $state(false);
	let loadedSections = $state<WorkspaceSection[]>([]);
	let loadingSections = $state<WorkspaceSection[]>([]);
	let selectedHarnessId = $state('');
	let activeSection = $state<WorkspaceSection>('overview');
	let selected = $state<Session | null>(null);
	let messages = $state<ConversationMessage[]>([]);
	let newestOffset = $state(0);
	let olderOffset = $state(0);
	let messageTotal = $state(0);
	let hasOlder = $state(false);
	let hasNewer = $state(false);
	let loading = $state(true);
	let loadingMoreSessions = $state(false);
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
	let focusedMessageId = $state<string | null>(null);
	let streamState = $state<StreamConnectionState>('connecting');
	let workspaceSearchOpen = $state(false);
	let WorkspaceSearchDialog = $state<typeof import('#lib/components/product/workspace-search.svelte').default | null>(null);
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

	async function openWorkspaceSearch() {
		WorkspaceSearchDialog ??= (await import('#lib/components/product/workspace-search.svelte')).default;
		workspaceSearchOpen = true;
	}

	function handleWorkspaceShortcut(event: KeyboardEvent) {
		if ((event.metaKey || event.ctrlKey) && event.key.toLocaleLowerCase() === 'k') {
			event.preventDefault();
			if (workspaceSearchOpen) workspaceSearchOpen = false;
			else void openWorkspaceSearch();
		}
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

	function matchingMessage(hit: SearchHit, query: string) {
		const normalized = query.toLocaleLowerCase();
		return messages.find((message) =>
			message.turn_id === hit.turn_id
			&& (hit.role === null || message.role === hit.role)
			&& (!normalized || message.content.toLocaleLowerCase().includes(normalized))
		) ?? messages.find((message) => message.turn_id === hit.turn_id);
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
			hasMoreSessions = sessionPage.has_more;
			const remembered = localStorage.getItem('turin.selectedHarness');
			selectedHarnessId = harnesses.some((harness) => harness.id === remembered)
				? remembered ?? ''
				: (harnesses.find((harness) => harness.id === 'default')?.id ?? harnesses[0]?.id ?? '');
			const initialAgents = agents.filter((agent) => agent.harness_id === selectedHarnessId);
			newAgentId = initialAgents[0]?.id ?? '';
		} catch (cause) {
			showError(cause, 'Turin could not be reached.');
		} finally {
			loading = false;
		}
	}

	async function selectHarness(harnessId: string) {
		if (harnessId === selectedHarnessId) return;
		selectedHarnessId = harnessId;
		activeSection = 'overview';
		localStorage.setItem('turin.selectedHarness', harnessId);
		const harnessAgents = agents.filter((agent) => agent.harness_id === harnessId);
		newAgentId = harnessAgents[0]?.id ?? '';
		cancelWindowRequest();
		unsubscribe?.();
		unsubscribe = null;
		selected = null;
		messages = [];
		messageTotal = 0;
		streamState = 'connecting';
	}

	function navigate(section: WorkspaceSection) {
		activeSection = section;
		cancelWindowRequest();
		unsubscribe?.();
		unsubscribe = null;
		selected = null;
		messages = [];
		messageTotal = 0;
		if (section === 'work') {
			selectedWorklist = null;
			selectedWorkItem = null;
			workItems = [];
		}
		void loadSection(section);
	}

	async function loadSection(section: WorkspaceSection) {
		if (!['work', 'memory'].includes(section) || loadedSections.includes(section) || loadingSections.includes(section)) return;
		loadingSections = [...loadingSections, section];
		try {
			if (section === 'work') worklists = (await turinWeb.listWorklists()).worklists;
			if (section === 'memory') {
				const page = await turinWeb.listMemories();
				memories = page.memories;
				memoryTotal = page.total;
			}
			loadedSections = [...loadedSections, section];
		} catch (cause) {
			showError(cause, `${section === 'work' ? 'Worklists' : 'Memories'} could not be loaded.`);
		} finally {
			loadingSections = loadingSections.filter((item) => item !== section);
		}
	}

	async function openWorklist(worklist: Worklist) {
		selectedWorklist = worklist;
		selectedWorkItem = null;
		workItems = [];
		loadingWorkItems = true;
		try {
			workItems = (await turinWeb.listWorklistItems(worklist.id)).items;
		} catch (cause) {
			showError(cause, 'Worklist items could not be loaded.');
		} finally {
			loadingWorkItems = false;
		}
	}

	async function openWorkItem(item: WorkItem) {
		selectedWorkItem = item;
		try {
			selectedWorkItem = await turinWeb.getWorkItem(item.id);
		} catch (cause) {
			showError(cause, 'Work item details could not be loaded.');
		}
	}

	async function controlWorkItem(action: WorkItemControlAction, reason?: string) {
		if (!selectedWorkItem || controllingWorkItem) return;
		controllingWorkItem = true;
		try {
			const updated = await turinWeb.controlWorkItem(selectedWorkItem.id, action, reason);
			selectedWorkItem = updated;
			workItems = workItems.map((item) => item.id === updated.id ? updated : item);
		} catch (cause) {
			showError(cause, 'The work item could not be updated.');
		} finally {
			controllingWorkItem = false;
		}
	}

	async function openWorkItemSession(item: WorkItem) {
		if (!item.claim_session_id || !item.claim_agent_id) return;
		const agent = agents.find((candidate) => candidate.id === item.claim_agent_id);
		if (agent && agent.harness_id !== selectedHarnessId) await selectHarness(agent.harness_id);
		const existing = sessions.find((session) => session.id === item.claim_session_id);
		const session = existing ?? {
			id: item.claim_session_id,
			title: item.title,
			agent_id: item.claim_agent_id,
			created_at: item.claimed_at ?? item.created_at,
			message_count: null,
			visibility: 'private',
			relation_kind: 'work_item'
		};
		if (!existing) sessions = [session, ...sessions];
		await selectSession(session);
	}

	async function loadMoreMemories() {
		if (loadingMoreMemories || memories.length >= memoryTotal) return;
		loadingMoreMemories = true;
		try {
			const page = await turinWeb.listMemories(100, memories.length);
			memories = [...memories, ...page.memories];
			memoryTotal = page.total;
		} catch (cause) {
			showError(cause, 'More memories could not be loaded.');
		} finally {
			loadingMoreMemories = false;
		}
	}

	async function loadMoreSessions() {
		if (loadingMoreSessions || !hasMoreSessions) return;
		loadingMoreSessions = true;
		try {
			const page = await turinWeb.listSessions(60, sessions.length);
			sessions = [...sessions, ...page.sessions];
			hasMoreSessions = page.has_more;
		} catch (cause) {
			showError(cause, 'More conversations could not be loaded.');
		} finally {
			loadingMoreSessions = false;
		}
	}

	async function selectSession(
		session: Session,
		signal?: AbortSignal,
		anchor?: { hit: SearchHit; query: string }
	) {
		activeSection = 'conversations';
		cancelWindowRequest();
		unsubscribe?.();
		unsubscribe = null;
		selected = session;
		messages = [];
		focusedMessageId = null;
		newestOffset = 0;
		olderOffset = 0;
		loadingMessages = true;
		error = null;
		try {
			const page = await turinWeb.loadMessages(session.id, {
				limit: PAGE_SIZE,
				turnId: anchor?.hit.turn_id ?? undefined,
				signal
			});
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
			if (anchor) {
				const target = matchingMessage(anchor.hit, anchor.query);
				focusedMessageId = target?.id ?? null;
				if (target) await transcriptView?.restoreMessageAnchor(target.id, (transcript?.clientHeight ?? 0) * 0.28);
			} else {
				await scrollToBottom();
			}
		} catch (cause) {
			showError(cause, 'The conversation could not be loaded.');
		} finally {
			loadingMessages = false;
		}
	}

	function beginConversation(agentId = newAgentId) {
		if (!agentId) return;
		activeSection = 'conversations';
		cancelWindowRequest();
		unsubscribe?.();
		unsubscribe = null;
		newAgentId = agentId;
		selected = {
			id: `${DRAFT_SESSION_PREFIX}${agentId}`,
			title: 'New conversation',
			agent_id: agentId,
			created_at: new Date().toISOString(),
			message_count: 0,
			visibility: 'private',
			relation_kind: null
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
		focusedMessageId = null;
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

	async function jumpToSearchHit(hit: SearchHit, query: string) {
		if (!selected || hit.turn_id === null) return;
		const resident = matchingMessage(hit, query);
		if (resident) {
			cancelWindowRequest();
			focusedMessageId = resident.id;
			await transcriptView?.restoreMessageAnchor(resident.id, (transcript?.clientHeight ?? 0) * 0.28);
			return;
		}

		const request = beginWindowRequest();
		try {
			const page = await turinWeb.loadMessages(selected.id, {
				limit: PAGE_SIZE,
				turnId: hit.turn_id,
				signal: request.controller.signal
			});
			if (!isCurrentWindowRequest(request)) return;
			messages = page.messages;
			newestOffset = page.offset;
			olderOffset = page.offset + page.messages.length;
			messageTotal = page.total;
			hasNewer = newestOffset > 0;
			hasOlder = olderOffset < page.total;
			const target = matchingMessage(hit, query);
			focusedMessageId = target?.id ?? null;
			if (target) await transcriptView?.restoreMessageAnchor(target.id, (transcript?.clientHeight ?? 0) * 0.28);
		} catch (cause) {
			if (isCurrentWindowRequest(request) && !isAbortError(cause)) showError(cause, 'That search result could not be loaded.');
		} finally {
			finishWindowRequest(request);
		}
	}

	async function openWorkspaceSearchHit(hit: SearchHit, query: string) {
		const targetAgent = agents.find((agent) => agent.id === hit.agent_id);
		if (targetAgent && targetAgent.harness_id !== selectedHarnessId) {
			await selectHarness(targetAgent.harness_id);
		}
		const existing = sessions.find((candidate) => candidate.id === hit.session_id);
		const session = existing ?? {
			id: hit.session_id,
			title: hit.title ?? 'Untitled conversation',
			agent_id: hit.agent_id,
			created_at: hit.created_at,
			message_count: null,
			visibility: 'private',
			relation_kind: null
		};
		if (!existing) sessions = [session, ...sessions];
		if (selected?.id === session.id && hit.turn_id !== null) {
			await jumpToSearchHit(hit, query);
			return;
		}
		await selectSession(
			session,
			undefined,
			hit.turn_id === null ? undefined : { hit, query }
		);
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

<svelte:window onkeydown={handleWorkspaceShortcut} />

<Sidebar.Provider class="h-svh min-h-0! flex-col overflow-hidden [--global-bar-height:3.5rem]">
	<HarnessBar {harnesses} {selectedHarnessId} session={selected} section={activeSection} onSelect={selectHarness} onNavigate={navigate} onSearch={() => void openWorkspaceSearch()} onDelete={() => selected && requestDelete(selected)} />
	<div class="flex min-h-0 flex-1">
		<WorkspaceNav active={activeSection} onNavigate={navigate} />
		{#if activeSection === 'conversations' && selected}
			<SessionRail agents={visibleAgents} sessions={visibleSessions} selectedId={selected.id} {loading} bind:search bind:newAgentId onCreate={beginConversation} onSelect={selectSession} onDelete={requestDelete} />
		{/if}

	<Sidebar.Inset class="h-full min-w-0 overflow-hidden bg-background">
		{#if error}
			<div class="flex items-center gap-3 border-b border-destructive/20 bg-destructive/5 px-4 py-2.5 text-sm text-destructive sm:px-5">
				<CircleAlert class="size-4 shrink-0" /><span class="min-w-0 flex-1">{error}</span>
				<Button variant="ghost" size="icon-sm" onclick={() => error = null} aria-label="Dismiss error"><X class="size-4" /></Button>
			</div>
		{/if}

		{#if activeSection === 'overview'}
			<WorkspaceOverview sessions={visibleSessions} agents={visibleAgents} onCreate={beginConversation} onSelect={selectSession} onNavigate={navigate} />
		{:else if activeSection === 'conversations' && !selected}
			<ConversationDashboard sessions={visibleSessions} agents={visibleAgents} {loading} loadingMore={loadingMoreSessions} hasMore={hasMoreSessions} onCreate={beginConversation} onSelect={selectSession} onDelete={requestDelete} onLoadMore={loadMoreSessions} />
		{:else if activeSection === 'conversations'}
			<div class="min-h-0 flex-1">
				<ConversationTranscript bind:this={transcriptView} bind:ref={transcript} session={selected} agentName={selectedAgentName} {messages} loading={loadingMessages} loadingWindow={loadingOlder} {hasOlder} {hasNewer} {messageTotal} {newestOffset} {olderOffset} {submitting} {streamMessageId} {focusedMessageId} onLoadOlder={loadOlder} onLoadNewer={loadNewer} onJumpToPosition={jumpToConversationPosition} onJumpToEnd={jumpToLatest} onSearchHit={jumpToSearchHit} onFork={forkFromMessage} onCreate={beginConversation} />
			</div>
			{#if selected}<MessageComposer bind:value={composer} agentName={selectedAgentName} model={agents.find((agent) => agent.id === selected?.agent_id)?.model ?? selected.agent_id} {submitting} connected={streamState === 'open'} onSend={sendMessage} />{/if}
		{:else if activeSection === 'work'}
			<WorkWorkspace {worklists} {selectedWorklist} {workItems} {selectedWorkItem} loading={loadingSections.includes('work') || loadingWorkItems} controlling={controllingWorkItem} onOpenWorklist={openWorklist} onCloseWorklist={() => { selectedWorklist = null; selectedWorkItem = null; workItems = []; }} onOpenItem={openWorkItem} onCloseItem={() => selectedWorkItem = null} onControlItem={controlWorkItem} onOpenSession={openWorkItemSession} />
		{:else}
			{#key activeSection}
				<CapabilityWorkspace section={activeSection} agents={visibleAgents} harness={harnesses.find((harness) => harness.id === selectedHarnessId)} {memories} {memoryTotal} loading={loadingSections.includes(activeSection)} {loadingMoreMemories} onLoadMoreMemories={loadMoreMemories} />
			{/key}
		{/if}
		</Sidebar.Inset>
	</div>
</Sidebar.Provider>

{#if WorkspaceSearchDialog}
	<WorkspaceSearchDialog bind:open={workspaceSearchOpen} {agents} onSelect={openWorkspaceSearchHit} />
{/if}

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
