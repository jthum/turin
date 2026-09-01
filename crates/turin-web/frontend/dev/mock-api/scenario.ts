import type { Agent, ConversationMessage, Harness, Memory, Session, WorkItem, Worklist } from '../../src/lib/api/contracts.js';

export type MockScenario = {
	agents: Agent[];
	harnesses: Harness[];
	worklists: Worklist[];
	workItems: Record<string, WorkItem[]>;
	memories: Memory[];
	sessions: Session[];
	messageCount(sessionId: string): number;
	messageAt(sessionId: string, index: number): ConversationMessage;
	responseFor(prompt: string): MockResponse;
};

export type MockStreamMode = 'normal' | 'slow' | 'error' | 'interrupt';

export type MockResponse = {
	text: string;
	mode: MockStreamMode;
	chunkDelayMs: number;
};

const BASE_TIME = Date.parse('2026-08-29T08:00:00.000Z');

export const MOCK_SESSION_IDS = {
	welcome: '019d52d1-1e01-7b4c-8f39-000000000001',
	context: '019d52d1-1e01-7b4c-8f39-000000000002',
	release: '019d52d1-1e01-7b4c-8f39-000000000003',
	docs: '019d52d1-1e01-7b4c-8f39-000000000004',
	security: '019d52d1-1e01-7b4c-8f39-000000000005',
	storage: '019d52d1-1e01-7b4c-8f39-000000000006',
	branches: '019d52d1-1e01-7b4c-8f39-000000000007',
	web: '019d52d1-1e01-7b4c-8f39-000000000008',
	ui: '019d52d1-1e01-7b4c-8f39-000000000009',
	performance: '019d52d1-1e01-7b4c-8f39-00000000000a',
	empty: '019d52d1-1e01-7b4c-8f39-00000000000b',
	research: '019d52d1-1e01-7b4c-8f39-00000000000c',
	long: '019d52d1-1e01-7b4c-8f39-00000000000d'
} as const;

function timestamp(index: number): string {
	return new Date(BASE_TIME + index * 45_000).toISOString();
}

function workItem(worklistId: string, item: Pick<WorkItem, 'id' | 'title' | 'kind' | 'status' | 'priority' | 'updated_at'> & Partial<WorkItem>): WorkItem {
	return {
		worklist_id: worklistId,
		parent_id: null,
		prompt: null,
		action_name: null,
		paused: false,
		pause_reason: null,
		pause_until_unix_ms: null,
		after: [],
		claim_agent_id: null,
		claim_session_id: null,
		claim_execution_id: null,
		claim_heartbeat_unix_ms: null,
		claimed_at: null,
		completed_at: null,
		failure_reason: null,
		created_at: item.updated_at,
		...item
	};
}

function memory(item: Pick<Memory, 'id' | 'scope_kind' | 'scope_key' | 'content' | 'created_at'> & Partial<Memory>): Memory {
	return {
		metadata: null,
		storage: 'lexical_only',
		embedding_key: null,
		embedding_dimensions: null,
		weight: 1,
		retrieval_count: 0,
		last_retrieved_at: null,
		superseded_at: null,
		superseded_by_id: null,
		...item
	};
}

const MESSAGE_PATTERN = [
	['user', 0], ['assistant', 0],
	['user', 1], ['assistant', 1],
	['user', 2], ['assistant', 2],
	['user', 3], ['assistant', 3],
	['user', 4], ['assistant', 4],
	['user', 5], ['tool', 5], ['assistant', 5],
	['user', 6], ['assistant', 6]
] as const;
const TURN_START_OFFSETS = [0, 2, 4, 6, 8, 10, 13] as const;

export function mockMessageIndexForTurn(turn: number) {
	const normalized = Math.max(1, turn) - 1;
	return Math.floor(normalized / 7) * MESSAGE_PATTERN.length + TURN_START_OFFSETS[normalized % 7];
}

export function mockTurnForMessageIndex(index: number) {
	const [, turnOffset] = MESSAGE_PATTERN[index % MESSAGE_PATTERN.length];
	return Math.floor(index / MESSAGE_PATTERN.length) * 7 + turnOffset + 1;
}

function generatedMessage(sessionId: string, index: number): ConversationMessage {
	const [role] = MESSAGE_PATTERN[index % MESSAGE_PATTERN.length];
	const turn = mockTurnForMessageIndex(index);
	const content = role === 'user'
		? userPrompt(turn)
		: role === 'tool'
			? `{"status":"success","files_checked":${3 + turn % 8},"warnings":${turn % 3},"checkpoint":${turn}}`
			: assistantResponse(turn);
	return {
		id: `${sessionId}-message-${index + 1}`,
		turn_id: `${sessionId}-turn-${turn}`,
		role,
		content,
		created_at: timestamp(index),
		token_count: Math.ceil(content.length / 4),
		...(role === 'assistant' ? {
			metrics: {
				input_tokens: 1_240 + turn * 17,
				output_tokens: Math.ceil(content.length / 4),
				cache_read_input_tokens: turn % 3 === 0 ? 840 + turn : 0,
				provider: 'minimax',
				model: 'MiniMax-M3'
			},
			...(turn % 4 === 0 ? { reasoning: { duration_ms: 2_400 + turn * 31, summary: 'Compared the current behavior with the runtime contract and ruled out broader changes.' } } : {})
		} : {})
	};
}

function userPrompt(turn: number): string {
	const prompts = [
		`Investigate development checkpoint ${turn} and identify the smallest useful next step.`,
		`Can you compare the current behavior at checkpoint ${turn} with the intended runtime contract?`,
		`Review checkpoint ${turn}. Focus on correctness first, then tell me what can remain simple.`
	];
	return prompts[turn % prompts.length];
}

function assistantResponse(turn: number): string {
	const responses = [
		`Checkpoint ${turn} is **bounded and reviewable**. The current behavior matches the primary path, but one edge needs attention.\n\n- Preserve the existing contract.\n- Add a focused regression test.\n- Avoid expanding the abstraction until another caller needs it.\n\nThe smallest next step is to validate the boundary before changing implementation ownership.`,
		`## Review outcome\n\nThe implementation is coherent, with two practical follow-ups:\n\n1. Verify the failure path independently.\n2. Keep the successful path allocation-free where possible.\n3. Document the invariant beside the owning module.\n\n> The important constraint is that a UI concern must not leak into Turin's kernel semantics.`,
		`The useful distinction is between **resident state** and **durable state**. Resident state may be discarded; durable state must survive a runtime restart.\n\n| Concern | Current shape | Recommendation |\n| --- | --- | --- |\n| Context | Bounded window | Keep |\n| Navigation | Latest intent wins | Keep |\n| Diagnostics | On demand | Avoid hot-path work |\n\nThis keeps the normal path lean while leaving advanced inspection possible.`,
		`I checked the relevant path and the fix can stay local:\n\n\`\`\`rust\nlet window = store.load_recent_path(target, limit).await?;\n\`\`\`\n\nNo new dependency or runtime-global state is required. The remaining risk is stale data arriving after a newer navigation request, which should be handled at the client boundary.`
	];
	return responses[turn % responses.length];
}

export function createMockScenario(): MockScenario {
	const largeMessageCount = Math.max(0, Number(process.env.TURIN_MOCK_MESSAGE_COUNT ?? 10_000));
	const streamMode = mockStreamMode(process.env.TURIN_MOCK_STREAM);
	const sessionSpecs = [
		[MOCK_SESSION_IDS.welcome, 'Building a focused Turin workspace', 'default', 8, null, null],
		[MOCK_SESSION_IDS.context, 'Context window strategy', 'default', 17, null, null],
		[MOCK_SESSION_IDS.release, 'Preparing the next Turin release', 'default', 43, null, null],
		[MOCK_SESSION_IDS.docs, 'Consumer-facing documentation', 'scout', 8, null, null],
		[MOCK_SESSION_IDS.security, 'Tool authorization review', 'default', 64, null, null],
		[MOCK_SESSION_IDS.storage, 'Persistence quality and integrity', 'scout', 126, 'delegated', MOCK_SESSION_IDS.context],
		[MOCK_SESSION_IDS.branches, 'Branching and linked sessions', 'default', 30, null, null],
		[MOCK_SESSION_IDS.web, 'Web product direction', 'scout', 83, null, null],
		[MOCK_SESSION_IDS.ui, 'Conversation interface polish', 'default', 15, null, null],
		[MOCK_SESSION_IDS.performance, 'Runtime memory and latency', 'scout', 220, 'delegated', MOCK_SESSION_IDS.storage],
		[MOCK_SESSION_IDS.empty, 'Ideas to revisit later', 'default', 0, null, null],
		[MOCK_SESSION_IDS.research, 'Runtime architecture review', 'reviewer', 32, 'delegated', MOCK_SESSION_IDS.docs],
		[MOCK_SESSION_IDS.long, `${largeMessageCount.toLocaleString()} message window test`, 'default', largeMessageCount, null, null]
	] as const;
	const counts = new Map<string, number>(sessionSpecs.map(([id, , , count]) => [id, count]));
	const titles = new Map<string, string>(sessionSpecs.map(([id, title]) => [id, title]));
	const sessions: Session[] = sessionSpecs.map(([id, title, agentId, count, relationKind, parentSessionId], index) => ({
		id,
		title,
		agent_id: agentId,
		created_at: timestamp(index * 12),
		message_count: count,
		visibility: parentSessionId ? 'contextual' : 'top_level',
		relation_kind: relationKind,
		parent_session_id: parentSessionId,
		parent_title: parentSessionId ? titles.get(parentSessionId) ?? null : null,
		origin_turn_id: parentSessionId ? `${parentSessionId}-turn-2` : null,
		latest_message_preview: count > 0 ? generatedMessage(id, count - 1).content.replace(/\s+/g, ' ').slice(0, 180) : null,
		latest_message_role: count > 0 ? generatedMessage(id, count - 1).role : null,
		latest_message_created_at: count > 0 ? generatedMessage(id, count - 1).created_at : null
	}));

	return {
		agents: [
			{ id: 'default', name: 'Turin', provider: 'minimax', model: 'MiniMax-M3', harness_id: 'default', enabled: true, running: true, active_tasks: 1, queued_tasks: 0, awaiting_results: 0 },
			{ id: 'scout', name: 'Scout', provider: 'minimax', model: 'MiniMax-M2.7', harness_id: 'default', enabled: true, running: false, active_tasks: 0, queued_tasks: 0, awaiting_results: 0 },
			{ id: 'reviewer', name: 'Reviewer', provider: 'minimax', model: 'MiniMax-M2.7', harness_id: 'research', enabled: true, running: true, active_tasks: 0, queued_tasks: 2, awaiting_results: 1 }
		],
		harnesses: [
			{ id: 'default', name: 'General', bound_agent_ids: ['default', 'scout'], has_ui: false },
			{ id: 'research', name: 'Research Desk', bound_agent_ids: ['reviewer'], has_ui: true }
		],
		worklists: [
			{ id: 'worklist-runtime', name: 'Runtime quality', scope: 'global', created_at: timestamp(4), updated_at: timestamp(52) },
			{ id: 'worklist-web', name: 'Web product runway', scope: 'harness:default', created_at: timestamp(18), updated_at: timestamp(60) }
		],
		workItems: {
			'worklist-runtime': [
				workItem('worklist-runtime', { id: 'item-integrity', title: 'Review persistence integrity failures', kind: 'task', status: 'pending', priority: 80, prompt: 'Audit malformed rows and graph-parent failures, then summarize any integrity gaps.', action_name: 'runtime.integrity_review', updated_at: timestamp(54) }),
				workItem('worklist-runtime', { id: 'item-context', title: 'Validate bounded context retrieval', kind: 'task', status: 'active', priority: 60, prompt: 'Exercise a long session and verify bounded ancestry retrieval.', claim_agent_id: 'default', claim_session_id: MOCK_SESSION_IDS.context, claim_execution_id: 'execution-context', claim_heartbeat_unix_ms: Date.now(), claimed_at: timestamp(55), updated_at: timestamp(56) }),
				workItem('worklist-runtime', { id: 'item-permissions', title: 'Recheck delegated tool permissions', kind: 'review', status: 'paused', priority: 45, paused: true, pause_reason: 'Awaiting operator input', prompt: 'Confirm the child agent receives no broader authority than its parent.', updated_at: timestamp(48) }),
				workItem('worklist-runtime', { id: 'item-index', title: 'Rebuild the stale search index', kind: 'task', status: 'failed', priority: 35, prompt: 'Rebuild the persisted search index and verify the new generation before activation.', failure_reason: 'The source database was locked by another maintenance task.', updated_at: timestamp(47) }),
				workItem('worklist-runtime', { id: 'item-release', title: 'Publish persistence findings', kind: 'task', status: 'done', priority: 20, after: ['item-integrity'], completed_at: timestamp(50), updated_at: timestamp(50) })
			],
			'worklist-web': [
				workItem('worklist-web', { id: 'item-workspace', title: 'Build a useful workspace shell', kind: 'task', status: 'active', priority: 90, prompt: 'Turn the exploratory Work surface into an operator workflow.', claim_agent_id: 'default', claim_session_id: MOCK_SESSION_IDS.web, claim_execution_id: 'execution-web', claim_heartbeat_unix_ms: 1, claimed_at: timestamp(61), updated_at: timestamp(62) }),
				workItem('worklist-web', { id: 'item-memory', title: 'Design scalable memory exploration', kind: 'task', status: 'pending', priority: 70, prompt: 'Design search-first memory inspection without loading all records.', updated_at: timestamp(63) })
			]
		},
		memories: [
			memory({ id: 'memory-architecture', scope_kind: 'harness', scope_key: 'default', content: 'Keep Turin core unopinionated; clients own presentation and navigation state.', metadata: { source_task: 'architecture-review', tags: ['principle'] }, storage: 'embedded', embedding_key: 'minilm', embedding_dimensions: 384, weight: 1, retrieval_count: 12, last_retrieved_at: timestamp(51), created_at: timestamp(8) }),
			memory({ id: 'memory-ui', scope_kind: 'agent', scope_key: 'default', content: 'Prefer focused product workflows over diagnostic dashboards that expose every runtime detail.', metadata: { source_task: 'ui-review', tags: ['product'] }, weight: 0.85, retrieval_count: 7, last_retrieved_at: timestamp(45), created_at: timestamp(20) }),
			memory({ id: 'memory-context', scope_kind: 'session', scope_key: MOCK_SESSION_IDS.context, content: 'Context retrieval should stop at the nearest semantic checkpoint before filling the remaining token budget.', metadata: { tags: ['context', 'performance'] }, weight: 1.2, retrieval_count: 18, created_at: timestamp(28) }),
			memory({ id: 'memory-old-ui', scope_kind: 'agent', scope_key: 'default', content: 'Render every runtime diagnostic on the default dashboard.', superseded_at: timestamp(19), superseded_by_id: 'memory-ui', weight: 0.6, retrieval_count: 2, created_at: timestamp(6) })
		],
		sessions,
		messageCount: (sessionId) => counts.get(sessionId) ?? 0,
		messageAt: generatedMessage,
		responseFor: (prompt) => ({
			text: `I received “${prompt.slice(0, 90)}${prompt.length > 90 ? '…' : ''}”.

### What happened

- The response streamed through Turin's browser event contract.
- The interface kept the active transcript window bounded.
- Markdown was rendered only after the stream completed.

You can use the mock to test **timing**, failure behavior, and long conversations without changing application components.`,
			mode: streamMode,
			chunkDelayMs: streamMode === 'slow' ? 350 : 35
		})
	};
}

function mockStreamMode(value: string | undefined): MockStreamMode {
	return value === 'slow' || value === 'error' || value === 'interrupt' ? value : 'normal';
}
