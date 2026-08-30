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

function generatedMessage(sessionId: string, index: number): ConversationMessage {
	const role = index % 2 === 0 ? 'user' : Math.floor(index / 2) % 7 === 5 ? 'tool' : 'assistant';
	const turn = Math.floor(index / 2) + 1;
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
		['session-welcome', 'Building a focused Turin workspace', 'default', 8, null],
		['session-context', 'Context window strategy', 'default', 18, null],
		['session-release', 'Preparing the next Turin release', 'default', 42, null],
		['session-docs', 'Consumer-facing documentation', 'scout', 7, null],
		['session-security', 'Tool authorization review', 'default', 64, null],
		['session-storage', 'Persistence quality and integrity', 'scout', 126, 'linked'],
		['session-branches', 'Branching and linked sessions', 'default', 31, null],
		['session-web', 'Web product direction', 'scout', 83, null],
		['session-ui', 'Conversation interface polish', 'default', 14, null],
		['session-performance', 'Runtime memory and latency', 'scout', 220, 'linked'],
		['session-empty', 'Ideas to revisit later', 'default', 0, null],
		['session-research', 'Runtime architecture review', 'reviewer', 32, 'linked'],
		['session-long', `${largeMessageCount.toLocaleString()} message window test`, 'default', largeMessageCount, null]
	] as const;
	const counts = new Map<string, number>(sessionSpecs.map(([id, , , count]) => [id, count]));
	const sessions: Session[] = sessionSpecs.map(([id, title, agentId, count, relationKind], index) => ({
		id,
		title,
		agent_id: agentId,
		created_at: timestamp(index * 12),
		message_count: count,
		visibility: 'private',
		relation_kind: relationKind
	}));

	return {
		agents: [
			{ id: 'default', name: 'Turin', provider: 'minimax', model: 'MiniMax-M3', harness_id: 'default', enabled: true },
			{ id: 'scout', name: 'Scout', provider: 'minimax', model: 'MiniMax-M2.7', harness_id: 'default', enabled: true },
			{ id: 'reviewer', name: 'Reviewer', provider: 'minimax', model: 'MiniMax-M2.7', harness_id: 'research', enabled: true }
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
				workItem('worklist-runtime', { id: 'item-context', title: 'Validate bounded context retrieval', kind: 'task', status: 'active', priority: 60, prompt: 'Exercise a long session and verify bounded ancestry retrieval.', claim_agent_id: 'default', claim_session_id: 'session-context', claim_execution_id: 'execution-context', claim_heartbeat_unix_ms: Date.now(), claimed_at: timestamp(55), updated_at: timestamp(56) }),
				workItem('worklist-runtime', { id: 'item-permissions', title: 'Recheck delegated tool permissions', kind: 'review', status: 'paused', priority: 45, paused: true, pause_reason: 'Awaiting operator input', prompt: 'Confirm the child agent receives no broader authority than its parent.', updated_at: timestamp(48) }),
				workItem('worklist-runtime', { id: 'item-release', title: 'Publish persistence findings', kind: 'task', status: 'done', priority: 20, after: ['item-integrity'], completed_at: timestamp(50), updated_at: timestamp(50) })
			],
			'worklist-web': [
				workItem('worklist-web', { id: 'item-workspace', title: 'Build a useful workspace shell', kind: 'task', status: 'active', priority: 90, prompt: 'Turn the exploratory Work surface into an operator workflow.', claim_agent_id: 'default', claim_session_id: 'session-web', claim_execution_id: 'execution-web', claim_heartbeat_unix_ms: 1, claimed_at: timestamp(61), updated_at: timestamp(62) }),
				workItem('worklist-web', { id: 'item-memory', title: 'Design scalable memory exploration', kind: 'task', status: 'pending', priority: 70, prompt: 'Design search-first memory inspection without loading all records.', updated_at: timestamp(63) })
			]
		},
		memories: [
			{ id: 'memory-architecture', scope_kind: 'harness', scope_key: 'default', content: 'Keep Turin core unopinionated; clients own presentation and navigation state.', storage: 'durable', weight: 1, retrieval_count: 12, created_at: timestamp(8) },
			{ id: 'memory-ui', scope_kind: 'agent', scope_key: 'default', content: 'Prefer focused product workflows over diagnostic dashboards that expose every runtime detail.', storage: 'durable', weight: 0.85, retrieval_count: 7, created_at: timestamp(20) }
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
