import type { Agent, ConversationMessage, Harness, Session } from '../../src/lib/api/contracts.js';

export type MockScenario = {
	agents: Agent[];
	harnesses: Harness[];
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
	const counts = new Map([
		['session-welcome', 8],
		['session-research', 32],
		['session-long', largeMessageCount]
	]);
	const sessions: Session[] = [
		{
			id: 'session-welcome',
			title: 'Building a focused Turin workspace',
			agent_id: 'default',
			created_at: timestamp(0),
			message_count: counts.get('session-welcome') ?? 0
		},
		{
			id: 'session-research',
			title: 'Runtime architecture review',
			agent_id: 'reviewer',
			created_at: timestamp(12),
			message_count: counts.get('session-research') ?? 0
		},
		{
			id: 'session-long',
			title: `${largeMessageCount.toLocaleString()} message window test`,
			agent_id: 'default',
			created_at: timestamp(24),
			message_count: largeMessageCount
		}
	];

	return {
		agents: [
			{ id: 'default', name: 'Turin', provider: 'minimax', model: 'MiniMax-M3', harness_id: 'default', enabled: true },
			{ id: 'reviewer', name: 'Reviewer', provider: 'minimax', model: 'MiniMax-M2.7', harness_id: 'research', enabled: true }
		],
		harnesses: [
			{ id: 'default', name: 'Default', bound_agent_ids: ['default'], has_ui: false },
			{ id: 'research', name: 'Research Desk', bound_agent_ids: ['reviewer'], has_ui: true }
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
