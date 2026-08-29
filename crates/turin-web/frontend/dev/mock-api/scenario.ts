import type { Agent, ConversationMessage, Session } from '../../src/lib/api/contracts.js';

export type MockScenario = {
	agents: Agent[];
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
	const role = index % 2 === 0 ? 'user' : 'assistant';
	const turn = Math.floor(index / 2) + 1;
	const content = role === 'user'
		? `Investigate development checkpoint ${turn} and identify the smallest useful next step.`
		: `Checkpoint ${turn} is bounded and reviewable. The next step is to validate the behavior against the runtime contract before expanding the surface.`;
	return {
		id: `${sessionId}-message-${index + 1}`,
		turn_id: `${sessionId}-turn-${turn}`,
		role,
		content,
		created_at: timestamp(index),
		token_count: Math.ceil(content.length / 4)
	};
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
			{ id: 'default', name: 'Turin', provider: 'minimax', model: 'MiniMax-M3', enabled: true },
			{ id: 'reviewer', name: 'Reviewer', provider: 'minimax', model: 'MiniMax-M2.7', enabled: true }
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
