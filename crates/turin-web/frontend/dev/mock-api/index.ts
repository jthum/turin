import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Plugin } from 'vite';
import type {
	ConversationEventMap,
	ConversationEventName,
	ConversationMessage,
	SearchHit,
	Session
} from '../../src/lib/api/contracts.js';
import { createMockScenario } from './scenario.js';

type Subscriber = { response: ServerResponse; sessionId: string };

function sendJson(response: ServerResponse, status: number, value: unknown): void {
	response.writeHead(status, {
		'Cache-Control': 'no-store',
		'Content-Type': 'application/json; charset=utf-8'
	});
	response.end(JSON.stringify(value));
}

async function readJson(request: IncomingMessage): Promise<Record<string, unknown>> {
	const chunks: Buffer[] = [];
	for await (const chunk of request) chunks.push(Buffer.from(chunk));
	return chunks.length === 0 ? {} : JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

export function turinMockApi(): Plugin {
	const scenario = createMockScenario();
	const sessions = new Map(scenario.sessions.map((session) => [session.id, session]));
	const appended = new Map<string, ConversationMessage[]>();
	const subscribers = new Set<Subscriber>();
	let nextId = 1;

	function publish<K extends ConversationEventName>(name: K, data: ConversationEventMap[K]): void {
		const frame = `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
		for (const subscriber of subscribers) {
			if (subscriber.sessionId === data.session_id) subscriber.response.write(frame);
		}
	}

	function sessionMessages(sessionId: string, limit: number, offset: number) {
		const generatedCount = scenario.messageCount(sessionId);
		const additions = appended.get(sessionId) ?? [];
		const total = generatedCount + additions.length;
		const resolvedOffset = Math.min(total, Math.max(0, offset));
		const end = Math.max(0, total - resolvedOffset);
		const start = Math.max(0, end - limit);
		const messages: ConversationMessage[] = [];
		for (let index = start; index < end; index += 1) {
			messages.push(
				index < generatedCount
					? scenario.messageAt(sessionId, index)
					: additions[index - generatedCount]
			);
		}
		return { messages, offset: resolvedOffset, total, has_more: start > 0 };
	}

	function sessionMessagesAroundTurn(sessionId: string, limit: number, turnId: string) {
		const generatedCount = scenario.messageCount(sessionId);
		const additions = appended.get(sessionId) ?? [];
		const total = generatedCount + additions.length;
		const generatedPrefix = `${sessionId}-turn-`;
		let targetIndex = turnId.startsWith(generatedPrefix)
			? (Number(turnId.slice(generatedPrefix.length)) - 1) * 2
			: additions.findIndex((message) => message.turn_id === turnId);
		if (!turnId.startsWith(generatedPrefix) && targetIndex >= 0) targetIndex += generatedCount;
		if (!Number.isFinite(targetIndex) || targetIndex < 0 || targetIndex >= total) return null;
		const start = Math.max(0, Math.min(total - limit, targetIndex - Math.floor(limit / 2)));
		const end = Math.min(total, start + limit);
		const messages: ConversationMessage[] = [];
		for (let index = start; index < end; index += 1) {
			messages.push(index < generatedCount ? scenario.messageAt(sessionId, index) : additions[index - generatedCount]);
		}
		return { messages, offset: total - end, total, has_more: start > 0 };
	}

	function workspaceSearch(query: string): SearchHit[] {
		const hits: SearchHit[] = [];
		for (const session of [...sessions.values()].reverse()) {
			if (session.title.toLowerCase().includes(query)) {
				hits.push({ kind: 'session', session_id: session.id, agent_id: session.agent_id, title: session.title, created_at: session.created_at, turn_id: null, turn_index: null, role: null, tool_name: null, event_type: null, snippet: session.title });
			}
			const total = scenario.messageCount(session.id);
			for (let index = total - 1; index >= 0 && hits.length < 50; index -= 1) {
				const message = scenario.messageAt(session.id, index);
				if (!message.content.toLowerCase().includes(query)) continue;
				hits.push({ kind: 'message', session_id: session.id, agent_id: session.agent_id, title: session.title, created_at: message.created_at, turn_id: message.turn_id, turn_index: Math.floor(index / 2), role: message.role, tool_name: null, event_type: null, snippet: message.content.slice(0, 220) });
			}
			if (hits.length >= 50) break;
		}
		if ('read_file persistence checkpoint'.includes(query)) {
			const session = sessions.get('session-storage');
			if (session) hits.push({ kind: 'tool_execution', session_id: session.id, agent_id: session.agent_id, title: session.title, created_at: session.created_at, turn_id: `${session.id}-turn-12`, turn_index: 11, role: null, tool_name: 'read_file', event_type: null, snippet: 'read_file persistence checkpoint and verify durable rows' });
		}
		if ('task completed runtime'.includes(query)) {
			const session = sessions.get('session-performance');
			if (session) hits.push({ kind: 'event', session_id: session.id, agent_id: session.agent_id, title: session.title, created_at: session.created_at, turn_id: `${session.id}-turn-8`, turn_index: 7, role: null, tool_name: null, event_type: 'task_completed', snippet: 'task_completed runtime diagnostic checkpoint' });
		}
		return hits.slice(0, 50);
	}

	return {
		name: 'turin-mock-api',
		configureServer(server) {
			server.middlewares.use(async (request, response, next) => {
				if (!request.url?.startsWith('/api/')) return next();
				const url = new URL(request.url, 'http://turin.local');
				const path = url.pathname;

				if (request.method === 'GET' && path === '/api/bootstrap') {
					return sendJson(response, 200, {
						web_version: 'mock',
						runtime: {
							connection_kind: 'mock', ready: true, version: 'mock', protocol_version: 1,
							issue_count: 0, agent_count: scenario.agents.length, harness_count: scenario.harnesses.length,
							running_agent_count: scenario.agents.length, active_task_count: 0
						}
					});
				}
				if (request.method === 'GET' && path === '/api/agents') {
					return sendJson(response, 200, { agents: scenario.agents });
				}
				if (request.method === 'GET' && path === '/api/harnesses') {
					return sendJson(response, 200, { harnesses: scenario.harnesses });
				}
				if (request.method === 'GET' && path === '/api/worklists') {
					return sendJson(response, 200, { worklists: scenario.worklists });
				}
				const worklistMatch = path.match(/^\/api\/worklists\/([^/]+)\/items$/);
				if (request.method === 'GET' && worklistMatch) {
					const worklistId = decodeURIComponent(worklistMatch[1]);
					return sendJson(response, 200, { worklist_id: worklistId, items: scenario.workItems[worklistId] ?? [] });
				}
				if (request.method === 'GET' && path === '/api/memories') {
					const limit = Number(url.searchParams.get('limit') ?? 100);
					const offset = Number(url.searchParams.get('offset') ?? 0);
					return sendJson(response, 200, {
						memories: scenario.memories.slice(offset, offset + limit), total: scenario.memories.length,
						offset, limit
					});
				}
				if (request.method === 'GET' && path === '/api/search/sessions') {
					const query = (url.searchParams.get('q') ?? '').trim().toLowerCase();
					const hits = [...sessions.values()].reverse()
						.filter((session) => session.title.toLowerCase().includes(query))
						.slice(0, 50)
						.map((session) => ({ kind: 'session' as const, session_id: session.id, agent_id: session.agent_id, title: session.title, created_at: session.created_at, turn_id: null, turn_index: null, role: null, tool_name: null, event_type: null, snippet: session.title }));
					return sendJson(response, 200, { hits });
				}
				if (request.method === 'GET' && path === '/api/search/workspace') {
					const query = (url.searchParams.get('q') ?? '').trim().toLowerCase();
					return sendJson(response, 200, { hits: workspaceSearch(query) });
				}
				if (request.method === 'GET' && path === '/api/sessions') {
					const limit = Number(url.searchParams.get('limit') ?? 50);
					const offset = Number(url.searchParams.get('offset') ?? 0);
					const all = [...sessions.values()].reverse();
					return sendJson(response, 200, {
						sessions: all.slice(offset, offset + limit), offset,
						has_more: offset + limit < all.length
					});
				}
				if (request.method === 'POST' && path === '/api/sessions') {
					const body = await readJson(request);
					const id = `session-created-${nextId++}`;
					const session: Session = {
						id, title: 'New conversation', agent_id: String(body.agent_id ?? 'default'),
						created_at: new Date().toISOString(), message_count: 0,
						visibility: 'private', relation_kind: null
					};
					sessions.set(id, session);
					return sendJson(response, 201, { session });
				}

				const match = path.match(/^\/api\/sessions\/([^/]+)(?:\/(messages|branches|search))?$/);
				if (match) {
					const sessionId = decodeURIComponent(match[1]);
					const session = sessions.get(sessionId);
					if (!session) return sendJson(response, 404, { error: 'Session not found' });
					if (request.method === 'GET' && match[2] === 'messages') {
						const limit = Math.min(200, Number(url.searchParams.get('limit') ?? 80));
						const turnId = url.searchParams.get('turn_id');
						const page = turnId
							? sessionMessagesAroundTurn(sessionId, limit, turnId)
							: sessionMessages(sessionId, limit, Number(url.searchParams.get('offset') ?? 0));
						return page
							? sendJson(response, 200, page)
							: sendJson(response, 404, { error: 'Turn not found on active path' });
					}
					if (request.method === 'GET' && match[2] === 'search') {
						const query = (url.searchParams.get('q') ?? '').trim().toLowerCase();
						const total = scenario.messageCount(sessionId);
						const extra = appended.get(sessionId) ?? [];
						const hits: SearchHit[] = [];
						for (const message of [...extra].reverse()) {
							if (hits.length >= 50 || !message.content.toLowerCase().includes(query)) continue;
							hits.push({ kind: 'message', session_id: sessionId, agent_id: session.agent_id, title: session.title, created_at: message.created_at, turn_id: message.turn_id, turn_index: null, role: message.role, tool_name: null, event_type: null, snippet: message.content.slice(0, 220) });
						}
						for (let index = total - 1; index >= 0 && hits.length < 50; index -= 1) {
							const message = scenario.messageAt(sessionId, index);
							if (!message.content.toLowerCase().includes(query)) continue;
							hits.push({ kind: 'message', session_id: sessionId, agent_id: session.agent_id, title: session.title, created_at: message.created_at, turn_id: message.turn_id, turn_index: Math.floor(index / 2), role: message.role, tool_name: null, event_type: null, snippet: message.content.slice(0, 220) });
						}
						return sendJson(response, 200, { hits });
					}
					if (request.method === 'PATCH' && !match[2]) {
						const body = await readJson(request);
						const updated = { ...session, title: String(body.title ?? '').trim() || session.title };
						sessions.set(sessionId, updated);
						return sendJson(response, 200, { session: updated });
					}
					if (request.method === 'DELETE' && !match[2]) {
						sessions.delete(sessionId);
						response.writeHead(204).end();
						return;
					}
					if (request.method === 'POST' && match[2] === 'branches') {
						const body = await readJson(request);
						return sendJson(response, 201, {
							branch: {
								branch_id: `mock-branch-${nextId++}`,
								name: `Fork from ${String(body.turn_id ?? 'turn')}`,
								active: Boolean(body.activate)
							}
						});
					}
					if (request.method === 'POST' && match[2] === 'messages') {
						const body = await readJson(request);
						const content = String(body.content ?? '').trim();
						const requestId = `mock-task-${nextId++}`;
						const messages = appended.get(sessionId) ?? [];
						messages.push({
							id: `${requestId}-user`, turn_id: `${requestId}-turn`, role: 'user', content,
							created_at: new Date().toISOString(), token_count: Math.ceil(content.length / 4)
						});
						appended.set(sessionId, messages);
						session.message_count = (session.message_count ?? 0) + 1;
						void streamMockResponse(session, requestId, content, messages, publish);
						return sendJson(response, 202, { request_id: requestId, session_id: sessionId });
					}
				}

				if (request.method === 'GET' && path === '/api/events') {
					const sessionId = url.searchParams.get('session_id');
					if (!sessionId) return sendJson(response, 400, { error: 'session_id is required' });
					response.writeHead(200, {
						'Cache-Control': 'no-store', 'Content-Type': 'text/event-stream',
						Connection: 'keep-alive'
					});
					response.write(': connected\n\n');
					const subscriber = { response, sessionId };
					subscribers.add(subscriber);
					request.on('close', () => subscribers.delete(subscriber));
					return;
				}

				return sendJson(response, 404, { error: 'Mock API route not found' });
			});
		}
	};

	async function streamMockResponse(
		session: Session,
		requestId: string,
		prompt: string,
		messages: ConversationMessage[],
		publishEvent: typeof publish
	): Promise<void> {
		publishEvent('conversation.task.started', {
			request_id: requestId, session_id: session.id, agent_id: session.agent_id
		});
		const response = scenario.responseFor(prompt);
		if (response.mode === 'error') {
			await delay(250);
			publishEvent('conversation.task.failed', {
				request_id: requestId, session_id: session.id,
				message: 'The mock provider rejected this request.', retryable: false
			});
			return;
		}
		const responseText = response.text;
		const messageId = `${requestId}-assistant`;
		publishEvent('conversation.message.started', {
			request_id: requestId, session_id: session.id, message_id: messageId
		});
		const chunks = responseText.match(/.{1,12}/g) ?? [];
		for (const [index, part] of chunks.entries()) {
			await delay(response.chunkDelayMs);
			publishEvent('conversation.message.delta', {
				request_id: requestId, session_id: session.id, message_id: messageId, delta: part
			});
			if (response.mode === 'interrupt' && index >= 3) {
				publishEvent('conversation.task.failed', {
					request_id: requestId, session_id: session.id,
					message: 'The mock stream was interrupted after partial output.', retryable: true
				});
				return;
			}
		}
		const message: ConversationMessage = {
			id: messageId, turn_id: `${requestId}-turn`, role: 'assistant', content: responseText,
			created_at: new Date().toISOString(), token_count: Math.ceil(responseText.length / 4)
		};
		messages.push(message);
		session.message_count = (session.message_count ?? 0) + 1;
		publishEvent('conversation.task.completed', { request_id: requestId, session_id: session.id });
	}
}

function delay(milliseconds: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
