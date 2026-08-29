import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Plugin } from 'vite';
import type {
	ConversationEventMap,
	ConversationEventName,
	ConversationMessage,
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
		const end = Math.max(0, total - offset);
		const start = Math.max(0, end - limit);
		const messages: ConversationMessage[] = [];
		for (let index = start; index < end; index += 1) {
			messages.push(
				index < generatedCount
					? scenario.messageAt(sessionId, index)
					: additions[index - generatedCount]
			);
		}
		return { messages, offset, total, has_more: start > 0 };
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
							issue_count: 0, agent_count: scenario.agents.length, harness_count: 1,
							running_agent_count: scenario.agents.length, active_task_count: 0
						}
					});
				}
				if (request.method === 'GET' && path === '/api/agents') {
					return sendJson(response, 200, { agents: scenario.agents });
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
						created_at: new Date().toISOString(), message_count: 0
					};
					sessions.set(id, session);
					return sendJson(response, 201, { session });
				}

				const match = path.match(/^\/api\/sessions\/([^/]+)(?:\/(messages))?$/);
				if (match) {
					const sessionId = decodeURIComponent(match[1]);
					const session = sessions.get(sessionId);
					if (!session) return sendJson(response, 404, { error: 'Session not found' });
					if (request.method === 'GET' && match[2] === 'messages') {
						return sendJson(response, 200, sessionMessages(
							sessionId,
							Math.min(200, Number(url.searchParams.get('limit') ?? 80)),
							Number(url.searchParams.get('offset') ?? 0)
						));
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
		const responseText = scenario.responseFor(prompt);
		const messageId = `${requestId}-assistant`;
		publishEvent('conversation.message.started', {
			request_id: requestId, session_id: session.id, message_id: messageId
		});
		for (const part of responseText.match(/.{1,12}/g) ?? []) {
			await new Promise((resolve) => setTimeout(resolve, 35));
			publishEvent('conversation.message.delta', {
				request_id: requestId, session_id: session.id, message_id: messageId, delta: part
			});
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
