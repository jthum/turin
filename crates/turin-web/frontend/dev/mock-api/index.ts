import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Plugin } from 'vite';
import type {
	ConversationEventMap,
	ConversationEventName,
	ConversationMessage,
	Memory,
	SearchHit,
	Session
} from '../../src/lib/api/contracts.js';
import { randomUUID } from 'node:crypto';
import { createMockScenario, mockMessageIndexForTurn, mockTurnForMessageIndex, MOCK_SESSION_IDS } from './scenario.js';

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

	function memoryResponse(memory: Memory): Memory {
		return {
			...memory,
			scope_display_name: memory.scope_kind === 'session'
				? scenario.sessions.find((session) => session.id === memory.scope_key)?.title ?? null
				: memory.scope_kind === 'agent'
					? scenario.agents.find((agent) => agent.id === memory.scope_key)?.name ?? null
					: memory.scope_kind === 'harness'
						? scenario.harnesses.find((harness) => harness.id === memory.scope_key)?.name ?? null
						: memory.scope_kind === 'global' ? 'All workspaces' : null
		};
	}

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
			? mockMessageIndexForTurn(Number(turnId.slice(generatedPrefix.length)))
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
				hits.push({ kind: 'message', session_id: session.id, agent_id: session.agent_id, title: session.title, created_at: message.created_at, turn_id: message.turn_id, turn_index: mockTurnForMessageIndex(index) - 1, role: message.role, tool_name: null, event_type: null, snippet: message.content.slice(0, 220) });
			}
			if (hits.length >= 50) break;
		}
		if ('read_file persistence checkpoint'.includes(query)) {
			const session = sessions.get(MOCK_SESSION_IDS.storage);
			if (session) hits.push({ kind: 'tool_execution', session_id: session.id, agent_id: session.agent_id, title: session.title, created_at: session.created_at, turn_id: `${session.id}-turn-12`, turn_index: 11, role: null, tool_name: 'read_file', event_type: null, snippet: 'read_file persistence checkpoint and verify durable rows' });
		}
		if ('task completed runtime'.includes(query)) {
			const session = sessions.get(MOCK_SESSION_IDS.performance);
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
				const agentMatch = path.match(/^\/api\/agents\/([^/]+)(?:\/(control))?$/);
				if (agentMatch) {
					const agentId = decodeURIComponent(agentMatch[1]);
					const agent = scenario.agents.find((candidate) => candidate.id === agentId);
					if (!agent) return sendJson(response, 404, { error: 'Agent not found.' });
					if (request.method === 'POST' && agentMatch[2] === 'control') {
						const body = await readJson(request);
						const action = String(body.action ?? '');
						if (action === 'enable') agent.enabled = true;
						else if (action === 'disable') {
							agent.enabled = false;
							agent.running = false;
							agent.active_tasks = 0;
							agent.queued_tasks = 0;
							agent.awaiting_results = 0;
						} else if (action !== 'reload') return sendJson(response, 409, { error: 'Unsupported agent operation.' });
					}
					if (request.method === 'GET' && agentMatch[2]) return sendJson(response, 404, { error: 'API route not found.' });
					if (request.method !== 'GET' && request.method !== 'POST') return sendJson(response, 405, { error: 'Method not allowed.' });
					return sendJson(response, 200, {
						...agent,
						directory: `/workspace/.turin/runtime/agents/${agent.id}`,
						system_prompt: agent.id === 'scout' ? 'Investigate narrowly and return concise evidence.' : null,
						idle_timeout_seconds: agent.id === 'scout' ? 180 : 300,
						has_local_harness: agent.harness_id.startsWith('agent::'),
						inference_contexts: [
							{ id: 'default', provider: agent.provider, model: agent.model, is_default: true },
							...(agent.id === 'default' ? [{ id: 'fast', provider: 'minimax', model: 'MiniMax-M2.7', is_default: false }] : [])
						],
						current_session_id: agent.running ? MOCK_SESSION_IDS.performance : null,
						issues: agent.id === 'reviewer' ? [{ path: '/workspace/.turin/runtime/agents/reviewer/config.toml', message: 'Example registry warning for the mock workflow.' }] : []
					});
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
				const workItemMatch = path.match(/^\/api\/work-items\/([^/]+)(?:\/(control))?$/);
				if (workItemMatch) {
					const workItemId = decodeURIComponent(workItemMatch[1]);
					const item = Object.values(scenario.workItems).flat().find((candidate) => candidate.id === workItemId);
					if (!item) return sendJson(response, 404, { error: 'Work item not found.' });
					if (request.method === 'GET' && !workItemMatch[2]) return sendJson(response, 200, item);
					if (request.method === 'POST' && workItemMatch[2] === 'control') {
						const body = await readJson(request);
						const action = String(body.action ?? '');
						if (action === 'pause' && item.status === 'pending' && !item.claim_execution_id) {
							item.status = 'paused';
							item.paused = true;
							item.pause_reason = typeof body.reason === 'string' && body.reason.trim() ? body.reason.trim() : 'Paused by operator';
						} else if (action === 'resume' && item.status === 'paused') {
							item.status = 'pending';
							item.paused = false;
							item.pause_reason = null;
							item.pause_until_unix_ms = null;
						} else if (action === 'release_stale' && item.status === 'active' && (item.claim_heartbeat_unix_ms ?? 0) < Date.now() - Number(body.stale_after_ms ?? 300_000)) {
							item.status = 'pending';
							item.claim_agent_id = null;
							item.claim_session_id = null;
							item.claim_execution_id = null;
							item.claim_heartbeat_unix_ms = null;
							item.claimed_at = null;
						} else {
							return sendJson(response, 409, { error: 'The work item changed or the operation is not valid.' });
						}
						item.updated_at = new Date().toISOString();
						return sendJson(response, 200, item);
					}
				}
				if (request.method === 'GET' && path === '/api/memories') {
					const limit = Number(url.searchParams.get('limit') ?? 100);
					const offset = Number(url.searchParams.get('offset') ?? 0);
					const query = (url.searchParams.get('q') ?? '').trim().toLowerCase();
					const scopeKind = url.searchParams.get('scope_kind');
					const scopeKey = url.searchParams.get('scope_key');
					const includeSuperseded = url.searchParams.get('include_superseded') === 'true';
					const visible = scenario.memories.filter((memory) =>
						(!query || memory.content.toLowerCase().includes(query))
						&& (!scopeKind || memory.scope_kind === scopeKind)
						&& (!scopeKey || memory.scope_key === scopeKey)
						&& (includeSuperseded || !memory.superseded_at)
					);
					const scopeKinds = [...new Set(scenario.memories.map((memory) => memory.scope_kind))].map((kind) => ({
						scope_kind: kind,
						count: scenario.memories.filter((memory) => memory.scope_kind === kind && (includeSuperseded || !memory.superseded_at)).length
					}));
					const memories = visible.slice(offset, offset + limit).map(memoryResponse);
					return sendJson(response, 200, {
						memories, scope_kinds: scopeKinds, total: visible.length, offset, limit
					});
				}
				const memoryMatch = path.match(/^\/api\/memories\/([^/]+)(?:\/(correct))?$/);
				if (memoryMatch) {
					const memoryId = decodeURIComponent(memoryMatch[1]);
					const index = scenario.memories.findIndex((memory) => memory.id === memoryId);
					if (index < 0) return sendJson(response, 404, { error: 'Memory not found.' });
					if (request.method === 'GET' && !memoryMatch[2]) return sendJson(response, 200, memoryResponse(scenario.memories[index]));
					if (request.method === 'POST' && memoryMatch[2] === 'correct') {
						const body = await readJson(request);
						const content = typeof body.content === 'string' ? body.content.trim() : '';
						if (!content || scenario.memories[index].superseded_at) return sendJson(response, 409, { error: 'Memory cannot be corrected.' });
						const original = scenario.memories[index];
						const replacement = { ...original, id: `memory-corrected-${nextId++}`, content, created_at: new Date().toISOString(), superseded_at: null, superseded_by_id: null, retrieval_count: 0, last_retrieved_at: null };
						original.superseded_at = new Date().toISOString();
						original.superseded_by_id = replacement.id;
						scenario.memories.unshift(replacement);
						return sendJson(response, 200, memoryResponse(replacement));
					}
					if (request.method === 'DELETE' && !memoryMatch[2]) {
						scenario.memories.splice(index, 1);
						for (const memory of scenario.memories) if (memory.superseded_by_id === memoryId) memory.superseded_by_id = null;
						return sendJson(response, 200, { id: memoryId, deleted: true });
					}
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
					const all = [...sessions.values()].filter((session) => !session.parent_session_id).reverse();
					return sendJson(response, 200, {
						sessions: all.slice(offset, offset + limit), offset,
						has_more: offset + limit < all.length
					});
				}
				if (request.method === 'POST' && path === '/api/sessions') {
					const body = await readJson(request);
					const id = randomUUID();
					const session: Session = {
						id, title: 'New conversation', agent_id: String(body.agent_id ?? 'default'),
						created_at: new Date().toISOString(), message_count: 0,
						visibility: 'private', relation_kind: null
					};
					sessions.set(id, session);
					return sendJson(response, 201, { session });
				}

				const match = path.match(/^\/api\/sessions\/([^/]+)(?:\/(messages|branches|search|linked))?$/);
				if (match) {
					const sessionId = decodeURIComponent(match[1]);
					const session = sessions.get(sessionId);
					if (!session) return sendJson(response, 404, { error: 'Session not found' });
					if (request.method === 'GET' && !match[2]) {
						return sendJson(response, 200, { session });
					}
					if (request.method === 'GET' && match[2] === 'linked') {
						const limit = Number(url.searchParams.get('limit') ?? 50);
						const offset = Number(url.searchParams.get('offset') ?? 0);
						const children = [...sessions.values()]
							.filter((candidate) => candidate.parent_session_id === sessionId)
							.reverse();
						return sendJson(response, 200, {
							sessions: children.slice(offset, offset + limit),
							offset,
							has_more: offset + limit < children.length
						});
					}
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
							hits.push({ kind: 'message', session_id: sessionId, agent_id: session.agent_id, title: session.title, created_at: message.created_at, turn_id: message.turn_id, turn_index: mockTurnForMessageIndex(index) - 1, role: message.role, tool_name: null, event_type: null, snippet: message.content.slice(0, 220) });
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
		session.latest_message_preview = responseText.replace(/\s+/g, ' ').slice(0, 180);
		session.latest_message_role = 'assistant';
		session.latest_message_created_at = message.created_at;
		publishEvent('conversation.task.completed', { request_id: requestId, session_id: session.id });
	}
}

function delay(milliseconds: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
