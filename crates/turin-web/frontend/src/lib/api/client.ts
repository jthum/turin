import type {
	Agent,
	AgentControlAction,
	AgentDetail,
	ConversationEventMap,
	ConversationEventName,
	CreatedSession,
	Harness,
	Memory,
	MemoryPage,
	MemoryListOptions,
	MessagePage,
	Session,
	SessionPage,
	SearchHit,
	SubmittedTask,
	WorkItem,
	WorkItemControlAction,
	Worklist
} from './contracts.js';

type EventHandlers = {
	[K in ConversationEventName]?: (event: ConversationEventMap[K]) => void;
};

export type StreamConnectionState = 'connecting' | 'open' | 'reconnecting';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	const response = await fetch(path, {
		...init,
		headers: {
			Accept: 'application/json',
			...(init?.body ? { 'Content-Type': 'application/json' } : {}),
			...init?.headers
		}
	});
	if (!response.ok) {
		const text = await response.text();
		let message = text;
		try {
			const body = JSON.parse(text) as { error?: unknown };
			if (typeof body.error === 'string') message = body.error;
		} catch {
			// Plain-text failures are already suitable for display.
		}
		throw new Error(message || `Turin returned ${response.status}`);
	}
	if (response.status === 204) return undefined as T;
	return response.json() as Promise<T>;
}

export class TurinWebClient {
	listHarnesses(signal?: AbortSignal): Promise<{ harnesses: Harness[] }> {
		return request('/api/harnesses', { signal });
	}

	listAgents(signal?: AbortSignal): Promise<{ agents: Agent[] }> {
		return request('/api/agents', { signal });
	}

	getAgent(agentId: string, signal?: AbortSignal): Promise<AgentDetail> {
		return request(`/api/agents/${encodeURIComponent(agentId)}`, { signal });
	}

	controlAgent(agentId: string, action: AgentControlAction): Promise<AgentDetail> {
		return request(`/api/agents/${encodeURIComponent(agentId)}/control`, {
			method: 'POST',
			body: JSON.stringify({ action })
		});
	}

	listWorklists(signal?: AbortSignal): Promise<{ worklists: Worklist[] }> {
		return request('/api/worklists', { signal });
	}

	listWorklistItems(worklistId: string, signal?: AbortSignal): Promise<{ worklist_id: string; items: WorkItem[] }> {
		return request(`/api/worklists/${encodeURIComponent(worklistId)}/items`, { signal });
	}

	getWorkItem(workItemId: string, signal?: AbortSignal): Promise<WorkItem> {
		return request(`/api/work-items/${encodeURIComponent(workItemId)}`, { signal });
	}

	controlWorkItem(workItemId: string, action: WorkItemControlAction, reason?: string): Promise<WorkItem> {
		return request(`/api/work-items/${encodeURIComponent(workItemId)}/control`, {
			method: 'POST',
			body: JSON.stringify({ action, reason })
		});
	}

	listMemories(options: MemoryListOptions = {}, signal?: AbortSignal): Promise<MemoryPage> {
		const params = new URLSearchParams({
			limit: String(options.limit ?? 100),
			offset: String(options.offset ?? 0)
		});
		if (options.query?.trim()) params.set('q', options.query.trim());
		if (options.scopeKind) params.set('scope_kind', options.scopeKind);
		if (options.scopeKey) params.set('scope_key', options.scopeKey);
		if (options.includeSuperseded) params.set('include_superseded', 'true');
		return request(`/api/memories?${params}`, { signal });
	}

	getMemory(memoryId: string, signal?: AbortSignal): Promise<Memory> {
		return request(`/api/memories/${encodeURIComponent(memoryId)}`, { signal });
	}

	correctMemory(memoryId: string, content: string): Promise<Memory> {
		return request(`/api/memories/${encodeURIComponent(memoryId)}/correct`, {
			method: 'POST', body: JSON.stringify({ content })
		});
	}

	deleteMemory(memoryId: string): Promise<{ id: string; deleted: boolean }> {
		return request(`/api/memories/${encodeURIComponent(memoryId)}`, { method: 'DELETE' });
	}

	listSessions(limit = 50, offset = 0, signal?: AbortSignal): Promise<SessionPage> {
		return request(`/api/sessions?limit=${limit}&offset=${offset}`, { signal });
	}

	listLinkedSessions(sessionId: string, limit = 50, offset = 0, signal?: AbortSignal): Promise<SessionPage> {
		return request(`/api/sessions/${encodeURIComponent(sessionId)}/linked?limit=${limit}&offset=${offset}`, { signal });
	}

	searchSessions(query: string, signal?: AbortSignal): Promise<{ hits: SearchHit[] }> {
		return request(`/api/search/sessions?q=${encodeURIComponent(query)}`, { signal });
	}

	searchWorkspace(query: string, signal?: AbortSignal): Promise<{ hits: SearchHit[] }> {
		return request(`/api/search/workspace?q=${encodeURIComponent(query)}`, { signal });
	}

	searchSessionMessages(sessionId: string, query: string, signal?: AbortSignal): Promise<{ hits: SearchHit[] }> {
		return request(`/api/sessions/${encodeURIComponent(sessionId)}/search?q=${encodeURIComponent(query)}`, { signal });
	}

	createSession(agentId: string): Promise<CreatedSession> {
		return request('/api/sessions', {
			method: 'POST',
			body: JSON.stringify({ agent_id: agentId })
		});
	}

	getSession(sessionId: string, signal?: AbortSignal): Promise<{ session: Session }> {
		return request(`/api/sessions/${encodeURIComponent(sessionId)}`, { signal });
	}

	loadMessages(
		sessionId: string,
		options: { limit?: number; offset?: number; total?: number; turnId?: string; signal?: AbortSignal } = {}
	): Promise<MessagePage> {
		const { limit = 80, offset = 0, total, turnId, signal } = options;
		const totalQuery = total === undefined ? '' : `&total=${total}`;
		const turnQuery = turnId === undefined ? '' : `&turn_id=${encodeURIComponent(turnId)}`;
		return request(
			`/api/sessions/${encodeURIComponent(sessionId)}/messages?limit=${limit}&offset=${offset}${totalQuery}${turnQuery}`,
			{ signal }
		);
	}

	renameSession(sessionId: string, title: string): Promise<{ session: Session }> {
		return request(`/api/sessions/${encodeURIComponent(sessionId)}`, {
			method: 'PATCH',
			body: JSON.stringify({ title })
		});
	}

	deleteSession(sessionId: string): Promise<void> {
		return request(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
	}

	createBranch(sessionId: string, turnId: string, activate: boolean): Promise<{ branch: unknown }> {
		return request(`/api/sessions/${encodeURIComponent(sessionId)}/branches`, {
			method: 'POST',
			body: JSON.stringify({ turn_id: turnId, activate })
		});
	}

	submitMessage(sessionId: string, content: string): Promise<SubmittedTask> {
		return request(`/api/sessions/${encodeURIComponent(sessionId)}/messages`, {
			method: 'POST',
			body: JSON.stringify({ content })
		});
	}

	subscribe(
		sessionId: string,
		handlers: EventHandlers,
		onConnectionChange?: (state: StreamConnectionState) => void
	): () => void {
		const source = new EventSource(`/api/events?session_id=${encodeURIComponent(sessionId)}`);
		onConnectionChange?.('connecting');
		source.onopen = () => onConnectionChange?.('open');
		source.onerror = () => onConnectionChange?.('reconnecting');
		for (const name of Object.keys(handlers) as ConversationEventName[]) {
			source.addEventListener(name, (event) => {
				const handler = handlers[name] as ((value: unknown) => void) | undefined;
				handler?.(JSON.parse((event as MessageEvent<string>).data));
			});
		}
		return () => {
			source.close();
			onConnectionChange?.('connecting');
		};
	}
}

export const turinWeb = new TurinWebClient();
