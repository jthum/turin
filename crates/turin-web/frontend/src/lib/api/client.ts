import type {
	Agent,
	ConversationEventMap,
	ConversationEventName,
	CreatedSession,
	MessagePage,
	Session,
	SessionPage,
	SubmittedTask
} from './contracts.js';

type EventHandlers = {
	[K in ConversationEventName]?: (event: ConversationEventMap[K]) => void;
};

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
		const message = await response.text();
		throw new Error(message || `Turin returned ${response.status}`);
	}
	if (response.status === 204) return undefined as T;
	return response.json() as Promise<T>;
}

export class TurinWebClient {
	listAgents(signal?: AbortSignal): Promise<{ agents: Agent[] }> {
		return request('/api/agents', { signal });
	}

	listSessions(limit = 50, offset = 0, signal?: AbortSignal): Promise<SessionPage> {
		return request(`/api/sessions?limit=${limit}&offset=${offset}`, { signal });
	}

	createSession(agentId: string): Promise<CreatedSession> {
		return request('/api/sessions', {
			method: 'POST',
			body: JSON.stringify({ agent_id: agentId })
		});
	}

	loadMessages(sessionId: string, limit = 80, offset = 0, signal?: AbortSignal): Promise<MessagePage> {
		return request(
			`/api/sessions/${encodeURIComponent(sessionId)}/messages?limit=${limit}&offset=${offset}`,
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

	submitMessage(sessionId: string, content: string): Promise<SubmittedTask> {
		return request(`/api/sessions/${encodeURIComponent(sessionId)}/messages`, {
			method: 'POST',
			body: JSON.stringify({ content })
		});
	}

	subscribe(sessionId: string, handlers: EventHandlers): () => void {
		const source = new EventSource(`/api/events?session_id=${encodeURIComponent(sessionId)}`);
		for (const name of Object.keys(handlers) as ConversationEventName[]) {
			source.addEventListener(name, (event) => {
				const handler = handlers[name] as ((value: unknown) => void) | undefined;
				handler?.(JSON.parse((event as MessageEvent<string>).data));
			});
		}
		return () => source.close();
	}
}

export const turinWeb = new TurinWebClient();
