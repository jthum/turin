import type {
	Agent,
	ConversationEventMap,
	ConversationEventName,
	CreatedSession,
	Harness,
	MessagePage,
	Session,
	SessionPage,
	SubmittedTask
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
		const message = await response.text();
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

	listSessions(limit = 50, offset = 0, signal?: AbortSignal): Promise<SessionPage> {
		return request(`/api/sessions?limit=${limit}&offset=${offset}`, { signal });
	}

	createSession(agentId: string): Promise<CreatedSession> {
		return request('/api/sessions', {
			method: 'POST',
			body: JSON.stringify({ agent_id: agentId })
		});
	}

	loadMessages(
		sessionId: string,
		options: { limit?: number; offset?: number; total?: number; signal?: AbortSignal } = {}
	): Promise<MessagePage> {
		const { limit = 80, offset = 0, total, signal } = options;
		const totalQuery = total === undefined ? '' : `&total=${total}`;
		return request(
			`/api/sessions/${encodeURIComponent(sessionId)}/messages?limit=${limit}&offset=${offset}${totalQuery}`,
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
