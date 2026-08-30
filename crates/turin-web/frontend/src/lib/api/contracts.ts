export type RuntimeHealth = {
	connection_kind: 'local' | 'remote' | 'mock';
	ready: boolean;
	version: string;
	protocol_version: number;
	issue_count: number;
	agent_count: number;
	harness_count: number;
	running_agent_count: number;
	active_task_count: number;
};

export type Bootstrap = {
	web_version: string;
	runtime: RuntimeHealth;
};

export type Agent = {
	id: string;
	name: string;
	provider: string;
	model: string;
	harness_id: string;
	enabled: boolean;
};

export type Harness = {
	id: string;
	name: string;
	bound_agent_ids: string[];
	has_ui: boolean;
};

export type Session = {
	id: string;
	title: string;
	agent_id: string;
	created_at: string;
	message_count: number | null;
};

export type SessionPage = {
	sessions: Session[];
	offset: number;
	has_more: boolean;
};

export type MessageRole = 'user' | 'assistant' | 'system' | 'tool';

export type ConversationMessage = {
	id: string;
	turn_id: string;
	role: MessageRole;
	content: string;
	created_at: string;
	token_count: number | null;
	metrics?: {
		input_tokens: number;
		output_tokens: number;
		cache_read_input_tokens?: number;
		cache_creation_input_tokens?: number;
		provider?: string;
		model?: string;
	};
	reasoning?: {
		duration_ms: number;
		summary: string;
	};
};

export type MessagePage = {
	messages: ConversationMessage[];
	offset: number;
	total: number;
	has_more: boolean;
};

export type CreatedSession = {
	session: Session;
};

export type SubmittedTask = {
	request_id: string;
	session_id: string;
};

export type ConversationEventMap = {
	'conversation.task.started': {
		request_id: string;
		session_id: string;
		agent_id: string;
	};
	'conversation.message.started': {
		request_id: string;
		session_id: string;
		message_id: string;
	};
	'conversation.message.delta': {
		request_id: string;
		session_id: string;
		message_id: string;
		delta: string;
	};
	'conversation.task.completed': {
		request_id: string;
		session_id: string;
	};
	'conversation.task.failed': {
		request_id: string;
		session_id: string;
		message: string;
		retryable: boolean;
	};
};

export type ConversationEventName = keyof ConversationEventMap;
