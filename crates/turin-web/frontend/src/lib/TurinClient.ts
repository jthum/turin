import type {
  LiveSession,
  SessionDetail,
  SessionSummary,
  TaskStatus,
  TurinEvent,
  TurinStatus,
  UiListRequest,
  UiListResult,
  JsonValue,
  MemoryList,
  WorklistDetail,
  WorklistItem,
} from "./types";

export interface EventSubscription {
  ready: Promise<void>;
  close(): void;
}

export interface TurinClient {
  status(): Promise<TurinStatus>;
  session(sessionId: string, messageLimit: number, messageOffset?: number): Promise<SessionDetail>;
  openSession(agentId: string): Promise<LiveSession>;
  resumeSession(sessionId: string, slotId?: string): Promise<LiveSession>;
  setSessionTitle(sessionId: string, title: string): Promise<SessionSummary>;
  submitTask(input: {
    agent_id?: string;
    session_id?: string;
    slot_id?: string;
    prompt: string;
  }): Promise<TaskStatus>;
  loadList(request: UiListRequest): Promise<UiListResult>;
  worklists(): Promise<WorklistDetail[]>;
  worklistItems(worklistId: string, limit?: number): Promise<WorklistItem[]>;
  memories(input?: {
    scopeKind?: string;
    scopeKey?: string;
    includeSuperseded?: boolean;
    limit?: number;
    offset?: number;
  }): Promise<MemoryList>;
  runAction(input: {
    action: string;
    harness_id?: string;
    agent_id?: string;
    params?: JsonValue;
  }): Promise<{ result: Record<string, JsonValue> }>;
  subscribe(
    listener: (event: TurinEvent) => void,
    options?: { sessionId?: string; slotId?: string },
  ): EventSubscription;
}
