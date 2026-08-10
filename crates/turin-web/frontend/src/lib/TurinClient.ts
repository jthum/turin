import type {
  LiveSession,
  SessionDetail,
  TaskStatus,
  TurinEvent,
  TurinStatus,
  UiListRequest,
  UiListResult,
  JsonValue,
} from "./types";

export interface EventSubscription {
  close(): void;
}

export interface TurinClient {
  status(): Promise<TurinStatus>;
  session(sessionId: string, messageLimit: number): Promise<SessionDetail>;
  openSession(agentId: string): Promise<LiveSession>;
  resumeSession(sessionId: string, slotId?: string): Promise<LiveSession>;
  submitTask(input: {
    agent_id?: string;
    session_id?: string;
    slot_id?: string;
    prompt: string;
  }): Promise<TaskStatus>;
  loadList(request: UiListRequest): Promise<UiListResult>;
  runAction(input: {
    action: string;
    harness_id?: string;
    agent_id?: string;
    params?: JsonValue;
  }): Promise<{ result: Record<string, JsonValue> }>;
  subscribe(
    listener: (event: TurinEvent) => void,
    options?: { sessionId?: string },
  ): EventSubscription;
}
