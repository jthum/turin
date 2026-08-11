import type { EventSubscription, TurinClient } from "./TurinClient";
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

interface ErrorEnvelope {
  error?: { code?: string; message?: string; details?: JsonValue };
}

export class HttpTurinClient implements TurinClient {
  constructor(private readonly baseUrl = "") {}

  status(): Promise<TurinStatus> {
    return this.request<TurinStatus>("/api/status");
  }

  async session(sessionId: string, messageLimit: number, messageOffset?: number): Promise<SessionDetail> {
    const params = new URLSearchParams({
      session_id: sessionId,
      message_limit: String(messageLimit),
    });
    if (messageOffset !== undefined) params.set("message_offset", String(messageOffset));
    const result = await this.request<{ detail: SessionDetail }>(
      `/api/session?${params}`,
    );
    return result.detail;
  }

  async openSession(agentId: string): Promise<LiveSession> {
    const result = await this.request<{ session: LiveSession }>("/api/sessions/open", {
      method: "POST",
      body: JSON.stringify({ agent_id: agentId }),
    });
    return result.session;
  }

  async resumeSession(sessionId: string, slotId?: string): Promise<LiveSession> {
    const result = await this.request<{ session: LiveSession }>("/api/sessions/resume", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, slot_id: slotId }),
    });
    return result.session;
  }

  async setSessionTitle(sessionId: string, title: string): Promise<SessionSummary> {
    const result = await this.request<{ session: SessionSummary }>("/api/session/title", {
      method: "PUT",
      body: JSON.stringify({ session_id: sessionId, title }),
    });
    return result.session;
  }

  async submitTask(input: {
    agent_id?: string;
    session_id?: string;
    slot_id?: string;
    prompt: string;
  }): Promise<TaskStatus> {
    const result = await this.request<{ task: TaskStatus }>("/api/tasks/submit", {
      method: "POST",
      body: JSON.stringify(input),
    });
    return result.task;
  }

  loadList(request: UiListRequest): Promise<UiListResult> {
    return this.request<UiListResult>("/api/ui/list", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  async worklists(): Promise<WorklistDetail[]> {
    const result = await this.request<{ worklists: WorklistDetail[] }>("/api/data/worklists");
    return result.worklists;
  }

  async worklistItems(worklistId: string, limit = 100): Promise<WorklistItem[]> {
    const params = new URLSearchParams({ id: worklistId, limit: String(limit) });
    const result = await this.request<{ list: { items: WorklistItem[] } }>(
      `/api/data/worklist-items?${params}`,
    );
    return result.list.items;
  }

  async memories(input: {
    scopeKind?: string;
    scopeKey?: string;
    includeSuperseded?: boolean;
    limit?: number;
    offset?: number;
  } = {}): Promise<MemoryList> {
    const params = new URLSearchParams({
      limit: String(input.limit ?? 100),
      offset: String(input.offset ?? 0),
    });
    if (input.scopeKind) params.set("scope_kind", input.scopeKind);
    if (input.scopeKey) params.set("scope_key", input.scopeKey);
    if (input.includeSuperseded) params.set("include_superseded", "true");
    const result = await this.request<{ list: MemoryList }>(`/api/data/memories?${params}`);
    return result.list;
  }

  runAction(input: {
    action: string;
    harness_id?: string;
    agent_id?: string;
    params?: JsonValue;
  }): Promise<{ result: Record<string, JsonValue> }> {
    return this.request("/api/actions/run", {
      method: "POST",
      body: JSON.stringify(input),
    });
  }

  subscribe(
    listener: (event: TurinEvent) => void,
    options: { sessionId?: string; slotId?: string } = {},
  ): EventSubscription {
    const params = new URLSearchParams();
    if (options.sessionId) params.set("session_id", options.sessionId);
    if (options.slotId) params.set("slot_id", options.slotId);
    const query = params.size ? `?${params}` : "";
    const source = new EventSource(`${this.baseUrl}/api/events${query}`);
    const ready = new Promise<void>(resolve => {
      source.addEventListener("open", () => resolve(), { once: true });
    });
    const eventTypes = [
      "runtime.snapshot",
      "runtime.rescanned",
      "ui.intent",
      "task.submitted",
      "task.updated",
      "task_start",
      "task_complete",
      "turn_start",
      "turn_end",
      "inference_request",
      "message_start",
      "thinking_delta",
      "message_delta",
      "message_end",
      "tool_call",
      "tool_result",
    ];
    for (const type of eventTypes) {
      source.addEventListener(type, raw => {
        const message = raw as MessageEvent<string>;
        try {
          listener({ type, data: JSON.parse(message.data) as Record<string, JsonValue> });
        } catch {
          listener({ type: "web.decode_error", data: { event_type: type } });
        }
      });
    }
    source.addEventListener("web.error", raw => {
      const message = raw as MessageEvent<string>;
      listener({ type: "web.error", data: { message: message.data } });
    });
    return { ready, close: () => source.close() };
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init.body ? { "Content-Type": "application/json" } : {}),
        ...init.headers,
      },
    });
    if (!response.ok) {
      let body: ErrorEnvelope = {};
      try {
        body = (await response.json()) as ErrorEnvelope;
      } catch {
        // The status text remains useful when an intermediary returns non-JSON.
      }
      throw new TurinHttpError(
        body.error?.message ?? `${response.status} ${response.statusText}`,
        response.status,
        body.error?.code,
        body.error?.details,
      );
    }
    return (await response.json()) as T;
  }
}

export class TurinHttpError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string,
    readonly details?: JsonValue,
  ) {
    super(message);
    this.name = "TurinHttpError";
  }
}
