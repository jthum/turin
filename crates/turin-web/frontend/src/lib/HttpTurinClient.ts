import type { EventSubscription, TurinClient } from "./TurinClient";
import type {
  LiveSession,
  SessionDetail,
  SessionBranch,
  SessionGraph,
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
  HarnessDetail,
  HarnessIssue,
  HarnessRuntime,
  HarnessValidation,
  ScheduleCreateInput,
  ScheduleJob,
  ScheduleRun,
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

  async sessionPath(sessionId: string, turnId: number, messageLimit = 24): Promise<SessionDetail> {
    const params = new URLSearchParams({
      session_id: sessionId,
      turn_id: String(turnId),
      message_limit: String(messageLimit),
    });
    const result = await this.request<{ detail: SessionDetail }>(`/api/session/path?${params}`);
    return result.detail;
  }

  async sessionGraph(sessionId: string): Promise<SessionGraph> {
    const params = new URLSearchParams({ session_id: sessionId });
    const result = await this.request<{ graph: SessionGraph }>(`/api/session/graph?${params}`);
    return result.graph;
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
    inference_context?: string;
  }): Promise<TaskStatus> {
    const result = await this.request<{ task: TaskStatus }>("/api/tasks/submit", {
      method: "POST",
      body: JSON.stringify(input),
    });
    return result.task;
  }

  async createBranch(input: {
    session_id: string;
    slot_id?: string;
    name: string;
    from_turn_id: number;
    activate?: boolean;
  }): Promise<SessionBranch> {
    const result = await this.request<{ branch: SessionBranch }>("/api/session/branches", {
      method: "POST",
      body: JSON.stringify(input),
    });
    return result.branch;
  }

  async checkoutBranch(input: {
    session_id: string;
    slot_id?: string;
    branch: string;
  }): Promise<SessionBranch> {
    const result = await this.request<{ branch: SessionBranch }>("/api/session/branches/checkout", {
      method: "POST",
      body: JSON.stringify(input),
    });
    return result.branch;
  }

  async sidestep(input: {
    session_id: string;
    slot_id?: string;
    prompt: string;
    mode: "ephemeral" | "fork_sibling";
    turn_id: number;
    timeout_ms?: number;
  }): Promise<TaskStatus> {
    const result = await this.request<{ task: TaskStatus }>("/api/tasks/sidestep", {
      method: "POST",
      body: JSON.stringify(input),
    });
    return result.task;
  }

  async promoteTask(input: { request_id: string; branch_name?: string }): Promise<SessionBranch> {
    const result = await this.request<{ branch: SessionBranch }>("/api/tasks/promote", {
      method: "POST",
      body: JSON.stringify(input),
    });
    return result.branch;
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

  async harnesses(): Promise<HarnessRuntime[]> {
    const result = await this.request<{ harnesses: HarnessRuntime[] }>("/api/harnesses");
    return result.harnesses;
  }

  harness(id: string): Promise<{ harness: HarnessDetail; issues: HarnessIssue[] }> {
    const params = new URLSearchParams({ id });
    return this.request(`/api/harness?${params}`);
  }

  createHarness(id: string): Promise<{ harness: HarnessDetail; issues: HarnessIssue[] }> {
    return this.request("/api/harnesses/create", {
      method: "POST",
      body: JSON.stringify({ id }),
    });
  }

  async validateHarness(id: string): Promise<HarnessValidation> {
    const result = await this.request<{ validation: HarnessValidation }>("/api/harnesses/validate", {
      method: "POST",
      body: JSON.stringify({ id }),
    });
    return result.validation;
  }

  reloadHarness(id: string): Promise<{ harness: HarnessDetail; issues: HarnessIssue[] }> {
    return this.request("/api/harnesses/reload", {
      method: "POST",
      body: JSON.stringify({ id }),
    });
  }

  async deleteHarness(id: string): Promise<void> {
    await this.request("/api/harnesses/delete", {
      method: "DELETE",
      body: JSON.stringify({ id }),
    });
  }

  async schedules(): Promise<ScheduleJob[]> {
    const result = await this.request<{ schedules: ScheduleJob[] }>("/api/operations/schedules");
    return result.schedules;
  }

  async scheduleRuns(id: string, limit = 50): Promise<ScheduleRun[]> {
    const params = new URLSearchParams({ id, limit: String(limit) });
    const result = await this.request<{ runs: { runs: ScheduleRun[] } }>(
      `/api/operations/schedule-runs?${params}`,
    );
    return result.runs.runs;
  }

  async createSchedule(input: ScheduleCreateInput): Promise<ScheduleJob> {
    const result = await this.request<{ schedule: ScheduleJob }>("/api/operations/schedules", {
      method: "POST",
      body: JSON.stringify(input),
    });
    return result.schedule;
  }

  async toggleSchedule(id: string, enabled: boolean): Promise<ScheduleJob> {
    const result = await this.request<{ schedule: ScheduleJob }>("/api/operations/schedules/toggle", {
      method: "POST",
      body: JSON.stringify({ id, enabled }),
    });
    return result.schedule;
  }

  async deleteSchedule(id: string): Promise<ScheduleJob> {
    const result = await this.request<{ schedule: ScheduleJob }>("/api/operations/schedules", {
      method: "DELETE",
      body: JSON.stringify({ id }),
    });
    return result.schedule;
  }

  async cancelTask(requestId: string): Promise<TaskStatus> {
    const result = await this.request<{ task: TaskStatus }>("/api/operations/tasks/cancel", {
      method: "POST",
      body: JSON.stringify({ request_id: requestId }),
    });
    return result.task;
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
      "perf.operation.started",
      "perf.operation.completed",
      "schedule.created",
      "schedule.updated",
      "schedule.enabled",
      "schedule.disabled",
      "schedule.deleted",
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
