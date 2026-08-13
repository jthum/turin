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
  HarnessSourceEntry,
  HarnessSourceFile,
  HarnessSourceOverlay,
  HarnessSourceSaveChange,
  HarnessSourceSaveResult,
  HarnessSourceValidation,
  ScheduleCreateInput,
  ScheduleJob,
  ScheduleRun,
} from "./types";

export interface EventSubscription {
  ready: Promise<void>;
  close(): void;
}

export interface TurinClient {
  status(): Promise<TurinStatus>;
  session(sessionId: string, messageLimit: number, messageOffset?: number): Promise<SessionDetail>;
  sessionPath(sessionId: string, turnId: number, messageLimit?: number): Promise<SessionDetail>;
  sessionGraph(sessionId: string): Promise<SessionGraph>;
  openSession(agentId: string): Promise<LiveSession>;
  resumeSession(sessionId: string, slotId?: string): Promise<LiveSession>;
  setSessionTitle(sessionId: string, title: string): Promise<SessionSummary>;
  deleteSession(sessionId: string): Promise<void>;
  submitTask(input: {
    agent_id?: string;
    session_id?: string;
    slot_id?: string;
    prompt: string;
    inference_context?: string;
  }): Promise<TaskStatus>;
  createBranch(input: {
    session_id: string;
    slot_id?: string;
    name: string;
    from_turn_id: number;
    activate?: boolean;
  }): Promise<SessionBranch>;
  checkoutBranch(input: {
    session_id: string;
    slot_id?: string;
    branch: string;
  }): Promise<SessionBranch>;
  sidestep(input: {
    session_id: string;
    slot_id?: string;
    prompt: string;
    mode: "ephemeral" | "fork_sibling";
    turn_id: number;
    timeout_ms?: number;
  }): Promise<TaskStatus>;
  promoteTask(input: { request_id: string; branch_name?: string }): Promise<SessionBranch>;
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
  harnesses(): Promise<HarnessRuntime[]>;
  harness(id: string): Promise<{ harness: HarnessDetail; issues: HarnessIssue[] }>;
  createHarness(id: string): Promise<{ harness: HarnessDetail; issues: HarnessIssue[] }>;
  validateHarness(id: string): Promise<HarnessValidation>;
  harnessSources(id: string): Promise<HarnessSourceEntry[]>;
  harnessSource(id: string, path: string): Promise<HarnessSourceFile>;
  validateHarnessSources(
    id: string,
    changes: HarnessSourceOverlay[],
  ): Promise<HarnessSourceValidation>;
  saveHarnessSources(
    id: string,
    changes: HarnessSourceSaveChange[],
  ): Promise<HarnessSourceSaveResult>;
  reloadHarness(id: string): Promise<{ harness: HarnessDetail; issues: HarnessIssue[] }>;
  deleteHarness(id: string): Promise<void>;
  schedules(): Promise<ScheduleJob[]>;
  scheduleRuns(id: string, limit?: number): Promise<ScheduleRun[]>;
  createSchedule(input: ScheduleCreateInput): Promise<ScheduleJob>;
  toggleSchedule(id: string, enabled: boolean): Promise<ScheduleJob>;
  deleteSchedule(id: string): Promise<ScheduleJob>;
  cancelTask(requestId: string): Promise<TaskStatus>;
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
