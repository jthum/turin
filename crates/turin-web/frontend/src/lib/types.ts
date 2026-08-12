export type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

export interface AgentSummary {
  id: string;
  enabled: boolean;
  provider: string;
  model: string;
  harness_ref: string;
}

export interface AgentRuntime {
  agent_id: string;
  running: boolean;
  active_tasks: number;
  queued_tasks: number;
  awaiting_results: number;
  current_session_id?: string | null;
  current_request_id?: string | null;
}

export interface LiveExecution {
  execution_id: string;
  context_target: JsonValue;
  visibility: string;
  durability: string;
  write_policy: string;
}

export interface LiveSession {
  agent_id: string;
  slot_id: string;
  session_id: string;
  running: boolean;
  active_tasks: number;
  queued_tasks: number;
  current_request_id?: string | null;
  execution: LiveExecution;
  conflict_policy: string;
  history?: { len: number; message_offset: number } | null;
}

export interface SessionSummary {
  internal_id: number;
  session_id: string;
  agent_id: string;
  metadata?: Record<string, JsonValue> | null;
  created_at: string;
}

export interface SessionMessage {
  id: number;
  turn_id: number;
  turn_index: number;
  role: string;
  content: JsonValue;
  token_count?: number | null;
  estimated_token_count?: number | null;
  created_at: string;
}

export interface InferenceRequestMetrics {
  provider: string;
  model: string;
  requested_context: string;
  resolved_context: string;
  compaction_mode: string;
  estimated_input_tokens_before_compaction: number;
  estimated_input_tokens: number;
  system_prompt_tokens: number;
  message_tokens: number;
  tool_definition_tokens: number;
  reusable_prefix_tokens: number;
  context_window_tokens: number;
  context_window_configured: boolean;
  input_budget_tokens: number;
  max_output_tokens?: number | null;
  thinking_budget_tokens?: number | null;
  available_message_count: number;
  sent_message_count: number;
  history_message_offset: number;
  checkpoint_covered_message_count: number;
  truncated_tool_results: number;
  dropped_messages: number;
  estimated_payload_bytes: number;
}

export interface SessionTurnEfficiency {
  turn_index: number;
  requests: SessionRequestEfficiency[];
  input_tokens: number;
  output_tokens: number;
  created_at: string;
}

export interface SessionRequestEfficiency {
  metrics?: InferenceRequestMetrics | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  cache_read_input_tokens?: number | null;
  cache_creation_input_tokens?: number | null;
  created_at: string;
}

export interface SessionEfficiency {
  total_input_tokens: number;
  total_output_tokens: number;
  total_cache_read_input_tokens: number;
  total_cache_creation_input_tokens: number;
  total_request_count: number;
  turns: SessionTurnEfficiency[];
  latest_compaction?: {
    covered_message_count: number;
    generated_at_turn_index: number;
    provider: string;
    model: string;
    created_at: string;
  } | null;
  provider_cache_metrics_available: boolean;
}

export interface ToolExecution {
  id: number;
  turn_index: number;
  tool_call_id: string;
  tool_name: string;
  args: JsonValue;
  output?: JsonValue;
  is_error: boolean;
  duration_ms?: number | null;
  verdict: string;
  created_at: string;
}

export interface SessionBranch {
  branch_id: string;
  name: string;
  head_turn_id?: number | null;
  head_turn_index?: number | null;
  source_turn_id?: number | null;
  origin_kind: string;
  origin_task_id?: string | null;
  origin_execution_id?: string | null;
  origin_metadata?: JsonValue | null;
  active: boolean;
  created_at: string;
}

export interface SessionGraphTurn {
  turn_id: number;
  turn_public_id: string;
  parent_turn_id?: number | null;
  turn_index: number;
  message_count: number;
  tool_execution_count: number;
  preview?: string | null;
  created_at: string;
}

export interface SessionGraph {
  session: SessionSummary;
  turns: SessionGraphTurn[];
  branches: SessionBranch[];
}

export interface SessionExecutionContext {
  execution_id: string;
  context_target: JsonValue;
  visibility: string;
  durability: string;
  write_policy: string;
}

export interface SessionTaskTurn {
  turn_index: number;
  task_turn_index: number;
  has_tool_calls?: boolean | null;
  started_at: string;
  completed_at?: string | null;
}

export interface SessionTaskExecution {
  task_id: string;
  trace_id: string;
  plan_id?: string | null;
  run_id?: string | null;
  agent_id: string;
  title?: string | null;
  prompt: string;
  status: string;
  queue_depth: number;
  task_turn_count: number;
  execution: SessionExecutionContext;
  turns: SessionTaskTurn[];
  branch_outcome?: JsonValue;
  error?: string | null;
  started_at: string;
  completed_at?: string | null;
}

export interface SessionPlanExecution {
  plan_id: string;
  title?: string | null;
  status: string;
  total_tasks: number;
  completed_tasks: number;
  started_at: string;
  completed_at?: string | null;
}

export interface SessionExecution {
  tasks: SessionTaskExecution[];
  plans: SessionPlanExecution[];
  event_limit: number;
  truncated: boolean;
}

export interface SessionDetail {
  session: SessionSummary;
  branches: SessionBranch[];
  messages: SessionMessage[];
  tool_executions: ToolExecution[];
  efficiency?: SessionEfficiency | null;
  execution: SessionExecution;
  message_window?: { offset: number; total: number } | null;
}

export interface PerfProcessMemory {
  rss_kb?: number | null;
  pss_kb?: number | null;
  pss_anon_kb?: number | null;
  pss_file_kb?: number | null;
  pss_shmem_kb?: number | null;
}

export interface PerfOperationSummary {
  operation_id: string;
  operation: string;
  session_id?: string | null;
  pid: number;
  build_profile?: string | null;
  started_at_ms: number;
  completed_at_ms: number;
  elapsed_us: number;
  outcome: string;
  start_fields: Record<string, JsonValue>;
  fields: Record<string, JsonValue>;
  memory_start?: PerfProcessMemory | null;
  memory_end?: PerfProcessMemory | null;
  memory_peak?: PerfProcessMemory | null;
  rss_delta_kb?: number | null;
  pss_delta_kb?: number | null;
}

export interface TaskStatus {
  request_id: string;
  agent_id: string;
  slot_id: string;
  trace_id: string;
  title?: string | null;
  prompt_preview: string;
  state: string;
  runtime_task_id?: string | null;
  execution: LiveExecution;
  status?: string | null;
  task_turn_count?: number | null;
  branch_outcome?: JsonValue | null;
  promotion_candidate?: { session_id: string; source_turn_id: number } | null;
  promoted_branch?: SessionBranch | null;
  output?: string | null;
  assistant_content?: JsonValue[] | null;
  error?: string | null;
}

export interface UiBadge {
  target: string;
  count?: number;
  label?: string;
  level?: "info" | "success" | "warning" | "error";
}

export interface UiMenuItem {
  id?: string;
  label: string;
  opens: string;
  icon?: string;
  badge?: string;
  items?: UiMenuItem[];
}

export interface UiMenu {
  title: string;
  items: UiMenuItem[];
}

export interface UiFormField {
  name: string;
  label: string;
  kind?: string;
  default?: JsonValue;
  required?: boolean;
  options?: JsonValue[];
}

export interface UiNode {
  kind: "section" | "text" | "action" | "list" | "activity" | "detail" | "form" | "report" | "chart";
  id?: string;
  title?: string;
  text?: string;
  label?: string;
  action?: string;
  params?: JsonValue;
  confirm?: boolean;
  source?: string;
  where?: Record<string, JsonValue>;
  fields?: Array<string | UiFormField>;
  sort?: string[];
  limit?: number;
  intent?: string;
  as?: string;
  prompt?: string;
  item_id?: string;
  nodes?: UiNode[];
}

export interface UiScreen {
  id: string;
  title: string;
  presentation?: string;
  nodes: UiNode[];
}

export interface UiPane extends UiScreen {}

export interface UiApp {
  id: string;
  source?: {
    harness_id?: string;
    agent_id?: string;
  };
  definition?: {
    id: string;
    title: string;
    about?: string;
    icon?: string;
  } | null;
  screens: Record<string, UiScreen>;
  panes: Record<string, UiPane>;
  menus: UiMenu[];
  opens_with?: string | null;
  badges: Record<string, UiBadge>;
}

export interface TurinStatus {
  web: {
    ready: boolean;
    version: string;
    bind: string;
    connection_kind: "local" | "remote";
    connection_target: string;
  };
  snapshot: {
    health: {
      ready: boolean;
      version: string;
      issue_count: number;
      agent_count: number;
      running_agent_count: number;
      active_task_count: number;
      queued_task_count: number;
      awaiting_result_count: number;
    };
    status: {
      registry: { agents: AgentSummary[]; issues: Array<{ path: string; message: string }> };
      agent_runtimes: AgentRuntime[];
    };
    live_sessions: LiveSession[];
    sessions: SessionSummary[];
    tasks: TaskStatus[];
  };
  ui: {
    apps: Record<string, UiApp>;
  };
}

export interface UiListRequest {
  source: string;
  where?: Record<string, JsonValue>;
  limit?: number;
}

export interface WorkItem {
  id: number;
  public_id?: string;
  kind?: string;
  status?: string;
  priority?: number;
  payload?: JsonValue;
  metadata?: Record<string, JsonValue>;
  action?: { name: string; params?: JsonValue } | null;
  [key: string]: unknown;
}

export interface WorklistDetail {
  id: number;
  public_id: string;
  name: string;
  scope_ref: string;
  metadata?: JsonValue;
  created_at: string;
  updated_at: string;
}

export interface WorklistItem {
  id: number;
  public_id: string;
  worklist_id: string;
  parent_id?: string | null;
  title: string;
  kind: string;
  status: string;
  paused: boolean;
  priority: number;
  metadata?: JsonValue;
  claim_agent_id?: string | null;
  completed_at?: string | null;
  failure_reason?: string | null;
  created_at: string;
  updated_at: string;
}

export interface MemoryDetail {
  public_id: string;
  scope_kind: string;
  scope_key: string;
  content: string;
  metadata?: JsonValue;
  storage: "embedded" | "lexical_only" | string;
  embedding_key?: string | null;
  embedding_dimensions?: number | null;
  weight: number;
  retrieval_count: number;
  last_retrieved_at?: string | null;
  superseded_at?: string | null;
  created_at: string;
}

export interface MemoryScope {
  scope_kind: string;
  scope_key: string;
  count: number;
}

export interface MemoryList {
  memories: MemoryDetail[];
  scopes: MemoryScope[];
  total: number;
  offset: number;
  limit: number;
}

export interface UiListResult {
  request: UiListRequest;
  list: { items: WorkItem[] };
}

export interface TurinEvent {
  type: string;
  data: Record<string, JsonValue>;
}
