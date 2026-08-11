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

export interface LiveSession {
  agent_id: string;
  slot_id: string;
  session_id: string;
  running: boolean;
  active_tasks: number;
  queued_tasks: number;
  current_request_id?: string | null;
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
  head_turn_index?: number | null;
  source_turn_id?: number | null;
  origin_kind: string;
  active: boolean;
  created_at: string;
}

export interface SessionDetail {
  session: SessionSummary;
  branches: SessionBranch[];
  messages: SessionMessage[];
  tool_executions: ToolExecution[];
  efficiency?: SessionEfficiency | null;
  message_window?: { offset: number; total: number } | null;
}

export interface TaskStatus {
  request_id: string;
  agent_id: string;
  slot_id: string;
  state: string;
  output?: string | null;
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
      active_task_count: number;
    };
    status: {
      registry: { agents: AgentSummary[]; issues: Array<{ path: string; message: string }> };
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
