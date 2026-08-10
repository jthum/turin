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
  created_at: string;
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

export interface UiListResult {
  request: UiListRequest;
  list: { items: WorkItem[] };
}

export interface TurinEvent {
  type: string;
  data: Record<string, JsonValue>;
}
