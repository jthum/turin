<script lang="ts">
  import type { AgentSummary, JsonValue, LiveSession, TaskStatus, TurinStatus } from "../lib/types";
  import { effectiveAgents } from "../lib/agents";
  import { humanize, sameSession, titleForSession } from "../lib/format";
  import Icon from "./Icon.svelte";

  export let status: TurinStatus;
  export let onSession: (sessionId: string) => void;
  export let onRefresh: () => Promise<void>;

  let selectedAgentId = "";
  let selectedTaskId: string | null = null;
  let refreshing = false;

  $: agents = effectiveAgents(status);
  $: if (!selectedAgentId || !agents.some(agent => agent.id === selectedAgentId)) {
    selectedAgentId = preferredAgent(agents);
  }
  $: selectedAgent = agents.find(agent => agent.id === selectedAgentId) ?? null;
  $: selectedRuntime = status.snapshot.status.agent_runtimes.find(runtime => runtime.agent_id === selectedAgentId);
  $: selectedSessions = status.snapshot.live_sessions.filter(session => session.agent_id === selectedAgentId);
  $: selectedTasks = newestFirst(status.snapshot.tasks.filter(task => task.agent_id === selectedAgentId));
  $: traceGroups = groupByTrace(selectedTasks);
  $: selectedTask = status.snapshot.tasks.find(task => task.request_id === selectedTaskId) ?? null;
  $: workingAgents = agents.filter(agent => agentLoad(agent.id) > 0).length;
  $: runningTasks = status.snapshot.tasks.filter(task => ["running", "cancelling"].includes(task.state)).length;
  $: queuedTasks = status.snapshot.tasks.filter(task => task.state === "queued").length;

  function preferredAgent(items: AgentSummary[]): string {
    return items.find(agent => agentLoad(agent.id) > 0)?.id
      ?? items.find(agent => agent.id === "default")?.id
      ?? items[0]?.id
      ?? "";
  }

  function agentLoad(agentId: string): number {
    const runtime = status.snapshot.status.agent_runtimes.find(item => item.agent_id === agentId);
    return (runtime?.active_tasks ?? 0) + (runtime?.queued_tasks ?? 0) + (runtime?.awaiting_results ?? 0);
  }

  function newestFirst(tasks: TaskStatus[]): TaskStatus[] {
    return [...tasks].sort((left, right) => right.request_id.localeCompare(left.request_id));
  }

  function groupByTrace(tasks: TaskStatus[]): Array<{ traceId: string; tasks: TaskStatus[] }> {
    const traces = new Map<string, TaskStatus[]>();
    for (const task of tasks) {
      const traceId = task.trace_id || task.request_id;
      const group = traces.get(traceId) ?? [];
      group.push(task);
      traces.set(traceId, group);
    }
    return Array.from(traces, ([traceId, grouped]) => ({ traceId, tasks: grouped })).slice(0, 12);
  }

  function shortId(value: string): string {
    const parts = value.split("_");
    const bare = value.includes("_") ? parts[parts.length - 1] ?? value : value;
    return bare.length > 9 ? bare.slice(-9) : bare;
  }

  function taskLabel(task: TaskStatus): string {
    return task.title?.trim() || task.prompt_preview || `Request ${shortId(task.request_id)}`;
  }

  function taskState(task: TaskStatus): string {
    return task.state === "completed" ? task.status ?? "completed" : task.state;
  }

  function taskTone(task: TaskStatus): "active" | "success" | "warning" | "danger" | "quiet" {
    const state = taskState(task);
    if (["running"].includes(state)) return "active";
    if (["success"].includes(state)) return "success";
    if (["queued", "cancelling", "cancelled", "conflict", "max_turns"].includes(state)) return "warning";
    if (["error", "killed", "rejected"].includes(state)) return "danger";
    return "quiet";
  }

  function sessionForTask(task: TaskStatus): LiveSession | undefined {
    return status.snapshot.live_sessions.find(session => session.agent_id === task.agent_id && session.slot_id === task.slot_id);
  }

  function sessionTitle(session: LiveSession): string {
    const persisted = status.snapshot.sessions.find(item => sameSession(item.session_id, session.session_id));
    return persisted ? titleForSession(persisted) : `Session ${shortId(session.session_id.split("@", 1)[0] ?? session.session_id)}`;
  }

  function contextTarget(target: JsonValue): string {
    if (!target || Array.isArray(target) || typeof target !== "object") return "Default context";
    const record = target as Record<string, JsonValue>;
    const kind = typeof record.kind === "string" ? record.kind : "context";
    if (kind === "branch_head") return record.branch_head_id == null ? "Active branch head" : `Branch head ${record.branch_head_id}`;
    if (kind === "turn_id") return `Exact turn ${record.turn_id}`;
    if (kind === "selected_path") return `Selected path · ${Array.isArray(record.turn_ids) ? record.turn_ids.length : 0} turns`;
    if (kind === "external_reference") return `External · ${String(record.reference ?? "reference")}`;
    if (kind === "summary_source") return `Summary source · turn ${record.source_turn_id}`;
    return humanize(kind);
  }

  function outcomeKind(value: JsonValue | undefined): string | null {
    if (!value || Array.isArray(value) || typeof value !== "object") return null;
    const kind = (value as Record<string, JsonValue>).kind;
    return typeof kind === "string" ? humanize(kind) : "Branch result";
  }

  function outputPreview(task: TaskStatus): string | null {
    return task.error || task.output || null;
  }

  async function refresh() {
    if (refreshing) return;
    refreshing = true;
    try {
      await onRefresh();
    } finally {
      refreshing = false;
    }
  }
</script>

<section class="orchestration-view">
  <header class="view-header orchestration-header">
    <div>
      <span class="eyebrow">Runtime-wide</span>
      <h1>Agent Orchestration</h1>
      <p>See which agents are awake, where work is running, and how delegated executions resolve.</p>
    </div>
    <div class="orchestration-header-actions">
      <span class:busy={runningTasks > 0} class="orchestration-live"><i></i>{runningTasks > 0 ? `${runningTasks} running` : "Runtime ready"}</span>
      <button class="icon-button" aria-label="Refresh orchestration" title="Refresh" disabled={refreshing} onclick={refresh}><Icon name="refresh" size={15} /></button>
    </div>
  </header>

  <div class="orchestration-scroll">
    <section class="orchestration-pulse" aria-label="Runtime summary">
      <div><span>Configured</span><strong>{agents.length}</strong><small>agents available</small></div>
      <div><span>Working</span><strong>{workingAgents}</strong><small>agents with work</small></div>
      <div><span>Live sessions</span><strong>{status.snapshot.live_sessions.length}</strong><small>independent slots</small></div>
      <div><span>In flight</span><strong>{runningTasks + queuedTasks}</strong><small>{runningTasks} running · {queuedTasks} queued</small></div>
    </section>

    <div class:with-inspector={Boolean(selectedTask)} class="orchestration-grid">
      <aside class="agent-roster">
        <header><span>Agents</span><small>{agents.length}</small></header>
        <div class="agent-list">
          {#each agents as agent (agent.id)}
            {@const runtime = status.snapshot.status.agent_runtimes.find(item => item.agent_id === agent.id)}
            {@const slots = status.snapshot.live_sessions.filter(session => session.agent_id === agent.id).length}
            {@const load = agentLoad(agent.id)}
            <button class:active={selectedAgentId === agent.id} onclick={() => { selectedAgentId = agent.id; selectedTaskId = null; }}>
              <span class:working={load > 0} class="agent-avatar">{agent.id.slice(0, 1).toUpperCase()}<i></i></span>
              <span class="agent-copy"><strong>{humanize(agent.id)}</strong><small>{agent.model || "Primary agent"}</small></span>
              <span class="agent-load">
                {#if load > 0}<b>{load}</b>{:else if runtime?.running}<i></i>{/if}
                <small>{slots ? `${slots} ${slots === 1 ? "slot" : "slots"}` : "cold"}</small>
              </span>
            </button>
          {:else}
            <div class="orchestration-empty compact"><strong>No agents configured</strong><span>Add an enabled agent to the Turin configuration.</span></div>
          {/each}
        </div>
      </aside>

      <main class="agent-stage">
        {#if selectedAgent}
          <header class="agent-stage-header">
            <div>
              <span class:working={Boolean(selectedRuntime?.running)} class="agent-presence"><i></i>{selectedRuntime?.running ? "Runtime awake" : "Starts on demand"}</span>
              <h2>{humanize(selectedAgent.id)}</h2>
              <p>{selectedAgent.provider && selectedAgent.model ? `${selectedAgent.provider} / ${selectedAgent.model}` : "Primary agent"}<span></span>{selectedAgent.harness_ref || "default"}</p>
            </div>
            <div class="agent-stage-counts">
              <span><strong>{selectedRuntime?.active_tasks ?? 0}</strong> active</span>
              <span><strong>{selectedRuntime?.queued_tasks ?? 0}</strong> queued</span>
              <span><strong>{selectedRuntime?.awaiting_results ?? 0}</strong> awaiting</span>
            </div>
          </header>

          <section class="slot-section">
            <div class="section-heading"><div><span class="eyebrow">Runtime slots</span><h3>Independent sessions</h3></div><small>{selectedSessions.length} live</small></div>
            <div class="slot-strip">
              {#each selectedSessions as session (session.slot_id)}
                <button class:working={session.active_tasks > 0} onclick={() => onSession(session.session_id)}>
                  <span class="slot-glyph"><i></i></span>
                  <span><strong>{sessionTitle(session)}</strong><small>{session.slot_id} · {session.active_tasks ? `${session.active_tasks} active` : session.queued_tasks ? `${session.queued_tasks} queued` : "idle"}</small></span>
                  <Icon name="chevron" size={13} />
                </button>
              {:else}
                <div class="orchestration-empty horizontal"><span class="empty-mark"><Icon name="activity" size={15} /></span><div><strong>No live session</strong><span>This agent stays cold until work is submitted.</span></div></div>
              {/each}
            </div>
          </section>

          <section class="trace-section">
            <div class="section-heading"><div><span class="eyebrow">Delegated work</span><h3>Execution traces</h3></div><small>{selectedTasks.length} retained</small></div>
            <div class="trace-list">
              {#each traceGroups as trace (trace.traceId)}
                <article class="trace-card">
                  <header><span><Icon name="route" size={14} />Trace {shortId(trace.traceId)}</span><small>{trace.tasks.length} {trace.tasks.length === 1 ? "execution" : "executions"}</small></header>
                  <div class="trace-flow">
                    {#each trace.tasks as task (task.request_id)}
                      <button class:selected={selectedTaskId === task.request_id} onclick={() => selectedTaskId = task.request_id}>
                        <span class={`task-state ${taskTone(task)}`}><i></i></span>
                        <span class="task-copy"><strong>{taskLabel(task)}</strong><small>{humanize(taskState(task))} · {task.slot_id}{task.task_turn_count ? ` · ${task.task_turn_count} turns` : ""}</small></span>
                        {#if task.state === "running"}<span class="task-running-bars"><i></i><i></i><i></i></span>{:else}<Icon name="chevron" size={13} />{/if}
                      </button>
                    {/each}
                  </div>
                </article>
              {:else}
                <div class="orchestration-empty trace-empty">
                  <span class="empty-mark"><Icon name="route" size={17} /></span>
                  <strong>No delegated work yet</strong>
                  <span>Peer-agent submissions, parallel executions, and their outcomes will appear here.</span>
                </div>
              {/each}
            </div>
          </section>
        {/if}
      </main>

      {#if selectedTask}
        <aside class="task-inspector">
          <header><div><span class="eyebrow">Execution detail</span><h2>{taskLabel(selectedTask)}</h2></div><button aria-label="Close execution detail" onclick={() => selectedTaskId = null}><Icon name="close" size={15} /></button></header>
          <div class="task-inspector-status"><span class={`task-state ${taskTone(selectedTask)}`}><i></i></span><div><strong>{humanize(taskState(selectedTask))}</strong><small>{humanize(selectedTask.agent_id)} · {selectedTask.slot_id}</small></div></div>

          <dl class="task-facts">
            <div><dt>Context</dt><dd>{contextTarget(selectedTask.execution.context_target)}</dd></div>
            <div><dt>Write policy</dt><dd>{humanize(selectedTask.execution.write_policy)}</dd></div>
            <div><dt>Visibility</dt><dd>{humanize(selectedTask.execution.visibility)}</dd></div>
            <div><dt>Durability</dt><dd>{humanize(selectedTask.execution.durability)}</dd></div>
            {#if selectedTask.task_turn_count !== null && selectedTask.task_turn_count !== undefined}<div><dt>Turns</dt><dd>{selectedTask.task_turn_count}</dd></div>{/if}
            {#if outcomeKind(selectedTask.branch_outcome)}<div><dt>Outcome</dt><dd>{outcomeKind(selectedTask.branch_outcome)}</dd></div>{/if}
          </dl>

          {#if outputPreview(selectedTask)}
            <section class:error={Boolean(selectedTask.error)} class="task-result"><span>{selectedTask.error ? "Error" : "Result"}</span><p>{outputPreview(selectedTask)}</p></section>
          {/if}

          <section class="task-identifiers">
            <span>Request <code>{selectedTask.request_id}</code></span>
            <span>Trace <code>{selectedTask.trace_id}</code></span>
            <span>Execution <code>{selectedTask.execution.execution_id}</code></span>
          </section>

          {#if sessionForTask(selectedTask)}
            {@const taskSession = sessionForTask(selectedTask)!}
            <button class="task-session-link" onclick={() => onSession(taskSession.session_id)}><span><strong>Open conversation</strong><small>{sessionTitle(taskSession)}</small></span><Icon name="chevron" size={14} /></button>
          {/if}
        </aside>
      {/if}
    </div>
  </div>
</section>
