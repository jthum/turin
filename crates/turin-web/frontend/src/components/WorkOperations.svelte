<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import type { EventSubscription, TurinClient } from "../lib/TurinClient";
  import { effectiveAgents } from "../lib/agents";
  import { fullDate, humanize, shortDate } from "../lib/format";
  import type {
    JsonValue,
    ScheduleCreateInput,
    ScheduleJob,
    ScheduleRun,
    TaskStatus,
    TurinStatus,
    WorklistDetail,
    WorklistItem,
  } from "../lib/types";
  import Icon from "./Icon.svelte";

  export let client: TurinClient;
  export let status: TurinStatus;
  export let onSession: (sessionId: string) => void;
  export let onStatusChanged: () => Promise<void>;

  let activeTab: "worklists" | "schedules" | "executions" = "worklists";
  let worklists: WorklistDetail[] = [];
  let selectedWorklistId = "";
  let worklistItems: WorklistItem[] = [];
  let selectedItemId = "";
  let itemFilter = "all";
  let schedules: ScheduleJob[] = [];
  let selectedScheduleId = "";
  let scheduleRuns: ScheduleRun[] = [];
  let selectedTaskId = "";
  let loading = true;
  let operation = "";
  let error = "";
  let notice = "";
  let scheduleDialog = false;
  let deleteScheduleDialog = false;
  let subscription: EventSubscription | null = null;

  let scheduleKind: "prompt" | "action" = "prompt";
  let scheduleAgent = "";
  let schedulePrompt = "";
  let scheduleAction = "";
  let scheduleActionParams = "{}";
  let scheduleRunAt = localDateTime(Date.now() + 60 * 60 * 1000);
  let scheduleRecurrence: "once" | "interval" | "daily" | "weekly" = "once";
  let scheduleInterval = 30;
  let scheduleIntervalUnit: "minutes" | "hours" | "days" = "minutes";
  let scheduleOverlap: "skip" | "queue" | "parallel" = "skip";
  let scheduleWorkKey = "";
  let scheduleMaxConcurrency = 1;
  let scheduleEnabled = true;

  $: agents = effectiveAgents(status);
  $: selectedWorklist = worklists.find(item => item.public_id === selectedWorklistId) ?? null;
  $: filteredItems = worklistItems.filter(item => itemFilter === "all" || itemState(item) === itemFilter);
  $: selectedItem = worklistItems.find(item => item.public_id === selectedItemId) ?? null;
  $: selectedSchedule = schedules.find(item => item.public_id === selectedScheduleId) ?? null;
  $: tasks = [...status.snapshot.tasks].sort((left, right) => right.request_id.localeCompare(left.request_id));
  $: selectedTask = tasks.find(task => task.request_id === selectedTaskId) ?? null;
  $: activeTaskCount = tasks.filter(task => ["queued", "running", "cancelling"].includes(task.state)).length;
  $: pendingItemCount = worklistItems.filter(item => itemState(item) === "pending").length;

  onMount(async () => {
    await loadAll();
    subscription = client.subscribe(event => {
      if (event.type.startsWith("schedule.")) void loadSchedules(selectedScheduleId);
      if (event.type === "task.updated" || event.type === "task.submitted") void onStatusChanged();
    });
  });

  onDestroy(() => subscription?.close());

  async function loadAll() {
    loading = true;
    error = "";
    try {
      await Promise.all([loadWorklists(), loadSchedules(), onStatusChanged()]);
    } catch (reason) {
      error = messageFor(reason);
    } finally {
      loading = false;
    }
  }

  async function loadWorklists(preferredId?: string) {
    worklists = await client.worklists();
    const next = preferredId && worklists.some(item => item.public_id === preferredId)
      ? preferredId
      : worklists.some(item => item.public_id === selectedWorklistId)
        ? selectedWorklistId
        : worklists[0]?.public_id ?? "";
    if (next) await selectWorklist(next);
    else {
      selectedWorklistId = "";
      worklistItems = [];
      selectedItemId = "";
    }
  }

  async function selectWorklist(id: string) {
    selectedWorklistId = id;
    worklistItems = await client.worklistItems(id, 250);
    if (!worklistItems.some(item => item.public_id === selectedItemId)) selectedItemId = "";
  }

  async function loadSchedules(preferredId?: string) {
    schedules = await client.schedules();
    const next = preferredId && schedules.some(item => item.public_id === preferredId)
      ? preferredId
      : schedules.some(item => item.public_id === selectedScheduleId)
        ? selectedScheduleId
        : schedules[0]?.public_id ?? "";
    if (next) await selectSchedule(next);
    else {
      selectedScheduleId = "";
      scheduleRuns = [];
    }
  }

  async function selectSchedule(id: string) {
    selectedScheduleId = id;
    scheduleRuns = await client.scheduleRuns(id, 50);
  }

  async function refreshActive() {
    if (loading) return;
    loading = true;
    error = "";
    try {
      if (activeTab === "worklists") await loadWorklists(selectedWorklistId);
      if (activeTab === "schedules") await loadSchedules(selectedScheduleId);
      if (activeTab === "executions") await onStatusChanged();
    } catch (reason) {
      error = messageFor(reason);
    } finally {
      loading = false;
    }
  }

  function itemState(item: WorklistItem): string {
    return item.paused ? "paused" : item.status;
  }

  function itemStatusCount(state: string): number {
    return worklistItems.filter(item => itemState(item) === state).length;
  }

  function selectItem(item: WorklistItem) {
    selectedItemId = selectedItemId === item.public_id ? "" : item.public_id;
  }

  function scheduleLabel(schedule: ScheduleJob): string {
    return schedule.kind === "action"
      ? schedule.action?.name ?? "Scheduled action"
      : schedule.prompt?.trim() || "Scheduled prompt";
  }

  function scheduleCadence(schedule: ScheduleJob): string {
    if (schedule.recurring_pattern) return humanize(schedule.recurring_pattern);
    if (schedule.interval_seconds) return `Every ${durationText(schedule.interval_seconds * 1000)}`;
    return "One time";
  }

  function scheduleState(schedule: ScheduleJob): string {
    if (!schedule.enabled) return "paused";
    if (schedule.active_run_count) return "running";
    if (schedule.failure_count && schedule.last_status === "error") return "error";
    return "scheduled";
  }

  function taskTone(task: TaskStatus): string {
    if (task.state === "running") return "active";
    if (task.state === "queued" || task.state === "cancelling") return "warning";
    if (task.status === "success") return "success";
    if (task.error || ["error", "rejected", "killed"].includes(task.status ?? task.state)) return "danger";
    return "quiet";
  }

  function canCancel(task: TaskStatus): boolean {
    return ["queued", "running"].includes(task.state);
  }

  function sessionForTask(task: TaskStatus): string | null {
    return status.snapshot.live_sessions.find(session => session.agent_id === task.agent_id && session.slot_id === task.slot_id)?.session_id ?? null;
  }

  async function toggleSelectedSchedule() {
    if (!selectedSchedule || operation) return;
    operation = "toggle";
    error = "";
    try {
      const updated = await client.toggleSchedule(selectedSchedule.public_id, !selectedSchedule.enabled);
      schedules = schedules.map(item => item.public_id === updated.public_id ? updated : item);
      notice = `${scheduleLabel(updated)} is now ${updated.enabled ? "enabled" : "paused"}.`;
    } catch (reason) {
      error = messageFor(reason);
    } finally {
      operation = "";
    }
  }

  async function deleteSelectedSchedule() {
    if (!selectedSchedule || operation) return;
    operation = "delete";
    error = "";
    try {
      const label = scheduleLabel(selectedSchedule);
      await client.deleteSchedule(selectedSchedule.public_id);
      deleteScheduleDialog = false;
      await loadSchedules();
      notice = `${label} was deleted.`;
    } catch (reason) {
      error = messageFor(reason);
    } finally {
      operation = "";
    }
  }

  function openScheduleDialog() {
    scheduleKind = "prompt";
    scheduleAgent = agents.find(agent => agent.id === "default")?.id ?? agents[0]?.id ?? "";
    schedulePrompt = "";
    scheduleAction = "";
    scheduleActionParams = "{}";
    scheduleRunAt = localDateTime(Date.now() + 60 * 60 * 1000);
    scheduleRecurrence = "once";
    scheduleInterval = 30;
    scheduleIntervalUnit = "minutes";
    scheduleOverlap = "skip";
    scheduleWorkKey = "";
    scheduleMaxConcurrency = 1;
    scheduleEnabled = true;
    error = "";
    scheduleDialog = true;
  }

  async function createSchedule() {
    if (operation) return;
    const nextRun = new Date(scheduleRunAt).getTime();
    if (!scheduleAgent || !Number.isFinite(nextRun)) {
      error = "Choose an agent and a valid first run time.";
      return;
    }
    let params: JsonValue = {};
    if (scheduleKind === "action") {
      try {
        params = JSON.parse(scheduleActionParams) as JsonValue;
      } catch {
        error = "Action parameters must be valid JSON.";
        return;
      }
    }
    const input: ScheduleCreateInput = {
      agent_id: scheduleAgent,
      next_run_unix_ms: nextRun,
      overlap_policy: scheduleOverlap,
      enabled: scheduleEnabled,
      ...(scheduleKind === "prompt" ? { prompt: schedulePrompt.trim() } : { action: { name: scheduleAction.trim(), params } }),
      ...(scheduleRecurrence === "interval" ? { interval_seconds: intervalSeconds() } : {}),
      ...(scheduleRecurrence === "daily" || scheduleRecurrence === "weekly" ? { recurring_pattern: scheduleRecurrence } : {}),
      ...(scheduleWorkKey.trim() ? { work_key: scheduleWorkKey.trim() } : {}),
      ...(scheduleMaxConcurrency > 0 ? { max_concurrency: scheduleMaxConcurrency } : {}),
    };
    if ((scheduleKind === "prompt" && !input.prompt) || (scheduleKind === "action" && !input.action?.name)) {
      error = scheduleKind === "prompt" ? "A scheduled prompt is required." : "An action name is required.";
      return;
    }
    operation = "create";
    error = "";
    try {
      const created = await client.createSchedule(input);
      scheduleDialog = false;
      await loadSchedules(created.public_id);
      notice = `${scheduleLabel(created)} was scheduled.`;
    } catch (reason) {
      error = messageFor(reason);
    } finally {
      operation = "";
    }
  }

  async function cancelTask(task: TaskStatus) {
    if (!canCancel(task) || operation) return;
    operation = `cancel:${task.request_id}`;
    error = "";
    try {
      await client.cancelTask(task.request_id);
      await onStatusChanged();
      notice = "Cancellation requested. Turin will stop at the next safe boundary.";
    } catch (reason) {
      error = messageFor(reason);
    } finally {
      operation = "";
    }
  }

  function intervalSeconds(): number {
    const multiplier = scheduleIntervalUnit === "minutes" ? 60 : scheduleIntervalUnit === "hours" ? 3600 : 86400;
    return Math.max(1, scheduleInterval) * multiplier;
  }

  function localDateTime(timestamp: number): string {
    const date = new Date(timestamp);
    return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
  }

  function dateFromUnix(timestamp: number | null | undefined): string {
    return timestamp ? fullDate(new Date(timestamp).toISOString()) : "Never";
  }

  function durationText(milliseconds: number | null | undefined): string {
    if (milliseconds === null || milliseconds === undefined) return "In progress";
    if (milliseconds < 1000) return `${milliseconds}ms`;
    if (milliseconds < 60_000) return `${(milliseconds / 1000).toFixed(1)}s`;
    if (milliseconds < 3_600_000) return `${Math.round(milliseconds / 60_000)}m`;
    return `${(milliseconds / 3_600_000).toFixed(1)}h`;
  }

  function jsonText(value: JsonValue | undefined): string {
    return value === undefined || value === null ? "No metadata" : JSON.stringify(value, null, 2);
  }

  function messageFor(reason: unknown): string {
    return reason instanceof Error ? reason.message : String(reason);
  }
</script>

<section class="operations-view">
  <header class="view-header operations-header">
    <div><span class="eyebrow">Durable runtime work</span><h1>Work Operations</h1><p>Watch queues, schedule automation, and intervene in live execution from one control surface.</p></div>
    <div class="operations-header-actions"><span class:busy={activeTaskCount > 0} class="operations-live"><i></i>{activeTaskCount ? `${activeTaskCount} in flight` : "Runtime steady"}</span><button class="secondary-button" disabled={loading} onclick={refreshActive}><Icon name="refresh" size={15} />Refresh</button></div>
  </header>

  <div class="operations-scroll">
    <div class="operations-content">
      <nav class="operations-tabs" aria-label="Work operations">
        <button class:active={activeTab === "worklists"} onclick={() => activeTab = "worklists"}><Icon name="database" size={14} /><span>Worklists</span><i>{worklists.length}</i></button>
        <button class:active={activeTab === "schedules"} onclick={() => activeTab = "schedules"}><Icon name="clock" size={14} /><span>Schedules</span><i>{schedules.length}</i></button>
        <button class:active={activeTab === "executions"} onclick={() => activeTab = "executions"}><Icon name="activity" size={14} /><span>Executions</span><i>{activeTaskCount}</i></button>
      </nav>

      {#if notice}<div class="operations-notice"><i></i><span>{notice}</span><button aria-label="Dismiss" onclick={() => notice = ""}><Icon name="close" size={13} /></button></div>{/if}
      {#if error}<div class="operations-error"><Icon name="activity" size={15} /><span>{error}</span><button aria-label="Dismiss" onclick={() => error = ""}><Icon name="close" size={13} /></button></div>{/if}

      {#if activeTab === "worklists"}
        <section class="operations-summary">
          <div><span>Collections</span><strong>{worklists.length}</strong><small>durable queues</small></div><div><span>Visible items</span><strong>{worklistItems.length}</strong><small>selected worklist</small></div><div><span>Pending</span><strong>{pendingItemCount}</strong><small>ready or dependency-blocked</small></div><div><span>Claimed</span><strong>{itemStatusCount("claimed")}</strong><small>currently owned</small></div>
        </section>
        <div class:with-inspector={Boolean(selectedItem)} class="worklist-workspace">
          <aside class="operations-roster"><header><span>Worklists</span><small>{worklists.length}</small></header><div>
            {#each worklists as worklist (worklist.public_id)}<button class:active={selectedWorklistId === worklist.public_id} onclick={() => selectWorklist(worklist.public_id)}><span class="ops-glyph"><Icon name="database" size={14} /></span><span><strong>{worklist.name}</strong><small>{worklist.scope_ref}</small></span><Icon name="chevron" size={13} /></button>{:else}<div class="ops-empty compact">No worklists yet.</div>{/each}
          </div></aside>
          <main class="worklist-stage">
            <header class="worklist-toolbar"><div><h2>{selectedWorklist?.name ?? "Worklist items"}</h2><span>{selectedWorklist?.scope_ref ?? "Select a worklist"}</span></div><div class="item-filters">{#each ["all", "pending", "claimed", "paused", "done", "failed"] as state}<button class:active={itemFilter === state} onclick={() => itemFilter = state}>{humanize(state)}{#if state !== "all"}<i>{itemStatusCount(state)}</i>{/if}</button>{/each}</div></header>
            <div class="work-items-table-wrap">
              {#if !filteredItems.length}<div class="ops-empty tall"><span class="empty-mark"><Icon name="database" size={16} /></span><strong>No matching work</strong><span>Try another status or let a harness add work to this collection.</span></div>{:else}<table class="operations-table"><thead><tr><th>Work</th><th>Kind</th><th>Status</th><th>Owner</th><th>Updated</th></tr></thead><tbody>{#each filteredItems as item (item.public_id)}<tr class:selected={selectedItemId === item.public_id} onclick={() => selectItem(item)}><td><strong>{item.title}</strong><small>{item.public_id}</small></td><td>{humanize(item.kind)}</td><td><span class={`ops-state ${itemState(item)}`}><i></i>{humanize(itemState(item))}</span></td><td>{item.claim_agent_id ?? "Unclaimed"}</td><td>{shortDate(item.updated_at)}</td></tr>{/each}</tbody></table>{/if}
            </div>
          </main>
          {#if selectedItem}<aside class="work-inspector"><header><div><span class="eyebrow">Work item</span><h2>{selectedItem.title}</h2></div><button aria-label="Close item detail" onclick={() => selectedItemId = ""}><Icon name="close" size={15} /></button></header><div class="work-inspector-scroll"><span class={`ops-state ${itemState(selectedItem)}`}><i></i>{humanize(itemState(selectedItem))}</span><dl><div><dt>Kind</dt><dd>{humanize(selectedItem.kind)}</dd></div><div><dt>Priority</dt><dd>{selectedItem.priority}</dd></div><div><dt>Owner</dt><dd>{selectedItem.claim_agent_id ?? "Unclaimed"}</dd></div><div><dt>Created</dt><dd>{fullDate(selectedItem.created_at)}</dd></div><div><dt>Updated</dt><dd>{fullDate(selectedItem.updated_at)}</dd></div>{#if selectedItem.claimed_at}<div><dt>Claimed</dt><dd>{fullDate(selectedItem.claimed_at)}</dd></div>{/if}{#if selectedItem.pause_reason}<div><dt>Paused</dt><dd>{selectedItem.pause_reason}</dd></div>{/if}{#if selectedItem.after?.length}<div><dt>Depends on</dt><dd>{selectedItem.after.join(", ")}</dd></div>{/if}</dl>{#if selectedItem.prompt}<section><span>Prompt</span><p>{selectedItem.prompt}</p></section>{/if}{#if selectedItem.action}<section><span>Action</span><strong>{selectedItem.action.name}</strong><pre>{jsonText(selectedItem.action.params)}</pre></section>{/if}{#if selectedItem.failure_reason}<section class="failure"><span>Failure</span><p>{selectedItem.failure_reason}</p></section>{/if}<details><summary>Metadata</summary><pre>{jsonText(selectedItem.metadata)}</pre></details><div class="work-identifiers"><span>Item <code>{selectedItem.public_id}</code></span>{#if selectedItem.claim_session_id}<span>Session <code>{selectedItem.claim_session_id}</code></span>{/if}{#if selectedItem.claim_execution_id}<span>Execution <code>{selectedItem.claim_execution_id}</code></span>{/if}</div></div></aside>{/if}
        </div>
      {:else if activeTab === "schedules"}
        <section class="operations-summary"><div><span>Scheduled</span><strong>{schedules.length}</strong><small>durable jobs</small></div><div><span>Enabled</span><strong>{schedules.filter(item => item.enabled).length}</strong><small>eligible to run</small></div><div><span>Active</span><strong>{schedules.reduce((sum, item) => sum + item.active_run_count, 0)}</strong><small>runs in progress</small></div><div><span>Failures</span><strong>{schedules.reduce((sum, item) => sum + item.failure_count, 0)}</strong><small>recorded attempts</small></div></section>
        <div class="schedule-actions"><div><span class="eyebrow">Automation</span><h2>Scheduled work</h2></div><button class="primary-button" onclick={openScheduleDialog}><Icon name="plus" size={14} />New schedule</button></div>
        <div class="schedule-workspace"><aside class="schedule-list">{#each schedules as schedule (schedule.public_id)}<button class:active={selectedScheduleId === schedule.public_id} onclick={() => selectSchedule(schedule.public_id)}><span class={`schedule-dot ${scheduleState(schedule)}`}><i></i></span><span><strong>{scheduleLabel(schedule)}</strong><small>{scheduleCadence(schedule)} · {humanize(schedule.agent_id)}</small></span><Icon name="chevron" size={13} /></button>{:else}<div class="ops-empty tall"><span class="empty-mark"><Icon name="clock" size={16} /></span><strong>No schedules</strong><span>Create a prompt or action that Turin should run later.</span></div>{/each}</aside>
          <main class="schedule-detail">{#if selectedSchedule}<header><div><span class={`ops-state ${scheduleState(selectedSchedule)}`}><i></i>{humanize(scheduleState(selectedSchedule))}</span><h2>{scheduleLabel(selectedSchedule)}</h2><p>{scheduleCadence(selectedSchedule)} · {humanize(selectedSchedule.overlap_policy)} overlap</p></div><div><button class="secondary-button" disabled={Boolean(operation)} onclick={toggleSelectedSchedule}>{selectedSchedule.enabled ? "Pause" : "Enable"}</button><button class="danger-button" disabled={Boolean(operation) || selectedSchedule.active_run_count > 0} onclick={() => deleteScheduleDialog = true}>Delete</button></div></header><section class="schedule-facts"><div><span>Next run</span><strong>{dateFromUnix(selectedSchedule.next_run_unix_ms)}</strong></div><div><span>Last run</span><strong>{dateFromUnix(selectedSchedule.last_run_unix_ms)}</strong></div><div><span>Agent / slot</span><strong>{selectedSchedule.agent_id} / {selectedSchedule.slot_id}</strong></div><div><span>Concurrency</span><strong>{selectedSchedule.max_concurrency ?? 1}{selectedSchedule.work_key ? ` · ${selectedSchedule.work_key}` : ""}</strong></div></section>{#if selectedSchedule.kind === "prompt"}<section class="schedule-payload"><span>Prompt</span><p>{selectedSchedule.prompt}</p></section>{:else}<section class="schedule-payload"><span>Action</span><strong>{selectedSchedule.action?.name}</strong><pre>{jsonText(selectedSchedule.action?.params)}</pre></section>{/if}<section class="schedule-runs"><header><div><span class="eyebrow">Execution history</span><h3>Recent runs</h3></div><small>{scheduleRuns.length}</small></header>{#each scheduleRuns as run (run.id)}<article><span class:active={run.active} class="run-glyph"><i></i></span><div><strong>{run.active ? "Running" : humanize(run.last_status ?? "completed")}</strong><small>{dateFromUnix(run.started_unix_ms)} · {durationText(run.duration_ms)}</small></div><code>{run.task_id}</code></article>{:else}<div class="ops-empty compact">This schedule has not run yet.</div>{/each}</section>{:else}<div class="ops-empty tall"><span class="empty-mark"><Icon name="clock" size={16} /></span><strong>Select a schedule</strong></div>{/if}</main>
        </div>
      {:else}
        <section class="operations-summary"><div><span>Retained</span><strong>{tasks.length}</strong><small>runtime task snapshots</small></div><div><span>Running</span><strong>{tasks.filter(item => item.state === "running").length}</strong><small>actively executing</small></div><div><span>Queued</span><strong>{tasks.filter(item => item.state === "queued").length}</strong><small>awaiting a slot</small></div><div><span>Agents</span><strong>{new Set(tasks.map(item => item.agent_id)).size}</strong><small>represented in work</small></div></section>
        <div class:with-inspector={Boolean(selectedTask)} class="execution-workspace"><main class="execution-list"><header><div><span class="eyebrow">Runtime ledger</span><h2>Executions</h2></div><small>{tasks.length} retained</small></header>{#each tasks as task (task.request_id)}<button class:selected={selectedTaskId === task.request_id} onclick={() => selectedTaskId = selectedTaskId === task.request_id ? "" : task.request_id}><span class={`task-state ${taskTone(task)}`}><i></i></span><span><strong>{task.title?.trim() || task.prompt_preview || "Untitled execution"}</strong><small>{humanize(task.agent_id)} · {humanize(task.status ?? task.state)} · {task.task_turn_count ?? 0} turns</small></span><code>{task.request_id.slice(-9)}</code><Icon name="chevron" size={13} /></button>{:else}<div class="ops-empty tall"><span class="empty-mark"><Icon name="activity" size={16} /></span><strong>No executions retained</strong><span>Agent tasks will appear here as they are submitted.</span></div>{/each}</main>{#if selectedTask}<aside class="execution-inspector"><header><div><span class="eyebrow">Execution</span><h2>{selectedTask.title?.trim() || selectedTask.prompt_preview}</h2></div><button aria-label="Close execution" onclick={() => selectedTaskId = ""}><Icon name="close" size={15} /></button></header><div class="execution-inspector-scroll"><span class={`ops-state ${taskTone(selectedTask)}`}><i></i>{humanize(selectedTask.status ?? selectedTask.state)}</span><dl><div><dt>Agent</dt><dd>{selectedTask.agent_id}</dd></div><div><dt>Slot</dt><dd>{selectedTask.slot_id}</dd></div><div><dt>Turns</dt><dd>{selectedTask.task_turn_count ?? 0}</dd></div><div><dt>Visibility</dt><dd>{humanize(selectedTask.execution.visibility)}</dd></div><div><dt>Durability</dt><dd>{humanize(selectedTask.execution.durability)}</dd></div><div><dt>Write policy</dt><dd>{humanize(selectedTask.execution.write_policy)}</dd></div></dl>{#if selectedTask.output || selectedTask.error}<section class:error={Boolean(selectedTask.error)}><span>{selectedTask.error ? "Error" : "Result"}</span><p>{selectedTask.error ?? selectedTask.output}</p></section>{/if}<div class="work-identifiers"><span>Request <code>{selectedTask.request_id}</code></span><span>Trace <code>{selectedTask.trace_id}</code></span><span>Execution <code>{selectedTask.execution.execution_id}</code></span></div>{#if sessionForTask(selectedTask)}<button class="inspector-link" onclick={() => onSession(sessionForTask(selectedTask)!)}>Open conversation<Icon name="chevron" size={13} /></button>{/if}{#if canCancel(selectedTask)}<button class="cancel-task-button" disabled={Boolean(operation)} onclick={() => cancelTask(selectedTask)}>{operation === `cancel:${selectedTask.request_id}` ? "Requesting cancellation..." : "Cancel execution"}</button>{/if}</div></aside>{/if}</div>
      {/if}
    </div>
  </div>
</section>

{#if scheduleDialog}
  <div class="overlay confirm-overlay" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) scheduleDialog = false; }}><div class="schedule-dialog" role="dialog" aria-modal="true" aria-labelledby="new-schedule-title"><header><div><span class="dialog-mark"><Icon name="clock" /></span><div><span class="eyebrow">Durable automation</span><h2 id="new-schedule-title">New schedule</h2></div></div><button class="dialog-close" aria-label="Close" onclick={() => scheduleDialog = false}><Icon name="close" /></button></header><div class="schedule-form-scroll"><div class="segmented-input"><button class:active={scheduleKind === "prompt"} onclick={() => scheduleKind = "prompt"}>Prompt</button><button class:active={scheduleKind === "action"} onclick={() => scheduleKind = "action"}>Action</button></div><label class="ops-field"><span>Agent</span><select bind:value={scheduleAgent}>{#each agents as agent (agent.id)}<option value={agent.id}>{humanize(agent.id)} · {agent.model}</option>{/each}</select></label>{#if scheduleKind === "prompt"}<label class="ops-field"><span>Prompt</span><textarea bind:value={schedulePrompt} rows="5" placeholder="Review pending work and summarize blockers"></textarea></label>{:else}<label class="ops-field"><span>Action name</span><input bind:value={scheduleAction} placeholder="worklist.dispatch_next" /></label><label class="ops-field"><span>Parameters (JSON)</span><textarea class="mono-input" bind:value={scheduleActionParams} rows="4"></textarea></label>{/if}<div class="ops-field-grid"><label class="ops-field"><span>First run</span><input type="datetime-local" bind:value={scheduleRunAt} /></label><label class="ops-field"><span>Recurrence</span><select bind:value={scheduleRecurrence}><option value="once">One time</option><option value="interval">Fixed interval</option><option value="daily">Daily</option><option value="weekly">Weekly</option></select></label></div>{#if scheduleRecurrence === "interval"}<div class="ops-field-grid interval-grid"><label class="ops-field"><span>Every</span><input type="number" min="1" bind:value={scheduleInterval} /></label><label class="ops-field"><span>Unit</span><select bind:value={scheduleIntervalUnit}><option value="minutes">Minutes</option><option value="hours">Hours</option><option value="days">Days</option></select></label></div>{/if}<div class="ops-field-grid"><label class="ops-field"><span>Overlap policy</span><select bind:value={scheduleOverlap}><option value="skip">Skip while active</option><option value="queue">Queue one rerun</option><option value="parallel">Allow parallel</option></select></label><label class="ops-field"><span>Max concurrency</span><input type="number" min="1" bind:value={scheduleMaxConcurrency} /></label></div><label class="ops-field"><span>Work key <small>optional shared capacity key</small></span><input bind:value={scheduleWorkKey} placeholder="release-ops" /></label><label class="ops-check"><input type="checkbox" bind:checked={scheduleEnabled} /><span><strong>Enable immediately</strong><small>Disabled schedules remain durable but do not run.</small></span></label></div><footer><button class="secondary-button" onclick={() => scheduleDialog = false}>Cancel</button><button class="primary-button" disabled={Boolean(operation)} onclick={createSchedule}>{operation === "create" ? "Scheduling..." : "Create schedule"}</button></footer></div></div>
{/if}

{#if deleteScheduleDialog && selectedSchedule}
  <div class="overlay confirm-overlay" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) deleteScheduleDialog = false; }}><div class="confirm-dialog studio-dialog" role="alertdialog" aria-modal="true" aria-labelledby="delete-schedule-title"><header><span class="dialog-mark danger"><Icon name="activity" /></span><button class="dialog-close" aria-label="Close" onclick={() => deleteScheduleDialog = false}><Icon name="close" /></button></header><h2 id="delete-schedule-title">Delete this schedule?</h2><p>{scheduleLabel(selectedSchedule)} and its durable definition will be removed. Historical task data remains separate.</p><div class="dialog-actions"><button onclick={() => deleteScheduleDialog = false}>Keep schedule</button><button class="danger-confirm" disabled={Boolean(operation)} onclick={deleteSelectedSchedule}>{operation === "delete" ? "Deleting..." : "Delete schedule"}</button></div></div></div>
{/if}
