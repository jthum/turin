<script lang="ts">
  import { onMount } from "svelte";
  import type { TurinClient } from "../lib/TurinClient";
  import type {
    HarnessDetail,
    HarnessIssue,
    HarnessRuntime,
    HarnessValidation,
    TurinStatus,
    UiApp,
  } from "../lib/types";
  import { humanize } from "../lib/format";
  import Icon from "./Icon.svelte";

  export let client: TurinClient;
  export let status: TurinStatus;
  export let onOpenApp: (appId: string) => void;
  export let onStatusChanged: () => Promise<void>;

  let harnesses: HarnessRuntime[] = [];
  let selectedId = "";
  let detail: HarnessDetail | null = null;
  let issues: HarnessIssue[] = [];
  let validation: HarnessValidation | null = null;
  let loading = true;
  let detailLoading = false;
  let operation = "";
  let error = "";
  let notice = "";
  let createOpen = false;
  let deleteOpen = false;
  let newHarnessId = "";

  $: selectedRuntime = harnesses.find(harness => harness.harness_id === selectedId) ?? null;
  $: selectedType = harnessType(selectedId);
  $: managed = selectedType === "Shared";
  $: canDelete = managed && (detail?.bound_agents.length ?? 0) === 0;
  $: harnessApps = Object.values(status.ui.apps).filter(app => app.source?.harness_id === selectedId) as UiApp[];
  $: healthyCount = harnesses.filter(harness => issueCount(harness.harness_id) === 0).length;

  onMount(loadHarnesses);

  function harnessType(id: string): "Bootstrap" | "Agent local" | "Shared" {
    if (id === "default") return "Bootstrap";
    if (id.startsWith("agent::")) return "Agent local";
    return "Shared";
  }

  function displayName(id: string): string {
    return id === "default" ? "Default harness" : humanize(id.replace(/^agent::/, ""));
  }

  function issueCount(id: string): number {
    return status.snapshot.status.registry.issues.filter(issue => {
      if (id === "default") return detail?.harness_id === id && issues.some(item => item.path === issue.path);
      return issue.path.includes(`/${id}/`) || issue.path.endsWith(`/${id}`);
    }).length;
  }

  async function loadHarnesses(preferredId?: string) {
    loading = true;
    error = "";
    try {
      harnesses = await client.harnesses();
      const nextId = preferredId && harnesses.some(item => item.harness_id === preferredId)
        ? preferredId
        : harnesses.some(item => item.harness_id === selectedId)
          ? selectedId
          : harnesses.find(item => item.harness_id === "default")?.harness_id ?? harnesses[0]?.harness_id ?? "";
      if (nextId) await selectHarness(nextId);
      else {
        selectedId = "";
        detail = null;
        issues = [];
      }
    } catch (reason) {
      error = messageFor(reason);
    } finally {
      loading = false;
    }
  }

  async function selectHarness(id: string) {
    selectedId = id;
    detailLoading = true;
    error = "";
    validation = null;
    try {
      const result = await client.harness(id);
      detail = result.harness;
      issues = result.issues;
    } catch (reason) {
      detail = null;
      issues = [];
      error = messageFor(reason);
    } finally {
      detailLoading = false;
    }
  }

  async function validateSelected() {
    if (!selectedId || operation) return;
    operation = "validate";
    error = "";
    notice = "";
    try {
      validation = await client.validateHarness(selectedId);
      notice = `${displayName(selectedId)} passed validation.`;
    } catch (reason) {
      validation = null;
      error = messageFor(reason);
    } finally {
      operation = "";
    }
  }

  async function reloadSelected() {
    if (!selectedId || operation) return;
    operation = "reload";
    error = "";
    notice = "";
    try {
      const result = await client.reloadHarness(selectedId);
      detail = result.harness;
      issues = result.issues;
      validation = null;
      await Promise.all([loadInventoryOnly(), onStatusChanged()]);
      notice = `${displayName(selectedId)} reloaded atomically.`;
    } catch (reason) {
      error = messageFor(reason);
    } finally {
      operation = "";
    }
  }

  async function createHarness() {
    const id = newHarnessId.trim();
    if (!id || operation) return;
    operation = "create";
    error = "";
    notice = "";
    try {
      const result = await client.createHarness(id);
      detail = result.harness;
      issues = result.issues;
      selectedId = result.harness.harness_id;
      createOpen = false;
      newHarnessId = "";
      await Promise.all([loadInventoryOnly(), onStatusChanged()]);
      notice = `${displayName(selectedId)} was created and loaded.`;
    } catch (reason) {
      error = messageFor(reason);
    } finally {
      operation = "";
    }
  }

  async function deleteSelected() {
    if (!selectedId || !canDelete || operation) return;
    const deleted = selectedId;
    operation = "delete";
    error = "";
    notice = "";
    try {
      await client.deleteHarness(deleted);
      deleteOpen = false;
      selectedId = "";
      detail = null;
      await Promise.all([loadHarnesses(), onStatusChanged()]);
      notice = `${displayName(deleted)} was deleted.`;
    } catch (reason) {
      error = messageFor(reason);
    } finally {
      operation = "";
    }
  }

  async function loadInventoryOnly() {
    harnesses = await client.harnesses();
  }

  function messageFor(reason: unknown): string {
    return reason instanceof Error ? reason.message : String(reason);
  }
</script>

<section class="studio-view">
  <header class="view-header studio-header">
    <div>
      <span class="eyebrow">Runtime authoring</span>
      <h1>Harness Studio</h1>
      <p>Inspect, validate, and operate the Lua applications shaping this Turin runtime.</p>
    </div>
    <div class="studio-header-actions">
      <button class="secondary-button" disabled={loading} onclick={() => loadHarnesses(selectedId)}><Icon name="refresh" size={15} />Refresh</button>
      <button class="primary-button" onclick={() => { createOpen = true; error = ""; }}><Icon name="plus" size={15} />New harness</button>
    </div>
  </header>

  <div class="studio-scroll">
    <section class="studio-pulse" aria-label="Harness summary">
      <div><span>Loaded</span><strong>{harnesses.length}</strong><small>runtime harnesses</small></div>
      <div><span>Healthy</span><strong>{healthyCount}</strong><small>without reported issues</small></div>
      <div><span>Applications</span><strong>{Object.keys(status.ui.apps).length}</strong><small>semantic UI surfaces</small></div>
      <div><span>Bindings</span><strong>{harnesses.reduce((sum, item) => sum + item.bound_agents.length, 0)}</strong><small>agent relationships</small></div>
    </section>

    {#if notice}<div class="studio-notice"><i></i><span>{notice}</span><button aria-label="Dismiss" onclick={() => notice = ""}><Icon name="close" size={13} /></button></div>{/if}
    {#if error}<div class="studio-error"><Icon name="activity" size={15} /><span>{error}</span><button aria-label="Dismiss" onclick={() => error = ""}><Icon name="close" size={13} /></button></div>{/if}

    <div class="studio-layout">
      <aside class="harness-roster">
        <header><span>Harnesses</span><small>{harnesses.length}</small></header>
        <div class="harness-list">
          {#if loading}
            <div class="studio-empty compact">Loading harnesses...</div>
          {:else}
            {#each harnesses as harness (harness.harness_id)}
              <button class:active={selectedId === harness.harness_id} onclick={() => selectHarness(harness.harness_id)}>
                <span class:issue={issueCount(harness.harness_id) > 0} class="harness-glyph"><Icon name={harness.harness_id.startsWith("agent::") ? "route" : "code"} size={15} /><i></i></span>
                <span><strong>{displayName(harness.harness_id)}</strong><small>{harnessType(harness.harness_id)} · {harness.loaded_scripts.length} scripts</small></span>
                <Icon name="chevron" size={13} />
              </button>
            {:else}
              <div class="studio-empty compact">No harnesses are loaded.</div>
            {/each}
          {/if}
        </div>
      </aside>

      <main class="harness-stage">
        {#if detailLoading}
          <div class="studio-empty tall"><span class="empty-mark"><Icon name="code" size={17} /></span><strong>Loading harness</strong></div>
        {:else if detail}
          <header class="harness-stage-header">
            <div>
              <span class="harness-kind">{selectedType}</span>
              <h2>{displayName(detail.harness_id)}</h2>
              <code title={detail.directory}>{detail.directory}</code>
            </div>
            <div class="harness-stage-actions">
              <button class="secondary-button" disabled={Boolean(operation)} onclick={validateSelected}>{operation === "validate" ? "Validating..." : "Validate"}</button>
              <button class="primary-button" disabled={Boolean(operation)} onclick={reloadSelected}>{operation === "reload" ? "Reloading..." : "Reload"}</button>
            </div>
          </header>

          <section class="harness-facts">
            <div><span>Scripts</span><strong>{detail.loaded_scripts.length}</strong><small>loaded modules</small></div>
            <div><span>Agents</span><strong>{detail.bound_agents.length}</strong><small>{detail.bound_agents.length ? detail.bound_agents.join(", ") : "not bound"}</small></div>
            <div><span>UI intents</span><strong>{detail.ui_intents.length}</strong><small>declared signals</small></div>
            <div><span>Apps</span><strong>{harnessApps.length}</strong><small>renderable surfaces</small></div>
          </section>

          <div class="harness-content-grid">
            <section class="studio-card scripts-card">
              <header><div><span class="eyebrow">Source map</span><h3>Loaded scripts</h3></div><small>{detail.loaded_scripts.length}</small></header>
              <div class="script-list">
                {#each detail.loaded_scripts as script (script)}
                  <div><span class="script-icon"><Icon name="code" size={14} /></span><span><strong>{script}</strong><small>Lua module</small></span></div>
                {:else}<div class="studio-empty compact">No Lua scripts were loaded.</div>{/each}
              </div>
              {#if detail.watched_roots.length}
                <footer><span>Watching</span>{#each detail.watched_roots as root}<code title={root}>{root}</code>{/each}</footer>
              {/if}
            </section>

            <section class="studio-card apps-card">
              <header><div><span class="eyebrow">Client surfaces</span><h3>Harness applications</h3></div><small>{harnessApps.length}</small></header>
              <div class="studio-app-list">
                {#each harnessApps as app (app.id)}
                  <button onclick={() => onOpenApp(app.id)}><span class="app-glyph"><Icon name="grid" size={15} /></span><span><strong>{app.definition?.title ?? humanize(app.id)}</strong><small>{Object.keys(app.screens).length} screens · {app.menus.length} menus</small></span><Icon name="chevron" size={13} /></button>
                {:else}<div class="studio-empty compact"><strong>No application declared</strong><span>This harness can still shape agents, tools, and runtime behavior.</span></div>{/each}
              </div>
            </section>

            <section class:has-issues={issues.length > 0} class="studio-card diagnostics-card">
              <header><div><span class="eyebrow">Diagnostics</span><h3>{issues.length ? `${issues.length} reported ${issues.length === 1 ? "issue" : "issues"}` : "No reported issues"}</h3></div><span class:healthy={!issues.length} class="health-pill"><i></i>{issues.length ? "Attention" : "Healthy"}</span></header>
              {#if validation}
                <div class="validation-result"><Icon name="check" size={15} /><span><strong>Validation passed</strong><small>{validation.script_count} scripts checked against the current runtime.</small></span></div>
              {/if}
              <div class="issue-list">
                {#each issues as issue (issue.path + issue.message)}
                  <article><span><Icon name="activity" size={14} /></span><div><strong>{issue.message}</strong><code title={issue.path}>{issue.path}</code></div></article>
                {:else}
                  {#if !validation}<div class="studio-empty compact">Validate before reloading to check the complete script set.</div>{/if}
                {/each}
              </div>
            </section>

            <section class="studio-card lifecycle-card">
              <header><div><span class="eyebrow">Lifecycle</span><h3>Runtime ownership</h3></div></header>
              <dl>
                <div><dt>Type</dt><dd>{selectedType}</dd></div>
                <div><dt>Reload</dt><dd>Atomic, preserving the active version on failure</dd></div>
                <div><dt>Agents</dt><dd>{detail.bound_agents.length ? detail.bound_agents.join(", ") : "No active bindings"}</dd></div>
              </dl>
              {#if managed}
                <footer><div><strong>Delete shared harness</strong><span>{canDelete ? "Removes its managed directory after confirmation." : "Unbind every agent before deletion."}</span></div><button class="danger-button" disabled={!canDelete || Boolean(operation)} onclick={() => deleteOpen = true}>Delete</button></footer>
              {:else}
                <footer><div><strong>Managed elsewhere</strong><span>{selectedType === "Bootstrap" ? "Configured by the bootstrap Turin config." : "Owned by the agent-local runtime directory."}</span></div></footer>
              {/if}
            </section>
          </div>
        {:else}
          <div class="studio-empty tall"><span class="empty-mark"><Icon name="code" size={17} /></span><strong>Select a harness</strong><span>Its scripts, bindings, applications, and diagnostics will appear here.</span></div>
        {/if}
      </main>
    </div>
  </div>
</section>

{#if createOpen}
  <div class="overlay confirm-overlay" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) createOpen = false; }}>
    <div class="confirm-dialog studio-dialog" role="dialog" aria-modal="true" aria-labelledby="create-harness-title">
      <header><span class="dialog-mark"><Icon name="code" /></span><button class="dialog-close" aria-label="Close" onclick={() => createOpen = false}><Icon name="close" /></button></header>
      <h2 id="create-harness-title">Create a shared harness</h2>
      <p>Turin will scaffold, discover, and load a managed Lua harness in the workspace.</p>
      <label class="studio-field"><span>Harness ID</span><input bind:value={newHarnessId} placeholder="research-desk" autocomplete="off" /></label>
      <div class="dialog-actions"><button onclick={() => createOpen = false}>Cancel</button><button class="primary" disabled={!newHarnessId.trim() || Boolean(operation)} onclick={createHarness}>{operation === "create" ? "Creating..." : "Create harness"}</button></div>
    </div>
  </div>
{/if}

{#if deleteOpen && detail}
  <div class="overlay confirm-overlay" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) deleteOpen = false; }}>
    <div class="confirm-dialog studio-dialog" role="alertdialog" aria-modal="true" aria-labelledby="delete-harness-title">
      <header><span class="dialog-mark danger"><Icon name="activity" /></span><button class="dialog-close" aria-label="Close" onclick={() => deleteOpen = false}><Icon name="close" /></button></header>
      <h2 id="delete-harness-title">Delete {displayName(detail.harness_id)}?</h2>
      <p>This removes the managed harness directory. Turin refuses the operation while agents remain bound.</p>
      <div class="dialog-actions"><button onclick={() => deleteOpen = false}>Keep harness</button><button class="danger-confirm" disabled={Boolean(operation)} onclick={deleteSelected}>{operation === "delete" ? "Deleting..." : "Delete harness"}</button></div>
    </div>
  </div>
{/if}
