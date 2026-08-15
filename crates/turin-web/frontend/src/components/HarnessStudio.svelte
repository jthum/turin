<script lang="ts">
  import { onMount } from "svelte";
  import type { TurinClient } from "../lib/TurinClient";
  import type {
    HarnessDetail,
    HarnessIssue,
    HarnessRuntime,
    HarnessSourceEntry,
    HarnessSourceSaveChange,
    HarnessSourceValidation,
    TurinStatus,
    UiApp,
  } from "../lib/types";
  import { humanize } from "../lib/format";
  import Icon from "./Icon.svelte";
  import LuaCodeEditor from "./LuaCodeEditor.svelte";

  export let client: TurinClient;
  export let status: TurinStatus;
  export let onOpenApp: (appId: string) => void;
  export let onStatusChanged: () => Promise<void>;

  type StudioMode = "source" | "overview" | "reference";
  type SourceBuffer = {
    path: string;
    source: string;
    originalSource: string | null;
    hash: string | null;
    deleted: boolean;
    isNew: boolean;
  };

  const referenceSections = [
    {
      title: "Lifecycle hooks",
      description: "Shape prompts and react to every stage of an agent turn.",
      docs: "docs/reference/hooks.md",
      code: `function on_turn_prepare(turn)\n  if turn.session.is_first_user_message then\n    turn.tools:include("set_conversation_title")\n  else\n    turn.tools:exclude("set_conversation_title")\n  end\n  return ALLOW\nend`,
    },
    {
      title: "Actions and tools",
      description: "Expose prose-like capabilities while keeping execution governed.",
      docs: "docs/reference/primitives.md",
      code: `action.define("release.approve", function(this, params)\n  return releases.approve(params.id)\nend)\n\ntool.declare("release_status", {\n  description = "Inspect release readiness",\n  params = { id = { type = "string", required = true } },\n  handler = function(args) return releases.status(args.id) end,\n})`,
    },
    {
      title: "Application surfaces",
      description: "Suggest semantic screens that every Turin client can interpret.",
      docs: "docs/guides/harness-guide.md",
      code: `local app = ui.app("Release Desk", { id = "release" })\n\napp:screen("queue", "Release Queue", function(screen)\n  screen:list("Open work", {\n    from = "worklists.release",\n    as = "table",\n  })\n  screen:action("Seed work", "release.seed")\nend)`,
    },
    {
      title: "Reusable modules",
      description: "Split substantial harnesses into small, testable Lua modules.",
      docs: "docs/reference/api-surface.md",
      code: `local policy = require("lib.policy")\nlocal release = require("workflows.release")\n\nreturn {\n  approve = policy.guard(release.approve),\n}`,
    },
  ];

  let harnesses: HarnessRuntime[] = [];
  let selectedId = "";
  let detail: HarnessDetail | null = null;
  let issues: HarnessIssue[] = [];
  let sources: HarnessSourceEntry[] = [];
  let buffers: Record<string, SourceBuffer> = {};
  let selectedPath = "";
  let mode: StudioMode = "source";
  let sourceValidation: HarnessSourceValidation | null = null;
  let validatedSignature = "";
  let loading = true;
  let detailLoading = false;
  let sourceLoading = false;
  let operation = "";
  let error = "";
  let notice = "";
  let createOpen = false;
  let deleteOpen = false;
  let newFileOpen = false;
  let newHarnessId = "";
  let newSourcePath = "";
  let editorLine = 1;
  let editorColumn = 1;

  $: selectedType = harnessType(selectedId);
  $: managed = selectedType === "Shared";
  $: canDelete = managed && (detail?.bound_agents.length ?? 0) === 0;
  $: harnessApps = Object.values(status.ui.apps).filter(app => app.source?.harness_id === selectedId) as UiApp[];
  $: dirtyBuffers = Object.values(buffers).filter(isDirty).sort((left, right) => left.path.localeCompare(right.path));
  $: dirtyCount = dirtyBuffers.length;
  $: candidateSignature = signatureFor(dirtyBuffers);
  $: canSave = dirtyCount > 0 && sourceValidation?.valid === true && validatedSignature === candidateSignature;
  $: activeBuffer = selectedPath ? buffers[selectedPath] ?? null : null;

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

  function isDirty(buffer: SourceBuffer): boolean {
    return buffer.deleted || buffer.isNew || buffer.source !== buffer.originalSource;
  }

  function pathIsDirty(path: string): boolean {
    const buffer = buffers[path];
    return buffer ? isDirty(buffer) : false;
  }

  function signatureFor(items: SourceBuffer[]): string {
    return JSON.stringify(items.map(item => [item.path, item.deleted ? null : item.source]));
  }

  function candidateChanges(): HarnessSourceSaveChange[] {
    return dirtyBuffers.map(buffer => ({
      path: buffer.path,
      source: buffer.deleted ? null : buffer.source,
      expected_hash: buffer.hash,
    }));
  }

  function invalidateCandidate() {
    sourceValidation = null;
    validatedSignature = "";
    notice = "";
  }

  function confirmDiscard(): boolean {
    return dirtyCount === 0 || window.confirm(`Discard ${dirtyCount} unsaved ${dirtyCount === 1 ? "file" : "files"}?`);
  }

  async function loadHarnesses(preferredId?: string) {
    if (!confirmDiscard()) return;
    loading = true;
    error = "";
    try {
      harnesses = await client.harnesses();
      const nextId = preferredId && harnesses.some(item => item.harness_id === preferredId)
        ? preferredId
        : harnesses.some(item => item.harness_id === selectedId)
          ? selectedId
          : harnesses.find(item => item.harness_id === "default")?.harness_id ?? harnesses[0]?.harness_id ?? "";
      if (nextId) await selectHarness(nextId, true, true);
      else resetSelection();
    } catch (reason) {
      error = messageFor(reason);
    } finally {
      loading = false;
    }
  }

  function resetSelection() {
    selectedId = "";
    selectedPath = "";
    detail = null;
    issues = [];
    sources = [];
    buffers = {};
    invalidateCandidate();
  }

  async function selectHarness(id: string, discardConfirmed = false, force = false) {
    if (!force && id === selectedId && detail) return;
    if (!discardConfirmed && !confirmDiscard()) return;
    selectedId = id;
    selectedPath = "";
    sources = [];
    buffers = {};
    detailLoading = true;
    error = "";
    invalidateCandidate();
    try {
      const [result] = await Promise.all([client.harness(id), loadSources(id)]);
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

  async function loadSources(id = selectedId, preferredPath = "") {
    if (!id) return;
    sourceLoading = true;
    try {
      sources = (await client.harnessSources(id)).sort((left, right) => left.path.localeCompare(right.path));
      const nextPath = sources.some(item => item.path === preferredPath)
        ? preferredPath
        : sources.find(item => item.path === "main.lua")?.path ?? sources[0]?.path ?? "";
      if (nextPath) await openSource(nextPath, id);
      else selectedPath = "";
    } finally {
      sourceLoading = false;
    }
  }

  async function openSource(path: string, id = selectedId) {
    selectedPath = path;
    if (buffers[path]) return;
    sourceLoading = true;
    error = "";
    try {
      const file = await client.harnessSource(id, path);
      buffers = {
        ...buffers,
        [path]: {
          path,
          source: file.source,
          originalSource: file.source,
          hash: file.hash,
          deleted: false,
          isNew: false,
        },
      };
    } catch (reason) {
      error = messageFor(reason);
    } finally {
      sourceLoading = false;
    }
  }

  function updateSource(source: string) {
    if (!activeBuffer) return;
    buffers = { ...buffers, [activeBuffer.path]: { ...activeBuffer, source, deleted: false } };
    invalidateCandidate();
  }

  function updateEditorCursor(line: number, column: number) {
    editorLine = line;
    editorColumn = column;
  }

  function validSourcePath(path: string): boolean {
    const parts = path.split("/");
    return path.endsWith(".lua") && !path.startsWith("/") && !path.includes("\\") && parts.every(part => part && part !== "." && part !== "..");
  }

  function createSource() {
    const path = newSourcePath.trim();
    if (!validSourcePath(path)) {
      error = "Use a harness-relative .lua path without empty, '.' or '..' segments.";
      return;
    }
    if (sources.some(item => item.path === path)) {
      error = `'${path}' already exists in this harness.`;
      return;
    }
    const source = `-- ${path}\n`;
    sources = [...sources, { path, hash: "", bytes: 0 }].sort((left, right) => left.path.localeCompare(right.path));
    buffers = { ...buffers, [path]: { path, source, originalSource: null, hash: null, deleted: false, isNew: true } };
    selectedPath = path;
    newSourcePath = "";
    newFileOpen = false;
    invalidateCandidate();
  }

  function revertSource() {
    if (!activeBuffer) return;
    if (activeBuffer.isNew) {
      const next = { ...buffers };
      delete next[activeBuffer.path];
      buffers = next;
      sources = sources.filter(item => item.path !== activeBuffer.path);
      selectedPath = sources.find(item => item.path === "main.lua")?.path ?? sources[0]?.path ?? "";
      if (selectedPath) void openSource(selectedPath);
    } else {
      buffers = { ...buffers, [activeBuffer.path]: { ...activeBuffer, source: activeBuffer.originalSource ?? "", deleted: false } };
    }
    invalidateCandidate();
  }

  function deleteSource() {
    if (!activeBuffer || !window.confirm(`Delete ${activeBuffer.path} when these changes are saved?`)) return;
    if (activeBuffer.isNew) {
      revertSource();
      return;
    }
    buffers = { ...buffers, [activeBuffer.path]: { ...activeBuffer, deleted: true } };
    invalidateCandidate();
  }

  async function validateCandidate() {
    if (!selectedId || operation) return;
    operation = "validate-source";
    error = "";
    notice = "";
    const signature = candidateSignature;
    try {
      sourceValidation = await client.validateHarnessSources(selectedId, candidateChanges());
      validatedSignature = signature;
      notice = dirtyCount
        ? `${dirtyCount} changed ${dirtyCount === 1 ? "file" : "files"} passed candidate validation.`
        : `${displayName(selectedId)} passed source validation.`;
    } catch (reason) {
      sourceValidation = null;
      validatedSignature = "";
      error = messageFor(reason);
    } finally {
      operation = "";
    }
  }

  async function saveCandidate() {
    if (!selectedId || !canSave || operation) return;
    operation = "save-source";
    error = "";
    notice = "";
    const previousPath = selectedPath;
    try {
      const result = await client.saveHarnessSources(selectedId, candidateChanges());
      buffers = {};
      sourceValidation = null;
      validatedSignature = "";
      await loadSources(selectedId, result.deleted.includes(previousPath) ? "main.lua" : previousPath);
      const changed = result.saved.length + result.deleted.length;
      notice = `${changed} ${changed === 1 ? "file" : "files"} persisted. Turin's watcher will validate and activate the new harness atomically.`;
    } catch (reason) {
      error = messageFor(reason);
    } finally {
      operation = "";
    }
  }

  async function validateRuntime() {
    if (!selectedId || operation) return;
    operation = "validate-runtime";
    error = "";
    notice = "";
    try {
      const result = await client.validateHarness(selectedId);
      notice = `${result.script_count} persisted scripts passed runtime validation.`;
    } catch (reason) {
      error = messageFor(reason);
    } finally {
      operation = "";
    }
  }

  async function reloadSelected() {
    if (!selectedId || operation || !confirmDiscard()) return;
    operation = "reload";
    error = "";
    notice = "";
    try {
      const result = await client.reloadHarness(selectedId);
      detail = result.harness;
      issues = result.issues;
      buffers = {};
      await Promise.all([loadInventoryOnly(), loadSources(selectedId, selectedPath), onStatusChanged()]);
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
      selectedId = result.harness.harness_id;
      detail = result.harness;
      issues = result.issues;
      buffers = {};
      createOpen = false;
      newHarnessId = "";
      mode = "source";
      await Promise.all([loadInventoryOnly(), loadSources(selectedId), onStatusChanged()]);
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
      resetSelection();
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

  function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  function fileName(path: string): string {
    const parts = path.split("/");
    return parts[parts.length - 1] ?? path;
  }

  function directoryName(path: string): string {
    const parts = path.split("/");
    return parts.length > 1 ? parts.slice(0, -1).join("/") : "Harness root";
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
      <p>Write, validate, and understand the Lua applications shaping this Turin runtime.</p>
    </div>
    <div class="studio-header-actions">
      <button class="secondary-button" disabled={loading || Boolean(operation)} onclick={() => loadHarnesses(selectedId)}><Icon name="refresh" size={15} />Refresh</button>
      <button class="primary-button" onclick={() => { createOpen = true; error = ""; }}><Icon name="plus" size={15} />New harness</button>
    </div>
  </header>

  <div class="studio-scroll">
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
              <div class="harness-title-line"><span class="harness-kind">{selectedType}</span>{#if dirtyCount}<span class="dirty-pill">{dirtyCount} unsaved</span>{/if}</div>
              <h2>{displayName(detail.harness_id)}</h2>
              <code title={detail.directory}>{detail.directory}</code>
            </div>
            <div class="harness-stage-actions">
              <button class="secondary-button" disabled={Boolean(operation) || dirtyCount > 0} title={dirtyCount ? "Save or discard source changes before reloading" : "Reload the persisted harness"} onclick={reloadSelected}><Icon name="refresh" size={14} />{operation === "reload" ? "Reloading..." : "Reload runtime"}</button>
            </div>
          </header>

          <nav class="studio-tabs" aria-label="Harness Studio views">
            <button class:active={mode === "source"} onclick={() => mode = "source"}><Icon name="code" size={14} />Source</button>
            <button class:active={mode === "overview"} onclick={() => mode = "overview"}><Icon name="activity" size={14} />Overview</button>
            <button class:active={mode === "reference"} onclick={() => mode = "reference"}><Icon name="spark" size={14} />API Reference</button>
          </nav>

          {#if mode === "source"}
            <section class="source-studio">
              <aside class="source-browser">
                <header><div><strong>Files</strong><span>{sources.length} Lua {sources.length === 1 ? "module" : "modules"}</span></div><button aria-label="New Lua file" title="New Lua file" onclick={() => newFileOpen = !newFileOpen}><Icon name="plus" size={14} /></button></header>
                {#if newFileOpen}
                  <form class="new-source-form" onsubmit={(event) => { event.preventDefault(); createSource(); }}>
                    <input bind:value={newSourcePath} placeholder="lib/policy.lua" />
                    <div><button type="button" onclick={() => { newFileOpen = false; newSourcePath = ""; }}>Cancel</button><button type="submit" class="primary-button">Add file</button></div>
                  </form>
                {/if}
                <div class="source-list">
                  {#each sources as source (source.path)}
                    <button class:active={selectedPath === source.path} class:dirty={pathIsDirty(source.path)} class:deleted={buffers[source.path]?.deleted} onclick={() => openSource(source.path)}>
                      <Icon name="code" size={14} />
                      <span><strong>{fileName(source.path)}</strong><small>{directoryName(source.path)}</small></span>
                      {#if pathIsDirty(source.path)}<i title="Unsaved change"></i>{:else}<small>{formatBytes(source.bytes)}</small>{/if}
                    </button>
                  {:else}
                    <div class="studio-empty compact">No Lua files found.</div>
                  {/each}
                </div>
                <footer><span>{dirtyCount ? `${dirtyCount} unsaved ${dirtyCount === 1 ? "file" : "files"}` : "All files saved"}</span><small>Changes stay in this browser until saved.</small></footer>
              </aside>

              <div class="source-editor">
                {#if sourceLoading && !activeBuffer}
                  <div class="studio-empty tall">Loading source...</div>
                {:else if activeBuffer}
                  <header>
                    <div><strong>{activeBuffer.path}</strong><span>{activeBuffer.deleted ? "Marked for deletion" : activeBuffer.isNew ? "New file" : isDirty(activeBuffer) ? "Modified locally" : "Saved source"}</span></div>
                    <div><button disabled={!isDirty(activeBuffer)} onclick={revertSource}>Revert</button><button class="source-delete" onclick={deleteSource}>Delete</button></div>
                  </header>
                  {#if activeBuffer.deleted}
                    <div class="deleted-source"><span><Icon name="activity" size={17} /></span><strong>{activeBuffer.path} will be deleted</strong><p>Revert this file to keep it, or validate and save the candidate to remove it.</p><button class="secondary-button" onclick={revertSource}>Keep file</button></div>
                  {:else}
                    {#key activeBuffer.path}
                      <LuaCodeEditor
                        value={activeBuffer.source}
                        ariaLabel={`Source for ${activeBuffer.path}`}
                        onChange={updateSource}
                        onSave={() => { if (canSave) void saveCandidate(); }}
                        onCursorChange={updateEditorCursor}
                      />
                    {/key}
                    <footer><span>Lua</span><span>Ln {editorLine}, Col {editorColumn}</span><span>{activeBuffer.source.split("\n").length} lines</span><span>{formatBytes(new TextEncoder().encode(activeBuffer.source).length)}</span><span class="editor-shortcut">Ctrl/⌘ F searches · Tab indents · Ctrl/⌘ S saves a validated candidate</span></footer>
                  {/if}
                {:else}
                  <div class="studio-empty tall"><span class="empty-mark"><Icon name="code" size={17} /></span><strong>Create the first module</strong><span>Every harness starts with a readable Lua file.</span><button class="secondary-button" onclick={() => newFileOpen = true}><Icon name="plus" size={14} />New file</button></div>
                {/if}
              </div>
            </section>

            <footer class="source-command-bar">
              <div class:validated={canSave}>
                <span>{#if canSave}<Icon name="check" size={15} />Candidate validated{:else}<Icon name="activity" size={15} />{dirtyCount ? "Validation required before save" : "Edit a file to create a candidate"}{/if}</span>
                <small>Validation includes every changed, added, and deleted module. Saving persists files; Turin activates them separately and atomically.</small>
              </div>
              <div><button class="secondary-button" disabled={Boolean(operation)} onclick={validateCandidate}>{operation === "validate-source" ? "Validating..." : dirtyCount ? `Validate ${dirtyCount} changes` : "Validate source"}</button><button class="primary-button" disabled={!canSave || Boolean(operation)} onclick={saveCandidate}>{operation === "save-source" ? "Saving..." : `Save ${dirtyCount || ""} ${dirtyCount === 1 ? "file" : "files"}`}</button></div>
            </footer>
          {:else if mode === "overview"}
            <div class="studio-overview">
              <section class="harness-facts">
                <div><span>Scripts</span><strong>{detail.loaded_scripts.length}</strong><small>active modules</small></div>
                <div><span>Agents</span><strong>{detail.bound_agents.length}</strong><small>{detail.bound_agents.length ? detail.bound_agents.join(", ") : "not bound"}</small></div>
                <div><span>UI intents</span><strong>{detail.ui_intents.length}</strong><small>declared signals</small></div>
                <div><span>Apps</span><strong>{harnessApps.length}</strong><small>client surfaces</small></div>
              </section>
              <div class="harness-content-grid">
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
                  <div class="issue-list">
                    {#each issues as issue (issue.path + issue.message)}
                      <article><span><Icon name="activity" size={14} /></span><div><strong>{issue.message}</strong><code title={issue.path}>{issue.path}</code></div></article>
                    {:else}<div class="studio-empty compact">The active harness has no reported diagnostics.</div>{/each}
                  </div>
                  <footer class="diagnostic-actions"><span>Persisted source</span><button class="secondary-button" disabled={Boolean(operation)} onclick={validateRuntime}>{operation === "validate-runtime" ? "Validating..." : "Validate runtime source"}</button></footer>
                </section>
                <section class="studio-card lifecycle-card wide">
                  <header><div><span class="eyebrow">Lifecycle</span><h3>Runtime ownership</h3></div></header>
                  <dl><div><dt>Type</dt><dd>{selectedType}</dd></div><div><dt>Directory</dt><dd><code>{detail.directory}</code></dd></div><div><dt>Reload</dt><dd>Atomic; the active version survives validation or activation failures</dd></div><div><dt>Watching</dt><dd>{detail.watched_roots.length ? detail.watched_roots.join(", ") : "No watched roots"}</dd></div></dl>
                  {#if managed}<footer><div><strong>Delete shared harness</strong><span>{canDelete ? "Removes its managed directory after confirmation." : "Unbind every agent before deletion."}</span></div><button class="danger-button" disabled={!canDelete || Boolean(operation)} onclick={() => deleteOpen = true}>Delete</button></footer>{/if}
                </section>
              </div>
            </div>
          {:else}
            <section class="reference-view">
              <header><span class="eyebrow">Authoring guide</span><h3>Build from small, composable primitives</h3><p>These examples are a working map, not a second documentation system. The durable reference paths are shown on every section.</p></header>
              <div class="reference-grid">
                {#each referenceSections as section (section.title)}
                  <article class="reference-card"><header><div><h4>{section.title}</h4><p>{section.description}</p></div><code>{section.docs}</code></header><pre><code>{section.code}</code></pre></article>
                {/each}
              </div>
            </section>
          {/if}
        {:else}
          <div class="studio-empty tall"><span class="empty-mark"><Icon name="code" size={17} /></span><strong>Select a harness</strong><span>Its source, runtime state, and authoring reference will appear here.</span></div>
        {/if}
      </main>
    </div>
  </div>
</section>

{#if createOpen}
  <div class="overlay confirm-overlay" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) createOpen = false; }}>
    <div class="confirm-dialog studio-dialog" role="dialog" aria-modal="true" aria-labelledby="create-harness-title">
      <header><span class="dialog-mark"><Icon name="code" /></span><button class="dialog-close" aria-label="Close" onclick={() => createOpen = false}><Icon name="close" /></button></header>
      <h2 id="create-harness-title">Create a shared harness</h2><p>Turin will scaffold, discover, and load a managed Lua harness in the workspace.</p>
      <label class="studio-field"><span>Harness ID</span><input bind:value={newHarnessId} placeholder="research-desk" autocomplete="off" /></label>
      <div class="dialog-actions"><button onclick={() => createOpen = false}>Cancel</button><button class="primary" disabled={!newHarnessId.trim() || Boolean(operation)} onclick={createHarness}>{operation === "create" ? "Creating..." : "Create harness"}</button></div>
    </div>
  </div>
{/if}

{#if deleteOpen && detail}
  <div class="overlay confirm-overlay" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) deleteOpen = false; }}>
    <div class="confirm-dialog studio-dialog" role="alertdialog" aria-modal="true" aria-labelledby="delete-harness-title">
      <header><span class="dialog-mark danger"><Icon name="activity" /></span><button class="dialog-close" aria-label="Close" onclick={() => deleteOpen = false}><Icon name="close" /></button></header>
      <h2 id="delete-harness-title">Delete {displayName(detail.harness_id)}?</h2><p>This removes the managed harness directory. Turin refuses the operation while agents remain bound.</p>
      <div class="dialog-actions"><button onclick={() => deleteOpen = false}>Keep harness</button><button class="danger-confirm" disabled={Boolean(operation)} onclick={deleteSelected}>{operation === "delete" ? "Deleting..." : "Delete harness"}</button></div>
    </div>
  </div>
{/if}
