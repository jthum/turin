<script lang="ts">
  import { onMount } from "svelte";
  import type { TurinClient } from "../lib/TurinClient";
  import type {
    MemoryList,
    MemoryScope,
    TurinStatus,
    WorklistDetail,
    WorklistItem,
  } from "../lib/types";
  import { shortDate, titleForSession } from "../lib/format";
  import Icon from "./Icon.svelte";

  type ExplorerTab = "worklists" | "memories" | "sessions";

  export let client: TurinClient;
  export let status: TurinStatus;
  export let onSession: (sessionId: string) => void;

  let activeTab: ExplorerTab = "worklists";
  let worklists: WorklistDetail[] = [];
  let selectedWorklistId = "";
  let worklistItems: WorklistItem[] = [];
  let worklistsLoading = true;
  let worklistItemsLoading = false;
  let worklistError = "";
  let memoryPage: MemoryList = { memories: [], scopes: [], total: 0, offset: 0, limit: 100 };
  let memoryScope = "";
  let selectedMemoryId = "";
  let includeSuperseded = false;
  let memoriesLoading = true;
  let memoryError = "";

  $: selectedWorklist = worklists.find(worklist => worklist.public_id === selectedWorklistId);
  $: selectedMemory = memoryPage.memories.find(memory => memory.public_id === selectedMemoryId)
    ?? memoryPage.memories[0];
  $: sessionCount = status.snapshot.sessions.length;

  onMount(() => {
    void Promise.all([loadWorklists(), loadMemories(0)]);
  });

  async function loadWorklists() {
    worklistsLoading = true;
    worklistError = "";
    try {
      worklists = await client.worklists();
      if (!worklists.some(worklist => worklist.public_id === selectedWorklistId)) {
        selectedWorklistId = worklists[0]?.public_id ?? "";
      }
      if (selectedWorklistId) await loadWorklistItems(selectedWorklistId);
      else worklistItems = [];
    } catch (reason) {
      worklistError = reason instanceof Error ? reason.message : String(reason);
    } finally {
      worklistsLoading = false;
    }
  }

  async function loadWorklistItems(id: string) {
    selectedWorklistId = id;
    worklistItemsLoading = true;
    worklistError = "";
    try {
      worklistItems = await client.worklistItems(id, 250);
    } catch (reason) {
      worklistItems = [];
      worklistError = reason instanceof Error ? reason.message : String(reason);
    } finally {
      worklistItemsLoading = false;
    }
  }

  async function loadMemories(offset = 0) {
    memoriesLoading = true;
    memoryError = "";
    try {
      const scope = selectedScope();
      memoryPage = await client.memories({
        ...(scope ? { scopeKind: scope.scope_kind, scopeKey: scope.scope_key } : {}),
        includeSuperseded,
        limit: 100,
        offset,
      });
      if (!memoryPage.memories.some(memory => memory.public_id === selectedMemoryId)) {
        selectedMemoryId = memoryPage.memories[0]?.public_id ?? "";
      }
    } catch (reason) {
      memoryError = reason instanceof Error ? reason.message : String(reason);
    } finally {
      memoriesLoading = false;
    }
  }

  function selectedScope(): MemoryScope | undefined {
    if (!memoryScope) return undefined;
    try {
      const [scope_kind, scope_key] = JSON.parse(memoryScope) as unknown[];
      return typeof scope_kind === "string" && typeof scope_key === "string"
        ? { scope_kind, scope_key, count: 0 }
        : undefined;
    } catch {
      return undefined;
    }
  }

  function scopeValue(scope: MemoryScope): string {
    return JSON.stringify([scope.scope_kind, scope.scope_key]);
  }

  function refreshActive() {
    if (activeTab === "worklists") void loadWorklists();
    if (activeTab === "memories") void loadMemories(memoryPage.offset);
  }

  function metadataText(value: unknown): string {
    if (value === undefined || value === null) return "No metadata";
    return JSON.stringify(value, null, 2);
  }
</script>

<section class="data-explorer">
  <header class="view-header data-explorer-header">
    <div>
      <span class="eyebrow">Runtime data</span>
      <h1>Data Explorer</h1>
      <p>Inspect Turin-owned operational data without changing retrieval state.</p>
    </div>
    <button class="secondary-button explorer-refresh" onclick={refreshActive}>
      <Icon name="refresh" size={15} />Refresh
    </button>
  </header>

  <div class="explorer-scroll">
    <div class="explorer-content">
      <nav class="explorer-tabs" aria-label="Data collections">
        <button class:active={activeTab === "worklists"} onclick={() => activeTab = "worklists"}>Worklists <span>{worklists.length}</span></button>
        <button class:active={activeTab === "memories"} onclick={() => activeTab = "memories"}>Memories <span>{memoryPage.total}</span></button>
        <button class:active={activeTab === "sessions"} onclick={() => activeTab = "sessions"}>Sessions <span>{sessionCount}</span></button>
      </nav>

      {#if activeTab === "worklists"}
        <div class="explorer-summary">
          <div><span>Collections</span><strong>{worklists.length}</strong><small>durable worklists</small></div>
          <div><span>Visible items</span><strong>{worklistItems.length}</strong><small>up to 250 rows</small></div>
          <div><span>Selected scope</span><strong class="summary-text">{selectedWorklist?.scope_ref ?? "-"}</strong><small>{selectedWorklist?.name ?? "No worklist"}</small></div>
        </div>

        <section class="explorer-panel explorer-split">
          <aside class="collection-list">
            <header><strong>Worklists</strong><span>{worklists.length}</span></header>
            {#if worklistsLoading}
              <div class="explorer-placeholder">Loading worklists...</div>
            {:else if !worklists.length}
              <div class="explorer-placeholder">No worklists have been created.</div>
            {:else}
              {#each worklists as worklist (worklist.public_id)}
                <button class:active={worklist.public_id === selectedWorklistId} onclick={() => loadWorklistItems(worklist.public_id)}>
                  <span>{worklist.name}</span>
                  <small>{worklist.scope_ref}</small>
                </button>
              {/each}
            {/if}
          </aside>
          <div class="collection-table">
            <header class="explorer-panel-header">
              <div><strong>{selectedWorklist?.name ?? "Worklist items"}</strong><span>{selectedWorklist ? `Updated ${shortDate(selectedWorklist.updated_at)}` : "Select a worklist"}</span></div>
              <span class="row-count">{worklistItems.length} rows</span>
            </header>
            {#if worklistError}
              <div class="explorer-error">{worklistError}</div>
            {:else if worklistItemsLoading}
              <div class="explorer-placeholder">Loading worklist items...</div>
            {:else if !worklistItems.length}
              <div class="empty-surface"><strong>No items in this worklist</strong><span>Items will appear here when the harness or runtime creates them.</span></div>
            {:else}
              <div class="table-scroll">
                <table class="explorer-table">
                  <thead><tr><th>Item</th><th>Kind</th><th>Status</th><th>Priority</th><th>Updated</th></tr></thead>
                  <tbody>
                    {#each worklistItems as item (item.public_id)}
                      <tr>
                        <td><strong>{item.title}</strong><small>{item.public_id}</small></td>
                        <td>{item.kind}</td>
                        <td><span class="status-cell">{item.paused ? "paused" : item.status}</span></td>
                        <td>{item.priority}</td>
                        <td>{shortDate(item.updated_at)}</td>
                      </tr>
                    {/each}
                  </tbody>
                </table>
              </div>
            {/if}
          </div>
        </section>
      {:else if activeTab === "memories"}
        <div class="explorer-toolbar">
          <label>
            <span>Scope</span>
            <select bind:value={memoryScope} onchange={() => loadMemories(0)}>
              <option value="">All scopes</option>
              {#each memoryPage.scopes as scope (scopeValue(scope))}
                <option value={scopeValue(scope)}>{scope.scope_kind} / {scope.scope_key} ({scope.count})</option>
              {/each}
            </select>
          </label>
          <label class="inline-check"><input type="checkbox" bind:checked={includeSuperseded} onchange={() => loadMemories(0)} /><span>Include superseded</span></label>
          <span class="toolbar-spacer"></span>
          <span class="range-label">{memoryPage.total ? `${memoryPage.offset + 1}-${Math.min(memoryPage.offset + memoryPage.memories.length, memoryPage.total)} of ${memoryPage.total}` : "0 memories"}</span>
          <button class="icon-button" aria-label="Previous memory page" disabled={memoryPage.offset === 0 || memoriesLoading} onclick={() => loadMemories(Math.max(0, memoryPage.offset - memoryPage.limit))}><Icon name="chevron" size={15} /></button>
          <button class="icon-button next-page" aria-label="Next memory page" disabled={memoryPage.offset + memoryPage.memories.length >= memoryPage.total || memoriesLoading} onclick={() => loadMemories(memoryPage.offset + memoryPage.limit)}><Icon name="chevron" size={15} /></button>
        </div>

        <section class="explorer-panel memory-layout">
          <div class="collection-table">
            <header class="explorer-panel-header"><div><strong>Memories</strong><span>Read-only inspection does not increase retrieval counts.</span></div><span class="row-count">{memoryPage.total} total</span></header>
            {#if memoryError}
              <div class="explorer-error">{memoryError}</div>
            {:else if memoriesLoading}
              <div class="explorer-placeholder">Loading memories...</div>
            {:else if !memoryPage.memories.length}
              <div class="empty-surface"><strong>No memories found</strong><span>Try another scope or store memories through a harness.</span></div>
            {:else}
              <div class="memory-list">
                {#each memoryPage.memories as memory (memory.public_id)}
                  <button class:active={memory.public_id === selectedMemory?.public_id} onclick={() => selectedMemoryId = memory.public_id}>
                    <span>{memory.content}</span>
                    <small>{memory.scope_kind} / {memory.scope_key} · {shortDate(memory.created_at)}</small>
                  </button>
                {/each}
              </div>
            {/if}
          </div>

          <aside class="memory-inspector">
            {#if selectedMemory}
              <header><span class="status-cell">{selectedMemory.storage.replace("_", " ")}</span><small>{shortDate(selectedMemory.created_at)}</small></header>
              <h2>Memory detail</h2>
              <p>{selectedMemory.content}</p>
              <dl>
                <div><dt>Scope</dt><dd>{selectedMemory.scope_kind} / {selectedMemory.scope_key}</dd></div>
                <div><dt>Weight</dt><dd>{selectedMemory.weight.toFixed(2)}</dd></div>
                <div><dt>Retrievals</dt><dd>{selectedMemory.retrieval_count}</dd></div>
                <div><dt>Last retrieved</dt><dd>{selectedMemory.last_retrieved_at ? shortDate(selectedMemory.last_retrieved_at) : "Never"}</dd></div>
                <div><dt>Embedding</dt><dd>{selectedMemory.embedding_key ? `${selectedMemory.embedding_key} · ${selectedMemory.embedding_dimensions ?? "?"}d` : "Lexical only"}</dd></div>
                <div><dt>ID</dt><dd class="mono-value">{selectedMemory.public_id}</dd></div>
              </dl>
              <details><summary>Metadata</summary><pre>{metadataText(selectedMemory.metadata)}</pre></details>
            {:else}
              <div class="explorer-placeholder">Select a memory to inspect it.</div>
            {/if}
          </aside>
        </section>
      {:else}
        <section class="explorer-panel collection-table sessions-table">
          <header class="explorer-panel-header"><div><strong>Sessions</strong><span>Conversation roots currently known to this state store.</span></div><span class="row-count">{sessionCount} rows</span></header>
          {#if !sessionCount}
            <div class="empty-surface"><strong>No sessions yet</strong><span>Start a conversation to create the first session.</span></div>
          {:else}
            <div class="table-scroll">
              <table class="explorer-table">
                <thead><tr><th>Session</th><th>Agent</th><th>Created</th><th></th></tr></thead>
                <tbody>
                  {#each status.snapshot.sessions as session (session.session_id)}
                    <tr>
                      <td><strong>{titleForSession(session)}</strong><small>{session.session_id}</small></td>
                      <td>{session.agent_id}</td>
                      <td>{shortDate(session.created_at)}</td>
                      <td><button class="table-action" onclick={() => onSession(session.session_id)}>Open conversation</button></td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {/if}
        </section>
      {/if}
    </div>
  </div>
</section>
