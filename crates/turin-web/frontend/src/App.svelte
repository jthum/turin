<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import AgentOrchestration from "./components/AgentOrchestration.svelte";
  import AssistantView from "./components/AssistantView.svelte";
  import DataExplorer from "./components/DataExplorer.svelte";
  import HarnessView from "./components/HarnessView.svelte";
  import HarnessStudio from "./components/HarnessStudio.svelte";
  import Icon from "./components/Icon.svelte";
  import Sidebar from "./components/Sidebar.svelte";
  import { HttpTurinClient } from "./lib/HttpTurinClient";
  import type { EventSubscription } from "./lib/TurinClient";
  import type { JsonValue, TurinEvent, TurinStatus, UiApp } from "./lib/types";

  const client = new HttpTurinClient();
  let status: TurinStatus | null = null;
  let loading = true;
  let error = "";
  let activeView = "assistant";
  let selectedSessionId: string | null = null;
  let sidebarOpen = false;
  let globalSubscription: EventSubscription | null = null;
  let eventRevision = 0;
  let latestUiIntent: JsonValue = null;
  let refreshTimer: number | undefined;

  $: activeApp = status?.ui.apps[activeView] as UiApp | undefined;
  $: if (status && !["assistant", "data", "harnesses", "operations"].includes(activeView)) connectGlobalEvents();
  $: if (["assistant", "data", "harnesses", "operations"].includes(activeView)) closeGlobalEvents();

  onMount(async () => {
    await refreshStatus();
  });

  onDestroy(() => {
    closeGlobalEvents();
    if (refreshTimer) window.clearTimeout(refreshTimer);
  });

  async function refreshStatus() {
    try {
      const next = await client.status();
      status = next;
      error = "";
      if (!["assistant", "data", "orchestration", "harnesses", "operations"].includes(activeView) && !next.ui.apps[activeView]) activeView = "assistant";
    } catch (reason) {
      error = reason instanceof Error ? reason.message : String(reason);
    } finally {
      loading = false;
    }
  }

  function connectGlobalEvents() {
    if (globalSubscription) return;
    globalSubscription = client.subscribe(handleGlobalEvent);
  }

  function closeGlobalEvents() {
    globalSubscription?.close();
    globalSubscription = null;
  }

  function handleGlobalEvent(event: TurinEvent) {
    if (event.type === "ui.intent") {
      latestUiIntent = event.data;
      eventRevision += 1;
    }
    if (["runtime.rescanned", "runtime.snapshot", "task.submitted", "task.updated", "ui.intent"].includes(event.type)) {
      if (refreshTimer) window.clearTimeout(refreshTimer);
      refreshTimer = window.setTimeout(() => void refreshStatus(), 180);
    }
  }

  function navigate(view: string) {
    activeView = view;
    sidebarOpen = false;
  }

  function selectSession(sessionId: string) {
    selectedSessionId = sessionId;
    activeView = "assistant";
    sidebarOpen = false;
  }

  function startNewConversation() {
    selectedSessionId = null;
    activeView = "assistant";
    sidebarOpen = false;
  }
</script>

{#if loading}
  <main class="boot-screen"><span class="brand-mark"><Icon name="spark" /></span><strong>Starting Turin</strong><i></i></main>
{:else if !status}
  <main class="fatal-screen"><span class="dialog-mark"><Icon name="activity" /></span><h1>Turin is out of reach</h1><p>{error}</p><button class="primary-button" onclick={refreshStatus}>Try again</button></main>
{:else}
  <div class="app-shell">
    <button class:open={sidebarOpen} class="sidebar-backdrop" aria-label="Close navigation" onclick={() => sidebarOpen = false}></button>
    <div class:open={sidebarOpen} class="sidebar-mobile-wrap">
      <Sidebar
        {status}
        {activeView}
        {selectedSessionId}
        onNavigate={navigate}
        onSession={selectSession}
        onNewConversation={startNewConversation}
        onClose={() => sidebarOpen = false}
      />
    </div>
    <div class="desktop-sidebar">
      <Sidebar
        {status}
        {activeView}
        {selectedSessionId}
        onNavigate={navigate}
        onSession={selectSession}
        onNewConversation={startNewConversation}
        onClose={() => sidebarOpen = false}
      />
    </div>

    <main class="app-stage">
      <button class="mobile-menu mobile-only" aria-label="Open navigation" onclick={() => sidebarOpen = true}><Icon name="menu" /></button>
      {#if activeView === "assistant"}
        <AssistantView
          {client}
          {status}
          {selectedSessionId}
          onSessionSelected={selectSession}
          onNewConversation={startNewConversation}
          onStatusChanged={refreshStatus}
        />
      {:else if activeView === "data"}
        <DataExplorer {client} {status} onSession={selectSession} />
      {:else if activeView === "orchestration"}
        <AgentOrchestration {status} onSession={selectSession} onRefresh={refreshStatus} />
      {:else if activeView === "harnesses"}
        <HarnessStudio {client} {status} onOpenApp={navigate} onStatusChanged={refreshStatus} />
      {:else if activeApp}
        <HarnessView {client} app={activeApp} {eventRevision} {latestUiIntent} />
      {/if}
    </main>
  </div>
{/if}
