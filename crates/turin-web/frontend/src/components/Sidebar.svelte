<script lang="ts">
  import type { SessionSummary, TurinStatus, UiApp } from "../lib/types";
  import { shortDate, titleForSession } from "../lib/format";
  import Icon from "./Icon.svelte";

  export let status: TurinStatus;
  export let activeView: string;
  export let selectedSessionId: string | null;
  export let collapsed = false;
  export let onNavigate: (view: string) => void;
  export let onSession: (sessionId: string) => void;
  export let onClose: () => void;

  $: apps = Object.values(status.ui.apps) as UiApp[];
  $: sessions = status.snapshot.sessions.slice(0, 8);
</script>

<aside class:collapsed class="sidebar">
  <div class="brand-row">
    <button class="brand" aria-label="Open assistant" onclick={() => onNavigate("assistant")}>
      <span class="brand-mark"><Icon name="spark" size={17} /></span>
      <span>Turin</span>
    </button>
    <button class="icon-button mobile-only" aria-label="Close navigation" onclick={onClose}>
      <Icon name="close" />
    </button>
  </div>

  <nav class="primary-nav" aria-label="Primary navigation">
    <button class:active={activeView === "assistant"} onclick={() => onNavigate("assistant")}>
      <Icon name="chat" /><span>Assistant</span>
    </button>
    {#each apps as app (app.id)}
      <button class:active={activeView === app.id} onclick={() => onNavigate(app.id)}>
        <Icon name="grid" /><span>{app.definition?.title ?? app.id}</span>
        {#if Object.keys(app.badges).length}
          <i class="nav-dot" aria-label="Has updates"></i>
        {/if}
      </button>
    {/each}
  </nav>

  <div class="sidebar-section">
    <div class="sidebar-label">
      <span>Recent conversations</span>
      <span>{sessions.length}</span>
    </div>
    <div class="conversation-list">
      {#each sessions as session (session.session_id)}
        <button
          class:active={activeView === "assistant" && selectedSessionId === session.session_id}
          onclick={() => onSession(session.session_id)}
        >
          <span class="conversation-title">{titleForSession(session)}</span>
          <span class="conversation-date">{shortDate(session.created_at)}</span>
        </button>
      {:else}
        <p class="sidebar-empty">Your conversations will appear here.</p>
      {/each}
    </div>
  </div>

  <div class="sidebar-footer">
    <span class:ready={status.snapshot.health.ready} class="status-dot"></span>
    <div>
      <strong>{status.snapshot.health.ready ? "Runtime ready" : "Runtime unavailable"}</strong>
      <span>{status.web.connection_kind} · v{status.snapshot.health.version}</span>
    </div>
  </div>
</aside>
