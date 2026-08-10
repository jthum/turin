<script lang="ts">
  import { tick } from "svelte";
  import type { TurinClient } from "../lib/TurinClient";
  import type { JsonValue, UiApp, UiNode, UiPane, UiScreen, WorkItem } from "../lib/types";
  import Icon from "./Icon.svelte";
  import NodeRenderer from "./NodeRenderer.svelte";
  import ScreenMenu from "./ScreenMenu.svelte";

  export let client: TurinClient;
  export let app: UiApp;
  export let eventRevision = 0;
  export let latestUiIntent: JsonValue = null;

  type PendingAction = { action: string; params: JsonValue };
  type Notice = { id: number; level: string; title: string; body?: string };

  let appId = "";
  let selectedScreenId = "";
  let activePaneId: string | null = null;
  let pendingAction: PendingAction | null = null;
  let runningAction = "";
  let refreshToken = 0;
  let appliedEventRevision = eventRevision;
  let notices: Notice[] = [];
  let noticeId = 0;
  let paneDialog: HTMLDivElement;
  let confirmDialog: HTMLDivElement;

  $: if (app.id !== appId) initializeApp();
  $: screen = app.screens[selectedScreenId] ?? Object.values(app.screens)[0];
  $: activePane = activePaneId ? app.panes[activePaneId] : undefined;
  $: if (eventRevision !== appliedEventRevision) {
    appliedEventRevision = eventRevision;
    applyIntent(latestUiIntent);
    refreshToken += 1;
  }

  function initializeApp() {
    appId = app.id;
    selectedScreenId = app.opens_with && app.screens[app.opens_with]
      ? app.opens_with
      : Object.keys(app.screens)[0] ?? "";
    activePaneId = null;
    pendingAction = null;
  }

  function selectScreen(screenId: string) {
    selectedScreenId = screenId;
    activePaneId = null;
  }

  function requestAction(action: string, params: JsonValue, confirm: boolean) {
    if (confirm) pendingAction = { action, params };
    else void runAction(action, params);
  }

  async function runAction(action: string, params: JsonValue) {
    pendingAction = null;
    runningAction = action;
    try {
      const response = await client.runAction({
        action,
        ...(app.source?.harness_id ? { harness_id: app.source.harness_id } : {}),
        ...(app.source?.agent_id ? { agent_id: app.source.agent_id } : {}),
        params,
      });
      const intents = response.result.ui_intents;
      if (Array.isArray(intents)) {
        for (const intent of intents) applyIntent(intent);
      }
      refreshToken += 1;
      pushNotice("success", "Action completed", humanAction(action));
    } catch (reason) {
      pushNotice("error", "Action failed", reason instanceof Error ? reason.message : String(reason));
    } finally {
      runningAction = "";
    }
  }

  function applyIntent(value: JsonValue) {
    if (!value || typeof value !== "object" || Array.isArray(value)) return;
    if (typeof value.app_id === "string" && value.app_id !== app.id) return;
    const type = value.type;
    if (type === "notify") {
      pushNotice(String(value.level ?? "info"), String(value.title ?? "Update"), typeof value.body === "string" ? value.body : undefined);
    } else if (type === "open" && typeof value.target === "string") {
      if (app.screens[value.target]) selectScreen(value.target);
    } else if (type === "show" && typeof value.target === "string") {
      if (app.screens[value.target]) selectScreen(value.target);
      else if (app.panes[value.target]) activePaneId = value.target;
    } else if (type === "refresh") {
      refreshToken += 1;
    } else if (type === "focus" && typeof value.target === "string") {
      void tick().then(() => document.getElementById(value.target as string)?.focus());
    }
  }

  function pushNotice(level: string, title: string, body?: string) {
    const notice: Notice = { id: ++noticeId, level, title, ...(body ? { body } : {}) };
    notices = [...notices.slice(-2), notice];
    window.setTimeout(() => notices = notices.filter(item => item.id !== notice.id), 5000);
  }

  function humanAction(action: string): string {
    const parts = action.split(".");
    return parts[parts.length - 1]?.replace(/_/g, " ") ?? action;
  }

  function paneNodes(pane: UiPane): UiNode[] {
    return pane.nodes ?? [];
  }

  function handleWindowKeydown(event: KeyboardEvent) {
    if (event.key === "Escape") {
      pendingAction = null;
      activePaneId = null;
      return;
    }
    const dialog = pendingAction ? confirmDialog : activePane ? paneDialog : null;
    if (event.key === "Tab" && dialog) trapFocus(event, dialog);
  }

  function trapFocus(event: KeyboardEvent, dialog: HTMLElement) {
    const controls = [...dialog.querySelectorAll<HTMLElement>(
      'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    )];
    if (!controls.length) return;
    const first = controls[0]!;
    const last = controls[controls.length - 1]!;
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function focusDialog(dialog: HTMLElement) {
    void tick().then(() => dialog.querySelector<HTMLElement>("button:not([disabled])")?.focus());
  }
</script>

<svelte:window onkeydown={handleWindowKeydown} />

<section class="harness-view">
  <header class="view-header harness-header">
    <div><span class="eyebrow">Harness application</span><h1>{app.definition?.title ?? app.id}</h1><p>{app.definition?.about ?? "A Turin-powered workspace."}</p></div>
    <button class="icon-button" aria-label="Refresh view" onclick={() => refreshToken += 1}><Icon name="refresh" /></button>
  </header>

  <div class="harness-layout">
    <aside class="screen-rail" aria-label="App screens">
      {#if app.menus.length}
        {#each app.menus as menu (menu.title)}
          <div class="menu-group"><span>{menu.title}</span><ScreenMenu {app} items={menu.items} {selectedScreenId} onSelect={selectScreen} /></div>
        {/each}
      {:else}
        <div class="screen-menu">
          {#each Object.values(app.screens) as candidate (candidate.id)}
            <button class:active={selectedScreenId === candidate.id} onclick={() => selectScreen(candidate.id)}><span>{candidate.title}</span><Icon name="chevron" size={14} /></button>
          {/each}
        </div>
      {/if}
    </aside>

    <main class="harness-screen">
      {#if screen}
        <div class="screen-title"><h2>{screen.title}</h2></div>
        <div class="node-stack">
          {#each screen.nodes ?? [] as node, index (`${node.id ?? node.kind}-${index}`)}
            <NodeRenderer {client} {app} {node} {refreshToken} running={Boolean(runningAction)} onAction={requestAction} />
          {/each}
        </div>
      {:else}
        <div class="welcome-state"><span class="welcome-mark"><Icon name="grid" /></span><h2>No screens declared</h2><p>This harness app has not declared a screen yet.</p></div>
      {/if}
    </main>
  </div>

  <div class="notice-stack" aria-live="polite">
    {#each notices as notice (notice.id)}
      <div class="toast" data-level={notice.level}><i></i><div><strong>{notice.title}</strong>{#if notice.body}<span>{notice.body}</span>{/if}</div></div>
    {/each}
  </div>
</section>

{#if activePane}
  <div class="overlay" role="presentation" onclick={event => { if (event.currentTarget === event.target) activePaneId = null; }}>
    <div class="pane-sheet" bind:this={paneDialog} role="dialog" aria-modal="true" aria-label={activePane.title} use:focusDialog>
      <header><div><span class="eyebrow">Details</span><h2>{activePane.title}</h2></div><button class="icon-button" aria-label="Close pane" onclick={() => activePaneId = null}><Icon name="close" /></button></header>
      <div class="node-stack">
        {#each paneNodes(activePane) as node, index (`pane-${node.id ?? node.kind}-${index}`)}
          <NodeRenderer {client} {app} {node} {refreshToken} running={Boolean(runningAction)} onAction={requestAction} />
        {/each}
      </div>
    </div>
  </div>
{/if}

{#if pendingAction}
  <div class="overlay confirm-overlay" role="presentation" onclick={event => { if (event.currentTarget === event.target) pendingAction = null; }}>
    <div class="confirm-dialog" bind:this={confirmDialog} role="alertdialog" aria-modal="true" aria-labelledby="confirm-title" use:focusDialog>
      <span class="dialog-mark"><Icon name="spark" /></span>
      <h2 id="confirm-title">Run this action?</h2>
      <p>Turin will run <strong>{humanAction(pendingAction.action)}</strong> using this harness.</p>
      <div class="dialog-actions"><button class="secondary-button" onclick={() => pendingAction = null}>Cancel</button><button class="primary-button" onclick={() => runAction(pendingAction!.action, pendingAction!.params)}>Run action</button></div>
    </div>
  </div>
{/if}
