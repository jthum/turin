<script lang="ts">
  import { onDestroy, tick } from "svelte";
  import type { EventSubscription, TurinClient } from "../lib/TurinClient";
  import type { JsonValue, SessionDetail, SessionMessage, ToolExecution, TurinEvent, TurinStatus } from "../lib/types";
  import { humanize, messageText, sameSession, titleForSession } from "../lib/format";
  import Icon from "./Icon.svelte";
  import Markdown from "./Markdown.svelte";

  const DATA_WINDOW_SIZE = 100;
  const DATA_WINDOW_OVERLAP = 30;
  const RENDER_WINDOW_SIZE = 30;
  const RENDER_WINDOW_STEP = 10;

  export let client: TurinClient;
  export let status: TurinStatus;
  export let selectedSessionId: string | null;
  export let onSessionSelected: (sessionId: string) => void;
  export let onNewConversation: () => void;
  export let onStatusChanged: () => Promise<void>;

  let detail: SessionDetail | null = null;
  let loadedSessionId: string | null = null;
  let messageOffset: number | undefined;
  let followLatest = true;
  let renderStart = 0;
  let loading = false;
  let slidingData = false;
  let sending = false;
  let error = "";
  let resumeFailed = false;
  let copiedSession = false;
  let prompt = "";
  let streamText = "";
  let pendingDelta = "";
  let optimisticPrompt = "";
  let transcript: HTMLDivElement;
  let composer: HTMLTextAreaElement;
  let subscription: EventSubscription | null = null;
  let refreshTimer: number | undefined;
  let deltaFrame: number | undefined;
  let requestVersion = 0;
  let copyTimer: number | undefined;
  let scrollFrame: number | undefined;
  let heightRevision = 0;
  let topSpacerHeight = 0;
  let bottomSpacerHeight = 0;
  const measuredHeights = new Map<number, number>();

  $: selectedSummary = status.snapshot.sessions.find(item => sameSession(item.session_id, selectedSessionId));
  $: selectedLive = status.snapshot.live_sessions.find(item => sameSession(item.session_id, selectedSessionId));
  $: agents = status.snapshot.status.registry.agents.filter(agent => agent.enabled);
  $: sessionReference = detail?.session.session_id ?? selectedSessionId;
  $: transcriptMessages = (detail?.messages ?? []).filter(isTranscriptMessage);
  $: renderEnd = Math.min(renderStart + RENDER_WINDOW_SIZE, transcriptMessages.length);
  $: renderedMessages = transcriptMessages.slice(renderStart, renderEnd);
  $: {
    heightRevision;
    topSpacerHeight = estimatedHeight(transcriptMessages.slice(0, renderStart));
    bottomSpacerHeight = estimatedHeight(transcriptMessages.slice(renderEnd));
  }
  $: if (selectedSessionId !== loadedSessionId) switchSession(selectedSessionId);

  onDestroy(() => {
    subscription?.close();
    if (refreshTimer) window.clearTimeout(refreshTimer);
    if (deltaFrame) window.cancelAnimationFrame(deltaFrame);
    if (copyTimer) window.clearTimeout(copyTimer);
    if (scrollFrame) window.cancelAnimationFrame(scrollFrame);
  });

  async function switchSession(sessionId: string | null) {
    loadedSessionId = sessionId;
    detail = null;
    streamText = "";
    optimisticPrompt = "";
    messageOffset = undefined;
    followLatest = true;
    renderStart = 0;
    measuredHeights.clear();
    error = "";
    resumeFailed = false;
    subscription?.close();
    subscription = null;
    if (!sessionId) return;
    await connectSessionEvents(sessionId);
    await loadDetail(false, true);
  }

  async function connectSessionEvents(sessionId: string, slotId?: string) {
    subscription?.close();
    subscription = client.subscribe(handleEvent, {
      sessionId,
      ...(slotId ? { slotId } : {}),
    });
    await waitForSubscription(subscription);
  }

  async function waitForSubscription(active: EventSubscription) {
    await Promise.race([
      active.ready,
      new Promise<void>(resolve => window.setTimeout(resolve, 750)),
    ]);
  }

  async function loadDetail(
    preserveScroll: boolean,
    retireStream = false,
    requestedOffset = followLatest ? undefined : messageOffset,
  ) {
    if (!selectedSessionId) return;
    const version = ++requestVersion;
    const anchor = preserveScroll ? captureAnchor() : null;
    loading = !detail;
    error = "";
    try {
      const next = await client.session(selectedSessionId, DATA_WINDOW_SIZE, requestedOffset);
      if (version !== requestVersion) return;
      const nextMessages = next.messages.filter(isTranscriptMessage);
      messageOffset = next.message_window?.offset;
      if (requestedOffset === undefined) followLatest = true;
      renderStart = anchor
        ? renderStartForAnchor(nextMessages, anchor.id)
        : Math.max(0, nextMessages.length - RENDER_WINDOW_SIZE);
      detail = next;
      optimisticPrompt = "";
      if (retireStream) {
        streamText = "";
        pendingDelta = "";
        if (deltaFrame) window.cancelAnimationFrame(deltaFrame);
        deltaFrame = undefined;
      }
      await tick();
      if (anchor) {
        restoreAnchor(anchor);
      } else {
        scrollToBottom();
      }
    } catch (reason) {
      if (version !== requestVersion) return;
      error = reason instanceof Error ? reason.message : String(reason);
    } finally {
      if (version === requestVersion) loading = false;
    }
  }

  async function slideData(direction: "older" | "newer") {
    const window = detail?.message_window;
    if (!window || slidingData) return;
    const shift = DATA_WINDOW_SIZE - DATA_WINDOW_OVERLAP;
    const latestOffset = Math.max(0, window.total - DATA_WINDOW_SIZE);
    const nextOffset = direction === "older"
      ? Math.max(0, window.offset - shift)
      : Math.min(latestOffset, window.offset + shift);
    if (nextOffset === window.offset) return;
    slidingData = true;
    try {
      await loadDetail(true, false, nextOffset);
      followLatest = direction === "newer" && nextOffset === latestOffset;
    } finally {
      slidingData = false;
    }
  }

  function handleEvent(event: TurinEvent) {
    if (event.type === "message_start") {
      streamText = "";
      pendingDelta = "";
      return;
    }
    if (event.type === "message_delta") {
      const delta = event.data.content_delta;
      if (typeof delta === "string") queueDelta(delta);
      return;
    }
    if (["message_end", "turn_end", "task_complete"].includes(event.type)) {
      scheduleDetailRefresh(event.type === "message_end" ? 80 : 180, true);
      void onStatusChanged();
    }
  }

  function queueDelta(delta: string) {
    pendingDelta += delta;
    if (deltaFrame) return;
    const shouldFollow = isNearBottom();
    deltaFrame = window.requestAnimationFrame(async () => {
      streamText += pendingDelta;
      pendingDelta = "";
      deltaFrame = undefined;
      if (shouldFollow) {
        await tick();
        scrollToBottom();
      }
    });
  }

  function scheduleDetailRefresh(delay: number, retireStream = false) {
    if (refreshTimer) window.clearTimeout(refreshTimer);
    refreshTimer = window.setTimeout(() => void loadDetail(false, retireStream), delay);
  }

  async function submit() {
    const value = prompt.trim();
    if (!value || sending) return;
    sending = true;
    error = "";
    resumeFailed = false;
    let resuming = false;
    try {
      let sessionId = selectedSessionId;
      let live = selectedLive;
      if (!sessionId) {
        const agentId = agents[0]?.id ?? status.snapshot.live_sessions[0]?.agent_id ?? "default";
        live = await client.openSession(agentId);
        sessionId = live.session_id;
        onSessionSelected(sessionId);
        await tick();
        if (subscription) {
          await waitForSubscription(subscription);
        } else {
          await connectSessionEvents(sessionId, live.slot_id);
        }
      } else {
        resuming = true;
        live = await client.resumeSession(sessionId, live?.slot_id);
        await connectSessionEvents(sessionId, live.slot_id);
        resuming = false;
      }
      optimisticPrompt = value;
      prompt = "";
      messageOffset = undefined;
      followLatest = true;
      await tick();
      scrollToBottom();
      await client.submitTask({
        session_id: sessionId,
        slot_id: live.slot_id,
        prompt: value,
      });
      await onStatusChanged();
      scheduleDetailRefresh(120);
    } catch (reason) {
      error = reason instanceof Error ? reason.message : String(reason);
      resumeFailed = resuming;
      if (!prompt) prompt = value;
      optimisticPrompt = "";
    } finally {
      sending = false;
      await tick();
      composer?.focus();
    }
  }

  function onComposerKeydown(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      void submit();
    }
  }

  function toolsFor(message: SessionMessage): ToolExecution[] {
    if (message.role.toLowerCase() !== "assistant") return [];
    const ids = toolCallIds(message.content);
    return detail?.tool_executions.filter(tool => ids.has(tool.tool_call_id)) ?? [];
  }

  function toolCallIds(content: JsonValue): Set<string> {
    const ids = new Set<string>();
    const visit = (value: JsonValue) => {
      if (Array.isArray(value)) {
        value.forEach(visit);
      } else if (value && typeof value === "object") {
        if (value.type === "tool_use" && typeof value.id === "string") ids.add(value.id);
      }
    };
    visit(content);
    return ids;
  }

  function isTranscriptMessage(message: SessionMessage): boolean {
    const role = message.role.toLowerCase();
    return role === "user" || role === "assistant";
  }

  function estimatedHeight(messages: SessionMessage[]): number {
    return messages.reduce((total, message) => {
      const fallback = message.role.toLowerCase() === "user" ? 88 : 150;
      return total + (measuredHeights.get(message.id) ?? fallback);
    }, 0);
  }

  function measureMessage(node: HTMLElement, messageId: number) {
    const update = () => {
      const margin = Number.parseFloat(getComputedStyle(node).marginBottom) || 0;
      const height = Math.ceil(node.getBoundingClientRect().height + margin);
      if (Math.abs((measuredHeights.get(messageId) ?? 0) - height) > 1) {
        measuredHeights.set(messageId, height);
        heightRevision += 1;
      }
    };
    const observer = new ResizeObserver(update);
    observer.observe(node);
    update();
    return { destroy: () => observer.disconnect() };
  }

  interface ScrollAnchor {
    id: number;
    top: number;
  }

  function captureAnchor(): ScrollAnchor | null {
    if (!transcript) return null;
    const rootTop = transcript.getBoundingClientRect().top;
    const nodes = transcript.querySelectorAll<HTMLElement>("[data-message-id]");
    for (const node of nodes) {
      const rect = node.getBoundingClientRect();
      if (rect.bottom > rootTop + 1) {
        return { id: Number(node.dataset.messageId), top: rect.top - rootTop };
      }
    }
    return null;
  }

  function restoreAnchor(anchor: ScrollAnchor) {
    if (!transcript) return;
    const node = transcript.querySelector<HTMLElement>(`[data-message-id="${anchor.id}"]`);
    if (!node) return;
    const rootTop = transcript.getBoundingClientRect().top;
    transcript.scrollTop += node.getBoundingClientRect().top - rootTop - anchor.top;
  }

  function renderStartForAnchor(messages: SessionMessage[], messageId: number): number {
    const index = messages.findIndex(message => message.id === messageId);
    if (index < 0) return Math.max(0, messages.length - RENDER_WINDOW_SIZE);
    return Math.min(
      Math.max(0, index - Math.floor(RENDER_WINDOW_STEP / 2)),
      Math.max(0, messages.length - RENDER_WINDOW_SIZE),
    );
  }

  async function slideRender(nextStart: number) {
    const bounded = Math.min(
      Math.max(0, nextStart),
      Math.max(0, transcriptMessages.length - RENDER_WINDOW_SIZE),
    );
    if (bounded === renderStart) return;
    const anchor = captureAnchor();
    renderStart = bounded;
    await tick();
    if (anchor) restoreAnchor(anchor);
  }

  function onTranscriptScroll() {
    if (scrollFrame || slidingData) return;
    scrollFrame = window.requestAnimationFrame(async () => {
      scrollFrame = undefined;
      if (!transcript || !detail) return;
      const nearRenderedTop = transcript.scrollTop < topSpacerHeight + 320;
      const nearRenderedBottom = transcript.scrollTop + transcript.clientHeight
        > transcript.scrollHeight - bottomSpacerHeight - 320;
      if (nearRenderedTop) {
        if (renderStart > 0) {
          await slideRender(renderStart - RENDER_WINDOW_STEP);
        } else if ((detail.message_window?.offset ?? 0) > 0) {
          await slideData("older");
        }
      } else if (nearRenderedBottom) {
        if (renderEnd < transcriptMessages.length) {
          await slideRender(renderStart + RENDER_WINDOW_STEP);
        } else if (
          detail.message_window
          && detail.message_window.offset + detail.messages.length < detail.message_window.total
        ) {
          await slideData("newer");
        }
      }
    });
  }

  async function copySessionReference() {
    if (!sessionReference) return;
    try {
      await navigator.clipboard.writeText(sessionReference);
      copiedSession = true;
      if (copyTimer) window.clearTimeout(copyTimer);
      copyTimer = window.setTimeout(() => copiedSession = false, 1400);
    } catch {
      error = "Could not copy the session reference. Select it from the header instead.";
    }
  }

  function isNearBottom(): boolean {
    if (!transcript) return true;
    return transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 120;
  }

  function scrollToBottom() {
    if (transcript) transcript.scrollTop = transcript.scrollHeight;
  }
</script>

<section class="assistant-view">
  <header class="view-header assistant-header">
    <div>
      <div class="title-line">
        <h1>{selectedSummary ? titleForSession(selectedSummary) : "Assistant"}</h1>
        {#if selectedLive?.active_tasks}
          <span class="working-badge"><i></i>Working</span>
        {/if}
      </div>
      {#if sessionReference}
        <button class="session-reference" title={sessionReference} onclick={copySessionReference}>
          <Icon name="copy" size={13} /><span>{copiedSession ? "Copied" : `Session ${sessionReference.split("@", 1)[0]}`}</span>
        </button>
      {:else}
        <p>A direct line to your Turin agents.</p>
      {/if}
    </div>
    <div class="header-actions">
      {#if selectedSessionId}
        <button class="new-conversation-header" onclick={onNewConversation}><Icon name="plus" size={15} />New conversation</button>
      {/if}
      {#if detail?.branches.length}
        <details class="branch-menu">
          <summary><Icon name="branch" size={16} />{detail.branches.find(branch => branch.active)?.name ?? "Branches"}</summary>
          <div class="branch-popover">
            <span class="popover-label">Conversation paths</span>
            {#each detail.branches as branch (branch.branch_id)}
              <div class:active={branch.active} class="branch-row">
                <span>{branch.name}</span>
                <small>{branch.head_turn_index === null ? "empty" : `turn ${branch.head_turn_index}`}</small>
              </div>
            {/each}
          </div>
        </details>
      {/if}
    </div>
  </header>

  <div class="transcript" bind:this={transcript} onscroll={onTranscriptScroll}>
    <div class="message-column">
      {#if loading}
        <div class="conversation-skeleton" aria-label="Loading conversation">
          <i></i><i></i><i></i>
        </div>
      {:else if !selectedSessionId && !optimisticPrompt}
        <div class="welcome-state">
          <span class="welcome-mark"><Icon name="spark" size={26} /></span>
          <h2>What should we work on?</h2>
          <p>Investigate a problem, build something, or ask Turin to coordinate a longer-running task.</p>
          <div class="prompt-suggestions">
            <button onclick={() => { prompt = "Review the current project and tell me what deserves attention next."; composer?.focus(); }}>Review this project</button>
            <button onclick={() => { prompt = "Help me investigate a difficult problem step by step."; composer?.focus(); }}>Investigate a problem</button>
          </div>
        </div>
      {:else if error && !detail}
        <div class="inline-error"><strong>Conversation unavailable</strong><span>{error}</span><button onclick={() => loadDetail(false)}>Retry</button></div>
      {:else}
        {#if slidingData}<div class="window-loading">Loading conversation history...</div>{/if}
        <div class="message-spacer" style={`height: ${topSpacerHeight}px`}></div>
        {#each renderedMessages as message (message.id)}
          {@const body = messageText(message.content)}
          {@const tools = toolsFor(message)}
          {#if isTranscriptMessage(message) && (body || tools.length)}
            <article
              class:user={message.role.toLowerCase() === "user"}
              class="message"
              data-message-id={message.id}
              use:measureMessage={message.id}
            >
              <div class="message-author">{message.role.toLowerCase() === "user" ? "You" : humanize(selectedSummary?.agent_id ?? message.role)}</div>
              {#if body}
                <div class="message-body">
                  {#if message.role.toLowerCase() === "user"}
                    <span class="plain-message">{body}</span>
                  {:else}
                    <Markdown source={body} />
                  {/if}
                </div>
              {/if}
              {#if tools.length}
                <div class="tool-stack">
                  {#each tools as tool (tool.id)}
                    <details class:error={tool.is_error} class="tool-call">
                      <summary><span><Icon name="activity" size={15} />{humanize(tool.tool_name)}</span><small>{tool.duration_ms ? `${tool.duration_ms} ms` : tool.verdict}</small></summary>
                      <pre>{tool.is_error ? messageText(tool.output ?? "") : JSON.stringify(tool.args, null, 2)}</pre>
                    </details>
                  {/each}
                </div>
              {/if}
            </article>
          {/if}
        {/each}
        <div class="message-spacer" style={`height: ${bottomSpacerHeight}px`}></div>
        {#if optimisticPrompt}
          <article class="message user optimistic"><div class="message-author">You</div><div class="message-body"><span class="plain-message">{optimisticPrompt}</span></div></article>
        {/if}
        {#if streamText || selectedLive?.active_tasks}
          <article class="message streaming">
            <div class="message-author">{humanize(selectedSummary?.agent_id ?? selectedLive?.agent_id ?? "Turin")}</div>
            <div class="message-body"><Markdown source={streamText} /><span class="stream-cursor"></span></div>
          </article>
        {/if}
      {/if}
    </div>
  </div>

  <footer class="composer-wrap">
    {#if error && detail}
      <div class="composer-error">
        <span>{error}</span>
        {#if resumeFailed}<button onclick={onNewConversation}>Start a new conversation</button>{/if}
      </div>
    {/if}
    <div class="composer">
      <textarea
        bind:this={composer}
        bind:value={prompt}
        rows="2"
        placeholder="Message Turin..."
        aria-label="Message Turin"
        onkeydown={onComposerKeydown}
      ></textarea>
      <div class="composer-footer">
        <span>Enter to send · Shift Enter for a new line</span>
        <button class="send-button" aria-label="Send message" disabled={!prompt.trim() || sending} onclick={submit}>
          <Icon name="send" size={17} />
        </button>
      </div>
    </div>
  </footer>
</section>
