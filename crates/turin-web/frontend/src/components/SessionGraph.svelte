<script lang="ts">
  import { tick } from "svelte";
  import type { TurinClient } from "../lib/TurinClient";
  import type {
    SessionBranch,
    SessionGraph as SessionGraphModel,
    SessionGraphTurn,
    TaskStatus,
  } from "../lib/types";
  import { fullDate, humanize, messageText } from "../lib/format";
  import Icon from "./Icon.svelte";
  import Markdown from "./Markdown.svelte";

  export let client: TurinClient;
  export let sessionId: string;
  export let slotId: string | undefined = undefined;
  export let onClose: () => void;
  export let onChanged: () => void | Promise<void>;

  let graph: SessionGraphModel | null = null;
  let loading = true;
  let action = "";
  let error = "";
  let notice = "";
  let selectedTurnId: number | null = null;
  let compact = false;
  let graphScroll: HTMLDivElement;
  let forkName = "";
  let activateFork = false;
  let sidestepPrompt = "";
  let sidestepMode: "ephemeral" | "fork_sibling" = "ephemeral";
  let sidestepResult: TaskStatus | null = null;
  let promotionName = "";

  $: layout = buildLayout(graph, compact);
  $: selectedTurn = graph?.turns.find(turn => turn.turn_id === selectedTurnId) ?? null;
  $: selectedHeads = graph?.branches.filter(branch => branch.head_turn_id === selectedTurnId) ?? [];
  $: sidestepOutput = sidestepResult
    ? sidestepResult.output ?? messageText(sidestepResult.assistant_content ?? [])
    : "";

  void loadGraph(true);

  async function loadGraph(centerActive = false) {
    loading = !graph;
    error = "";
    try {
      graph = await client.sessionGraph(sessionId);
      const retained = graph.turns.some(turn => turn.turn_id === selectedTurnId);
      if (!retained) {
        selectedTurnId = graph.branches.find(branch => branch.active)?.head_turn_id
          ?? graph.turns[graph.turns.length - 1]?.turn_id
          ?? null;
        const initialTurn = graph.turns.find(turn => turn.turn_id === selectedTurnId);
        forkName = initialTurn ? `fork-turn-${initialTurn.turn_index}` : "";
      }
      await tick();
      if (centerActive) centerSelected();
    } catch (reason) {
      error = reason instanceof Error ? reason.message : String(reason);
    } finally {
      loading = false;
    }
  }

  function selectTurn(turn: SessionGraphTurn) {
    selectedTurnId = turn.turn_id;
    forkName = `fork-turn-${turn.turn_index}`;
    sidestepResult = null;
    notice = "";
  }

  function centerSelected() {
    if (!graphScroll || selectedTurnId === null) return;
    const node = graphScroll.querySelector<HTMLElement>(`[data-turn-id="${selectedTurnId}"]`);
    node?.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });
  }

  async function createFork() {
    if (!selectedTurn || !forkName.trim() || action) return;
    action = "fork";
    error = "";
    notice = "";
    try {
      const branch = await client.createBranch({
        session_id: sessionId,
        ...(slotId ? { slot_id: slotId } : {}),
        name: forkName.trim(),
        from_turn_id: selectedTurn.turn_id,
        activate: activateFork,
      });
      notice = activateFork ? `Created and checked out ${branch.name}.` : `Created ${branch.name}.`;
      await loadGraph();
      await onChanged();
    } catch (reason) {
      error = reason instanceof Error ? reason.message : String(reason);
    } finally {
      action = "";
    }
  }

  async function checkout(branch: SessionBranch) {
    if (branch.active || action) return;
    action = `checkout:${branch.branch_id}`;
    error = "";
    notice = "";
    try {
      await client.checkoutBranch({
        session_id: sessionId,
        ...(slotId ? { slot_id: slotId } : {}),
        branch: branch.branch_id,
      });
      notice = `Checked out ${branch.name}.`;
      await loadGraph();
      await onChanged();
    } catch (reason) {
      error = reason instanceof Error ? reason.message : String(reason);
    } finally {
      action = "";
    }
  }

  async function runSidestep() {
    if (!selectedTurn || !sidestepPrompt.trim() || action) return;
    action = "sidestep";
    error = "";
    notice = "";
    sidestepResult = null;
    try {
      sidestepResult = await client.sidestep({
        session_id: sessionId,
        prompt: sidestepPrompt.trim(),
        mode: sidestepMode,
        turn_id: selectedTurn.turn_id,
        timeout_ms: 180_000,
      });
      if (sidestepResult.error) throw new Error(sidestepResult.error);
      if (sidestepMode === "fork_sibling") {
        notice = "The sidestep was retained as a sibling path.";
        await loadGraph();
        await onChanged();
      } else if (sidestepResult.promotion_candidate) {
        promotionName = `kept-turn-${selectedTurn.turn_index}`;
      }
    } catch (reason) {
      error = reason instanceof Error ? reason.message : String(reason);
    } finally {
      action = "";
    }
  }

  async function promoteSidestep() {
    if (!sidestepResult?.promotion_candidate || action) return;
    action = "promote";
    error = "";
    try {
      const branch = await client.promoteTask({
        request_id: sidestepResult.request_id,
        ...(promotionName.trim() ? { branch_name: promotionName.trim() } : {}),
      });
      notice = `Promoted the aside to ${branch.name}.`;
      sidestepResult = { ...sidestepResult, promoted_branch: branch };
      await loadGraph();
      await onChanged();
    } catch (reason) {
      error = reason instanceof Error ? reason.message : String(reason);
    } finally {
      action = "";
    }
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === "Escape") onClose();
  }

  function nodeForBranch(branch: SessionBranch): LayoutNode | undefined {
    return layout.nodes.find(node => node.turn.turn_id === branch.head_turn_id);
  }

  function branchOffset(branch: SessionBranch): number {
    if (!graph || branch.head_turn_id === null || branch.head_turn_id === undefined) return 0;
    return graph.branches.filter(candidate => candidate.head_turn_id === branch.head_turn_id).indexOf(branch);
  }

  interface LayoutNode {
    turn: SessionGraphTurn;
    x: number;
    y: number;
    activePath: boolean;
  }

  interface LayoutEdge {
    source: LayoutNode;
    target: LayoutNode;
    activePath: boolean;
  }

  function buildLayout(model: SessionGraphModel | null, dense: boolean) {
    if (!model?.turns.length) return { nodes: [] as LayoutNode[], edges: [] as LayoutEdge[], width: 720, height: 480 };
    const turnById = new Map(model.turns.map(turn => [turn.turn_id, turn]));
    const children = new Map<number, SessionGraphTurn[]>();
    const roots: SessionGraphTurn[] = [];
    for (const turn of model.turns) {
      if (turn.parent_turn_id !== null && turn.parent_turn_id !== undefined && turnById.has(turn.parent_turn_id)) {
        const siblings = children.get(turn.parent_turn_id) ?? [];
        siblings.push(turn);
        children.set(turn.parent_turn_id, siblings);
      } else {
        roots.push(turn);
      }
    }
    const activePath = new Set<number>();
    let activeTurnId = model.branches.find(branch => branch.active)?.head_turn_id;
    while (activeTurnId !== null && activeTurnId !== undefined) {
      if (activePath.has(activeTurnId)) break;
      activePath.add(activeTurnId);
      activeTurnId = turnById.get(activeTurnId)?.parent_turn_id;
    }
    const laneById = new Map<number, number>();
    let nextLane = 1;
    function assign(turn: SessionGraphTurn, lane: number) {
      laneById.set(turn.turn_id, lane);
      const descendants = [...(children.get(turn.turn_id) ?? [])].sort((left, right) => {
        const activeDifference = Number(activePath.has(right.turn_id)) - Number(activePath.has(left.turn_id));
        return activeDifference || left.created_at.localeCompare(right.created_at) || left.turn_id - right.turn_id;
      });
      descendants.forEach((child, index) => assign(child, index === 0 ? lane : nextLane++));
    }
    [...roots]
      .sort((left, right) => left.created_at.localeCompare(right.created_at) || left.turn_id - right.turn_id)
      .forEach((root, index) => assign(root, index === 0 ? 0 : nextLane++));

    const xStep = dense ? 142 : 184;
    const yStep = dense ? 58 : 82;
    const nodes = model.turns.map(turn => ({
      turn,
      x: 62 + (laneById.get(turn.turn_id) ?? 0) * xStep,
      y: 62 + turn.turn_index * yStep,
      activePath: activePath.has(turn.turn_id),
    }));
    const layoutById = new Map(nodes.map(node => [node.turn.turn_id, node]));
    const edges: LayoutEdge[] = [];
    for (const target of nodes) {
      const source = target.turn.parent_turn_id === null || target.turn.parent_turn_id === undefined
        ? undefined
        : layoutById.get(target.turn.parent_turn_id);
      if (source) edges.push({ source, target, activePath: source.activePath && target.activePath });
    }
    const maxLane = Math.max(...laneById.values(), 0);
    const maxDepth = Math.max(...model.turns.map(turn => turn.turn_index), 0);
    return {
      nodes,
      edges,
      width: Math.max(720, 150 + (maxLane + 1) * xStep),
      height: Math.max(480, 150 + (maxDepth + 1) * yStep),
    };
  }

  function edgePath(edge: LayoutEdge): string {
    const bend = Math.max(18, (edge.target.y - edge.source.y) * 0.48);
    return `M ${edge.source.x} ${edge.source.y} C ${edge.source.x} ${edge.source.y + bend}, ${edge.target.x} ${edge.target.y - bend}, ${edge.target.x} ${edge.target.y}`;
  }
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="graph-overlay" role="dialog" aria-modal="true" aria-label="Session graph">
  <button class="graph-backdrop" aria-label="Close session graph" onclick={onClose}></button>
  <section class="graph-workspace">
    <header class="graph-header">
      <div>
        <span class="eyebrow">Conversation topology</span>
        <h2>Session Graph</h2>
        <p>Inspect durable paths, fork exact turns, or explore an idea without moving the active conversation.</p>
      </div>
      <div class="graph-header-actions">
        <span>{graph?.turns.length ?? 0} turns · {graph?.branches.length ?? 0} paths</span>
        <button class:active={compact} onclick={() => compact = !compact}>{compact ? "Expanded" : "Compact"}</button>
        <button title="Center selected turn" aria-label="Center selected turn" onclick={centerSelected}><Icon name="refresh" size={15} /></button>
        <button title="Close session graph" aria-label="Close session graph" onclick={onClose}><Icon name="close" size={16} /></button>
      </div>
    </header>

    {#if error}<div class="graph-alert error"><strong>Action failed</strong><span>{error}</span></div>{/if}
    {#if notice}<div class="graph-alert success"><strong>Graph updated</strong><span>{notice}</span></div>{/if}

    <div class="graph-body">
      <section class="map-panel">
        <div class="map-toolbar">
          <div><i class="active-line"></i>Active path <i></i>Other paths</div>
          <span>Every node is a durable turn</span>
        </div>
        <div class="graph-scroll" bind:this={graphScroll}>
          {#if loading}
            <div class="graph-loading"><i></i><span>Reading turn topology...</span></div>
          {:else if graph?.turns.length}
            <div class="graph-canvas" style={`width:${layout.width}px;height:${layout.height}px`}>
              <svg width={layout.width} height={layout.height} aria-hidden="true">
                {#each layout.edges as edge (`${edge.source.turn.turn_id}:${edge.target.turn.turn_id}`)}
                  <path class:active={edge.activePath} d={edgePath(edge)} />
                {/each}
              </svg>
              {#each layout.nodes as node (node.turn.turn_id)}
                <button
                  class="turn-node"
                  class:active-path={node.activePath}
                  class:selected={node.turn.turn_id === selectedTurnId}
                  class:has-tools={node.turn.tool_execution_count > 0}
                  data-turn-id={node.turn.turn_id}
                  style={`left:${node.x}px;top:${node.y}px`}
                  title={node.turn.preview ?? `Turn ${node.turn.turn_index}`}
                  onclick={() => selectTurn(node.turn)}
                >
                  <span>{node.turn.turn_index}</span>
                </button>
              {/each}
              {#if graph}
                {#each graph.branches as branch (branch.branch_id)}
                  {@const node = nodeForBranch(branch)}
                  {#if node}
                    <button
                      class="branch-tag"
                      class:active={branch.active}
                      style={`left:${node.x + 23}px;top:${node.y - 13 + branchOffset(branch) * 24}px`}
                      onclick={() => { selectedTurnId = node.turn.turn_id; }}
                    >
                      <Icon name="branch" size={12} />{branch.name}{branch.active ? " · active" : ""}
                    </button>
                  {/if}
                {/each}
              {/if}
            </div>
          {:else}
            <div class="graph-empty"><Icon name="branch" size={22} /><strong>No turns yet</strong><span>Send a message to create the first durable turn.</span></div>
          {/if}
        </div>
      </section>

      <aside class="graph-inspector">
        {#if selectedTurn}
          <header class="turn-heading">
            <span>Selected node</span>
            <h3>Turn {selectedTurn.turn_index}</h3>
            <time title={fullDate(selectedTurn.created_at)}>{fullDate(selectedTurn.created_at)}</time>
          </header>
          <div class="turn-stats">
            <div><strong>{selectedTurn.message_count}</strong><span>messages</span></div>
            <div><strong>{selectedTurn.tool_execution_count}</strong><span>tools</span></div>
            <div><strong>{selectedHeads.length}</strong><span>heads</span></div>
          </div>
          {#if selectedTurn.preview}<p class="turn-preview">{selectedTurn.preview}</p>{/if}

          {#if selectedHeads.length}
            <section class="inspector-section">
              <span class="section-label">Paths at this turn</span>
              <div class="path-list">
                {#each selectedHeads as branch (branch.branch_id)}
                  <div class:active={branch.active}>
                    <span><Icon name="branch" size={13} /><strong>{branch.name}</strong><small>{humanize(branch.origin_kind)}</small></span>
                    {#if branch.active}
                      <b>Active</b>
                    {:else}
                      <button disabled={Boolean(action)} onclick={() => checkout(branch)}>{action === `checkout:${branch.branch_id}` ? "Switching" : "Check out"}</button>
                    {/if}
                  </div>
                {/each}
              </div>
            </section>
          {/if}

          <section class="inspector-section">
            <span class="section-label">Create a durable path</span>
            <form class="fork-form" onsubmit={event => { event.preventDefault(); void createFork(); }}>
              <input bind:value={forkName} placeholder={`fork-turn-${selectedTurn.turn_index}`} aria-label="New branch name" />
              <label><input type="checkbox" bind:checked={activateFork} />Check out after creating</label>
              <button class="primary-action" disabled={!forkName.trim() || Boolean(action)}>{action === "fork" ? "Creating path..." : "Fork from this turn"}</button>
            </form>
          </section>

          <section class="inspector-section aside-section">
            <span class="section-label">Ask from here</span>
            <div class="mode-picker">
              <button class:active={sidestepMode === "ephemeral"} onclick={() => sidestepMode = "ephemeral"}>
                <strong>Private aside</strong><span>Explore first; promote only if useful.</span>
              </button>
              <button class:active={sidestepMode === "fork_sibling"} onclick={() => sidestepMode = "fork_sibling"}>
                <strong>Sibling path</strong><span>Retain the answer as a durable branch.</span>
              </button>
            </div>
            <textarea bind:value={sidestepPrompt} rows="3" placeholder="What should Turin explore from this point?" aria-label="Sidestep prompt"></textarea>
            <button class="primary-action" disabled={!sidestepPrompt.trim() || Boolean(action)} onclick={runSidestep}>
              {action === "sidestep" ? "Exploring..." : sidestepMode === "ephemeral" ? "Run private aside" : "Create sibling exploration"}
            </button>
          </section>

          {#if sidestepResult}
            <section class="sidestep-result">
              <header><span>Aside result</span><b>{humanize(sidestepResult.state)}</b></header>
              {#if sidestepOutput}<div class="result-body"><Markdown source={sidestepOutput} /></div>{/if}
              {#if sidestepResult.promotion_candidate && !sidestepResult.promoted_branch}
                <div class="promote-row">
                  <input bind:value={promotionName} placeholder="Branch name" aria-label="Promotion branch name" />
                  <button disabled={Boolean(action)} onclick={promoteSidestep}>{action === "promote" ? "Keeping..." : "Keep as branch"}</button>
                </div>
              {:else if sidestepResult.promoted_branch}
                <p class="promoted-note"><Icon name="branch" size={13} />Kept as {sidestepResult.promoted_branch.name}</p>
              {/if}
            </section>
          {/if}
        {:else}
          <div class="inspector-empty"><Icon name="branch" size={22} /><strong>Select a turn</strong><span>Inspect its context and available paths.</span></div>
        {/if}
      </aside>
    </div>
  </section>
</div>

<style>
  .graph-overlay { position: fixed; z-index: 80; inset: 0; display: grid; place-items: center; padding: 22px; }
  .graph-backdrop { position: absolute; inset: 0; width: 100%; height: 100%; border: 0; border-radius: 0; background: color-mix(in srgb, var(--ink) 32%, transparent); backdrop-filter: blur(5px); cursor: default; }
  .graph-workspace { position: relative; display: flex; flex-direction: column; width: min(1240px, 100%); height: min(850px, 100%); overflow: hidden; border: 1px solid var(--line-strong); border-radius: 18px; background: var(--surface-raised); box-shadow: var(--shadow-lg); }
  .graph-header { display: flex; flex: 0 0 auto; align-items: center; justify-content: space-between; gap: 24px; min-height: 92px; padding: 18px 20px 17px 23px; border-bottom: 1px solid var(--line); background: color-mix(in srgb, var(--surface) 92%, transparent); }
  .eyebrow, .section-label { color: var(--faint); font-size: 9px; font-weight: 750; letter-spacing: .1em; text-transform: uppercase; }
  .graph-header h2 { margin: 1px 0 0; font-size: 20px; font-weight: 690; letter-spacing: -.035em; }
  .graph-header p { margin: 2px 0 0; color: var(--muted); font-size: 10px; }
  .graph-header-actions { display: flex; align-items: center; gap: 7px; }
  .graph-header-actions > span { margin-right: 5px; color: var(--faint); font-size: 9px; }
  .graph-header-actions button { display: grid; place-items: center; min-width: 32px; height: 32px; padding: 0 9px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface-raised); color: var(--muted); font-size: 9px; font-weight: 650; }
  .graph-header-actions button:hover, .graph-header-actions button.active { border-color: color-mix(in srgb, var(--accent) 40%, var(--line)); color: var(--accent-strong); }
  .graph-alert { display: flex; flex: 0 0 auto; align-items: center; gap: 8px; min-height: 35px; padding: 7px 22px; border-bottom: 1px solid var(--line); font-size: 9px; }
  .graph-alert.error { background: color-mix(in srgb, var(--danger) 8%, var(--surface)); color: var(--danger); }
  .graph-alert.success { background: color-mix(in srgb, var(--success) 8%, var(--surface)); color: var(--success); }
  .graph-alert span { color: var(--muted); }
  .graph-body { display: grid; grid-template-columns: minmax(0, 1fr) 350px; min-height: 0; flex: 1; }
  .map-panel { display: flex; min-width: 0; min-height: 0; flex-direction: column; background: radial-gradient(circle at 18% 5%, color-mix(in srgb, var(--accent-soft) 48%, transparent), transparent 28%), var(--surface-muted); }
  .map-toolbar { display: flex; flex: 0 0 auto; align-items: center; justify-content: space-between; min-height: 42px; padding: 8px 15px; border-bottom: 1px solid var(--line); color: var(--faint); font-size: 9px; }
  .map-toolbar > div { display: flex; align-items: center; gap: 6px; }
  .map-toolbar i { display: inline-block; width: 16px; height: 2px; margin-left: 7px; background: var(--line-strong); }
  .map-toolbar i:first-child { margin-left: 0; background: var(--accent); }
  .graph-scroll { min-height: 0; flex: 1; overflow: auto; scrollbar-color: var(--line-strong) transparent; }
  .graph-canvas { position: relative; min-width: 100%; min-height: 100%; background-image: radial-gradient(circle, color-mix(in srgb, var(--faint) 25%, transparent) .8px, transparent .8px); background-size: 18px 18px; }
  .graph-canvas svg { position: absolute; inset: 0; overflow: visible; pointer-events: none; }
  .graph-canvas path { fill: none; stroke: var(--line-strong); stroke-width: 2; }
  .graph-canvas path.active { stroke: var(--accent); stroke-width: 2.5; }
  .turn-node { position: absolute; z-index: 2; display: grid; place-items: center; width: 28px; height: 28px; padding: 0; border: 2px solid var(--line-strong); border-radius: 50%; background: var(--surface-raised); color: var(--muted); box-shadow: 0 2px 7px color-mix(in srgb, var(--ink) 10%, transparent); font-size: 8px; font-weight: 750; font-variant-numeric: tabular-nums; transform: translate(-50%, -50%); }
  .turn-node::after { position: absolute; right: -3px; bottom: -3px; width: 7px; height: 7px; border: 2px solid var(--surface-raised); border-radius: 50%; background: transparent; content: ""; }
  .turn-node.has-tools::after { background: var(--warning); }
  .turn-node.active-path { border-color: var(--accent); color: var(--accent-strong); }
  .turn-node.selected { outline: 4px solid color-mix(in srgb, var(--accent) 18%, transparent); border-color: var(--accent); background: var(--accent); color: white; transform: translate(-50%, -50%) scale(1.12); }
  .branch-tag { position: absolute; z-index: 3; display: flex; align-items: center; gap: 5px; max-width: 145px; height: 24px; padding: 3px 7px; overflow: hidden; border: 1px solid var(--line); border-radius: 7px; background: color-mix(in srgb, var(--surface-raised) 95%, transparent); color: var(--muted); box-shadow: var(--shadow-sm); font-size: 8px; font-weight: 650; text-overflow: ellipsis; white-space: nowrap; }
  .branch-tag.active { border-color: color-mix(in srgb, var(--accent) 35%, var(--line)); background: var(--accent-soft); color: var(--accent-strong); }
  .graph-loading, .graph-empty, .inspector-empty { display: grid; height: 100%; place-content: center; justify-items: center; color: var(--faint); text-align: center; }
  .graph-loading i { width: 26px; height: 26px; margin-bottom: 8px; border: 2px solid var(--line); border-top-color: var(--accent); border-radius: 50%; animation: spin .8s linear infinite; }
  .graph-loading span, .graph-empty span, .inspector-empty span { margin-top: 4px; font-size: 9px; }
  .graph-empty strong, .inspector-empty strong { margin-top: 8px; color: var(--ink); font-size: 11px; }
  .graph-inspector { min-height: 0; padding: 19px; overflow: auto; border-left: 1px solid var(--line); background: var(--surface-raised); }
  .turn-heading > span { color: var(--faint); font-size: 9px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
  .turn-heading h3 { margin: 2px 0 0; font-size: 21px; letter-spacing: -.035em; }
  .turn-heading time { color: var(--faint); font-size: 9px; }
  .turn-stats { display: grid; grid-template-columns: repeat(3, 1fr); margin: 14px 0 0; overflow: hidden; border: 1px solid var(--line); border-radius: 10px; }
  .turn-stats > div { display: grid; gap: 1px; padding: 9px 10px; border-right: 1px solid var(--line); }
  .turn-stats > div:last-child { border-right: 0; }
  .turn-stats strong { font-size: 15px; }
  .turn-stats span { color: var(--faint); font-size: 8px; text-transform: uppercase; }
  .turn-preview { margin: 11px 0 0; padding: 10px 11px; border-left: 2px solid var(--accent); border-radius: 0 8px 8px 0; background: var(--surface-muted); color: var(--muted); font-size: 10px; line-height: 1.55; }
  .inspector-section { display: grid; gap: 8px; margin-top: 18px; }
  .path-list { display: grid; gap: 5px; }
  .path-list > div { display: flex; align-items: center; justify-content: space-between; gap: 8px; min-height: 38px; padding: 6px 7px 6px 9px; border: 1px solid var(--line); border-radius: 9px; background: var(--surface); }
  .path-list > div.active { border-color: color-mix(in srgb, var(--accent) 30%, var(--line)); background: var(--accent-soft); }
  .path-list span { display: flex; min-width: 0; align-items: center; gap: 6px; }
  .path-list strong { overflow: hidden; font-size: 10px; text-overflow: ellipsis; white-space: nowrap; }
  .path-list small { color: var(--faint); font-size: 8px; }
  .path-list b { color: var(--accent-strong); font-size: 8px; }
  .path-list button, .promote-row button { padding: 5px 7px; border: 1px solid var(--line); border-radius: 7px; background: var(--surface-raised); color: var(--muted); font-size: 8px; font-weight: 650; }
  .fork-form { display: grid; gap: 7px; }
  .fork-form > input, .aside-section textarea, .promote-row input { width: 100%; border: 1px solid var(--line); border-radius: 8px; outline: 0; background: var(--surface); color: var(--ink); font-size: 10px; }
  .fork-form > input, .promote-row input { height: 34px; padding: 7px 9px; }
  .aside-section textarea { min-height: 72px; padding: 9px; resize: vertical; line-height: 1.5; }
  .fork-form input:focus, .aside-section textarea:focus, .promote-row input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 10%, transparent); }
  .fork-form label { display: flex; align-items: center; gap: 6px; color: var(--muted); font-size: 9px; }
  .primary-action { min-height: 34px; border: 1px solid var(--ink); border-radius: 8px; background: var(--ink); color: var(--surface-raised); font-size: 9px; font-weight: 680; }
  .primary-action:hover { background: color-mix(in srgb, var(--ink) 88%, var(--accent)); }
  .mode-picker { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .mode-picker button { display: grid; gap: 2px; min-height: 58px; padding: 8px; border: 1px solid var(--line); border-radius: 9px; background: var(--surface); color: var(--muted); text-align: left; }
  .mode-picker button.active { border-color: color-mix(in srgb, var(--accent) 45%, var(--line)); background: var(--accent-soft); color: var(--accent-strong); }
  .mode-picker strong { font-size: 9px; }
  .mode-picker span { color: var(--faint); font-size: 8px; line-height: 1.35; }
  .sidestep-result { margin-top: 17px; overflow: hidden; border: 1px solid color-mix(in srgb, var(--accent) 28%, var(--line)); border-radius: 11px; background: var(--surface); }
  .sidestep-result > header { display: flex; align-items: center; justify-content: space-between; padding: 8px 10px; border-bottom: 1px solid var(--line); background: var(--accent-soft); color: var(--accent-strong); font-size: 9px; font-weight: 700; }
  .sidestep-result > header b { font-size: 8px; }
  .result-body { max-height: 230px; padding: 11px; overflow: auto; font-size: 10px; }
  .promote-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 6px; padding: 8px; border-top: 1px solid var(--line); }
  .promoted-note { display: flex; align-items: center; gap: 6px; margin: 0; padding: 9px 10px; border-top: 1px solid var(--line); color: var(--success); font-size: 9px; font-weight: 650; }
  @keyframes spin { to { transform: rotate(360deg); } }
  @media (max-width: 900px) {
    .graph-overlay { padding: 0; }
    .graph-workspace { width: 100%; height: 100%; border: 0; border-radius: 0; }
    .graph-header p, .graph-header-actions > span { display: none; }
    .graph-body { grid-template-columns: 1fr; grid-template-rows: minmax(280px, 48%) minmax(0, 1fr); }
    .graph-inspector { border-top: 1px solid var(--line); border-left: 0; }
  }
</style>
