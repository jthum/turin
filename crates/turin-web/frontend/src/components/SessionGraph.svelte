<script lang="ts">
  import { onMount, tick } from "svelte";
  import type { TurinClient } from "../lib/TurinClient";
  import type {
    SessionBranch,
    SessionDetail,
    SessionGraph as SessionGraphModel,
    SessionGraphTurn,
    SessionMessage,
    TaskStatus,
  } from "../lib/types";
  import { fullDate, humanize, messageText } from "../lib/format";
  import Icon from "./Icon.svelte";
  import Markdown from "./Markdown.svelte";

  export let client: TurinClient;
  export let sessionId: string;
  export let slotId: string | undefined = undefined;
  export let initialTurnIndex: number | null = null;
  export let initialMode: "inspect" | "compare" | "explore" = "inspect";
  export let onClose: () => void;
  export let onChanged: () => void | Promise<void>;

  const PATH_MESSAGE_LIMIT = 24;

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
  let inspectorMode: "inspect" | "compare" | "explore" = initialMode;
  let initialSelectionApplied = false;
  let pathDetail: SessionDetail | null = null;
  let pathLoading = false;
  let pathError = "";
  let pathRequestVersion = 0;
  let comparisonLeftBranchId = "";
  let comparisonRightBranchId = "";
  let comparisonLeftDetail: SessionDetail | null = null;
  let comparisonRightDetail: SessionDetail | null = null;
  let comparisonLoading = false;
  let comparisonError = "";
  let comparisonRequestVersion = 0;
  let workspace: HTMLElement;
  let closeButton: HTMLButtonElement;
  let previouslyFocused: HTMLElement | null = null;
  const pathCache = new Map<number, SessionDetail>();
  const pathRequests = new Map<number, Promise<SessionDetail>>();

  $: layout = buildLayout(graph, compact, selectedTurnId);
  $: selectedTurn = graph?.turns.find(turn => turn.turn_id === selectedTurnId) ?? null;
  $: selectedHeads = graph?.branches.filter(branch => branch.head_turn_id === selectedTurnId) ?? [];
  $: selectedOnActivePath = layout.nodes.find(node => node.turn.turn_id === selectedTurnId)?.activePath ?? false;
  $: branchHeads = graph?.branches.filter(branch => branch.head_turn_id !== null && branch.head_turn_id !== undefined) ?? [];
  $: comparisonLeftBranch = branchHeads.find(branch => branch.branch_id === comparisonLeftBranchId) ?? null;
  $: comparisonRightBranch = branchHeads.find(branch => branch.branch_id === comparisonRightBranchId) ?? null;
  $: comparison = buildBranchComparison(graph, comparisonLeftBranch, comparisonRightBranch);
  $: comparisonLeftMessages = comparisonMessages(comparisonLeftDetail, comparison?.leftTurns ?? []);
  $: comparisonRightMessages = comparisonMessages(comparisonRightDetail, comparison?.rightTurns ?? []);
  $: comparisonLeftTools = comparisonTools(comparisonLeftDetail, comparison?.leftTurns ?? []);
  $: comparisonRightTools = comparisonTools(comparisonRightDetail, comparison?.rightTurns ?? []);
  $: comparisonLeftLoadedRows = comparisonRows(comparisonLeftDetail, comparison?.leftTurns ?? []);
  $: comparisonRightLoadedRows = comparisonRows(comparisonRightDetail, comparison?.rightTurns ?? []);
  $: pathMessages = (pathDetail?.messages ?? []).filter(message => {
    const role = message.role.toLowerCase();
    return role === "user" || role === "assistant";
  });
  $: pathTurns = groupContextTurns(pathMessages);
  $: pathTools = pathDetail?.tool_executions ?? [];
  $: omittedPathMessages = pathDetail?.message_window?.offset ?? 0;
  $: sidestepOutput = sidestepResult
    ? sidestepResult.output ?? messageText(sidestepResult.assistant_content ?? [])
    : "";

  void loadGraph(true);

  onMount(() => {
    previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    void tick().then(() => closeButton?.focus());
    return () => previouslyFocused?.focus();
  });

  async function loadGraph(centerActive = false) {
    loading = !graph;
    error = "";
    try {
      graph = await client.sessionGraph(sessionId);
      const retained = graph.turns.some(turn => turn.turn_id === selectedTurnId);
      if (!retained) {
        const requestedTurn = !initialSelectionApplied && initialTurnIndex !== null
          ? activePathTurnAtIndex(graph, initialTurnIndex)
          : null;
        selectedTurnId = requestedTurn?.turn_id
          ?? graph.branches.find(branch => branch.active)?.head_turn_id
          ?? graph.turns[graph.turns.length - 1]?.turn_id
          ?? null;
        const initialTurn = graph.turns.find(turn => turn.turn_id === selectedTurnId);
        forkName = initialTurn ? `fork-turn-${initialTurn.turn_index}` : "";
      }
      if (!initialSelectionApplied) {
        inspectorMode = initialMode;
        initialSelectionApplied = true;
      }
      syncComparisonSelection();
      if (inspectorMode === "compare") {
        void loadComparison();
      } else if (selectedTurnId !== null) {
        void loadPath(selectedTurnId);
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
    inspectorMode = "inspect";
    void loadPath(turn.turn_id);
  }

  async function loadPath(turnId: number, force = false) {
    const requestVersion = ++pathRequestVersion;
    pathError = "";
    const cached = !force ? pathCache.get(turnId) : undefined;
    if (cached) {
      pathDetail = cached;
      pathLoading = false;
      return;
    }
    pathLoading = true;
    pathDetail = null;
    try {
      const detail = await getPathDetail(turnId, force);
      if (requestVersion !== pathRequestVersion) return;
      pathDetail = detail;
    } catch (reason) {
      if (requestVersion !== pathRequestVersion) return;
      pathError = reason instanceof Error ? reason.message : String(reason);
    } finally {
      if (requestVersion === pathRequestVersion) pathLoading = false;
    }
  }

  async function getPathDetail(turnId: number, force = false): Promise<SessionDetail> {
    if (force) pathCache.delete(turnId);
    const cached = pathCache.get(turnId);
    if (cached) return cached;
    const pending = pathRequests.get(turnId);
    if (pending) return pending;
    const request = client.sessionPath(sessionId, turnId, PATH_MESSAGE_LIMIT);
    pathRequests.set(turnId, request);
    try {
      const detail = await request;
      pathCache.set(turnId, detail);
      return detail;
    } finally {
      if (pathRequests.get(turnId) === request) pathRequests.delete(turnId);
    }
  }

  function syncComparisonSelection() {
    const heads = graph?.branches.filter(branch => branch.head_turn_id !== null && branch.head_turn_id !== undefined) ?? [];
    if (!heads.length) {
      comparisonLeftBranchId = "";
      comparisonRightBranchId = "";
      return;
    }
    if (!heads.some(branch => branch.branch_id === comparisonLeftBranchId)) {
      comparisonLeftBranchId = heads.find(branch => branch.active)?.branch_id ?? heads[0]!.branch_id;
    }
    if (!heads.some(branch => branch.branch_id === comparisonRightBranchId) || comparisonRightBranchId === comparisonLeftBranchId) {
      comparisonRightBranchId = heads.find(branch => branch.head_turn_id === selectedTurnId && branch.branch_id !== comparisonLeftBranchId)?.branch_id
        ?? heads.find(branch => branch.branch_id !== comparisonLeftBranchId)?.branch_id
        ?? "";
    }
  }

  function showComparison() {
    syncComparisonSelection();
    inspectorMode = "compare";
    focusComparedHead();
    void loadComparison();
  }

  function showInspection() {
    inspectorMode = "inspect";
    if (selectedTurnId !== null) void loadPath(selectedTurnId);
  }

  function selectComparisonBranch(side: "left" | "right", event: Event) {
    const branchId = (event.currentTarget as HTMLSelectElement).value;
    if (side === "left") comparisonLeftBranchId = branchId;
    else comparisonRightBranchId = branchId;
    focusComparedHead();
    void loadComparison();
  }

  function swapComparison() {
    const left = comparisonLeftBranchId;
    comparisonLeftBranchId = comparisonRightBranchId;
    comparisonRightBranchId = left;
    focusComparedHead();
    void loadComparison();
  }

  function focusComparedHead() {
    const branch = graph?.branches.find(candidate => candidate.branch_id === comparisonRightBranchId);
    if (branch?.head_turn_id === null || branch?.head_turn_id === undefined) return;
    selectedTurnId = branch.head_turn_id;
    void tick().then(centerSelected);
  }

  async function loadComparison(force = false) {
    const left = graph?.branches.find(branch => branch.branch_id === comparisonLeftBranchId);
    const right = graph?.branches.find(branch => branch.branch_id === comparisonRightBranchId);
    const leftTurnId = left?.head_turn_id;
    const rightTurnId = right?.head_turn_id;
    const requestVersion = ++comparisonRequestVersion;
    comparisonError = "";
    comparisonLeftDetail = null;
    comparisonRightDetail = null;
    if (leftTurnId === null || leftTurnId === undefined || rightTurnId === null || rightTurnId === undefined) {
      comparisonLoading = false;
      return;
    }
    comparisonLoading = true;
    try {
      const [leftDetail, rightDetail] = await Promise.all([
        getPathDetail(leftTurnId, force),
        getPathDetail(rightTurnId, force),
      ]);
      if (requestVersion !== comparisonRequestVersion) return;
      comparisonLeftDetail = leftDetail;
      comparisonRightDetail = rightDetail;
    } catch (reason) {
      if (requestVersion !== comparisonRequestVersion) return;
      comparisonError = reason instanceof Error ? reason.message : String(reason);
    } finally {
      if (requestVersion === comparisonRequestVersion) comparisonLoading = false;
    }
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
    if (event.key === "Escape") {
      onClose();
      return;
    }
    if (event.key !== "Tab" || !workspace) return;
    const focusable = [...workspace.querySelectorAll<HTMLElement>(
      'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    )];
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (!first || !last) return;
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function nodeForBranch(branch: SessionBranch): LayoutNode | undefined {
    return layout.nodes.find(node => node.turn.turn_id === branch.head_turn_id);
  }

  function branchOffset(branch: SessionBranch): number {
    if (!graph || branch.head_turn_id === null || branch.head_turn_id === undefined) return 0;
    return graph.branches.filter(candidate => candidate.head_turn_id === branch.head_turn_id).indexOf(branch);
  }

  interface BranchComparison {
    commonAncestor: SessionGraphTurn | null;
    leftTurns: SessionGraphTurn[];
    rightTurns: SessionGraphTurn[];
    leftMessageRows: number;
    rightMessageRows: number;
    leftToolRuns: number;
    rightToolRuns: number;
  }

  function buildBranchComparison(
    model: SessionGraphModel | null,
    left: SessionBranch | null,
    right: SessionBranch | null,
  ): BranchComparison | null {
    if (!model || left?.head_turn_id === null || left?.head_turn_id === undefined
      || right?.head_turn_id === null || right?.head_turn_id === undefined) return null;
    const turnById = new Map(model.turns.map(turn => [turn.turn_id, turn]));
    const leftPath = ancestry(turnById, left.head_turn_id);
    const rightPath = ancestry(turnById, right.head_turn_id);
    let sharedIndex = -1;
    const sharedLength = Math.min(leftPath.length, rightPath.length);
    for (let index = 0; index < sharedLength; index += 1) {
      if (leftPath[index]?.turn_id !== rightPath[index]?.turn_id) break;
      sharedIndex = index;
    }
    const leftTurns = leftPath.slice(sharedIndex + 1);
    const rightTurns = rightPath.slice(sharedIndex + 1);
    return {
      commonAncestor: sharedIndex >= 0 ? leftPath[sharedIndex] ?? null : null,
      leftTurns,
      rightTurns,
      leftMessageRows: leftTurns.reduce((total, turn) => total + turn.message_count, 0),
      rightMessageRows: rightTurns.reduce((total, turn) => total + turn.message_count, 0),
      leftToolRuns: leftTurns.reduce((total, turn) => total + turn.tool_execution_count, 0),
      rightToolRuns: rightTurns.reduce((total, turn) => total + turn.tool_execution_count, 0),
    };
  }

  function ancestry(turnById: Map<number, SessionGraphTurn>, headTurnId: number): SessionGraphTurn[] {
    const path: SessionGraphTurn[] = [];
    const visited = new Set<number>();
    let turnId: number | null | undefined = headTurnId;
    while (turnId !== null && turnId !== undefined && !visited.has(turnId)) {
      visited.add(turnId);
      const turn = turnById.get(turnId);
      if (!turn) break;
      path.push(turn);
      turnId = turn.parent_turn_id;
    }
    return path.reverse();
  }

  function comparisonRows(detail: SessionDetail | null, turns: SessionGraphTurn[]) {
    const indexes = new Set(turns.map(turn => turn.turn_index));
    return (detail?.messages ?? []).filter(message => indexes.has(message.turn_index));
  }

  function comparisonMessages(detail: SessionDetail | null, turns: SessionGraphTurn[]) {
    return comparisonRows(detail, turns).filter(message => {
      const role = message.role.toLowerCase();
      return role === "user" || role === "assistant";
    });
  }

  function comparisonTools(detail: SessionDetail | null, turns: SessionGraphTurn[]) {
    const indexes = new Set(turns.map(turn => turn.turn_index));
    return (detail?.tool_executions ?? []).filter(tool => indexes.has(tool.turn_index));
  }

  interface ContextTurnGroup {
    turnIndex: number;
    messages: SessionMessage[];
  }

  function groupContextTurns(messages: SessionMessage[]): ContextTurnGroup[] {
    const groups: ContextTurnGroup[] = [];
    for (const message of messages) {
      const current = groups[groups.length - 1];
      if (current?.turnIndex === message.turn_index) current.messages.push(message);
      else groups.push({ turnIndex: message.turn_index, messages: [message] });
    }
    return groups;
  }

  function activePathTurnAtIndex(model: SessionGraphModel, turnIndex: number): SessionGraphTurn | null {
    const turnById = new Map(model.turns.map(turn => [turn.turn_id, turn]));
    let turnId = model.branches.find(branch => branch.active)?.head_turn_id;
    const visited = new Set<number>();
    while (turnId !== null && turnId !== undefined && !visited.has(turnId)) {
      visited.add(turnId);
      const turn = turnById.get(turnId);
      if (!turn) return null;
      if (turn.turn_index === turnIndex) return turn;
      turnId = turn.parent_turn_id;
    }
    return null;
  }

  function branchSource(branch: SessionBranch): string | null {
    if (branch.source_turn_id === null || branch.source_turn_id === undefined) return null;
    const source = graph?.turns.find(turn => turn.turn_id === branch.source_turn_id);
    return source ? `Turn ${source.turn_index}` : `Turn id ${branch.source_turn_id}`;
  }

  interface LayoutNode {
    turn: SessionGraphTurn;
    x: number;
    y: number;
    activePath: boolean;
    inspectedPath: boolean;
  }

  interface LayoutEdge {
    source: LayoutNode;
    target: LayoutNode;
    activePath: boolean;
    inspectedPath: boolean;
  }

  function buildLayout(model: SessionGraphModel | null, dense: boolean, inspectedTurnId: number | null) {
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
    const inspectedPath = new Set<number>();
    let inspectedId: number | null | undefined = inspectedTurnId;
    while (inspectedId !== null && inspectedId !== undefined) {
      if (inspectedPath.has(inspectedId)) break;
      inspectedPath.add(inspectedId);
      inspectedId = turnById.get(inspectedId)?.parent_turn_id;
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
      inspectedPath: inspectedPath.has(turn.turn_id),
    }));
    const layoutById = new Map(nodes.map(node => [node.turn.turn_id, node]));
    const edges: LayoutEdge[] = [];
    for (const target of nodes) {
      const source = target.turn.parent_turn_id === null || target.turn.parent_turn_id === undefined
        ? undefined
        : layoutById.get(target.turn.parent_turn_id);
      if (source) edges.push({
        source,
        target,
        activePath: source.activePath && target.activePath,
        inspectedPath: source.inspectedPath && target.inspectedPath,
      });
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

<div class="graph-overlay" role="dialog" aria-modal="true" aria-labelledby="session-graph-title">
  <button class="graph-backdrop" aria-label="Close session graph" onclick={onClose}></button>
  <section class="graph-workspace" bind:this={workspace}>
    <header class="graph-header">
      <div>
        <span class="eyebrow">Conversation topology</span>
        <h2 id="session-graph-title">{inspectorMode === "compare" ? "Compare Paths" : "Session Graph"}</h2>
        <p>{inspectorMode === "compare" ? "Review two branch histories without checking either one out." : "Inspect durable paths, fork exact turns, or explore an idea without moving the active conversation."}</p>
      </div>
      <div class="graph-header-actions">
        <span>{graph?.turns.length ?? 0} turns · {graph?.branches.length ?? 0} paths</span>
        <button class:active={compact} onclick={() => compact = !compact}>{compact ? "Expanded" : "Compact"}</button>
        <button title="Center selected turn" aria-label="Center selected turn" onclick={centerSelected}><Icon name="refresh" size={15} /></button>
        <button bind:this={closeButton} title="Close session graph" aria-label="Close session graph" onclick={onClose}><Icon name="close" size={16} /></button>
      </div>
    </header>

    {#if error}<div class="graph-alert error"><strong>Action failed</strong><span>{error}</span></div>{/if}
    {#if notice}<div class="graph-alert success"><strong>Graph updated</strong><span>{notice}</span></div>{/if}

    <div class="graph-body" class:comparison-open={inspectorMode === "compare"}>
      <section class="map-panel">
        <div class="map-toolbar">
          <div><i class="active-line"></i>Active <i class="inspected-line"></i>Inspected <i></i>Other</div>
          <span>Selection does not change the active path</span>
        </div>
        <div class="graph-scroll" bind:this={graphScroll}>
          {#if loading}
            <div class="graph-loading"><i></i><span>Reading turn topology...</span></div>
          {:else if graph?.turns.length}
            <div class="graph-canvas" style={`width:${layout.width}px;height:${layout.height}px`}>
              <svg width={layout.width} height={layout.height} aria-hidden="true">
                {#each layout.edges as edge (`${edge.source.turn.turn_id}:${edge.target.turn.turn_id}`)}
                  <path class:active={edge.activePath} class:inspected={edge.inspectedPath && !edge.activePath} d={edgePath(edge)} />
                {/each}
              </svg>
              {#each layout.nodes as node (node.turn.turn_id)}
                <button
                  class="turn-node"
                  class:active-path={node.activePath}
                  class:inspected-path={node.inspectedPath && !node.activePath}
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
                      onclick={() => selectTurn(node.turn)}
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
          {#if inspectorMode === "compare"}
            <header class="turn-heading comparison-heading">
              <div>
                <span>Read-only comparison</span>
                <h3>Branch heads</h3>
                <time>No path will be activated or changed.</time>
              </div>
              <span class="path-state">Active path unchanged</span>
            </header>
          {:else}
            <header class="turn-heading">
              <div>
                <span>Inspecting durable path</span>
                <h3>Turn {selectedTurn.turn_index}</h3>
                <time title={fullDate(selectedTurn.created_at)}>{fullDate(selectedTurn.created_at)}</time>
              </div>
              <span class:active={selectedOnActivePath} class="path-state">{selectedOnActivePath ? "On active path" : "Active path unchanged"}</span>
            </header>
            <div class="turn-stats">
              <div><strong>{selectedTurn.message_count}</strong><span>messages</span></div>
              <div><strong>{selectedTurn.tool_execution_count}</strong><span>tools</span></div>
              <div><strong>{selectedHeads.length}</strong><span>heads</span></div>
            </div>
          {/if}
          <nav class="inspector-tabs" aria-label="Turn inspector mode">
            <button class:active={inspectorMode === "inspect"} onclick={showInspection}>Inspect</button>
            <button
              class:active={inspectorMode === "compare"}
              disabled={branchHeads.length < 2}
              title={branchHeads.length < 2 ? "Create another branch to compare paths" : "Compare two branch heads"}
              onclick={showComparison}
            >Compare</button>
            <button class:active={inspectorMode === "explore"} onclick={() => inspectorMode = "explore"}>Explore from here</button>
          </nav>

          {#if inspectorMode === "inspect"}
            {#if selectedHeads.length}
              <section class="inspector-section">
                <span class="section-label">Branch heads here</span>
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

            <section class="inspector-section context-section">
              <div class="context-heading">
                <span class="section-label">Conversation path through this turn</span>
                <button disabled={pathLoading} onclick={() => loadPath(selectedTurn.turn_id, true)}><Icon name="refresh" size={12} />Refresh</button>
              </div>
              {#if pathLoading}
                <div class="path-loading"><i></i><span>Loading this path...</span></div>
              {:else if pathError}
                <div class="path-error"><strong>Path unavailable</strong><span>{pathError}</span><button onclick={() => loadPath(selectedTurn.turn_id, true)}>Retry</button></div>
              {:else if pathDetail}
                <div class="context-explainer">
                  <Icon name="branch" size={14} />
                  <div><strong>Continuation path at Turn {selectedTurn.turn_index}</strong><span>A fork or aside starts from this durable history. Compaction and context-window policy may reduce what reaches the model.</span></div>
                </div>
                <div class="context-summary">
                  <span>{pathDetail.message_window?.total ?? pathDetail.messages.length} message rows</span>
                  <span>{pathTools.length} tool runs in window</span>
                </div>
                {#if omittedPathMessages > 0}<p class="path-omitted">{omittedPathMessages} earlier rows are outside this preview.</p>{/if}
                <div class="context-turns">
                  {#each pathTurns as turn (turn.turnIndex)}
                    <section class:current={turn.turnIndex === selectedTurn.turn_index} class="context-turn">
                      <header><span>Turn {turn.turnIndex}</span>{#if turn.turnIndex === selectedTurn.turn_index}<b>Selected point</b>{/if}</header>
                      {#each turn.messages as message (message.id)}
                        {@const body = messageText(message.content)}
                        {#if body}
                          <article class:user={message.role.toLowerCase() === "user"} class="context-entry">
                            <strong>{message.role.toLowerCase() === "user" ? "Your message" : `Assistant response · ${humanize(graph?.session.agent_id ?? "assistant")}`}</strong>
                            {#if message.role.toLowerCase() === "assistant"}
                              <div class="path-markdown"><Markdown source={body} /></div>
                            {:else}
                              <p>{body}</p>
                            {/if}
                          </article>
                        {/if}
                      {/each}
                    </section>
                  {/each}
                  {#if !pathTurns.length}<p class="path-empty">No conversational messages are stored on this path yet.</p>{/if}
                </div>
                {#if pathTools.length}
                  <details class="path-tools">
                    <summary>{pathTools.length} tool {pathTools.length === 1 ? "execution" : "executions"}</summary>
                    {#each pathTools as tool (tool.id)}
                      <div><Icon name="activity" size={12} /><span>{humanize(tool.tool_name)}</span><small>Turn {tool.turn_index}{tool.duration_ms !== null && tool.duration_ms !== undefined ? ` · ${tool.duration_ms} ms` : ""}</small></div>
                    {/each}
                  </details>
                {/if}
              {/if}
            </section>
          {:else if inspectorMode === "compare"}
            <section class="comparison-controls">
              <div class="context-heading">
                <span class="section-label">Choose two durable heads</span>
                <button disabled={comparisonLoading} onclick={() => loadComparison(true)}><Icon name="refresh" size={12} />Refresh</button>
              </div>
              <div class="comparison-pickers">
                <label>
                  <span>Reference</span>
                  <select value={comparisonLeftBranchId} onchange={event => selectComparisonBranch("left", event)}>
                    {#each branchHeads as branch (branch.branch_id)}
                      <option value={branch.branch_id} disabled={branch.branch_id === comparisonRightBranchId}>{branch.name}{branch.active ? " · active" : ""}</option>
                    {/each}
                  </select>
                </label>
                <button class="swap-button" aria-label="Swap compared branches" title="Swap compared branches" onclick={swapComparison}>Swap</button>
                <label>
                  <span>Candidate</span>
                  <select value={comparisonRightBranchId} onchange={event => selectComparisonBranch("right", event)}>
                    {#each branchHeads as branch (branch.branch_id)}
                      <option value={branch.branch_id} disabled={branch.branch_id === comparisonLeftBranchId}>{branch.name}{branch.active ? " · active" : ""}</option>
                    {/each}
                  </select>
                </label>
              </div>
            </section>

            {#if comparisonLoading}
              <div class="path-loading comparison-state"><i></i><span>Reading both exact paths...</span></div>
            {:else if comparisonError}
              <div class="path-error comparison-state"><strong>Comparison unavailable</strong><span>{comparisonError}</span><button onclick={() => loadComparison(true)}>Retry</button></div>
            {:else if comparison && comparisonLeftBranch && comparisonRightBranch}
              <div class="common-ancestor">
                <Icon name="branch" size={15} />
                <div>
                  <strong>{comparison.leftTurns.length === 0 && comparison.rightTurns.length === 0
                    ? "Both labels point to the same head"
                    : comparison.commonAncestor
                      ? `Shared through Turn ${comparison.commonAncestor.turn_index}`
                      : "No shared ancestor found"}</strong>
                  <span>{comparison.commonAncestor
                    ? fullDate(comparison.commonAncestor.created_at)
                    : "The loaded topology contains separate roots."}</span>
                </div>
              </div>

              <div class="comparison-columns">
                <section class="comparison-branch">
                  <header>
                    <div><span>Reference</span><strong title={comparisonLeftBranch.name}>{comparisonLeftBranch.name}</strong></div>
                    {#if comparisonLeftBranch.active}<b>Active</b>{/if}
                  </header>
                  <div class="branch-provenance">
                    <span>{humanize(comparisonLeftBranch.origin_kind)}</span>
                    {#if branchSource(comparisonLeftBranch)}<span>from {branchSource(comparisonLeftBranch)}</span>{/if}
                    {#if comparisonLeftBranch.origin_task_id}<span title={comparisonLeftBranch.origin_task_id}>task {comparisonLeftBranch.origin_task_id}</span>{/if}
                    {#if comparisonLeftBranch.origin_execution_id}<span title={comparisonLeftBranch.origin_execution_id}>run {comparisonLeftBranch.origin_execution_id}</span>{/if}
                  </div>
                  <div class="comparison-stats">
                    <span><strong>{comparison.leftTurns.length}</strong> unique turns</span>
                    <span><strong>{comparison.leftMessageRows}</strong> message rows</span>
                    <span><strong>{comparison.leftToolRuns}</strong> tool runs</span>
                  </div>
                  {#if comparison.leftMessageRows > comparisonLeftLoadedRows.length}
                    <p class="comparison-clipped">Latest bounded preview. {comparison.leftMessageRows - comparisonLeftLoadedRows.length} earlier branch-only rows are not shown.</p>
                  {/if}
                  <div class="comparison-transcript">
                    {#each comparisonLeftMessages as message (message.id)}
                      {@const body = messageText(message.content)}
                      {#if body}
                        <article class:user={message.role.toLowerCase() === "user"}>
                          <header><strong>{message.role.toLowerCase() === "user" ? "You" : humanize(graph?.session.agent_id ?? "assistant")}</strong><span>Turn {message.turn_index}</span></header>
                          {#if message.role.toLowerCase() === "assistant"}<div class="comparison-markdown"><Markdown source={body} /></div>{:else}<p>{body}</p>{/if}
                        </article>
                      {/if}
                    {/each}
                    {#if !comparisonLeftMessages.length}<p class="path-empty">{comparison.leftTurns.length ? "No conversational messages in this bounded preview." : "No unique turns after the shared head."}</p>{/if}
                  </div>
                  {#if comparisonLeftTools.length}
                    <details class="path-tools comparison-tools">
                      <summary>{comparisonLeftTools.length}{comparison.leftToolRuns > comparisonLeftTools.length ? ` of ${comparison.leftToolRuns}` : ""} branch-only tool runs</summary>
                      {#each comparisonLeftTools as tool (tool.id)}
                        <div><Icon name="activity" size={12} /><span>{humanize(tool.tool_name)}</span><small>Turn {tool.turn_index}</small></div>
                      {/each}
                    </details>
                  {/if}
                </section>

                <section class="comparison-branch candidate">
                  <header>
                    <div><span>Candidate</span><strong title={comparisonRightBranch.name}>{comparisonRightBranch.name}</strong></div>
                    {#if comparisonRightBranch.active}<b>Active</b>{/if}
                  </header>
                  <div class="branch-provenance">
                    <span>{humanize(comparisonRightBranch.origin_kind)}</span>
                    {#if branchSource(comparisonRightBranch)}<span>from {branchSource(comparisonRightBranch)}</span>{/if}
                    {#if comparisonRightBranch.origin_task_id}<span title={comparisonRightBranch.origin_task_id}>task {comparisonRightBranch.origin_task_id}</span>{/if}
                    {#if comparisonRightBranch.origin_execution_id}<span title={comparisonRightBranch.origin_execution_id}>run {comparisonRightBranch.origin_execution_id}</span>{/if}
                  </div>
                  <div class="comparison-stats">
                    <span><strong>{comparison.rightTurns.length}</strong> unique turns</span>
                    <span><strong>{comparison.rightMessageRows}</strong> message rows</span>
                    <span><strong>{comparison.rightToolRuns}</strong> tool runs</span>
                  </div>
                  {#if comparison.rightMessageRows > comparisonRightLoadedRows.length}
                    <p class="comparison-clipped">Latest bounded preview. {comparison.rightMessageRows - comparisonRightLoadedRows.length} earlier branch-only rows are not shown.</p>
                  {/if}
                  <div class="comparison-transcript">
                    {#each comparisonRightMessages as message (message.id)}
                      {@const body = messageText(message.content)}
                      {#if body}
                        <article class:user={message.role.toLowerCase() === "user"}>
                          <header><strong>{message.role.toLowerCase() === "user" ? "You" : humanize(graph?.session.agent_id ?? "assistant")}</strong><span>Turn {message.turn_index}</span></header>
                          {#if message.role.toLowerCase() === "assistant"}<div class="comparison-markdown"><Markdown source={body} /></div>{:else}<p>{body}</p>{/if}
                        </article>
                      {/if}
                    {/each}
                    {#if !comparisonRightMessages.length}<p class="path-empty">{comparison.rightTurns.length ? "No conversational messages in this bounded preview." : "No unique turns after the shared head."}</p>{/if}
                  </div>
                  {#if comparisonRightTools.length}
                    <details class="path-tools comparison-tools">
                      <summary>{comparisonRightTools.length}{comparison.rightToolRuns > comparisonRightTools.length ? ` of ${comparison.rightToolRuns}` : ""} branch-only tool runs</summary>
                      {#each comparisonRightTools as tool (tool.id)}
                        <div><Icon name="activity" size={12} /><span>{humanize(tool.tool_name)}</span><small>Turn {tool.turn_index}</small></div>
                      {/each}
                    </details>
                  {/if}
                </section>
              </div>
            {:else}
              <p class="path-empty comparison-state">Create a second durable branch to compare paths.</p>
            {/if}
          {:else}
            <p class="explore-note"><Icon name="branch" size={14} />Both operations start from Turn {selectedTurn.turn_index}. Only checkout changes the active conversation.</p>
            <section class="inspector-section">
              <span class="section-label">Create a durable path</span>
              <form class="fork-form" onsubmit={event => { event.preventDefault(); void createFork(); }}>
                <input bind:value={forkName} placeholder={`fork-turn-${selectedTurn.turn_index}`} aria-label="New branch name" />
                <label><input type="checkbox" bind:checked={activateFork} />Check out after creating</label>
                <button class="primary-action" disabled={!forkName.trim() || Boolean(action)}>{action === "fork" ? "Creating path..." : "Fork from this turn"}</button>
              </form>
            </section>

            <section class="inspector-section aside-section">
              <span class="section-label">Ask from this context</span>
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
  .graph-workspace { position: relative; display: flex; flex-direction: column; width: min(1480px, 100%); height: min(920px, 100%); overflow: hidden; border: 1px solid var(--line-strong); border-radius: 18px; background: var(--surface-raised); box-shadow: var(--shadow-lg); }
  .graph-header { display: flex; flex: 0 0 auto; align-items: center; justify-content: space-between; gap: 24px; min-height: 92px; padding: 18px 20px 17px 23px; border-bottom: 1px solid var(--line); background: color-mix(in srgb, var(--surface) 92%, transparent); }
  .eyebrow, .section-label { color: var(--faint); font-size: 10px; font-weight: 750; letter-spacing: .09em; text-transform: uppercase; }
  .graph-header h2 { margin: 1px 0 0; font-size: 20px; font-weight: 690; letter-spacing: -.035em; }
  .graph-header p { margin: 3px 0 0; color: var(--muted); font-size: 11px; }
  .graph-header-actions { display: flex; align-items: center; gap: 7px; }
  .graph-header-actions > span { margin-right: 5px; color: var(--faint); font-size: 10px; }
  .graph-header-actions button { display: grid; place-items: center; min-width: 32px; height: 32px; padding: 0 9px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface-raised); color: var(--muted); font-size: 10px; font-weight: 650; }
  .graph-header-actions button:hover, .graph-header-actions button.active { border-color: color-mix(in srgb, var(--accent) 40%, var(--line)); color: var(--accent-strong); }
  .graph-alert { display: flex; flex: 0 0 auto; align-items: center; gap: 8px; min-height: 35px; padding: 7px 22px; border-bottom: 1px solid var(--line); font-size: 10px; }
  .graph-alert.error { background: color-mix(in srgb, var(--danger) 8%, var(--surface)); color: var(--danger); }
  .graph-alert.success { background: color-mix(in srgb, var(--success) 8%, var(--surface)); color: var(--success); }
  .graph-alert span { color: var(--muted); }
  .graph-body { display: grid; grid-template-columns: minmax(0, 1fr) 450px; min-height: 0; flex: 1; }
  .graph-body.comparison-open { grid-template-columns: minmax(0, 1fr); }
  .graph-body.comparison-open .map-panel { display: none; }
  .graph-body.comparison-open .graph-inspector { padding: 22px 26px 26px; border-left: 0; }
  .map-panel { display: flex; min-width: 0; min-height: 0; flex-direction: column; background: radial-gradient(circle at 18% 5%, color-mix(in srgb, var(--accent-soft) 48%, transparent), transparent 28%), var(--surface-muted); }
  .map-toolbar { display: flex; flex: 0 0 auto; align-items: center; justify-content: space-between; min-height: 42px; padding: 8px 15px; border-bottom: 1px solid var(--line); color: var(--faint); font-size: 10px; }
  .map-toolbar > div { display: flex; align-items: center; gap: 6px; }
  .map-toolbar i { display: inline-block; width: 16px; height: 2px; margin-left: 7px; background: var(--line-strong); }
  .map-toolbar i:first-child { margin-left: 0; }
  .map-toolbar i.active-line { background: var(--accent); }
  .map-toolbar i.inspected-line { background: var(--blue); }
  .graph-scroll { min-height: 0; flex: 1; overflow: auto; scrollbar-color: var(--line-strong) transparent; }
  .graph-canvas { position: relative; min-width: 100%; min-height: 100%; background-image: radial-gradient(circle, color-mix(in srgb, var(--faint) 25%, transparent) .8px, transparent .8px); background-size: 18px 18px; }
  .graph-canvas svg { position: absolute; inset: 0; overflow: visible; pointer-events: none; }
  .graph-canvas path { fill: none; stroke: var(--line-strong); stroke-width: 2; }
  .graph-canvas path.inspected { stroke: var(--blue); stroke-dasharray: 5 4; }
  .graph-canvas path.active { stroke: var(--accent); stroke-width: 2.5; }
  .turn-node { position: absolute; z-index: 2; display: grid; place-items: center; width: 30px; height: 30px; padding: 0; border: 2px solid var(--line-strong); border-radius: 50%; background: var(--surface-raised); color: var(--muted); box-shadow: 0 2px 7px color-mix(in srgb, var(--ink) 10%, transparent); font-size: 9px; font-weight: 750; font-variant-numeric: tabular-nums; transform: translate(-50%, -50%); }
  .turn-node::after { position: absolute; right: -3px; bottom: -3px; width: 7px; height: 7px; border: 2px solid var(--surface-raised); border-radius: 50%; background: transparent; content: ""; }
  .turn-node.has-tools::after { background: var(--warning); }
  .turn-node.active-path { border-color: var(--accent); color: var(--accent-strong); }
  .turn-node.inspected-path { border-color: var(--blue); color: var(--blue); }
  .turn-node.selected { outline: 4px solid color-mix(in srgb, var(--accent) 18%, transparent); border-color: var(--accent); background: var(--accent); color: white; transform: translate(-50%, -50%) scale(1.12); }
  .branch-tag { position: absolute; z-index: 3; display: flex; align-items: center; gap: 5px; max-width: 155px; height: 25px; padding: 3px 8px; overflow: hidden; border: 1px solid var(--line); border-radius: 7px; background: color-mix(in srgb, var(--surface-raised) 95%, transparent); color: var(--muted); box-shadow: var(--shadow-sm); font-size: 9px; font-weight: 650; text-overflow: ellipsis; white-space: nowrap; }
  .branch-tag.active { border-color: color-mix(in srgb, var(--accent) 35%, var(--line)); background: var(--accent-soft); color: var(--accent-strong); }
  .graph-loading, .graph-empty, .inspector-empty { display: grid; height: 100%; place-content: center; justify-items: center; color: var(--faint); text-align: center; }
  .graph-loading i { width: 26px; height: 26px; margin-bottom: 8px; border: 2px solid var(--line); border-top-color: var(--accent); border-radius: 50%; animation: spin .8s linear infinite; }
  .graph-loading span, .graph-empty span, .inspector-empty span { margin-top: 4px; font-size: 10px; }
  .graph-empty strong, .inspector-empty strong { margin-top: 8px; color: var(--ink); font-size: 12px; }
  .graph-inspector { min-height: 0; padding: 19px; overflow: auto; border-left: 1px solid var(--line); background: var(--surface-raised); }
  .turn-heading { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; }
  .turn-heading > div > span { color: var(--faint); font-size: 10px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
  .turn-heading h3 { margin: 2px 0 0; font-size: 21px; letter-spacing: -.035em; }
  .turn-heading time { color: var(--faint); font-size: 10px; }
  .path-state { flex: 0 0 auto; margin-top: 2px; padding: 4px 7px; border: 1px solid color-mix(in srgb, var(--blue) 30%, var(--line)); border-radius: 99px; background: color-mix(in srgb, var(--blue) 7%, var(--surface)); color: var(--blue); font-size: 9px; font-weight: 680; }
  .path-state.active { border-color: color-mix(in srgb, var(--accent) 30%, var(--line)); background: var(--accent-soft); color: var(--accent-strong); }
  .turn-stats { display: grid; grid-template-columns: repeat(3, 1fr); margin: 14px 0 0; overflow: hidden; border: 1px solid var(--line); border-radius: 10px; }
  .turn-stats > div { display: grid; gap: 1px; padding: 9px 10px; border-right: 1px solid var(--line); }
  .turn-stats > div:last-child { border-right: 0; }
  .turn-stats strong { font-size: 15px; }
  .turn-stats span { color: var(--faint); font-size: 9px; text-transform: uppercase; }
  .inspector-tabs { display: grid; grid-template-columns: .8fr .85fr 1.2fr; gap: 4px; margin-top: 14px; padding: 3px; border: 1px solid var(--line); border-radius: 10px; background: var(--surface-muted); }
  .inspector-tabs button { min-height: 31px; border: 0; border-radius: 7px; background: transparent; color: var(--muted); font-size: 10px; font-weight: 680; }
  .inspector-tabs button.active { background: var(--surface-raised); color: var(--ink); box-shadow: var(--shadow-sm); }
  .inspector-tabs button:disabled { color: var(--faint); cursor: not-allowed; opacity: .55; }
  .inspector-section { display: grid; gap: 8px; margin-top: 18px; }
  .path-list { display: grid; gap: 5px; }
  .path-list > div { display: flex; align-items: center; justify-content: space-between; gap: 8px; min-height: 38px; padding: 6px 7px 6px 9px; border: 1px solid var(--line); border-radius: 9px; background: var(--surface); }
  .path-list > div.active { border-color: color-mix(in srgb, var(--accent) 30%, var(--line)); background: var(--accent-soft); }
  .path-list span { display: flex; min-width: 0; align-items: center; gap: 6px; }
  .path-list strong { overflow: hidden; font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
  .path-list small { color: var(--faint); font-size: 9px; }
  .path-list b { color: var(--accent-strong); font-size: 9px; }
  .path-list button, .promote-row button { padding: 5px 7px; border: 1px solid var(--line); border-radius: 7px; background: var(--surface-raised); color: var(--muted); font-size: 9px; font-weight: 650; }
  .context-section { padding-top: 2px; }
  .context-heading { display: flex; align-items: center; justify-content: space-between; }
  .context-heading button { display: flex; align-items: center; gap: 4px; border: 0; background: transparent; color: var(--faint); font-size: 9px; }
  .context-heading button:hover { color: var(--ink); }
  .path-loading { display: flex; align-items: center; gap: 7px; min-height: 46px; padding: 10px; border: 1px solid var(--line); border-radius: 9px; color: var(--faint); font-size: 10px; }
  .path-loading i { width: 14px; height: 14px; border: 2px solid var(--line); border-top-color: var(--blue); border-radius: 50%; animation: spin .8s linear infinite; }
  .path-error { display: grid; gap: 3px; padding: 10px; border: 1px solid color-mix(in srgb, var(--danger) 25%, var(--line)); border-radius: 9px; background: color-mix(in srgb, var(--danger) 5%, var(--surface)); color: var(--danger); font-size: 10px; }
  .path-error span { color: var(--muted); }
  .path-error button { justify-self: start; margin-top: 4px; border: 0; background: transparent; color: var(--danger); font-size: 9px; font-weight: 700; }
  .context-summary { display: flex; align-items: center; gap: 6px; }
  .context-summary span { padding: 3px 6px; border-radius: 99px; background: var(--surface-muted); color: var(--faint); font-size: 9px; }
  .path-omitted { margin: -1px 0 0; color: var(--faint); font-size: 9px; }
  .context-explainer { display: flex; align-items: flex-start; gap: 8px; padding: 9px 10px; border: 1px solid color-mix(in srgb, var(--blue) 18%, var(--line)); border-radius: 9px; background: color-mix(in srgb, var(--blue) 4%, var(--surface)); }
  .context-explainer > :global(.icon) { flex: 0 0 auto; margin-top: 1px; color: var(--blue); }
  .context-explainer > div { display: grid; gap: 1px; }
  .context-explainer strong { color: var(--ink); font-size: 10px; }
  .context-explainer span { color: var(--muted); font-size: 9px; line-height: 1.4; }
  .context-turns { display: grid; gap: 8px; max-height: 390px; padding-right: 3px; overflow: auto; scrollbar-color: var(--line-strong) transparent; }
  .context-turn { position: relative; display: grid; gap: 0; overflow: hidden; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); }
  .context-turn.current { border-color: color-mix(in srgb, var(--blue) 32%, var(--line)); box-shadow: inset 3px 0 var(--blue); }
  .context-turn > header { display: flex; align-items: center; justify-content: space-between; min-height: 28px; padding: 5px 9px; border-bottom: 1px solid var(--line); background: var(--surface-muted); color: var(--faint); font-size: 9px; font-weight: 700; }
  .context-turn > header b { color: var(--blue); font-size: 8px; }
  .context-entry { display: grid; gap: 4px; padding: 8px 9px; border-bottom: 1px solid var(--line); }
  .context-entry:last-child { border-bottom: 0; }
  .context-entry.user { background: color-mix(in srgb, var(--blue) 3%, var(--surface)); }
  .context-entry > strong { color: var(--faint); font-size: 8px; font-weight: 720; letter-spacing: .05em; text-transform: uppercase; }
  .context-entry > p, .context-entry .path-markdown { margin: 0; padding: 0; color: var(--muted); font-size: 11px; line-height: 1.55; white-space: pre-wrap; }
  .path-empty { margin: 0; padding: 18px; border: 1px dashed var(--line); border-radius: 9px; color: var(--faint); text-align: center; font-size: 10px; }
  .path-tools { overflow: hidden; border: 1px solid var(--line); border-radius: 9px; background: var(--surface); }
  .path-tools summary { padding: 8px 9px; color: var(--muted); cursor: pointer; font-size: 10px; font-weight: 650; }
  .path-tools > div { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; align-items: center; gap: 6px; padding: 7px 9px; border-top: 1px solid var(--line); color: var(--muted); font-size: 9px; }
  .path-tools > div span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .path-tools > div small { color: var(--faint); }
  .comparison-controls { display: grid; gap: 8px; margin-top: 16px; }
  .comparison-pickers { display: grid; grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr); align-items: end; gap: 7px; }
  .comparison-pickers label { display: grid; min-width: 0; gap: 4px; color: var(--faint); font-size: 9px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
  .comparison-pickers select { min-width: 0; height: 34px; padding: 0 28px 0 8px; border: 1px solid var(--line); border-radius: 8px; outline: 0; background: var(--surface); color: var(--ink); font: inherit; font-size: 10px; font-weight: 650; text-transform: none; }
  .comparison-pickers select:focus { border-color: var(--blue); box-shadow: 0 0 0 3px color-mix(in srgb, var(--blue) 10%, transparent); }
  .swap-button { height: 34px; padding: 0 9px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface-raised); color: var(--muted); font-size: 9px; font-weight: 700; }
  .swap-button:hover { border-color: color-mix(in srgb, var(--blue) 35%, var(--line)); color: var(--blue); }
  .comparison-state { margin-top: 12px; }
  .common-ancestor { display: flex; align-items: center; gap: 9px; margin-top: 12px; padding: 9px 10px; border: 1px solid color-mix(in srgb, var(--blue) 20%, var(--line)); border-radius: 10px; background: color-mix(in srgb, var(--blue) 4%, var(--surface)); }
  .common-ancestor :global(.icon) { flex: 0 0 auto; color: var(--blue); }
  .common-ancestor div { display: grid; min-width: 0; gap: 1px; }
  .common-ancestor strong { color: var(--ink); font-size: 10px; }
  .common-ancestor span { color: var(--faint); font-size: 9px; }
  .comparison-columns { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 9px; margin-top: 9px; }
  .comparison-branch { display: flex; min-width: 0; flex-direction: column; gap: 8px; padding: 10px; border: 1px solid var(--line); border-radius: 11px; background: var(--surface); }
  .comparison-branch.candidate { border-color: color-mix(in srgb, var(--blue) 23%, var(--line)); }
  .comparison-branch > header { display: flex; min-width: 0; align-items: flex-start; justify-content: space-between; gap: 8px; }
  .comparison-branch > header div { display: grid; min-width: 0; gap: 1px; }
  .comparison-branch > header span { color: var(--faint); font-size: 8px; font-weight: 750; letter-spacing: .08em; text-transform: uppercase; }
  .comparison-branch > header strong { overflow: hidden; color: var(--ink); font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
  .comparison-branch > header b { flex: 0 0 auto; padding: 3px 6px; border-radius: 99px; background: var(--accent-soft); color: var(--accent-strong); font-size: 8px; }
  .branch-provenance { display: flex; min-width: 0; flex-wrap: wrap; gap: 4px; }
  .branch-provenance span { max-width: 100%; padding: 3px 5px; overflow: hidden; border-radius: 5px; background: var(--surface-muted); color: var(--faint); font-family: var(--font-mono); font-size: 8px; text-overflow: ellipsis; white-space: nowrap; }
  .comparison-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); overflow: hidden; border: 1px solid var(--line); border-radius: 8px; }
  .comparison-stats span { display: grid; gap: 1px; padding: 6px; border-right: 1px solid var(--line); color: var(--faint); font-size: 8px; line-height: 1.25; }
  .comparison-stats span:last-child { border-right: 0; }
  .comparison-stats strong { color: var(--ink); font-size: 11px; }
  .comparison-clipped { margin: 0; padding: 6px 7px; border-radius: 7px; background: color-mix(in srgb, var(--warning) 8%, var(--surface)); color: var(--muted); font-size: 8px; line-height: 1.4; }
  .comparison-transcript { display: grid; align-content: start; gap: 7px; max-height: min(54vh, 570px); overflow: auto; scrollbar-color: var(--line-strong) transparent; }
  .comparison-transcript article { min-width: 0; overflow: hidden; border: 1px solid var(--line); border-radius: 8px; background: var(--surface-raised); }
  .comparison-transcript article.user { border-color: color-mix(in srgb, var(--blue) 18%, var(--line)); background: color-mix(in srgb, var(--blue) 3%, var(--surface)); }
  .comparison-transcript article > header { display: flex; justify-content: space-between; gap: 6px; padding: 5px 7px; border-bottom: 1px solid var(--line); color: var(--faint); font-size: 8px; }
  .comparison-transcript article > header strong { color: var(--muted); font-size: 8px; }
  .comparison-transcript article > p, .comparison-markdown { max-height: 116px; margin: 0; padding: 7px 8px; overflow: auto; color: var(--muted); font-size: 9px; line-height: 1.48; white-space: pre-wrap; }
  .comparison-tools { margin-top: auto; }
  .comparison-tools summary { font-size: 9px; }
  .explore-note { display: flex; align-items: flex-start; gap: 7px; margin: 13px 0 0; padding: 9px 10px; border: 1px solid color-mix(in srgb, var(--blue) 18%, var(--line)); border-radius: 9px; background: color-mix(in srgb, var(--blue) 4%, var(--surface)); color: var(--muted); font-size: 10px; line-height: 1.45; }
  .explore-note :global(.icon) { flex: 0 0 auto; margin-top: 1px; color: var(--blue); }
  .fork-form { display: grid; gap: 7px; }
  .fork-form > input, .aside-section textarea, .promote-row input { width: 100%; border: 1px solid var(--line); border-radius: 8px; outline: 0; background: var(--surface); color: var(--ink); font-size: 11px; }
  .fork-form > input, .promote-row input { height: 34px; padding: 7px 9px; }
  .aside-section textarea { min-height: 72px; padding: 9px; resize: vertical; line-height: 1.5; }
  .fork-form input:focus, .aside-section textarea:focus, .promote-row input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 10%, transparent); }
  .fork-form label { display: flex; align-items: center; gap: 6px; color: var(--muted); font-size: 10px; }
  .primary-action { min-height: 35px; border: 1px solid var(--ink); border-radius: 8px; background: var(--ink); color: var(--surface-raised); font-size: 10px; font-weight: 680; }
  .primary-action:hover { background: color-mix(in srgb, var(--ink) 88%, var(--accent)); }
  .mode-picker { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .mode-picker button { display: grid; gap: 2px; min-height: 58px; padding: 8px; border: 1px solid var(--line); border-radius: 9px; background: var(--surface); color: var(--muted); text-align: left; }
  .mode-picker button.active { border-color: color-mix(in srgb, var(--accent) 45%, var(--line)); background: var(--accent-soft); color: var(--accent-strong); }
  .mode-picker strong { font-size: 10px; }
  .mode-picker span { color: var(--faint); font-size: 9px; line-height: 1.4; }
  .sidestep-result { margin-top: 17px; overflow: hidden; border: 1px solid color-mix(in srgb, var(--accent) 28%, var(--line)); border-radius: 11px; background: var(--surface); }
  .sidestep-result > header { display: flex; align-items: center; justify-content: space-between; padding: 8px 10px; border-bottom: 1px solid var(--line); background: var(--accent-soft); color: var(--accent-strong); font-size: 10px; font-weight: 700; }
  .sidestep-result > header b { font-size: 9px; }
  .result-body { max-height: 230px; padding: 11px; overflow: auto; font-size: 11px; }
  .promote-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 6px; padding: 8px; border-top: 1px solid var(--line); }
  .promoted-note { display: flex; align-items: center; gap: 6px; margin: 0; padding: 9px 10px; border-top: 1px solid var(--line); color: var(--success); font-size: 10px; font-weight: 650; }
  @keyframes spin { to { transform: rotate(360deg); } }
  @media (max-width: 900px) {
    .graph-overlay { padding: 0; }
    .graph-workspace { width: 100%; height: 100%; border: 0; border-radius: 0; }
    .graph-header p, .graph-header-actions > span { display: none; }
    .graph-body, .graph-body.comparison-open { grid-template-columns: 1fr; grid-template-rows: minmax(280px, 48%) minmax(0, 1fr); }
    .graph-inspector { border-top: 1px solid var(--line); border-left: 0; }
    .comparison-columns { grid-template-columns: 1fr; }
  }
</style>
