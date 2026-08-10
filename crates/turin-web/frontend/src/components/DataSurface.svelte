<script lang="ts">
  import type { TurinClient } from "../lib/TurinClient";
  import type { UiNode, WorkItem } from "../lib/types";
  import { displayValue, humanize } from "../lib/format";

  export let client: TurinClient;
  export let node: UiNode;
  export let refreshToken = 0;
  export let onItemAction: (item: WorkItem) => void;

  let items: WorkItem[] = [];
  let loading = true;
  let error = "";
  let loadedKey = "";
  let selected = 0;

  $: requestKey = JSON.stringify([node.source, node.where, node.limit, refreshToken]);
  $: if (node.source && requestKey !== loadedKey) load(requestKey);
  $: fields = fieldNames(node, items);
  $: groups = groupCounts(items, groupField(node));
  $: selectedItem = items[selected];

  async function load(key: string) {
    loadedKey = key;
    loading = true;
    error = "";
    try {
      const response = await client.loadList({
        source: node.source!,
        ...(node.where ? { where: node.where } : {}),
        ...(node.limit ? { limit: node.limit } : {}),
      });
      if (loadedKey !== key) return;
      items = response.list.items;
      selected = Math.min(selected, Math.max(0, items.length - 1));
    } catch (reason) {
      if (loadedKey !== key) return;
      error = reason instanceof Error ? reason.message : String(reason);
    } finally {
      if (loadedKey === key) loading = false;
    }
  }

  function fieldNames(current: UiNode, rows: WorkItem[]): string[] {
    const declared = current.fields?.filter((field): field is string => typeof field === "string") ?? [];
    if (declared.length) return declared.slice(0, 6);
    const first = rows[0];
    if (!first) return ["kind", "status", "priority"];
    return ["kind", "status", "priority", "public_id"].filter(field => first[field] !== undefined);
  }

  function groupField(current: UiNode): string {
    if (current.intent === "kind_breakdown") return "kind";
    if (current.intent === "priority_breakdown") return "priority";
    return "status";
  }

  function groupCounts(rows: WorkItem[], field: string): Array<{ label: string; count: number }> {
    const counts = new Map<string, number>();
    for (const row of rows) {
      const label = displayValue(row[field]);
      counts.set(label, (counts.get(label) ?? 0) + 1);
    }
    return [...counts.entries()].map(([label, count]) => ({ label, count })).sort((a, b) => b.count - a.count);
  }
</script>

<section class:data-kind={node.kind} class="data-surface surface-card">
  <header class="surface-header">
    <div><span class="surface-kicker">{node.kind}</span><h2>{node.title}</h2></div>
    {#if !loading && !error}<span class="row-count">{items.length} {items.length === 1 ? "item" : "items"}</span>{/if}
  </header>

  {#if loading}
    <div class="surface-loading"><i></i><i></i><i></i></div>
  {:else if error}
    <div class="surface-error"><strong>Data could not be loaded</strong><span>{error}</span><button onclick={() => load(requestKey)}>Try again</button></div>
  {:else if !items.length}
    <div class="empty-surface"><strong>No items yet</strong><span>This view will update when its source has data.</span></div>
  {:else if node.kind === "chart" || node.kind === "report"}
    <div class="report-grid">
      <div class="report-total"><span>Total</span><strong>{items.length}</strong><small>{node.prompt ?? `Grouped by ${humanize(groupField(node))}`}</small></div>
      <div class="bar-chart">
        {#each groups as group (group.label)}
          <div class="bar-row">
            <span>{humanize(group.label)}</span>
            <div class="bar-track"><i style={`width: ${Math.max(7, (group.count / items.length) * 100)}%`}></i></div>
            <strong>{group.count}</strong>
          </div>
        {/each}
      </div>
    </div>
  {:else if node.kind === "detail"}
    <div class="detail-layout">
      <div class="detail-list">
        {#each items.slice(0, 10) as item, index (item.public_id ?? item.id)}
          <button class:active={selected === index} onclick={() => selected = index}>
            <strong>{displayValue(item.kind ?? item.public_id ?? item.id)}</strong>
            <span>{displayValue(item.status)}</span>
          </button>
        {/each}
      </div>
      {#if selectedItem}
        <dl class="detail-values">
          {#each Object.entries(selectedItem).filter(([key]) => !["payload", "metadata", "action"].includes(key)).slice(0, 10) as [key, value]}
            <div><dt>{humanize(key)}</dt><dd>{displayValue(value)}</dd></div>
          {/each}
        </dl>
      {/if}
    </div>
  {:else if node.kind === "activity"}
    <div class="activity-list">
      {#each items.slice(0, node.limit ?? 12) as item (item.public_id ?? item.id)}
        <div class="activity-row"><i></i><div><strong>{humanize(displayValue(item.kind ?? "Update"))}</strong><span>{displayValue(item.status)} · priority {displayValue(item.priority)}</span></div></div>
      {/each}
    </div>
  {:else if node.as === "cards" || node.intent === "cards"}
    <div class="card-grid">
      {#each items as item (item.public_id ?? item.id)}
        <article class="item-card">
          <span class="item-status">{displayValue(item.status)}</span>
          <h3>{humanize(displayValue(item.kind ?? item.public_id ?? item.id))}</h3>
          <p>{displayValue(item.payload)}</p>
          {#if item.action}<button class="secondary-button" onclick={() => onItemAction(item)}>{humanize(item.action.name)}</button>{/if}
        </article>
      {/each}
    </div>
  {:else}
    <div class="table-scroll">
      <table>
        <thead><tr>{#each fields as field}<th>{humanize(field)}</th>{/each}<th><span class="sr-only">Actions</span></th></tr></thead>
        <tbody>
          {#each items as item (item.public_id ?? item.id)}
            <tr>
              {#each fields as field}<td><span class:status-cell={field === "status"}>{displayValue(item[field])}</span></td>{/each}
              <td>{#if item.action}<button class="table-action" onclick={() => onItemAction(item)}>Run</button>{/if}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</section>
