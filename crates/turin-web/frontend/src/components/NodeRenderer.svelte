<script lang="ts">
  import type { TurinClient } from "../lib/TurinClient";
  import type { JsonValue, UiApp, UiNode, WorkItem } from "../lib/types";
  import DataSurface from "./DataSurface.svelte";
  import FormSurface from "./FormSurface.svelte";

  export let client: TurinClient;
  export let app: UiApp;
  export let node: UiNode;
  export let refreshToken = 0;
  export let running = false;
  export let onAction: (action: string, params: JsonValue, confirm: boolean) => void;

  function itemAction(item: WorkItem) {
    if (item.action) onAction(item.action.name, item.action.params ?? {}, true);
  }

  function formSubmit(action: string, params: Record<string, JsonValue>) {
    onAction(action, params, false);
  }
</script>

{#if node.kind === "section"}
  <section class="node-section" id={node.id}>
    <header><h2>{node.title}</h2></header>
    <div class="node-stack">
      {#each node.nodes ?? [] as child, index (`${child.id ?? child.kind}-${index}`)}
        <svelte:self {client} {app} node={child} {refreshToken} {running} {onAction} />
      {/each}
    </div>
  </section>
{:else if node.kind === "text"}
  <p class="prose-node" id={node.id}>{node.text}</p>
{:else if node.kind === "action"}
  <button class="primary-button action-node" id={node.id} disabled={running} onclick={() => onAction(node.action!, node.params ?? {}, node.confirm ?? false)}>
    {running ? "Running..." : node.label}
  </button>
{:else if node.kind === "form"}
  <FormSurface {node} {running} onSubmit={formSubmit} />
{:else}
  <DataSurface {client} {node} {refreshToken} onItemAction={itemAction} />
{/if}
