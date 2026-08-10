<script lang="ts">
  import type { UiApp, UiMenuItem } from "../lib/types";
  import Icon from "./Icon.svelte";

  export let app: UiApp;
  export let items: UiMenuItem[];
  export let selectedScreenId: string;
  export let depth = 0;
  export let onSelect: (screenId: string) => void;

  function target(item: UiMenuItem): string | null {
    if (app.screens[item.opens]) return item.opens;
    return Object.values(app.screens).find(screen => screen.title === item.opens)?.id ?? null;
  }
</script>

<div class:sub-menu={depth > 0} class="screen-menu">
  {#each items as item (item.id ?? `${item.label}-${item.opens}`)}
    {@const screenId = target(item)}
    {#if screenId}
      <button class:active={selectedScreenId === screenId} onclick={() => onSelect(screenId)}>
        <span>{item.label}</span>
        {#if item.badge || app.badges[item.badge ?? item.opens]}
          <i>{item.badge ?? app.badges[item.opens]?.label ?? app.badges[item.opens]?.count}</i>
        {/if}
        <Icon name="chevron" size={14} />
      </button>
    {:else}
      <div class="menu-heading">{item.label}</div>
    {/if}
    {#if item.items?.length}
      <svelte:self {app} items={item.items} {selectedScreenId} depth={depth + 1} {onSelect} />
    {/if}
  {/each}
</div>
