<script lang="ts">
  export let name: "spark" | "chat" | "grid" | "database" | "branch" | "route" | "refresh" | "send" | "chevron" | "menu" | "close" | "plus" | "activity" | "copy" | "edit" = "spark";
  export let size = 18;
</script>

<svg
  class="icon"
  width={size}
  height={size}
  viewBox="0 0 24 24"
  fill="none"
  stroke="currentColor"
  stroke-width="1.8"
  stroke-linecap="round"
  stroke-linejoin="round"
  aria-hidden="true"
>
  {#if name === "spark"}
    <path d="m12 3 1.5 4.4a4.8 4.8 0 0 0 3.1 3.1L21 12l-4.4 1.5a4.8 4.8 0 0 0-3.1 3.1L12 21l-1.5-4.4a4.8 4.8 0 0 0-3.1-3.1L3 12l4.4-1.5a4.8 4.8 0 0 0 3.1-3.1L12 3Z" />
  {:else if name === "chat"}
    <path d="M20 15a4 4 0 0 1-4 4H8l-5 2 1.4-4.2A7 7 0 0 1 3 12V8a4 4 0 0 1 4-4h9a4 4 0 0 1 4 4v7Z" />
  {:else if name === "grid"}
    <rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" />
  {:else if name === "database"}
    <ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6" />
  {:else if name === "branch"}
    <circle cx="6" cy="5" r="2" /><circle cx="18" cy="7" r="2" /><circle cx="6" cy="19" r="2" /><path d="M6 7v10M8 10h5a5 5 0 0 0 5-5" />
  {:else if name === "route"}
    <circle cx="5" cy="6" r="2" /><circle cx="19" cy="18" r="2" /><path d="M7 6h5a3 3 0 0 1 3 3v0a3 3 0 0 1-3 3H9a3 3 0 0 0-3 3v1M17 18h-4" />
  {:else if name === "refresh"}
    <path d="M20 7v5h-5M4 17v-5h5" /><path d="M6.1 8a7 7 0 0 1 11.3-2.2L20 8M4 16l2.6 2.2A7 7 0 0 0 17.9 16" />
  {:else if name === "send"}
    <path d="m22 2-7 20-4-9-9-4 20-7Z" /><path d="M22 2 11 13" />
  {:else if name === "chevron"}
    <path d="m9 18 6-6-6-6" />
  {:else if name === "menu"}
    <path d="M4 7h16M4 12h16M4 17h16" />
  {:else if name === "close"}
    <path d="m6 6 12 12M18 6 6 18" />
  {:else if name === "plus"}
    <path d="M12 5v14M5 12h14" />
  {:else if name === "activity"}
    <path d="M3 12h4l2.5-7 5 14 2.5-7h4" />
  {:else if name === "copy"}
    <rect x="8" y="8" width="11" height="11" rx="2" /><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" />
  {:else if name === "edit"}
    <path d="M12 20h9" /><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L8 18l-4 1 1-4Z" />
  {/if}
</svg>
