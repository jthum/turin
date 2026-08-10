<script lang="ts">
  import { renderMarkdown } from "../lib/markdown";

  export let source: string;

  $: html = renderMarkdown(source);
</script>

<div class="markdown-body">{@html html}</div>
