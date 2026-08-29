<script lang="ts">
	import './layout.css';
	import favicon from '#lib/assets/favicon.svg';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>Turin</title>
	<meta
		name="description"
		content="A focused workspace for durable, governed agentic applications."
	/>
</svelte:head>

{@render children()}
