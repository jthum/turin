<script lang="ts">
	import { onMount } from 'svelte';
	import { Bot, CircleDot, MessageSquareText, RefreshCw } from '@lucide/svelte';
	import { Badge } from '#lib/components/ui/badge/index.js';
	import { Button } from '#lib/components/ui/button/index.js';
	import { Separator } from '#lib/components/ui/separator/index.js';
	import { Skeleton } from '#lib/components/ui/skeleton/index.js';
	import { loadBootstrap, type Bootstrap } from '#lib/api/bootstrap.js';

	let bootstrap = $state<Bootstrap | null>(null);
	let error = $state<string | null>(null);
	let loading = $state(true);

	async function refresh(signal?: AbortSignal) {
		loading = true;
		error = null;
		try {
			bootstrap = await loadBootstrap(signal);
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Turin could not be reached.';
		} finally {
			loading = false;
		}
	}

	onMount(() => {
		const controller = new AbortController();
		void refresh(controller.signal);
		return () => controller.abort();
	});
</script>

<div class="app-shell">
	<aside class="sidebar">
		<div class="brand">
			<div class="brand-mark">T</div>
			<span>Turin</span>
		</div>

		<nav aria-label="Primary navigation">
			<a class="nav-item active" href="/" aria-current="page">
				<MessageSquareText />
				<span>Workspace</span>
			</a>
		</nav>

		<div class="sidebar-footer">
			<Separator />
			<div class="connection-summary">
				<span class:online={bootstrap?.runtime.ready} class="status-dot"></span>
				<div>
					<strong>{bootstrap?.runtime.ready ? 'Runtime ready' : 'Runtime unavailable'}</strong>
					<span>{bootstrap?.runtime.connection_kind ?? 'Checking connection'}</span>
				</div>
			</div>
		</div>
	</aside>

	<main>
		<header>
			<div class="mobile-brand">
				<div class="brand-mark">T</div>
				<span>Turin</span>
			</div>
			<div class="header-title">
				<span>Workspace</span>
				{#if bootstrap}
					<Badge variant={bootstrap.runtime.ready ? 'secondary' : 'destructive'}>
						{bootstrap.runtime.connection_kind}
					</Badge>
				{/if}
			</div>
			<Button variant="ghost" size="icon" onclick={() => refresh()} aria-label="Refresh runtime status">
				<RefreshCw class={loading ? 'spin' : undefined} />
			</Button>
		</header>

		<section class="stage">
			<div class="welcome">
				<div class="welcome-icon"><Bot /></div>
				<h1>Your Turin workspace</h1>
				<p>
					A focused place for durable conversations, agent activity, and the work that grows
					around them.
				</p>

				{#if loading && !bootstrap}
					<div class="status-card loading-card" aria-label="Connecting to Turin">
						<Skeleton class="h-4 w-28" />
						<Skeleton class="h-3 w-52" />
					</div>
				{:else if error}
					<div class="status-card error-card">
						<div>
							<strong>Connection unavailable</strong>
							<span>{error}</span>
						</div>
						<Button variant="outline" size="sm" onclick={() => refresh()}>Try again</Button>
					</div>
				{:else if bootstrap}
					<div class="status-card">
						<CircleDot class="connected-icon" />
						<div>
							<strong>Connected to Turin {bootstrap.runtime.version}</strong>
							<span>
								{bootstrap.runtime.agent_count} configured
								{bootstrap.runtime.agent_count === 1 ? 'agent' : 'agents'}
								{#if bootstrap.runtime.issue_count > 0}
									· {bootstrap.runtime.issue_count} runtime
									{bootstrap.runtime.issue_count === 1 ? 'issue' : 'issues'}
								{/if}
							</span>
						</div>
					</div>
				{/if}
			</div>
		</section>
	</main>
</div>

<style>
	:global(body) {
		margin: 0;
		min-width: 320px;
		min-height: 100vh;
		overflow: hidden;
	}

	:global(button),
	:global(a) {
		-webkit-tap-highlight-color: transparent;
	}

	.app-shell {
		display: grid;
		grid-template-columns: 248px minmax(0, 1fr);
		min-height: 100vh;
		background:
			radial-gradient(circle at 65% 32%, color-mix(in oklch, var(--accent) 72%, transparent), transparent 34rem),
			var(--background);
	}

	.sidebar {
		display: flex;
		flex-direction: column;
		padding: 22px 16px 18px;
		border-right: 1px solid var(--border);
		background: color-mix(in oklch, var(--sidebar) 94%, transparent);
		backdrop-filter: blur(18px);
	}

	.brand,
	.mobile-brand {
		display: flex;
		align-items: center;
		gap: 10px;
		font-size: 15px;
		font-weight: 680;
		letter-spacing: -0.02em;
	}

	.brand {
		padding: 0 8px 24px;
	}

	.brand-mark {
		display: grid;
		width: 29px;
		height: 29px;
		place-items: center;
		border-radius: 9px;
		background: var(--foreground);
		color: var(--background);
		font-size: 13px;
		font-weight: 760;
	}

	nav {
		flex: 1;
	}

	.nav-item {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 9px 11px;
		border-radius: 10px;
		color: var(--muted-foreground);
		font-size: 13px;
		font-weight: 520;
		text-decoration: none;
	}

	.nav-item :global(svg) {
		width: 16px;
		height: 16px;
	}

	.nav-item.active {
		background: var(--accent);
		color: var(--accent-foreground);
	}

	.sidebar-footer {
		display: grid;
		gap: 16px;
	}

	.connection-summary {
		display: grid;
		grid-template-columns: 8px minmax(0, 1fr);
		align-items: center;
		gap: 10px;
		padding: 0 8px;
	}

	.connection-summary div {
		display: grid;
		gap: 2px;
		min-width: 0;
	}

	.connection-summary strong {
		font-size: 12px;
		font-weight: 600;
	}

	.connection-summary span:not(.status-dot) {
		color: var(--muted-foreground);
		font-size: 11px;
		text-transform: capitalize;
	}

	.status-dot {
		width: 7px;
		height: 7px;
		border-radius: 999px;
		background: var(--muted-foreground);
	}

	.status-dot.online {
		background: oklch(0.68 0.17 155);
		box-shadow: 0 0 0 3px oklch(0.68 0.17 155 / 14%);
	}

	main {
		display: grid;
		grid-template-rows: 64px minmax(0, 1fr);
		min-width: 0;
	}

	header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0 24px;
		border-bottom: 1px solid color-mix(in oklch, var(--border) 75%, transparent);
		background: color-mix(in oklch, var(--background) 78%, transparent);
		backdrop-filter: blur(18px);
	}

	.header-title {
		display: flex;
		align-items: center;
		gap: 10px;
		font-size: 13px;
		font-weight: 610;
	}

	.mobile-brand {
		display: none;
	}

	.stage {
		display: grid;
		place-items: center;
		padding: 32px;
	}

	.welcome {
		display: grid;
		justify-items: center;
		width: min(100%, 520px);
		text-align: center;
	}

	.welcome-icon {
		display: grid;
		width: 50px;
		height: 50px;
		margin-bottom: 22px;
		place-items: center;
		border: 1px solid var(--border);
		border-radius: 16px;
		background: var(--card);
		box-shadow: 0 14px 36px oklch(0 0 0 / 7%);
	}

	.welcome-icon :global(svg) {
		width: 22px;
		height: 22px;
	}

	h1 {
		margin: 0;
		font-size: clamp(28px, 4vw, 38px);
		font-weight: 660;
		letter-spacing: -0.045em;
		line-height: 1.05;
	}

	.welcome > p {
		max-width: 440px;
		margin: 14px 0 28px;
		color: var(--muted-foreground);
		font-size: 14px;
		line-height: 1.65;
	}

	.status-card {
		display: flex;
		align-items: center;
		gap: 12px;
		width: min(100%, 390px);
		padding: 14px 15px;
		border: 1px solid var(--border);
		border-radius: 14px;
		background: color-mix(in oklch, var(--card) 92%, transparent);
		box-shadow: 0 12px 32px oklch(0 0 0 / 4%);
		text-align: left;
	}

	.status-card div {
		display: grid;
		flex: 1;
		gap: 3px;
	}

	.status-card strong {
		font-size: 13px;
		font-weight: 610;
	}

	.status-card span {
		color: var(--muted-foreground);
		font-size: 12px;
	}

	.loading-card {
		display: grid;
		gap: 8px;
	}

	.error-card {
		border-color: color-mix(in oklch, var(--destructive) 28%, var(--border));
	}

	:global(.connected-icon) {
		width: 18px;
		height: 18px;
		color: oklch(0.58 0.17 155);
	}

	:global(.spin) {
		animation: spin 0.8s linear infinite;
	}

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}

	@media (max-width: 760px) {
		.app-shell {
			display: block;
		}

		.sidebar {
			display: none;
		}

		main {
			min-height: 100vh;
		}

		.mobile-brand {
			display: flex;
		}

		.header-title > span {
			display: none;
		}

		header {
			padding: 0 16px;
		}

		.stage {
			padding: 24px;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		:global(.spin) {
			animation: none;
		}
	}
</style>
