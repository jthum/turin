<script lang="ts">
	import { MessageSquare, PanelLeft, Plus, Search } from '@lucide/svelte';
	import { Button } from '#lib/components/ui/button/index.js';
	import type { Agent, Bootstrap, Session } from '#lib/api/contracts.js';

	let {
		bootstrap,
		agents,
		sessions,
		selectedId,
		loading,
		search = $bindable(),
		newAgentId = $bindable(),
		onCreate,
		onSelect
	}: {
		bootstrap: Bootstrap | null;
		agents: Agent[];
		sessions: Session[];
		selectedId: string | null;
		loading: boolean;
		search: string;
		newAgentId: string;
		onCreate: () => void;
		onSelect: (session: Session) => void;
	} = $props();

	let filtered = $derived(
		sessions.filter((session) => session.title.toLowerCase().includes(search.toLowerCase()))
	);
</script>

<aside class="session-rail">
	<div class="rail-brand">
		<a href="/" class="brand-lockup" aria-label="Turin home"><span class="brand-mark">T</span><strong>Turin</strong></a>
		<Button variant="ghost" size="icon-sm" aria-label="Collapse sidebar"><PanelLeft /></Button>
	</div>
	<div class="new-session">
		<Button class="new-button" onclick={onCreate} disabled={!newAgentId}><Plus /> New conversation</Button>
		<select bind:value={newAgentId} aria-label="Agent for new conversation">
			{#each agents as agent}<option value={agent.id}>{agent.name} · {agent.model}</option>{/each}
		</select>
	</div>
	<label class="session-search"><Search /><input bind:value={search} placeholder="Search conversations" /></label>
	<div class="session-list" aria-label="Conversations">
		<span class="section-label">Recent</span>
		{#if loading}
			{#each Array(5) as _}<div class="session-skeleton"></div>{/each}
		{:else if filtered.length === 0}
			<p class="rail-empty">No conversations found.</p>
		{:else}
			{#each filtered as session (session.id)}
				<button class="session-row" class:active={selectedId === session.id} onclick={() => onSelect(session)}>
					<MessageSquare /><span><strong>{session.title}</strong><small>{session.message_count?.toLocaleString() ?? '—'} messages</small></span>
				</button>
			{/each}
		{/if}
	</div>
	<div class="rail-footer">
		<span class:online={bootstrap?.runtime.ready} class="runtime-dot"></span>
		<span><strong>{bootstrap?.runtime.ready ? 'Runtime ready' : 'Runtime unavailable'}</strong><small>{bootstrap?.runtime.connection_kind ?? 'Connecting'}</small></span>
	</div>
</aside>

<style>
	.session-rail { display: flex; min-width: 0; height: 100%; flex-direction: column; border-right: 1px solid #e5e5e1; background: #f2f2ef; }
	.rail-brand { display: flex; height: 68px; align-items: center; justify-content: space-between; padding: 0 18px; }
	.brand-lockup { display: flex; align-items: center; gap: 10px; color: inherit; text-decoration: none; font-size: 15px; }
	.brand-mark { display: grid; width: 30px; height: 30px; place-items: center; border-radius: 9px; background: #191918; color: white; font-size: 13px; font-weight: 750; box-shadow: 0 1px 1px #0002; }
	.rail-brand :global(svg) { width: 17px; height: 17px; }
	.new-session { display: grid; gap: 7px; padding: 6px 14px 12px; }
	:global(.new-button) { width: 100%; justify-content: flex-start; box-shadow: 0 1px 1px #0001; }
	.new-session select { width: 100%; border: 0; background: transparent; color: #74746f; padding: 3px 8px; font-size: 11px; outline: none; }
	.session-search { display: flex; height: 34px; align-items: center; gap: 8px; margin: 0 14px 14px; padding: 0 10px; border: 1px solid #dfdfda; border-radius: 9px; background: #fafaf8; color: #999992; }
	.session-search:focus-within { border-color: #b9b9b2; box-shadow: 0 0 0 3px #00000008; }
	.session-search :global(svg) { width: 14px; }
	.session-search input { width: 100%; border: 0; outline: 0; background: transparent; color: #2b2b29; font-size: 12px; }
	.session-list { min-height: 0; flex: 1; overflow-y: auto; padding: 0 9px; }
	.section-label { display: block; padding: 6px 11px 7px; color: #999992; font-size: 10px; font-weight: 650; letter-spacing: .08em; text-transform: uppercase; }
	.session-row { display: flex; width: 100%; align-items: flex-start; gap: 9px; border: 0; border-radius: 9px; background: transparent; padding: 9px 10px; color: #60605b; text-align: left; cursor: pointer; }
	.session-row:hover { background: #e9e9e5; color: #242422; }
	.session-row.active { background: white; color: #191918; box-shadow: 0 1px 2px #0000000b, inset 0 0 0 1px #e2e2dd; }
	.session-row :global(svg) { width: 14px; height: 14px; margin-top: 2px; flex: none; }
	.session-row span { min-width: 0; display: grid; gap: 3px; }
	.session-row strong { overflow: hidden; font-size: 12px; font-weight: 560; text-overflow: ellipsis; white-space: nowrap; }
	.session-row small { color: #9a9a94; font-size: 10px; }
	.session-skeleton { height: 49px; margin: 4px; border-radius: 9px; background: linear-gradient(100deg, #e7e7e3 35%, #f1f1ed 50%, #e7e7e3 65%); background-size: 300% 100%; animation: shimmer 1.5s infinite; }
	.rail-empty { padding: 16px 11px; color: #8c8c86; font-size: 12px; }
	.rail-footer { display: flex; align-items: center; gap: 10px; margin: 8px 14px 14px; padding: 11px; border-top: 1px solid #dfdfda; }
	.runtime-dot { width: 7px; height: 7px; border-radius: 50%; background: #c9c9c4; }
	.runtime-dot.online { background: #22a565; box-shadow: 0 0 0 3px #22a56518; }
	.rail-footer > span:last-child { display: grid; gap: 1px; }
	.rail-footer strong { font-size: 11px; font-weight: 550; }
	.rail-footer small { color: #969690; font-size: 10px; text-transform: capitalize; }
	@keyframes shimmer { from { background-position: 100% 0; } to { background-position: 0 0; } }
</style>
