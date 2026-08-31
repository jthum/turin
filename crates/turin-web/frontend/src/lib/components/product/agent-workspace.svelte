<script lang="ts">
	import { Bot, CircleAlert, RefreshCw, Search, ShieldCheck } from '@lucide/svelte';
	import type { Agent, AgentControlAction, AgentDetail } from '#lib/api/contracts.js';
	import * as AlertDialog from '#lib/components/ui/alert-dialog/index.js';
	import { Badge } from '#lib/components/ui/badge/index.js';
	import { Button } from '#lib/components/ui/button/index.js';
	import * as Card from '#lib/components/ui/card/index.js';
	import { Input } from '#lib/components/ui/input/index.js';
	import * as Sheet from '#lib/components/ui/sheet/index.js';
	import * as Table from '#lib/components/ui/table/index.js';

	let {
		agents, selectedAgent, loading, mutating, onOpen, onClose, onControl, onOpenSession
	}: {
		agents: Agent[];
		selectedAgent: AgentDetail | null;
		loading: boolean;
		mutating: boolean;
		onOpen: (agent: Agent) => void;
		onClose: () => void;
		onControl: (action: AgentControlAction) => void;
		onOpenSession: (agent: AgentDetail) => void;
	} = $props();

	let query = $state('');
	let availabilityOpen = $state(false);
	let filtered = $derived(agents.filter((agent) => {
		const needle = query.trim().toLowerCase();
		return !needle || `${agent.name} ${agent.id} ${agent.provider} ${agent.model} ${agent.harness_id}`.toLowerCase().includes(needle);
	}));

	function workload(agent: Agent) {
		const total = agent.active_tasks + agent.queued_tasks + agent.awaiting_results;
		if (total === 0) return 'Idle';
		const parts = [];
		if (agent.active_tasks) parts.push(`${agent.active_tasks} active`);
		if (agent.queued_tasks) parts.push(`${agent.queued_tasks} queued`);
		if (agent.awaiting_results) parts.push(`${agent.awaiting_results} awaiting`);
		return parts.join(' · ');
	}
</script>

<div class="h-full overflow-y-auto bg-muted/20">
	<div class="mx-auto flex w-full max-w-[92rem] flex-col gap-6 px-5 py-8 lg:px-8 lg:py-10">
		<header>
			<h1 class="font-heading text-3xl font-semibold tracking-tight">Agents</h1>
			<p class="mt-2 max-w-2xl text-sm text-muted-foreground">Inspect who can take work, understand current runtime pressure, and deliberately control availability.</p>
		</header>

		<Card.Root class="overflow-hidden shadow-none">
			<Card.Header class="gap-4 border-b bg-background sm:flex-row sm:items-center sm:justify-between">
				<div><Card.Title>Runtime roster</Card.Title><Card.Description>Configured agents for the selected harness.</Card.Description></div>
				<div class="relative w-full sm:w-72"><Search class="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" /><Input bind:value={query} class="pl-9" placeholder="Search agents…" /></div>
			</Card.Header>
			<Card.Content class="p-0">
				{#if loading}
					<div class="grid min-h-64 place-items-center text-sm text-muted-foreground">Loading agents…</div>
				{:else if filtered.length === 0}
					<div class="grid min-h-64 place-items-center px-6 text-center"><div><Bot class="mx-auto mb-3 size-6 text-muted-foreground" /><p class="font-medium">No agents found</p><p class="mt-1 text-sm text-muted-foreground">Try another search or select a different harness.</p></div></div>
				{:else}
					<Table.Root>
						<Table.Header><Table.Row><Table.Head>Agent</Table.Head><Table.Head>Model</Table.Head><Table.Head>Workload</Table.Head><Table.Head class="w-28">State</Table.Head></Table.Row></Table.Header>
						<Table.Body>
							{#each filtered as agent}
								<Table.Row class="cursor-pointer" onclick={() => onOpen(agent)}>
									<Table.Cell><div class="flex items-center gap-3"><span class="grid size-9 shrink-0 place-items-center rounded-xl bg-primary/10 text-primary"><Bot class="size-4" /></span><span><span class="block font-medium">{agent.name}</span><span class="block text-xs text-muted-foreground">{agent.id}</span></span></div></Table.Cell>
									<Table.Cell><span class="block text-sm">{agent.model}</span><span class="block text-xs text-muted-foreground">{agent.provider}</span></Table.Cell>
									<Table.Cell><span class:font-medium={agent.active_tasks > 0} class="text-sm">{workload(agent)}</span></Table.Cell>
									<Table.Cell>{#if !agent.enabled}<Badge variant="outline">Disabled</Badge>{:else if agent.running}<Badge>Running</Badge>{:else}<Badge variant="secondary">Ready</Badge>{/if}</Table.Cell>
								</Table.Row>
							{/each}
						</Table.Body>
					</Table.Root>
				{/if}
			</Card.Content>
		</Card.Root>
	</div>
</div>

<Sheet.Root open={selectedAgent !== null} onOpenChange={(open) => !open && onClose()}>
	<Sheet.Content class="w-full overflow-y-auto sm:max-w-xl">
		{#if selectedAgent}
			<Sheet.Header class="border-b pb-5">
				<div class="flex items-start gap-3"><span class="grid size-10 shrink-0 place-items-center rounded-xl bg-primary/10 text-primary"><Bot class="size-5" /></span><div class="min-w-0"><Sheet.Title>{selectedAgent.name}</Sheet.Title><Sheet.Description>{selectedAgent.provider} / {selectedAgent.model}</Sheet.Description></div></div>
			</Sheet.Header>

			<div class="space-y-7 py-6">
				<div class="grid grid-cols-3 gap-2">
					<div class="rounded-xl border bg-muted/20 p-3"><span class="text-xs text-muted-foreground">Active</span><strong class="mt-1 block text-xl">{selectedAgent.active_tasks}</strong></div>
					<div class="rounded-xl border bg-muted/20 p-3"><span class="text-xs text-muted-foreground">Queued</span><strong class="mt-1 block text-xl">{selectedAgent.queued_tasks}</strong></div>
					<div class="rounded-xl border bg-muted/20 p-3"><span class="text-xs text-muted-foreground">Awaiting</span><strong class="mt-1 block text-xl">{selectedAgent.awaiting_results}</strong></div>
				</div>

				{#if selectedAgent.current_session_id}<section><h3 class="mb-3 text-sm font-semibold">Current activity</h3><div class="flex items-center justify-between gap-4 rounded-xl border bg-primary/5 p-4"><div><p class="text-sm font-medium">Conversation in progress</p><p class="mt-1 max-w-72 truncate text-xs text-muted-foreground">{selectedAgent.current_session_id}</p></div><Button size="sm" variant="outline" onclick={() => onOpenSession(selectedAgent)}>Open conversation</Button></div></section>{/if}

				<section><h3 class="mb-3 text-sm font-semibold">Configuration</h3><dl class="divide-y rounded-xl border bg-background text-sm"><div class="flex justify-between gap-4 px-4 py-3"><dt class="text-muted-foreground">Harness</dt><dd class="text-right font-medium">{selectedAgent.harness_id}</dd></div><div class="flex justify-between gap-4 px-4 py-3"><dt class="text-muted-foreground">Idle timeout</dt><dd class="text-right font-medium">{selectedAgent.idle_timeout_seconds ? `${selectedAgent.idle_timeout_seconds}s` : 'Runtime default'}</dd></div><div class="flex justify-between gap-4 px-4 py-3"><dt class="text-muted-foreground">Harness source</dt><dd class="text-right font-medium">{selectedAgent.has_local_harness ? 'Agent-local' : 'Shared'}</dd></div></dl></section>

				{#if selectedAgent.inference_contexts.length > 0}<section><h3 class="mb-3 text-sm font-semibold">Inference routes</h3><div class="space-y-2">{#each selectedAgent.inference_contexts as context}<div class="flex items-center justify-between gap-4 rounded-xl border bg-background px-4 py-3"><div><p class="text-sm font-medium">{context.id}</p><p class="text-xs text-muted-foreground">{context.provider} / {context.model}</p></div>{#if context.is_default}<Badge variant="secondary">Default route</Badge>{/if}</div>{/each}</div></section>{/if}

				{#if selectedAgent.system_prompt}<section><h3 class="mb-3 text-sm font-semibold">System instruction</h3><p class="whitespace-pre-wrap rounded-xl border bg-muted/20 p-4 text-sm leading-6">{selectedAgent.system_prompt}</p></section>{/if}

				<section><h3 class="mb-3 text-sm font-semibold">Registry health</h3>{#if selectedAgent.issues.length === 0}<div class="flex items-center gap-3 rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4 text-sm"><ShieldCheck class="size-5 text-emerald-600" /><span>No configuration issues reported.</span></div>{:else}<div class="space-y-2">{#each selectedAgent.issues as issue}<div class="rounded-xl border border-destructive/20 bg-destructive/5 p-4"><p class="flex items-center gap-2 text-sm font-medium text-destructive"><CircleAlert class="size-4" />{issue.message}</p><p class="mt-2 break-all text-xs text-muted-foreground">{issue.path}</p></div>{/each}</div>{/if}</section>
			</div>

			<Sheet.Footer class="border-t pt-5 sm:justify-between">
				<Button variant="outline" onclick={() => onControl('reload')} disabled={mutating}><RefreshCw class={mutating ? 'animate-spin' : ''} />Reload</Button>
				<Button variant={selectedAgent.enabled ? 'destructive' : 'default'} onclick={() => availabilityOpen = true} disabled={mutating}>{selectedAgent.enabled ? 'Disable agent' : 'Enable agent'}</Button>
			</Sheet.Footer>
		{/if}
	</Sheet.Content>
</Sheet.Root>

<AlertDialog.Root bind:open={availabilityOpen}>
	<AlertDialog.Content>
		<AlertDialog.Header><AlertDialog.Media><Bot /></AlertDialog.Media><AlertDialog.Title>{selectedAgent?.enabled ? 'Disable this agent?' : 'Enable this agent?'}</AlertDialog.Title><AlertDialog.Description>{selectedAgent?.enabled ? 'New work will no longer be admitted for this agent. Existing runtime reconciliation follows Turin’s agent lifecycle rules.' : 'The agent will become available for new conversations and delegated work.'}</AlertDialog.Description></AlertDialog.Header>
		<AlertDialog.Footer><AlertDialog.Cancel>Cancel</AlertDialog.Cancel><AlertDialog.Action variant={selectedAgent?.enabled ? 'destructive' : 'default'} onclick={() => { onControl(selectedAgent?.enabled ? 'disable' : 'enable'); availabilityOpen = false; }}>{selectedAgent?.enabled ? 'Disable' : 'Enable'}</AlertDialog.Action></AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
