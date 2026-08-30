<script lang="ts">
	import { ArrowRight, Bot, Brain, ListTodo, MessageSquare, Plus } from '@lucide/svelte';
	import type { Agent, Session } from '#lib/api/contracts.js';
	import { Button } from '#lib/components/ui/button/index.js';
	import * as Card from '#lib/components/ui/card/index.js';
	import type { WorkspaceSection } from '#lib/workspace.js';

	let {
		sessions, agents, onCreate, onSelect, onNavigate
	}: {
		sessions: Session[];
		agents: Agent[];
		onCreate: (agentId?: string) => void;
		onSelect: (session: Session) => void;
		onNavigate: (section: WorkspaceSection) => void;
	} = $props();

	let recent = $derived(sessions.slice(0, 6));

	function relativeTime(value: string) {
		const seconds = Math.max(1, Math.floor((Date.now() - Date.parse(value)) / 1000));
		if (seconds < 3600) return `${Math.max(1, Math.floor(seconds / 60))}m ago`;
		if (seconds < 86_400) return `${Math.floor(seconds / 3600)}h ago`;
		return `${Math.floor(seconds / 86_400)}d ago`;
	}
</script>

<div class="h-full overflow-y-auto bg-muted/20">
	<div class="mx-auto flex w-full max-w-[92rem] flex-col gap-6 px-5 py-8 lg:px-8 lg:py-10">
		<header class="flex flex-col justify-between gap-5 rounded-2xl border border-primary/15 bg-gradient-to-br from-primary/10 via-background to-background p-6 sm:flex-row sm:items-end lg:p-8">
			<div>
				<h1 class="font-heading text-3xl font-semibold tracking-tight">What are you working on?</h1>
				<p class="mt-2 text-sm text-muted-foreground">Start a new conversation or continue one that is already in motion.</p>
			</div>
			<Button onclick={() => onCreate(agents[0]?.id)} disabled={agents.length === 0}><Plus />New conversation</Button>
		</header>

		<div class:grid={agents.length > 1} class="gap-4 xl:grid-cols-[minmax(0,1.6fr)_minmax(20rem,0.7fr)]">
			<Card.Root class="shadow-none">
				<Card.Header>
					<Card.Title>Recent conversations</Card.Title>
					<Card.Description>Continue from the latest message.</Card.Description>
					<Card.Action><Button variant="ghost" size="sm" onclick={() => onNavigate('conversations')}>View all<ArrowRight /></Button></Card.Action>
				</Card.Header>
				<Card.Content class="space-y-1">
					{#if recent.length === 0}<div class="grid min-h-56 place-items-center text-sm text-muted-foreground">No conversations yet.</div>{/if}
					{#each recent as session}
						<button class="group flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left transition-colors hover:bg-muted" onclick={() => onSelect(session)}>
							<span class="grid size-9 shrink-0 place-items-center rounded-xl bg-muted transition-colors group-hover:bg-background"><MessageSquare class="size-4 text-muted-foreground" /></span>
							<span class="min-w-0 flex-1"><span class="block truncate text-sm font-medium">{session.title}</span><span class="mt-0.5 block text-xs text-muted-foreground">{agents.find((agent) => agent.id === session.agent_id)?.name ?? session.agent_id}</span></span>
							<span class="text-xs text-muted-foreground">{relativeTime(session.created_at)}</span>
						</button>
					{/each}
				</Card.Content>
			</Card.Root>

			{#if agents.length > 1}<Card.Root class="shadow-none">
				<Card.Header><Card.Title>Start with an agent</Card.Title><Card.Description>Each conversation keeps its own context.</Card.Description></Card.Header>
				<Card.Content class="space-y-2">
					{#each agents as agent}
						<button class="group flex w-full items-center gap-3 rounded-xl border bg-background px-3 py-3 text-left transition-colors hover:bg-muted/60" onclick={() => onCreate(agent.id)}>
							<span class="grid size-9 place-items-center rounded-xl bg-primary/10 text-primary"><Bot class="size-4" /></span>
							<span class="min-w-0 flex-1"><span class="block truncate text-sm font-medium">{agent.name}</span><span class="block truncate text-xs text-muted-foreground">{agent.model}</span></span>
							<ArrowRight class="size-4 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
						</button>
					{/each}
				</Card.Content>
			</Card.Root>{/if}
		</div>

		<div class="grid gap-4 md:grid-cols-2">
			<Card.Root class="shadow-none">
				<Card.Header><span class="mb-2 grid size-9 place-items-center rounded-xl bg-primary/10 text-primary"><ListTodo class="size-4" /></span><Card.Title>Work</Card.Title><Card.Description>Inspect durable queues and the items waiting inside them.</Card.Description><Card.Action><Button variant="outline" size="sm" onclick={() => onNavigate('work')}>Open work<ArrowRight /></Button></Card.Action></Card.Header>
			</Card.Root>
			<Card.Root class="shadow-none">
				<Card.Header><span class="mb-2 grid size-9 place-items-center rounded-xl bg-chart-2/10 text-chart-2"><Brain class="size-4" /></span><Card.Title>Memory</Card.Title><Card.Description>Search durable knowledge and inspect how it is scoped.</Card.Description><Card.Action><Button variant="outline" size="sm" onclick={() => onNavigate('memory')}>Open memory<ArrowRight /></Button></Card.Action></Card.Header>
			</Card.Root>
		</div>
	</div>
</div>
